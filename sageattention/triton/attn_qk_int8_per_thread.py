"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch, math
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    q_scales_rows,
    qo_len, kv_len,
    K_ptrs, K_scale_base,
    V_ptrs, stride_kn, stride_vn, 
    start_m, mask_ptrs, stride_maskn,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    K_WARP: tl.constexpr,
    K_SLICES: tl.constexpr,
):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_block = None
        skip = False
        if mask_ptrs is not None:
            if mask_ptrs.dtype.element_ty == tl.int1:
                mask_block = tl.load(
                    mask_ptrs + start_n * stride_maskn,
                    mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n),
                    other=False
                )
                if tl.max(mask_block) == 0:
                    skip = True
            else:
                mask_block = tl.load(
                    mask_ptrs + start_n * stride_maskn,
                    mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n),
                    other=-1.0e6
                )

        if not skip:
            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs, mask=k_mask)

            col_abs = start_n + offs_n
            k_blk   = col_abs // K_WARP
            grp_in8 = (col_abs % 8) // (8 // K_SLICES)
            k_idx   = k_blk * K_SLICES + grp_in8
            k_scales_cols = tl.load(K_scale_base + k_idx)

            qk = tl.dot(q, k).to(tl.float32)
            qk = 1.4426950408889634 / (HEAD_DIM ** 0.5) * qk * (q_scales_rows[:, None] * k_scales_cols[None, :])

            if mask_block is not None:
                if mask_block.dtype == tl.int1:
                    qk = qk + tl.where(mask_block, 0, -1.0e6)
                else:
                    qk = qk + mask_block
            else:
                qk += tl.where(k_mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
            p = p.to(tl.float16)
            acc += tl.dot(p, v, out_dtype=tl.float16)
            m_i = m_ij

        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out, mask, Lse, 
    stride_qz, stride_qh, stride_qn,
    stride_kz, stride_kh, stride_kn,  
    stride_vz, stride_vh, stride_vn,  
    stride_oz, stride_oh, stride_on, 
    stride_maskz, stride_maskh, stride_maskm, stride_maskn,
    qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
    # sm_scale: tl.float32,
    HEAD_DIM: tl.constexpr,  
    BLOCK_M: tl.constexpr,  
    BLOCK_N: tl.constexpr,  
    STAGE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    Q_TILES_PER_SEQ: int,
    K_TILES_PER_SEQ: int,
    Q_WARP: tl.constexpr,
    Q_SLICES: tl.constexpr,
    K_WARP: tl.constexpr,
    K_SLICES: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_base = Q_scale + (off_z * H + off_h) * Q_TILES_PER_SEQ

    H_kv = H // num_kv_groups
    off_h_kv = off_h // num_kv_groups
    k_scale_base = K_scale + (off_z * H_kv + off_h_kv) * K_TILES_PER_SEQ

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    K_ptrs = K + (off_z * stride_kz + off_h_kv * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    V_ptrs = V + (off_z * stride_vz + off_h_kv * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]

    if mask is None:
        mask_ptrs = None
    else:
        mask_ptrs = mask + (off_z * stride_maskz + off_h * stride_maskh) + offs_m[:, None] * stride_maskm + offs_n[None, :] * stride_maskn

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=(offs_m[:, None] < qo_len))

    row_blk   = offs_m // Q_WARP
    row_slice = (offs_m % Q_WARP) % Q_SLICES
    q_idx     = row_blk * Q_SLICES + row_slice
    q_scales_rows = tl.load(q_scale_base + q_idx)

    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q, q_scales_rows, qo_len, kv_len,
        K_ptrs, k_scale_base, V_ptrs, stride_kn, stride_vn,
        start_m, mask_ptrs, stride_maskn,
        BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n,
        K_WARP, K_SLICES
    )

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))


def forward(q, k, v, q_scale, k_scale, sm_scale, tensor_layout="HND", attn_mask=None, output_dtype=torch.float16, return_lse=False):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1

    Q_WARP = 32
    Q_SLICES = 8
    K_WARP = 64
    K_SLICES = 4

    Q_TILES_PER_SEQ = q_scale.size(-1)
    K_TILES_PER_SEQ = k_scale.size(-1)

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    print(f"q.shape:{q.shape}")
    print(f"k.shape:{k.shape}")
    if attn_mask is not None:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = attn_mask.stride(0), attn_mask.stride(1), attn_mask.stride(2), attn_mask.stride(3)
    else:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = 0, 0, 0, 0

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o, attn_mask, lse,
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage, RETURN_LSE=return_lse,
        Q_TILES_PER_SEQ=Q_TILES_PER_SEQ,
        K_TILES_PER_SEQ=K_TILES_PER_SEQ,
        Q_WARP=Q_WARP, Q_SLICES=Q_SLICES,
        K_WARP=K_WARP, K_SLICES=K_SLICES,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4)

    return o, lse