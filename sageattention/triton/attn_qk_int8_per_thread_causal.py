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
    q_scales_rows,                 # [BLOCK_M] 行向量（新）
    kv_len,
    K_ptrs, K_scale_base,          # K_scale_base 指向 (b, h_kv) 这条带的 scale 起始（新）
    V_ptrs, stride_kn, stride_vn,
    start_m,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    K_WARP: tl.constexpr,          # e.g., 64
    K_SLICES: tl.constexpr,        # e.g., 4
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        # 注意：K/V 指针仍需前移；但 K 的 scale 不再用线性 ptr 累加，而是每次“按列号”计算索引
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # --- 1) 读取 K tile
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_mask)          # [D, BLOCK_N]

        # --- 2) 计算 K 列方向的 per-thread scale 向量 k_scales_cols[BLOCK_N]
        col_abs = start_n + offs_n                # [BLOCK_N] 绝对列号 j
        k_blk   = col_abs // K_WARP               # 第几个 WARPK 子块
        # 每个 WARPK 子块里，以 mod-8 的余数来分组。K_SLICES=4 → 2 个余数组成一个组
        grp_in8 = (col_abs % 8) // (8 // K_SLICES)   # 0..K_SLICES-1
        k_idx   = k_blk * K_SLICES + grp_in8         # [BLOCK_N]
        k_scales_cols = tl.load(K_scale_base + k_idx)  # [BLOCK_N] 标量向量

        # --- 3) 计算分数 qk 并逐元素缩放
        qk = tl.dot(q, k).to(tl.float32)          # [BLOCK_M, BLOCK_N]
        qk = qk * (q_scales_rows[:, None] * k_scales_cols[None, :])

        # --- 4) 因果/越界 mask
        mask = k_mask
        if STAGE == 2:
            mask &= offs_m[:, None] >= (start_n + offs_n[None, :])
        qk += tl.where(mask, 0, float('-inf'))

        # --- 5) 在线 softmax 及累积
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # --- 6) 聚合 V
        v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)

        # --- 7) 前移到下一列块
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out, Lse, sm_scale,
    stride_qz, stride_qh, stride_qn,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_oz, stride_oh, stride_on,
    Q_TILES_PER_SEQ: int,
    K_TILES_PER_SEQ: int,
    Q_WARP: tl.constexpr,
    Q_SLICES: tl.constexpr,
    K_WARP: tl.constexpr,
    K_SLICES: tl.constexpr,
    qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    RETURN_LSE: tl.constexpr
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    # 这里不再用 cdiv(..., BLOCK_M/N) 当作 scale 的第三维长度，
    # 而是用调用方传入的 Q_TILES_PER_SEQ / K_TILES_PER_SEQ。
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

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)

    # === 关键：为本 CTA 的 BLOCK_M 行计算 per-thread Q scales 行向量 ===
    # 行 i 所属的 WARPQ 子块：i // Q_WARP
    # 行 i 在子块内的切片： (i % Q_WARP) % 8  （Q_SLICES=8）
    row_blk   = offs_m // Q_WARP
    row_slice = (offs_m % Q_WARP) % Q_SLICES
    q_idx     = row_blk * Q_SLICES + row_slice              # [BLOCK_M]
    q_scales_rows = tl.load(q_scale_base + q_idx)           # [BLOCK_M]

    # 扫描列块：先“对角线之前”，再“对角线”块
    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q,
        q_scales_rows, kv_len,
        K_ptrs, k_scale_base, V_ptrs, stride_kn, stride_vn,
        start_m,
        BLOCK_M, HEAD_DIM, BLOCK_N,
        4 - STAGE, offs_m, offs_n,
        K_WARP, K_SLICES
    )

    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q,
        q_scales_rows, kv_len,
        K_ptrs, k_scale_base, V_ptrs, stride_kn, stride_vn,
        start_m,
        BLOCK_M, HEAD_DIM, BLOCK_N,
        2, offs_m, offs_n,
        K_WARP, K_SLICES
    )

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))


def forward(q, k, v, q_scale, k_scale, sm_scale, tensor_layout="HND", output_dtype=torch.float16, return_lse=False):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 3

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
    
    assert qo_len == kv_len, "qo_len and kv_len must be equal for causal attention"

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    Q_WARP = 32
    Q_SLICES = 8
    K_WARP = 64
    K_SLICES = 4

    q_tiles = q_scale.size(2)   # ceil_div(qo_len, BLKQ) * (BLKQ//Q_WARP) * 8
    k_tiles = k_scale.size(2)   # ceil_div(kv_len, BLKK) * (BLKK//K_WARP) * 4

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b   )
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o, lse, sm_scale,
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        q_tiles, k_tiles, Q_WARP, Q_SLICES, K_WARP, K_SLICES,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage,  
        RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4)

    return o, lse