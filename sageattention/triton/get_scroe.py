import torch, math
import triton
import triton.language as tl

@triton.jit
def _qk_fwd_inner(
    Scores,
    q,
    q_scales_rows,
    qo_len, kv_len,
    K_ptrs, K_scale_base,
    stride_kn,
    mask_ptrs, stride_maskn,
    start_m,
    stride_sz, stride_sh, stride_sm, stride_sn,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    offs_m: tl.constexpr, offs_n: tl.constexpr,
    K_WARP: tl.constexpr, K_SLICES: tl.constexpr,
    sm_scale: tl.float32,
    APPLY_MASK: tl.constexpr,
):
    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_mask)
        col_abs = start_n + offs_n
        k_blk   = col_abs // K_WARP
        grp_in8 = (col_abs % 8) // (8 // K_SLICES)
        k_idx   = k_blk * K_SLICES + grp_in8
        k_scales_cols = tl.load(K_scale_base + k_idx).to(tl.float32)

        qk = tl.dot(q, k, out_dtype=tl.int32)
        # q_scales_rows = q_scales_rows.to(tl.float32)
        # qk *= q_scales_rows[:, None]
        # qk *= k_scales_cols[None, :]
        # qk *= sm_scale

        # if APPLY_MASK:
        #     if mask_ptrs is not None:
        #         if mask_ptrs.dtype.element_ty == tl.int1:
        #             mask_block = tl.load(
        #                 mask_ptrs + start_n * stride_maskn,
        #                 mask=(offs_m[:, None] < qo_len) & k_mask,
        #                 other=False
        #             )
        #             qk = qk + tl.where(mask_block, 0, -1.0e6)
        #         else:
        #             mask_block = tl.load(
        #                 mask_ptrs + start_n * stride_maskn,
        #                 mask=(offs_m[:, None] < qo_len) & k_mask,
        #                 other=-1.0e6
        #             )
        #             qk = qk + mask_block
        #     else:
        #         qk += tl.where(k_mask, 0, -1.0e6)

        S_ptrs = Scores + offs_m[:, None] * stride_sm + (start_n + offs_n)[None, :] * stride_sn
        tl.store(S_ptrs, qk.to(Scores.type.element_ty),
                 mask=(offs_m[:, None] < qo_len) & k_mask)

        K_ptrs += BLOCK_N * stride_kn


@triton.jit
def _qk_fwd(
    Q, K, Q_scale, K_scale,
    Scores, mask,                
    stride_qz, stride_qh, stride_qn,
    stride_kz, stride_kh, stride_kn,
    stride_sz, stride_sh, stride_sm, stride_sn,  
    stride_maskz, stride_maskh, stride_maskm, stride_maskn,
    qo_len, kv_len,
    H: tl.constexpr, num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    Q_TILES_PER_SEQ: int, K_TILES_PER_SEQ: int,
    Q_WARP: tl.constexpr, Q_SLICES: tl.constexpr,
    K_WARP: tl.constexpr, K_SLICES: tl.constexpr,
    sm_scale: tl.float32,         
    APPLY_MASK: tl.constexpr,     
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

    Scores_base = Scores + (off_z * stride_sz + off_h * stride_sh)

    if mask is None:
        mask_ptrs = None
    else:
        mask_ptrs = mask + (off_z * stride_maskz + off_h * stride_maskh) + \
                    offs_m[:, None] * stride_maskm + offs_n[None, :] * stride_maskn

    q = tl.load(Q_ptrs, mask=(offs_m[:, None] < qo_len))

    row_blk   = offs_m // Q_WARP
    row_slice = (offs_m % Q_WARP) % Q_SLICES
    q_idx     = row_blk * Q_SLICES + row_slice
    q_scales_rows = tl.load(q_scale_base + q_idx)

    _qk_fwd_inner(
        Scores_base, q, q_scales_rows, qo_len, kv_len,
        K_ptrs, k_scale_base, stride_kn,
        mask_ptrs, stride_maskn,
        start_m,
        stride_sz, stride_sh, stride_sm, stride_sn,
        BLOCK_M, HEAD_DIM, BLOCK_N, offs_m, offs_n,
        K_WARP, K_SLICES, sm_scale, APPLY_MASK
    )

def qk_forward(q, k, q_scale, k_scale, sm_scale,
               tensor_layout="HND", attn_mask=None,
                apply_mask=False):
    BLOCK_M, BLOCK_N = 128, 64
    Q_WARP, Q_SLICES = 32, 8
    K_WARP, K_SLICES = 64, 4
    out_dtype=torch.float32
    if tensor_layout == "HND":
        b, h_q, M, D = q.shape
        _, h_kv, N, _ = k.shape
        assert h_q % h_kv == 0
        Scores = torch.empty((b, h_q, M, N), dtype=torch.int32, device=q.device)

        sqz, sqh, sqn = q.stride(0), q.stride(1), q.stride(2)
        skz, skh, skn = k.stride(0), k.stride(1), k.stride(2)
        ssz, ssh, ssm, ssn = Scores.stride(0), Scores.stride(1), Scores.stride(2), Scores.stride(3)
    elif tensor_layout == "NHD":
        b, M, h_q, D = q.shape
        _, N, h_kv, _ = k.shape
        assert h_q % h_kv == 0
        Scores = torch.empty((b, h_q, M, N), dtype=out_dtype, device=q.device)
        
        sqz, sqh, sqn = q.stride(0), q.stride(2), q.stride(1)
        skz, skh, skn = k.stride(0), k.stride(2), k.stride(1)
        ssz, ssh, ssm, ssn = Scores.stride(0), Scores.stride(1), Scores.stride(2), Scores.stride(3)
    else:
        raise ValueError("tensor_layout not supported")

    if attn_mask is not None:
        smz, smh, smm, smn = attn_mask.stride(0), attn_mask.stride(1), attn_mask.stride(2), attn_mask.stride(3)
    else:
        smz = smh = smm = smn = 0

    grid = (triton.cdiv(M, BLOCK_M), h_q, b)
    _qk_fwd[grid](
        q, k, q_scale, k_scale,
        Scores, attn_mask,
        sqz, sqh, sqn,
        skz, skh, skn,
        ssz, ssh, ssm, ssn,
        smz, smh, smm, smn,
        M, N, h_q, h_q // h_kv,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        Q_TILES_PER_SEQ=q_scale.size(-1), K_TILES_PER_SEQ=k_scale.size(-1),
        Q_WARP=Q_WARP, Q_SLICES=Q_SLICES,
        K_WARP=K_WARP, K_SLICES=K_SLICES,
        sm_scale=float(sm_scale),            
        APPLY_MASK=apply_mask,
        num_warps=4 if D == 64 else 8,
        num_stages=3 if D == 64 else 4,
    )

    # ln2 = math.log(2.0)
    # m = Scores.amax(dim=-1, keepdim=True)
    # z = Scores - m
    # z = torch.where(torch.isfinite(z), z, torch.full_like(z, float('-inf')))

    # # base-2 softmax: exp(z * ln2)
    # w = torch.exp(z * ln2)
    # denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    # probs = (w / denom).nan_to_num(0.0, 0.0, 0.0)

    return Scores
    # return probs
