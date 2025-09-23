import os, time, csv
import torch
import torch.nn.functional as F

# ========= 配置 =========
file1 = "/home/tmp/rightqk/qk_int32_first.pt"  # a
file2 = "/home/tmp/triton/qk.pt"               # b
HEAD_INDEX = 0                                  # 要分析的 head 下标
SAVE_CSV = True
CSV_PATH = f"row_cosine_head{HEAD_INDEX}.csv"
PRINT_WORST_K = 10                              # 打印余弦最差的前 K 行
ROW_BLOCK = None                                # 行分块大小；None=整块。内存紧张可设为 1024/2048

# 从 dict/state_dict 中挑张量时优先匹配这些 key 片段
PREFERRED_KEYS = ["qk", "attn", "logits", "t", "q_k", "scores"]

# ========= 基础工具 =========
def stat_file(p):
    try:
        st = os.stat(p)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
        return mtime, st.st_size
    except FileNotFoundError:
        return "NA", -1

def _prod(shape):
    n = 1
    for x in shape:
        n *= int(x)
    return int(n)

def _pick_from_dict(d):
    tensors = [(k, v) for k, v in d.items() if torch.is_tensor(v)]
    if not tensors:
        for k, v in d.items():
            if isinstance(v, dict):
                sub = _pick_from_dict(v)
                if sub is not None:
                    return sub
        return None
    # 1) key 命中优先
    for pref in PREFERRED_KEYS:
        cand = [(k, v) for k, v in tensors if pref in str(k).lower()]
        if cand:
            cand.sort(key=lambda kv: (-(kv[1].dim()==4), -kv[1].numel()))
            return cand[0]
    # 2) 否则优先 4D，再按 numel
    tensors.sort(key=lambda kv: (-(kv[1].dim()==4), -kv[1].numel()))
    return tensors[0]

def load_any_tensor(path):
    """
    兼容：Tensor / [tensor, shape] / dict / TorchScript -> (tensor, source_desc)
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    # 直接 tensor
    if torch.is_tensor(obj):
        return obj, "raw-tensor"

    # [tensor, shape] 打包
    if isinstance(obj, (list, tuple)) and len(obj)==2 and torch.is_tensor(obj[0]) and torch.is_tensor(obj[1]):
        t, shp = obj
        shp = tuple(int(x) for x in shp.tolist())
        if t.dim()==1 and t.numel()==_prod(shp):
            t = t.view(*shp)
        return t, "packed-[tensor,shape]"

    # dict / state_dict
    if isinstance(obj, dict):
        picked = _pick_from_dict(obj)
        if picked is None:
            raise TypeError(f"{path} dict has no tensor to pick.")
        name, t = picked
        return t, f"dict:{name}"

    # TorchScript
    try:
        mod = torch.jit.load(path, map_location="cpu")
        sd = mod.state_dict()
        picked = _pick_from_dict(sd)
        if picked is None:
            raise TypeError(f"{path} torchscript state_dict has no tensor.")
        name, t = picked
        return t, f"torchscript:{name}"
    except Exception:
        pass

    raise TypeError(f"{path} contains unsupported object: {type(obj)}")

def coerce_pair_shapes_to_4d(t1, t2):
    """
    若一边 4D、一边 1D 且 numel 一致，则把 1D 在内存中 view 成 4D；
    返回 (t1, t2, shape4d)
    """
    if t1.numel()!=t2.numel():
        raise AssertionError(f"Numel mismatch: {t1.numel()} vs {t2.numel()}")
    shape4d = None
    if t1.dim()==4:
        shape4d = tuple(t1.shape)
        if t2.dim()==1 and t2.numel()==_prod(shape4d):
            print(f"[fix] reshape t2 {tuple(t2.shape)} -> {shape4d} (RAM only)")
            t2 = t2.view(*shape4d)
    elif t2.dim()==4:
        shape4d = tuple(t2.shape)
        if t1.dim()==1 and t1.numel()==_prod(shape4d):
            print(f"[fix] reshape t1 {tuple(t1.shape)} -> {shape4d} (RAM only)")
            t1 = t1.view(*shape4d)
    return t1, t2, shape4d

# ========= 核心：用 F.cosine_similarity 逐行余弦 =========
def cosine_rows_F(A_mn: torch.Tensor, B_mn: torch.Tensor, row_block=None):
    """
    对 (M,N) 的两矩阵逐行算余弦（vectorized；可分块）。
    - NaN/Inf 屏蔽策略：对任一侧非有限的位置，在两侧都置 0，从而不参与点积和范数
    - 零范数处理：both-zero -> 1.0；one-zero -> NaN
    返回：cos (M,), reason(list[str]：ok/both-zero/one-zero)
    """
    assert A_mn.shape == B_mn.shape and A_mn.dim()==2
    M, N = A_mn.shape

    # 统一 dtype 为 float32（更省内存；F.cosine_similarity 默认 float32）
    A = A_mn.to(torch.float32)
    B = B_mn.to(torch.float32)

    cos_all = torch.empty(M, dtype=torch.float32)
    reasons = ["ok"] * M

    def process_rows(a, b, start):
        # 屏蔽 NaN/Inf：对任一侧非有限的位置在两侧都置 0
        finite = torch.isfinite(a) & torch.isfinite(b)   # (m, n)
        if not finite.all():
            a = torch.where(finite, a, torch.zeros((), dtype=a.dtype))
            b = torch.where(finite, b, torch.zeros((), dtype=b.dtype))

        # 行范数
        n1 = a.norm(dim=1)  # (m,)
        n2 = b.norm(dim=1)
        both_zero = (n1 == 0) & (n2 == 0)
        one_zero  = ((n1 == 0) ^ (n2 == 0))

        # 正常行：直接用 F.cosine_similarity
        safe = ~(both_zero | one_zero)
        if safe.any():
            cos_blk = F.cosine_similarity(a[safe], b[safe], dim=1).clamp(-1, 1)
            cos_all[start:start + safe.numel()][safe.nonzero(as_tuple=False).flatten()] = cos_blk

        # 特殊行
        if both_zero.any():
            idx = both_zero.nonzero(as_tuple=False).flatten() + start
            cos_all[idx] = 1.0
            for i in idx.tolist(): reasons[i] = "both-zero"
        if one_zero.any():
            idx = one_zero.nonzero(as_tuple=False).flatten() + start
            cos_all[idx] = float("nan")
            for i in idx.tolist(): reasons[i] = "one-zero"

    if row_block is None:
        process_rows(A, B, 0)
    else:
        for start in range(0, M, row_block):
            end = min(start + row_block, M)
            process_rows(A[start:end], B[start:end], start)

    return cos_all, reasons

# ========= 主流程 =========
if __name__ == "__main__":
    # 文件信息
    for p in (file1, file2):
        m, s = stat_file(p)
        print(f"[file] {p}  mtime={m}  size={s}")

    # 加载
    t1_raw, src1 = load_any_tensor(file1)
    t2_raw, src2 = load_any_tensor(file2)
    print(f"[raw] t1(src={src1}): dtype={t1_raw.dtype}, shape={tuple(t1_raw.shape)}, stride={tuple(t1_raw.stride())}, contig={t1_raw.is_contiguous()}")
    print(f"[raw] t2(src={src2}): dtype={t2_raw.dtype}, shape={tuple(t2_raw.shape)}, stride={tuple(t2_raw.stride())}, contig={t2_raw.is_contiguous()}")

    # 统一到 4D (1,H,M,N)
    t1, t2, shape4d = coerce_pair_shapes_to_4d(t1_raw, t2_raw)
    if shape4d is None or len(shape4d) != 4 or shape4d[0] != 1:
        raise RuntimeError(f"需要 4D 形状 (1,H,M,N)，当前 shape1={tuple(t1.shape)}, shape2={tuple(t2.shape)}")

    _, H, M, N = t1.shape
    if not (0 <= HEAD_INDEX < H):
        raise ValueError(f"HEAD_INDEX 越界: {HEAD_INDEX}, H={H}")

    print(f"[info] 选择 head={HEAD_INDEX}, 矩阵大小 = {M} x {N}")
    A = t1[0, HEAD_INDEX]  # (M, N), int32
    B = t2[0, HEAD_INDEX]  # (M, N), int32

    # 用 F.cosine_similarity 逐行计算
    cos, reasons = cosine_rows_F(A, B, row_block=ROW_BLOCK)  # cos: (M,), float32
    cos64 = cos.to(torch.float64)  # 统计用

    finite_mask = torch.isfinite(cos64)
    finite_cos = cos64[finite_mask]

    if finite_cos.numel() > 0:
        mean_cos = float(finite_cos.mean().item())
        q10  = float(torch.quantile(finite_cos, 0.10).item())
        q50  = float(torch.quantile(finite_cos, 0.50).item())
        q90  = float(torch.quantile(finite_cos, 0.90).item())
        min_cos = float(finite_cos.min().item())
        max_cos = float(finite_cos.max().item())
    else:
        mean_cos = q10 = q50 = q90 = min_cos = max_cos = float("nan")

    print("====== 逐行余弦相似度（head 级, F.cosine_similarity）======")
    print(f"总行数: {M}")
    print(f"cosine mean: {mean_cos:.6f}, min: {min_cos:.6f}, median: {q50:.6f}, p10: {q10:.6f}, p90: {q90:.6f}, max: {max_cos:.6f}")

    # 最差若干行（按 cos 从小到大，忽略 NaN）
    idxs = torch.arange(M)[finite_mask]
    vals = cos64[finite_mask]
    worst_k = min(PRINT_WORST_K, int(finite_mask.sum().item()))
    if worst_k > 0:
        order = torch.argsort(vals)[:worst_k]
        print(f"------ 最差的 {worst_k} 行 ------")
        for i in range(worst_k):
            row_i = int(idxs[order[i]].item())
            cv = float(vals[order[i]].item())
            print(f"row {row_i:5d}: cos={cv:.6f}, reason={reasons[row_i]}")

    # CSV 导出
    if SAVE_CSV:
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_idx", "cos_sim", "reason"])
            for i in range(M):
                w.writerow([i, float(cos[i].item()) if cos[i]==cos[i] else "nan", reasons[i]])
        print(f"[csv] per-row cosine written to {CSV_PATH}")

    # ========= 一键对照诊断 =========
    print("\n====== 对照诊断 ======")
    with torch.no_grad():
        # 1) 行级比例修正后的相对 L1 误差（检查是否“每行只差一个缩放系数”）
        A32 = A.to(torch.float32)
        B32 = B.to(torch.float32)
        denom = (B32 * B32).sum(dim=1)               # (M,)
        num   = (A32 * B32).sum(dim=1)               # (M,)
        alpha = torch.where(denom > 0, num / denom, torch.zeros_like(denom))  # (M,)
        A_hat = torch.where(denom[:, None] > 0, alpha[:, None] * B32, torch.zeros_like(B32))
        # 行内相对 L1：对 denom==0 的行，分母+eps避免除零
        eps = 1e-12
        row_rel_err = (A32 - A_hat).abs().sum(dim=1) / (B32.abs().sum(dim=1) + eps)
        # 过滤极端 NaN（理论上不会出现）
        row_rel_err = row_rel_err[torch.isfinite(row_rel_err)]
        if row_rel_err.numel() > 0:
            print(f"[diag] 行级比例修正后的相对L1误差：mean={row_rel_err.mean().item():.6e}, "
                  f"p90={torch.quantile(row_rel_err, 0.90).item():.6e}, "
                  f"max={row_rel_err.max().item():.6e}")
        else:
            print("[diag] 行级比例修正后的相对L1误差：NA（无有效行）")

        # 2) 扁平后的整 head 余弦（float64）
        a_flat = A.to(torch.float64).reshape(-1)
        b_flat = B.to(torch.float64).reshape(-1)
        num_f = torch.dot(a_flat, b_flat)
        den_f = a_flat.norm() * b_flat.norm()
        cos_flat = float((num_f / den_f).clamp(-1, 1).item())
        print(f"[diag] 扁平后的整head余弦: {cos_flat:.6f}")

        # 3) 按行归一后再扁平的余弦（消除行间尺度差）
        An = A32 / (A32.norm(dim=1, keepdim=True) + 1e-12)
        Bn = B32 / (B32.norm(dim=1, keepdim=True) + 1e-12)
        af = An.to(torch.float64).reshape(-1)
        bf = Bn.to(torch.float64).reshape(-1)
        cos_flat_row_normed = float((torch.dot(af, bf) / (af.norm() * bf.norm())).clamp(-1, 1).item())
        print(f"[diag] 按行归一后再扁平的余弦: {cos_flat_row_normed:.6f}")

        # 4) 快速相等/零位点核对
        eq_ratio = (A == B).to(torch.float64).mean().item()
        a0b1 = ((A == 0) & (B != 0)).to(torch.float64).mean().item()
        b0a1 = ((B == 0) & (A != 0)).to(torch.float64).mean().item()
        print(f"[diag] equal%={eq_ratio*100:.4f}%, a==0&b!=0%={a0b1*100:.4f}%, b==0&a!=0%={b0a1*100:.4f}%")

        # 5) F.cosine_similarity 直接对整 head（(1, M*N)）一次
        cos_one_shot = F.cosine_similarity(A32.view(1, -1), B32.view(1, -1), dim=1).clamp(-1, 1).item()
        print(f"[diag] F.cosine_similarity(整head一次) = {cos_one_shot:.6f}")
