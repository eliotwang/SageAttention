import os, time, csv
import torch
import torch.nn.functional as F

# ========= 配置 =========
file1 = "/home/tmp/rightqk/qk_int32_first.pt"  # A 源
file2 = "/home/tmp/triton/o.pt"               # B 源
HEAD_INDEX = 0                                  # 要分析的 head 下标
ROW_BLOCK = None                                # 行分块大小；None=整块
PRINT_WORST_K = 10                              # 打印最差 K 行

# 输出文件
SAVE_CSV = True
COS_CSV        = f"row_cosine_head{HEAD_INDEX}.csv"
ALPHA_CSV      = f"row_alpha_head{HEAD_INDEX}.csv"
ZERO_SEG_CSV   = f"row_zero_segments_head{HEAD_INDEX}.csv"

# 从 dict/state_dict 中挑张量时优先匹配这些 key 片段
PREFERRED_KEYS = ["qk", "attn", "logits", "t", "q_k", "scores"]

# ========= 工具 =========
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

# ========= 逐行余弦（F.cosine_similarity） =========
def cosine_rows_F(A_mn: torch.Tensor, B_mn: torch.Tensor, row_block=None):
    """
    对 (M,N) 的两矩阵逐行算余弦（vectorized；可分块）。
    - NaN/Inf 屏蔽：任一侧非有限 -> 两侧置 0（不参与点积与范数）
    - 零范数处理：both-zero -> 1.0；one-zero -> NaN
    返回：cos (M,), reasons(list[str]：ok/both-zero/one-zero)
    """
    assert A_mn.shape == B_mn.shape and A_mn.dim()==2
    M, _ = A_mn.shape
    A = A_mn.to(torch.float32)
    B = B_mn.to(torch.float32)
    cos_all = torch.empty(M, dtype=torch.float32)
    reasons = ["ok"] * M

    def process_rows(a, b, start):
        m = a.size(0)
        finite = torch.isfinite(a) & torch.isfinite(b)
        if not finite.all():
            a = torch.where(finite, a, torch.zeros((), dtype=a.dtype))
            b = torch.where(finite, b, torch.zeros((), dtype=b.dtype))
        n1 = a.norm(dim=1)
        n2 = b.norm(dim=1)
        both_zero = (n1 == 0) & (n2 == 0)
        one_zero  = ((n1 == 0) ^ (n2 == 0))
        safe = ~(both_zero | one_zero)
        if safe.any():
            safe_idx = torch.arange(m)[safe]
            cos_blk = F.cosine_similarity(a[safe], b[safe], dim=1).clamp(-1, 1)
            cos_all[start + safe_idx] = cos_blk
        if both_zero.any():
            idx = torch.arange(m)[both_zero] + start
            cos_all[idx] = 1.0
            for i in idx.tolist(): reasons[i] = "both-zero"
        if one_zero.any():
            idx = torch.arange(m)[one_zero] + start
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

    # —— 逐行余弦 —— #
    cos, reasons = cosine_rows_F(A, B, row_block=ROW_BLOCK)
    cos64 = cos.to(torch.float64)
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

    print("====== 逐行余弦相似度（head 级）======")
    print(f"总行数: {M}")
    print(f"cosine mean: {mean_cos:.6f}, min: {min_cos:.6f}, median: {q50:.6f}, p10: {q10:.6f}, p90: {q90:.6f}, max: {max_cos:.6f}")
    worst_k = min(PRINT_WORST_K, int(finite_mask.sum().item()))
    if worst_k > 0:
        idxs = torch.arange(M)[finite_mask]
        vals = finite_cos
        order = torch.argsort(vals)[:worst_k]
        print(f"------ 余弦最差的 {worst_k} 行 ------")
        for i in range(worst_k):
            row_i = int(idxs[order[i]].item())
            cv = float(vals[order[i]].item())
            print(f"row {row_i:5d}: cos={cv:.6f}, reason={reasons[row_i]}")

    if SAVE_CSV:
        with open(COS_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_idx", "cos_sim", "reason"])
            for i in range(M):
                w.writerow([i, float(cos[i].item()) if cos[i]==cos[i] else "nan", reasons[i]])
        print(f"[csv] per-row cosine written to {COS_CSV}")

    # ========= 行尺度 αᵢ 及诊断 =========
    print("\n====== 行尺度 αᵢ 诊断 ======")
    A32 = A.to(torch.float32); B32 = B.to(torch.float32)
    denom = (B32 * B32).sum(dim=1)                # (M,)
    num   = (A32 * B32).sum(dim=1)                # (M,)
    alpha = torch.where(denom > 0, num / denom, torch.zeros_like(denom))  # (M,)

    amask = torch.isfinite(alpha)
    alpha_f = alpha[amask]
    M_eff = int(amask.sum().item())
    if M_eff > 0:
        a_mean = float(alpha_f.mean().item())
        a_min  = float(alpha_f.min().item())
        a_max  = float(alpha_f.max().item())
        a_q10  = float(torch.quantile(alpha_f, 0.10).item())
        a_q50  = float(torch.quantile(alpha_f, 0.50).item())
        a_q90  = float(torch.quantile(alpha_f, 0.90).item())
        a_std  = float(alpha_f.std(unbiased=False).item())
        a_cv   = a_std / (abs(a_mean) + 1e-12)
        neg_pct = float((alpha_f < 0).to(torch.float64).mean().item()) * 100.0

        print(f"[alpha] count={M_eff}/{M}, mean={a_mean:.6f}, std={a_std:.6f}, CV={a_cv:.6f}")
        print(f"[alpha] min={a_min:.6f}, p10={a_q10:.6f}, median={a_q50:.6f}, p90={a_q90:.6f}, max={a_max:.6f}")
        print(f"[alpha] negative ratio={neg_pct:.4f}%")
        med = a_q50
        dev = (alpha_f - med).abs()
        k = min(PRINT_WORST_K, M_eff)
        worst_idx = torch.argsort(dev, descending=True)[:k]
        print(f"------ |alpha - median| 最大的 {k} 行 ------")
        full_idx = torch.arange(M)[amask]
        for i in range(k):
            ridx = int(full_idx[worst_idx[i]].item())
            aval = float(alpha_f[worst_idx[i]].item())
            print(f"row {ridx:5d}: alpha={aval:.6f}, |alpha-med|={float(dev[worst_idx[i]].item()):.6f}")
    else:
        print("[alpha] 无有效 alpha（大概率整行 B 为 0）。")

    # 导出 ALPHA_CSV（简版：行号 + alpha）
    if SAVE_CSV:
        with open(ALPHA_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_idx", "alpha"])
            for i in range(M):
                ai = float(alpha[i].item()) if torch.isfinite(alpha[i]) else "nan"
                w.writerow([i, ai])
        print(f"[csv] per-row alpha written to {ALPHA_CSV}")

    # ========= A 中整行全 0 行与连续区间统计 =========
    print("\n====== A 中整行全 0 统计 ======")
    row_zero = (A == 0).all(dim=1)                # (M,), True 表示该行全 0
    num_zero_rows = int(row_zero.sum().item())
    first_zero_idx = int(torch.nonzero(row_zero, as_tuple=False)[0].item()) if num_zero_rows > 0 else -1
    last_zero_idx  = int(torch.nonzero(row_zero, as_tuple=False)[-1].item()) if num_zero_rows > 0 else -1
    ratio_zero = num_zero_rows / M if M > 0 else float("nan")

    print(f"总行数: {M}")
    print(f"整行全 0 行数: {num_zero_rows} ({ratio_zero:.2%})")
    if num_zero_rows > 0:
        print(f"第一条全 0 行索引: {first_zero_idx}")
        print(f"最后一条全 0 行索引: {last_zero_idx}")
    else:
        print("未发现整行全 0 的行。")

    # 连续全 0 区间 [start, end]（含 end）
    segments = []
    in_seg = False
    start = -1
    for i in range(M):
        if row_zero[i].item() and not in_seg:
            in_seg = True
            start = i
        if in_seg and (i == M-1 or not row_zero[i+1].item()):
            end = i
            segments.append((start, end, end - start + 1))
            in_seg = False

    print("\n------ 连续全 0 区间（最多展示前 20 个）------")
    for s, (st, ed, ln) in enumerate(segments[:20], 1):
        print(f"{s:2d}. [{st}, {ed}]  长度={ln}")
    print(f"区间总数: {len(segments)}")

    # 写 CSV
    if SAVE_CSV:
        with open(ZERO_SEG_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seg_idx", "row_start", "row_end", "length"])
            for si, (st, ed, ln) in enumerate(segments, 1):
                w.writerow([si, st, ed, ln])
        print(f"[csv] zero-row segments written to {ZERO_SEG_CSV}")

    # ========= 整 head 复核（便于与旧统计对齐） =========
    a_flat = A.to(torch.float64).reshape(-1)
    b_flat = B.to(torch.float64).reshape(-1)
    cos_flat = float((torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())).clamp(-1, 1).item())
    An = A.to(torch.float32) / (A.to(torch.float32).norm(dim=1, keepdim=True) + 1e-12)
    Bn = B.to(torch.float32) / (B.to(torch.float32).norm(dim=1, keepdim=True) + 1e-12)
    af = An.to(torch.float64).reshape(-1)
    bf = Bn.to(torch.float64).reshape(-1)
    cos_flat_row_normed = float((torch.dot(af, bf) / (af.norm() * bf.norm())).clamp(-1, 1).item())

    print("\n====== 整 head 复核 ======")
    print(f"[flat] 扁平余弦: {cos_flat:.6f}")
    print(f"[flat] 按行归一后扁平余弦: {cos_flat_row_normed:.6f}")
