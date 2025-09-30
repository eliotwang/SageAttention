import os, time, csv
import torch

# ========= 配置 =========
file1 = "/home/tmp/newv/vafter_first.pt"  # a
file2 = "/home/tmp/newv/vbefore.pt"               # b
HEAD_INDEX = 0                                  # 要比较的 head
SAVE_CSV = True
CSV_PATH = f"row_cosine_head{HEAD_INDEX}.csv"
PRINT_WORST_K = 10                               # 打印余弦最差的前 K 行（按 cos 从小到大）
CHUNK_ROWS = None                                # 可选：矢量化时分块行数，None=整块；如内存紧，可设 1024/2048

# 从 dict/state_dict 中优先挑选的 key 片段
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
    for pref in PREFERRED_KEYS:
        cand = [(k, v) for k, v in tensors if pref in str(k).lower()]
        if cand:
            cand.sort(key=lambda kv: (-(kv[1].dim()==4), -kv[1].numel()))
            return cand[0]
    tensors.sort(key=lambda kv: (-(kv[1].dim()==4), -kv[1].numel()))
    return tensors[0]

def load_any_tensor(path):
    """
    通吃：Tensor / [tensor,shape] / dict / TorchScript -> (tensor, source_desc)
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

# ========= 余弦计算：矢量化快速路径 & 安全逐行 =========
def row_cosine_vectorized(A_mn: torch.Tensor, B_mn: torch.Tensor, chunk_rows=None):
    """
    矢量化逐行余弦：要求输入不含 NaN/Inf。
    - 使用 float64 累加（通过 sum(..., dtype=torch.float64) 实现）
    - 对零范数行进行稳健处理
    返回：cos (M,), reason (list[str]) 仅对零范数行填充，否则 'ok'
    """
    assert A_mn.shape == B_mn.shape and A_mn.dim()==2
    M, N = A_mn.shape
    cos = torch.empty(M, dtype=torch.float64)
    reasons = ["ok"] * M

    def process_block(a, b, start):
        # 转为 float32 再用 float64 累加可以少占一半内存；直接一口气 float64 也行但更占内存
        a32 = a.to(torch.float32)
        b32 = b.to(torch.float32)
        dot = (a32 * b32).sum(dim=1, dtype=torch.float64)          # (m,)
        n1s = (a32 * a32).sum(dim=1, dtype=torch.float64)          # (m,)
        n2s = (b32 * b32).sum(dim=1, dtype=torch.float64)          # (m,)
        n1 = torch.sqrt(n1s)
        n2 = torch.sqrt(n2s)

        both_zero = (n1 == 0) & (n2 == 0)
        one_zero  = ((n1 == 0) ^ (n2 == 0))

        # 正常行
        denom = n1 * n2
        safe = ~(both_zero | one_zero)
        if safe.any():
            cos_block = (dot[safe] / denom[safe]).clamp(-1, 1)
            cos[start:start+safe.numel()][safe.nonzero(as_tuple=False).flatten()] = cos_block

        # 特殊行
        if both_zero.any():
            idx = both_zero.nonzero(as_tuple=False).flatten() + start
            cos[idx] = 1.0
            for i in idx.tolist(): reasons[i] = "both-zero"
        if one_zero.any():
            idx = one_zero.nonzero(as_tuple=False).flatten() + start
            cos[idx] = float("nan")
            for i in idx.tolist(): reasons[i] = "one-zero"

        # 给 safe 的 reason 标记为 ok（默认就是 ok）
        return

    if chunk_rows is None:
        if not (torch.isfinite(A_mn).all() and torch.isfinite(B_mn).all()):
            raise ValueError("Found NaN/Inf, vectorized path requires finite inputs.")
        process_block(A_mn, B_mn, 0)
    else:
        # 分块（按行切块），适合内存吃紧时
        # 这里也要求整块都有限；如果需要块内过滤，可退回安全逐行版本
        if not (torch.isfinite(A_mn).all() and torch.isfinite(B_mn).all()):
            raise ValueError("Found NaN/Inf, vectorized path requires finite inputs.")
        for start in range(0, M, chunk_rows):
            end = min(start + chunk_rows, M)
            process_block(A_mn[start:end], B_mn[start:end], start)

    return cos, reasons

def safe_cosine_row(a_row: torch.Tensor, b_row: torch.Tensor):
    """
    安全逐行版本：过滤 NaN/Inf，float64 计算，零范数稳健处理。
    返回 (cos, reason, valid_elems)
    """
    mask = torch.isfinite(a_row) & torch.isfinite(b_row)
    av = a_row[mask]; bv = b_row[mask]
    valid = av.numel()
    if valid == 0:
        return float("nan"), "no-valid", 0
    av64 = av.to(torch.float64); bv64 = bv.to(torch.float64)
    n1 = torch.linalg.norm(av64); n2 = torch.linalg.norm(bv64)
    if n1 == 0 and n2 == 0: return 1.0, "both-zero", valid
    if n1 == 0 or n2 == 0:  return float("nan"), "one-zero", valid
    cos = (torch.dot(av64, bv64) / (n1 * n2)).clamp(-1, 1).item()
    return float(cos), "ok", valid

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
    A = t1[0, HEAD_INDEX]  # (M, N)
    B = t2[0, HEAD_INDEX]  # (M, N)

    # —— 先尝试矢量化快速路径 —— #
    use_vector = False
    try:
        if torch.isfinite(A).all() and torch.isfinite(B).all():
            use_vector = True
    except RuntimeError:
        # 某些 dtype 不支持 isfinite（这里是 int32，没问题）；留作兜底
        use_vector = False

    if use_vector:
        try:
            cos_vec, reasons = row_cosine_vectorized(A, B, chunk_rows=CHUNK_ROWS)
            row_cos = [(i, float(cos_vec[i].item()), reasons[i], N) for i in range(M)]
            print("[path] vectorized")
        except ValueError:
            # 遇到 NaN/Inf，回退逐行安全路径
            use_vector = False

    # —— 回退到安全逐行路径（处理 NaN/Inf 的情况） —— #
    if not use_vector:
        print("[path] safe-per-row")
        row_cos = []
        total_valid = 0
        nan_rows = both_zero_rows = one_zero_rows = 0
        for i in range(M):
            cos, reason, valid = safe_cosine_row(A[i], B[i])
            row_cos.append((i, cos, reason, valid))
            total_valid += valid
            if (i+1) % 1000 == 0:
                print(f"[progress] rows processed: {i+1}/{M}")

    # —— 汇总统计 —— #
    cos_vals = [c for (_, c, _, _) in row_cos if c == c]  # 过滤 NaN
    if len(cos_vals) > 0:
        tcos = torch.tensor(cos_vals, dtype=torch.float64)
        mean_cos = float(tcos.mean().item())
        min_cos = float(tcos.min().item())
        max_cos = float(tcos.max().item())
        q10 = float(torch.quantile(tcos, 0.10).item())
        q50 = float(torch.quantile(tcos, 0.50).item())
        q90 = float(torch.quantile(tcos, 0.90).item())

    else:
        mean_cos = min_cos = max_cos = q10 = q50 = q90 = float("nan")

    print("====== 逐行余弦相似度（head 级）======")
    print(f"总行数: {M}")
    print(f"cosine mean: {mean_cos:.6f}, min: {min_cos:.6f}, median: {q50:.6f}, p10: {q10:.6f}, p90: {q90:.6f}, max: {max_cos:.6f}")

    # 最差若干行
    worst_k = min(PRINT_WORST_K, M)
    ranked = sorted([(i, c, r, v) for (i, c, r, v) in row_cos if c == c], key=lambda x: x[1])
    print(f"------ 最差的 {worst_k} 行 ------")
    for idx, (i, cos, reason, valid) in enumerate(ranked[:worst_k]):
        print(f"row {i:5d}: cos={cos:.6f}, reason={reason}, valid={valid}")

    # —— CSV 导出 —— #
    if SAVE_CSV:
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_idx", "cos_sim", "reason", "valid_elems"])
            for idx, cos, reason, valid in row_cos:
                w.writerow([idx, cos, reason, valid])
        print(f"[csv] per-row cosine written to {CSV_PATH}")
