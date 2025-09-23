import os, time, csv
import torch

# ========= 配置 =========
file1 = "/home/tmp/rightqk/qk_int32_first.pt"  # a
file2 = "/home/tmp/triton/qk.pt"               # b
CHUNK_ELEMS = 8_000_000                        # 分块大小（按内存调）
REL_THRESH = 0.002                             # 相对误差阈值（None 表示不统计相对误差）
EPS = 1e-8
ONLY_PRINT_TOP = None  # 例如设为 5 只打印“最差”的前 5 个 head；None 打印所有 head

# CSV 导出
WRITE_CSV = True
PER_HEAD_CSV = "per_head_metrics.csv"
GLOBAL_CSV   = "global_summary.csv"

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
    """从 dict 里挑一个 tensor：优先 key 命中 + 4D + 最大 numel。"""
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
    """通吃：Tensor / [tensor,shape] / dict / TorchScript -> (tensor, source_desc)"""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(obj):
        return obj, "raw-tensor"
    if isinstance(obj, (list, tuple)) and len(obj)==2 and torch.is_tensor(obj[0]) and torch.is_tensor(obj[1]):
        t, shp = obj
        shp = tuple(int(x) for x in shp.tolist())
        if t.dim()==1 and t.numel()==_prod(shp):
            t = t.view(*shp)
        return t, "packed-[tensor,shape]"
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
    """若一边 4D、一边 1D 且 numel 一致，则把 1D 在内存中 view 成 4D；返回 (t1, t2, shape4d)"""
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

# ========= 逐 head 统计 =========
def head_stats(t1_4d, t2_4d, rel_thresh=REL_THRESH, chunk_elems=8_000_000):
    """t1_4d/t2_4d: 形状必须是 (1, H, M, N)"""
    assert t1_4d.dim()==4 and t2_4d.dim()==4 and t1_4d.shape==t2_4d.shape and t1_4d.shape[0]==1
    _, H, M, N = t1_4d.shape
    results = []  # 每个 head 的统计字典

    for h in range(H):
        a4 = t1_4d[0, h]
        b4 = t2_4d[0, h]
        a = a4.reshape(-1)
        b = b4.reshape(-1)
        N_head = a.numel()

        total_valid = 0
        equal_count = 0
        zero_b_nonzero_a = 0
        zero_a_nonzero_b = 0
        nan1 = nan2 = inf1 = inf2 = 0

        rel_over = 0
        rel_sum = 0.0
        rel_max = 0.0

        dot64 = torch.zeros((), dtype=torch.float64)
        n1sq64 = torch.zeros((), dtype=torch.float64)
        n2sq64 = torch.zeros((), dtype=torch.float64)

        for start in range(0, N_head, chunk_elems):
            end = min(start + chunk_elems, N_head)
            av = a[start:end]
            bv = b[start:end]

            nan1 += torch.isnan(av).sum().item()
            nan2 += torch.isnan(bv).sum().item()
            inf1 += torch.isinf(av).sum().item()
            inf2 += torch.isinf(bv).sum().item()

            mask = torch.isfinite(av) & torch.isfinite(bv)
            if not mask.any():
                continue
            av = av[mask]
            bv = bv[mask]
            total_valid += av.numel()

            equal_count      += (av == bv).sum().item()
            zero_b_nonzero_a += ((bv == 0) & (av != 0)).sum().item()
            zero_a_nonzero_b += ((av == 0) & (bv != 0)).sum().item()

            if rel_thresh is not None:
                rel = (av - bv).abs() / (bv.abs() + EPS)
                rel_over += (rel > rel_thresh).sum().item()
                rel_sum  += rel.sum().item()
                m = rel.max().item()
                if m > rel_max: rel_max = m

            av64 = av.to(torch.float64)
            bv64 = bv.to(torch.float64)
            dot64 += torch.dot(av64, bv64)
            n1sq64 += (av64 * av64).sum()
            n2sq64 += (bv64 * bv64).sum()

        if n1sq64 == 0 and n2sq64 == 0:
            cos_sim = 1.0; reason = "both zero after filtering"
        elif n1sq64 == 0 or n2sq64 == 0:
            cos_sim = float("nan"); reason = "one zero-norm after filtering"
        else:
            den = torch.sqrt(n1sq64) * torch.sqrt(n2sq64)
            cos_sim = float((dot64 / den).clamp(-1, 1).item()); reason = "ok"

        res = {
            "head": h,
            "total": int(N_head),
            "valid": int(total_valid),
            "nan1": int(nan1), "inf1": int(inf1),
            "nan2": int(nan2), "inf2": int(inf2),
            "eq_ratio": (equal_count/total_valid if total_valid>0 else float("nan")),
            "b0_a1_ratio": (zero_b_nonzero_a/total_valid if total_valid>0 else float("nan")),
            "a0_b1_ratio": (zero_a_nonzero_b/total_valid if total_valid>0 else float("nan")),
            "rel_over_ratio": (rel_over/total_valid if (rel_thresh is not None and total_valid>0) else None),
            "rel_mean": (rel_sum/total_valid if (rel_thresh is not None and total_valid>0) else None),
            "rel_max": (rel_max if rel_thresh is not None else None),
            "cos_sim": cos_sim,
            "reason": reason,
        }
        results.append(res)
    return results

def print_head_table(results, rel_thresh=REL_THRESH, top=None):
    def fmt_pct(x): return f"{x*100:.4f}%" if x==x else "nan"
    header = ["head","valid","eq%","b==0&a!=0%","a==0&b!=0%","cos_sim"]
    use_rel = rel_thresh is not None
    if use_rel:
        header = ["head","valid",f"rel>{rel_thresh}%","rel_mean","rel_max","eq%","b==0&a!=0%","a==0&b!=0%","cos_sim"]
    print("====== per-head ======")
    print("\t".join(header))

    rows = []
    for r in results:
        if use_rel:
            row = [
                f"{r['head']}",
                f"{r['valid']}",
                f"{fmt_pct(r['rel_over_ratio'])}",
                f"{r['rel_mean']:.6e}" if r['rel_mean'] is not None else "NA",
                f"{r['rel_max']:.6e}"  if r['rel_max']  is not None else "NA",
                f"{fmt_pct(r['eq_ratio'])}",
                f"{fmt_pct(r['b0_a1_ratio'])}",
                f"{fmt_pct(r['a0_b1_ratio'])}",
                f"{r['cos_sim']:.6f}" if r['cos_sim']==r['cos_sim'] else "nan",
            ]
        else:
            row = [
                f"{r['head']}",
                f"{r['valid']}",
                f"{fmt_pct(r['eq_ratio'])}",
                f"{fmt_pct(r['b0_a1_ratio'])}",
                f"{fmt_pct(r['a0_b1_ratio'])}",
                f"{r['cos_sim']:.6f}" if r['cos_sim']==r['cos_sim'] else "nan",
            ]
        rows.append(row)

    if top is not None:
        if use_rel:
            rows.sort(key=lambda rr: -float(rr[2][:-1]))  # 按 rel_over% 降序
        # 否则就不排序，直接截断
        rows = rows[:top]

    for row in rows:
        print("\t".join(row))
    print()

def print_global_from_heads(results, rel_thresh=REL_THRESH):
    tot_valid = sum(r["valid"] for r in results)
    if tot_valid == 0:
        print("====== 全局（由 per-head 聚合）======")
        print("有效元素为 0")
        return

    def wavg(key):
        num = 0.0
        den = 0
        for r in results:
            val = r[key]
            if val is None or val!=val:  # None 或 NaN
                continue
            num += val * r["valid"]
            den += r["valid"]
        return (num / den) if den>0 else float("nan")

    eq_ratio = wavg("eq_ratio")
    b0a1_ratio = wavg("b0_a1_ratio")
    a0b1_ratio = wavg("a0_b1_ratio")
    cos_sim = sum((r["cos_sim"] if r["cos_sim"]==r["cos_sim"] else 0.0) * r["valid"] for r in results) / tot_valid

    print("====== 全局（由 per-head 聚合）======")
    print(f"有效元素个数: {tot_valid}")
    print(f"相等元素占比: {eq_ratio*100:.4f}%")
    print(f"b==0 且 a!=0 占比: {b0a1_ratio*100:.4f}%")
    print(f"a==0 且 b!=0 占比: {a0b1_ratio*100:.4f}%")
    if rel_thresh is not None:
        rel_over_ratio = wavg("rel_over_ratio")
        rel_mean = wavg("rel_mean")
        rel_max = max((r["rel_max"] for r in results if r["rel_max"] is not None), default=float("nan"))
        print(f"相对误差 > {rel_thresh:.4f} 的比例: {rel_over_ratio*100:.4f}%")
        print(f"平均相对误差: {rel_mean:.6e}")
        print(f"最大相对误差: {rel_max:.6e}")
    print(f"余弦相似度（加权）: {cos_sim:.6f}")
    print()

def write_csv_per_head(results, path, rel_thresh=REL_THRESH):
    use_rel = rel_thresh is not None
    if use_rel:
        fields = ["head","valid","eq_ratio","b0_a1_ratio","a0_b1_ratio","rel_over_ratio","rel_mean","rel_max","cos_sim","nan1","inf1","nan2","inf2"]
    else:
        fields = ["head","valid","eq_ratio","b0_a1_ratio","a0_b1_ratio","cos_sim","nan1","inf1","nan2","inf2"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, None) for k in fields}
            w.writerow(row)
    print(f"[csv] per-head metrics written to {path}")

def write_csv_global(results, path, rel_thresh=REL_THRESH):
    tot_valid = sum(r["valid"] for r in results)
    def wavg(key):
        num = 0.0; den = 0
        for r in results:
            val = r[key]
            if val is None or val!=val:  # None 或 NaN
                continue
            num += val * r["valid"]; den += r["valid"]
        return (num/den) if den>0 else float("nan")
    eq_ratio = wavg("eq_ratio")
    b0a1_ratio = wavg("b0_a1_ratio")
    a0b1_ratio = wavg("a0_b1_ratio")
    rel_over_ratio = wavg("rel_over_ratio") if rel_thresh is not None else None
    rel_mean = wavg("rel_mean") if rel_thresh is not None else None
    rel_max = max((r["rel_max"] for r in results if r["rel_max"] is not None), default=float("nan")) if rel_thresh is not None else None
    cos_sim = sum((r["cos_sim"] if r["cos_sim"]==r["cos_sim"] else 0.0) * r["valid"] for r in results) / (tot_valid if tot_valid>0 else 1)

    fields = ["valid_total","eq_ratio","b0_a1_ratio","a0_b1_ratio","cos_sim"]
    row = {
        "valid_total": tot_valid,
        "eq_ratio": eq_ratio,
        "b0_a1_ratio": b0a1_ratio,
        "a0_b1_ratio": a0b1_ratio,
        "cos_sim": cos_sim,
    }
    if rel_thresh is not None:
        fields[1:1] = ["rel_over_ratio","rel_mean","rel_max"]  # 插入在 eq_ratio 前
        row.update({
            "rel_over_ratio": rel_over_ratio,
            "rel_mean": rel_mean,
            "rel_max": rel_max,
        })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    print(f"[csv] global summary written to {path}")

# ========= 主流程 =========
if __name__ == "__main__":
    m1, s1 = stat_file(file1)
    m2, s2 = stat_file(file2)
    print(f"[file] {file1}  mtime={m1}  size={s1}")
    print(f"[file] {file2}  mtime={m2}  size={s2}")

    t1_raw, src1 = load_any_tensor(file1)
    t2_raw, src2 = load_any_tensor(file2)

    print(f"[raw] t1(src={src1}): dtype={t1_raw.dtype}, shape={tuple(t1_raw.shape)}, stride={tuple(t1_raw.stride())}, contig={t1_raw.is_contiguous()}")
    print(f"[raw] t2(src={src2}): dtype={t2_raw.dtype}, shape={tuple(t2_raw.shape)}, stride={tuple(t2_raw.stride())}, contig={t2_raw.is_contiguous()}")

    # 若一边是 1D，另一边是 4D，则把 1D view 成 4D（仅内存）
    t1, t2, shape4d = coerce_pair_shapes_to_4d(t1_raw, t2_raw)
    if shape4d is None:
        raise RuntimeError("两侧至少有一侧需要是 4D (1,H,M,N) 才能逐 head 分析。")

    # 逐 head 统计
    results = head_stats(t1, t2, rel_thresh=REL_THRESH, chunk_elems=CHUNK_ELEMS)
    print_head_table(results, rel_thresh=REL_THRESH, top=ONLY_PRINT_TOP)

    # 全局（由 per-head 聚合）
    print_global_from_heads(results, rel_thresh=REL_THRESH)

    # CSV 导出
    if WRITE_CSV:
        write_csv_per_head(results, PER_HEAD_CSV, rel_thresh=REL_THRESH)
        write_csv_global(results, GLOBAL_CSV,   rel_thresh=REL_THRESH)
