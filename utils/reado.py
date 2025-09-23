#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, argparse, time
import torch
import numpy as np

# ============== 参数 ==============
PREFERRED_KEYS = ["o", "out", "attn_out", "sv", "sv_out", "value_proj_out"]  # dict/state_dict 里优先挑选的键
TARGET_SHAPE = (1, 30, 8866, 64)  # 期望的形状（若为 1D 且 numel 一致会在内存中 reshape）
# =================================

def _prod(shape):
    n = 1
    for x in shape:
        n *= int(x)
    return int(n)

def _stat_file(p):
    try:
        st = os.stat(p)
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)), st.st_size
    except FileNotFoundError:
        return "NA", -1

def _pick_from_dict(d):
    """
    从 dict 里挑一个张量：
      1) 优先 key 命中 PREFERRED_KEYS
      2) 其次优先 4D
      3) 再按 numel 最大
    支持递归子 dict。
    """
    tensors = [(k, v) for k, v in d.items() if torch.is_tensor(v)]
    if not tensors:
        for k, v in d.items():
            if isinstance(v, dict):
                sub = _pick_from_dict(v)
                if sub is not None:
                    return sub
        return None

    # 先按 key 偏好
    for pref in PREFERRED_KEYS:
        cand = [(k, v) for k, v in tensors if pref in str(k).lower()]
        if cand:
            cand.sort(key=lambda kv: (-(kv[1].dim() == 4), -kv[1].numel()))
            return cand[0]

    # 否则优先 4D，再按 numel
    tensors.sort(key=lambda kv: (-(kv[1].dim() == 4), -kv[1].numel()))
    return tensors[0]

def load_any_tensor(path):
    """
    通吃加载 .pt：
      - 直接 Tensor
      - [tensor, shape]
      - dict/state_dict（递归挑选张量）
      - TorchScript（jit.load 后从 state_dict 里挑）
    返回: (tensor, src_desc)
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    # 直接 tensor
    if torch.is_tensor(obj):
        return obj, "raw-tensor"

    # [tensor, shape]
    if isinstance(obj, (list, tuple)) and len(obj) == 2 and torch.is_tensor(obj[0]) and torch.is_tensor(obj[1]):
        t, shp = obj
        shp = tuple(int(x) for x in obj[1].tolist())
        if t.dim() == 1 and t.numel() == _prod(shp):
            t = t.view(*shp)
        return t, "packed-[tensor,shape]"

    # dict（包括普通 torch.save(dict) 或 state_dict）
    if isinstance(obj, dict):
        picked = _pick_from_dict(obj)
        if picked is None:
            raise TypeError(f"{path} dict has no tensor to pick.")
        name, t = picked
        return t, f"dict:{name}"

    # TorchScript：从 state_dict 选择
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

def coerce_to_target_shape_if_needed(t, target_shape):
    """
    若 t 是 1D 且元素数与 target_shape 一致，则仅在内存中 view 成 target_shape。
    不改文件。
    """
    if t.dim() == 1 and t.numel() == _prod(target_shape):
        print(f"[fix] reshape in RAM: {tuple(t.shape)} -> {target_shape}")
        return t.view(*target_shape)
    return t

def row_stats_numpy_row(row_np):
    """
    对单行 ndarray 计算：
      - zero_count
      - nan_count
      - inf_count
      - abs_min_nonzero / abs_max_nonzero（忽略 0 与非有限值）
    返回 dict（min/max 若无有效元素 -> 'NaN' 字符串）
    """
    zero_count = int(np.sum(row_np == 0))
    nan_count  = int(np.isnan(row_np).sum())
    inf_count  = int(np.isinf(row_np).sum())

    finite = np.isfinite(row_np)
    nonzero = (row_np != 0)
    valid_mask = finite & nonzero

    if valid_mask.any():
        abs_vals = np.abs(row_np[valid_mask])
        # 用 float 写入；NaN/Inf 显式字符串在写 CSV 前处理
        abs_min = float(abs_vals.min())
        abs_max = float(abs_vals.max())
        min_out = f"{abs_min:.9g}"
        max_out = f"{abs_max:.9g}"
    else:
        # 显式字符串，避免空字段
        min_out = "NaN"
        max_out = "NaN"

    return {
        "zero_count": zero_count,
        "nan_count":  nan_count,
        "inf_count":  inf_count,
        "abs_min_nonzero": min_out,
        "abs_max_nonzero": max_out,
    }

def write_head_csv(head_idx, mat_2d, out_dir):
    """
    对单个 head 的 (M,N) 做逐行统计并写 CSV。
    列：row, zero_count, nan_count, inf_count, abs_min_nonzero, abs_max_nonzero
    NaN/Inf 会以字符串 "NaN"/"Inf"/"-Inf" 写出，不会出现空字符串 ""。
    """
    M, N = mat_2d.shape
    csv_path = os.path.join(out_dir, f"o_head{head_idx:02d}.csv")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "zero_count", "nan_count", "inf_count", "abs_min_nonzero", "abs_max_nonzero"])

        # 逐行处理，内存友好
        for i in range(M):
            row = mat_2d[i].to(torch.float32).cpu().numpy()
            st = row_stats_numpy_row(row)
            w.writerow([i, st["zero_count"], st["nan_count"], st["inf_count"],
                        st["abs_min_nonzero"], st["abs_max_nonzero"]])

    print(f"[csv] head {head_idx:02d}: wrote {csv_path}")

def summarize_head(head_idx, mat_2d):
    """
    生成 head 级别的汇总：
      - total_rows
      - total_zero / total_nan / total_inf
      - rows_all_invalid（该行全为 0 或非有限值）
      - global_abs_min_nonzero / global_abs_max_nonzero（跨全行，忽略 0 和非有限）
    返回 dict（min/max 若无有效元素 -> 'NaN'）
    """
    M, N = mat_2d.shape
    total_zero = total_nan = total_inf = 0
    rows_all_invalid = 0
    global_min = None
    global_max = None

    for i in range(M):
        row = mat_2d[i].to(torch.float32).cpu().numpy()
        zero_count = int(np.sum(row == 0))
        nan_count  = int(np.isnan(row).sum())
        inf_count  = int(np.isinf(row).sum())

        total_zero += zero_count
        total_nan  += nan_count
        total_inf  += inf_count

        finite = np.isfinite(row)
        nonzero = (row != 0)
        valid_mask = finite & nonzero
        if not valid_mask.any():
            rows_all_invalid += 1
        else:
            abs_vals = np.abs(row[valid_mask])
            mn = float(abs_vals.min())
            mx = float(abs_vals.max())
            global_min = mn if (global_min is None or mn < global_min) else global_min
            global_max = mx if (global_max is None or mx > global_max) else global_max

    out = {
        "head": head_idx,
        "total_rows": M,
        "total_zero": total_zero,
        "total_nan": total_nan,
        "total_inf": total_inf,
        "rows_all_invalid": rows_all_invalid,
        "global_abs_min_nonzero": f"{global_min:.9g}" if global_min is not None else "NaN",
        "global_abs_max_nonzero": f"{global_max:.9g}" if global_max is not None else "NaN",
    }
    return out

def write_global_csv(all_heads_summary, out_dir):
    path = os.path.join(out_dir, "o_heads_summary.csv")
    fields = ["head","total_rows","total_zero","total_nan","total_inf",
              "rows_all_invalid","global_abs_min_nonzero","global_abs_max_nonzero"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_heads_summary:
            w.writerow(row)
    print(f"[csv] wrote global summary -> {path}")

def main():
    ap = argparse.ArgumentParser(description="Per-head, per-row stats for svgemm output o (shape (1,H,8866,64))")
    ap.add_argument("--pt", default="/home/tmp/qkroc/o_first.pt", help="输入 .pt 路径（o 的输出）")
    ap.add_argument("--out", default="/home/tmp/oscv", help="CSV 输出目录")
    ap.add_argument("--expectH", type=int, default=30, help="期望 head 数（默认 30）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    mtime, size = _stat_file(args.pt)
    print(f"[file] {args.pt}  mtime={mtime}  size={size}")

    # 读取 o
    o, src = load_any_tensor(args.pt)
    print(f"[load] src={src}, dtype={o.dtype}, shape={tuple(o.shape)}, stride={tuple(o.stride())}, contig={o.is_contiguous()}")

    # 需要的话，把 1D reshape 回 (1,H,8866,64)
    tgt = (1, args.expectH, TARGET_SHAPE[2], TARGET_SHAPE[3])
    o = coerce_to_target_shape_if_needed(o, tgt)

    # 基本形状检查
    assert o.dim() == 4 and o.shape[0] == 1, f"expect (1,H,8866,64), got {tuple(o.shape)}"
    H = o.shape[1]
    M, N = o.shape[2], o.shape[3]
    assert (M, N) == (TARGET_SHAPE[2], TARGET_SHAPE[3]), f"expect last dims (8866,64), got {(M,N)}"
    if H != args.expectH:
        print(f"[warn] heads H={H} != expectH={args.expectH}, 继续按实际 H={H} 处理。")

    # 逐 head 输出 CSV + 汇总
    summaries = []
    for h in range(H):
        mat = o[0, h]  # (8866,64)
        write_head_csv(h, mat, args.out)
        summaries.append(summarize_head(h, mat))

    write_global_csv(summaries, args.out)
    print(f"[DONE] wrote {H} head CSVs and a global summary in: {args.out}")

if __name__ == "__main__":
    main()
