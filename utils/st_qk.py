#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
st_qk.py — Robust Tensor Loader & Inspector + CSV Dump

用法示例：
  # 只打印基本信息
  python utils/st_qk.py --pt /home/tmp/newv/vafter_first.pt

  # 导出前 2048 个数到 csv
  python utils/st_qk.py --pt /home/tmp/newv/vafter_first.pt --dump-csv /home/tmp/newv/out.csv --num-values 2048
"""

import argparse
import os
import time
import zipfile
from typing import Any, Dict, Tuple, Optional

import torch
import numpy as np
import pandas as pd


# -------------------- helpers: flatten & pick --------------------

def _flatten_tensors(obj: Any, pool: Dict[str, torch.Tensor], prefix: str = ""):
    if isinstance(obj, torch.Tensor):
        pool[prefix or "tensor"] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            _flatten_tensors(v, pool, name)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            name = f"{prefix}[{i}]" if prefix else f"[{i}]"
            _flatten_tensors(v, pool, name)


def _pick_best_tensor(pool: Dict[str, torch.Tensor],
                      prefer_dtype: Optional[torch.dtype] = None,
                      prefer_ndim: Optional[int] = None) -> Tuple[str, torch.Tensor]:
    if not pool:
        raise RuntimeError("候选张量池为空")

    def score(name: str, t: torch.Tensor):
        s = 0
        if prefer_ndim is not None:
            s -= abs((t.ndim or 0) - prefer_ndim) * 10
            if t.ndim == prefer_ndim:
                s += 15
        if any(k in (name or "").lower() for k in ("v", "o", "out", "output", "logits", "scores", "probs")):
            s += 5
        if t.ndim in (4, 2):
            s += 2
        if t.numel() > 0:
            s += 1
        if prefer_dtype is not None and t.dtype == prefer_dtype:
            s += 1
        return (s, t.numel())

    return max(pool.items(), key=lambda kv: score(kv[0], kv[1]))


# -------------------- robust loader --------------------

def load_tensor_any(path: str,
                    map_location: str = "cpu",
                    prefer_dtype: Optional[torch.dtype] = None,
                    prefer_ndim: Optional[int] = 4) -> Tuple[torch.Tensor, str]:
    obj = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(obj, torch.Tensor):
        return obj, "pt:tensor"

    if isinstance(obj, dict):
        if "tensor" in obj and isinstance(obj["tensor"], torch.Tensor):
            return obj["tensor"], "dict:tensor"
        if obj.get("kind") == "tensor_dump" and "tensor" in obj and isinstance(obj["tensor"], torch.Tensor):
            return obj["tensor"], "dict:tensor_dump"

    if isinstance(obj, (list, tuple, dict)):
        pool: Dict[str, torch.Tensor] = {}
        _flatten_tensors(obj, pool)
        if pool:
            k, t = _pick_best_tensor(pool, prefer_dtype, prefer_ndim)
            return t, f"container:{k}"

    try:
        if zipfile.is_zipfile(path):
            try:
                m = torch.jit.load(path, map_location=map_location)
                pool: Dict[str, torch.Tensor] = {}
                try:
                    for k, v in dict(m.state_dict()).items():
                        if isinstance(v, torch.Tensor):
                            pool[f"sd:{k}"] = v
                except Exception:
                    pass
                try:
                    for k, v in m.named_buffers():
                        if isinstance(v, torch.Tensor):
                            pool[f"buf:{k}"] = v
                except Exception:
                    pass
                try:
                    for k, p in m.named_parameters():
                        if isinstance(p, torch.nn.Parameter):
                            pool[f"par:{k}"] = p.data
                except Exception:
                    pass
                pool4 = {k: v for k, v in pool.items() if v.ndim == prefer_ndim}
                if pool4:
                    k, t = _pick_best_tensor(pool4, prefer_dtype, prefer_ndim)
                    return t, f"torchscript:{k}"
                if pool:
                    k, t = _pick_best_tensor(pool, prefer_dtype, prefer_ndim)
                    return t, f"torchscript:{k}"
                try:
                    out = m()
                    if isinstance(out, torch.Tensor):
                        return out, "torchscript:forward()"
                    pool2: Dict[str, torch.Tensor] = {}
                    _flatten_tensors(out, pool2)
                    if pool2:
                        k, t = _pick_best_tensor(pool2, prefer_dtype, prefer_ndim)
                        return t, f"torchscript:forward:{k}"
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass

    raise RuntimeError(f"无法从 {path} 提取 Tensor（读取到类型: {type(obj)}）")


# -------------------- pretty print --------------------

def _dtype_name(dt: torch.dtype) -> str:
    try:
        return str(dt)
    except Exception:
        return repr(dt)


def print_tensor_info(t: torch.Tensor, src: str, path: str):
    shape = tuple(t.shape)
    stride = tuple(t.stride())
    print(f"[raw] src={src}")
    print(f"      path={path}")
    print(f"      dtype={_dtype_name(t.dtype)}, shape={shape}, stride={stride}, contig={t.is_contiguous()}")


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="/home/tmp/lastv/o.pt")
    ap.add_argument("--map-location", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--prefer-ndim", type=int, default=4)
    ap.add_argument("--prefer-dtype", type=str, default="float16")
    ap.add_argument("--dump-pt", type=str, default=None)
    ap.add_argument("--dump-npy", type=str, default=None)
    ap.add_argument("--dump-csv", type=str, default="/home/tmp/lastv/o.csv", help="把前 num-values 个元素保存到 csv")
    ap.add_argument("--num-values", type=int, default=10000, help="csv 导出的元素个数")
    args = ap.parse_args()

    prefer_dtype_map = {
        "float16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "float": torch.float32,
        "uint8": torch.uint8, "int8": torch.int8
    }
    prefer_dtype = prefer_dtype_map.get(args.prefer_dtype.lower(), None)

    st = os.stat(args.pt)
    print(f"[file] {args.pt}  mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}  size={st.st_size}")

    t, how = load_tensor_any(args.pt,
                             map_location=args.map_location,
                             prefer_dtype=prefer_dtype,
                             prefer_ndim=args.prefer_ndim)

    print_tensor_info(t, how, args.pt)

    if args.dump_pt:
        torch.save(t, args.dump_pt)
        print(f"[dump] wrote pure tensor to {args.dump_pt}")

    if args.dump_npy:
        np.save(args.dump_npy, t.detach().to(torch.float32).cpu().numpy())
        print(f"[dump] wrote float32 .npy to {args.dump_npy}")

    if args.dump_csv:
        flat = t.flatten().detach().to(torch.float32).cpu().numpy()
        n = min(args.num_values, flat.shape[0])
        
        if not np.issubdtype(flat.dtype, np.floating):
            print(f"[warn] tensor dtype is {flat.dtype}; exporting raw values (no float conversion).")
            # 将原始值直接转为 Python 可打印形式
            rows = [(i, int(flat[i])) for i in range(n)]
            df = pd.DataFrame(rows, columns=["index", "value"])
            df.to_csv(args.dump_csv, index=False)
            print(f"[dump] wrote first {n} raw values to {args.dump_csv}")
        else:
            # flat 已为浮点：给出统计并导出，NaN 用 'NaN' 显示
            num_nan = int(np.isnan(flat).sum())
            num_inf = int(np.isinf(flat).sum())
            print(f"[diag] exporting {n} floats (finite={int(np.isfinite(flat).sum())}, NaN={num_nan}, Inf={num_inf})")

            # 方案 A: 简单导出，NaN 用 'NaN'
            df = pd.DataFrame(flat[:n], columns=["value"])
            df.to_csv(args.dump_csv, index=False, na_rep="NaN")
            print(f"[dump] wrote first {n} values to {args.dump_csv} (NaN written as 'NaN')")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
