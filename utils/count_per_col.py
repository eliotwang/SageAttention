#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, math, time, zipfile
from typing import Any, Dict, Tuple, Optional
import torch
import pandas as pd

def _flatten_tensors(obj: Any, pool: Dict[str, torch.Tensor], prefix: str = ""):
    if isinstance(obj, torch.Tensor):
        pool[prefix or "tensor"] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_tensors(v, pool, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten_tensors(v, pool, f"{prefix}[{i}]" if prefix else f"[{i}]")

def pick_best_tensor(pool: Dict[str, torch.Tensor],
                     prefer_dtype: Optional[torch.dtype] = None,
                     prefer_ndim: Optional[int] = None) -> Tuple[str, torch.Tensor]:
    def score(name: str, t: torch.Tensor) -> Tuple[int, int, int, int]:
        s0 = 0
        if prefer_ndim is not None:
            s0 -= abs((t.ndim or 0) - prefer_ndim) * 10
            if t.ndim == prefer_ndim: s0 += 15
        if prefer_dtype is not None:
            s0 += 10 if t.dtype == prefer_dtype else 0
        s1 = int(t.numel())
        kw = ("v", "o", "out", "output", "t", "logits", "scores", "probs")
        s2 = 5 if any(k in (name or "").lower() for k in kw) else 0
        s3 = 2 if t.ndim in (4, 2) else 0
        return (s0, s2, s3, s1)
    if not pool: raise RuntimeError("候选张量池为空")
    return max(pool.items(), key=lambda kv: score(kv[0], kv[1]))

def _extract_from_module(m, prefer_dtype=None, prefer_ndim=None, expect_shape=None):
    pool: Dict[str, torch.Tensor] = {}
    try: pool.update({f"sd:{k}": v for k, v in m.state_dict().items()})
    except: pass
    try: pool.update({f"buf:{k}": v for k, v in m.named_buffers()})
    except: pass
    try: pool.update({f"par:{k}": p.data for k, p in m.named_parameters()})
    except: pass
    # 形状优先：如果有 expect_shape，就先过滤 shape 一致的
    if expect_shape is not None:
        pool2 = {k:v for k,v in pool.items() if tuple(v.shape)==tuple(expect_shape)}
        if pool2:
            k,t = pick_best_tensor(pool2, prefer_dtype, prefer_ndim)
            return t, f"module:{k}"
    if pool:
        k,t = pick_best_tensor(pool, prefer_dtype, prefer_ndim)
        return t, f"module:{k}"
    # 退路：零参 forward()
    try:
        res = m()
        if isinstance(res, torch.Tensor):
            if expect_shape is None or tuple(res.shape)==tuple(expect_shape):
                return res, "module:forward"
        pool2: Dict[str, torch.Tensor] = {}
        _flatten_tensors(res, pool2)
        if expect_shape is not None:
            pool2 = {k:v for k,v in pool2.items() if tuple(v.shape)==tuple(expect_shape)}
        if pool2:
            k,t = pick_best_tensor(pool2, prefer_dtype, prefer_ndim)
            return t, f"module:forward:{k}"
    except: pass
    return None

def load_tensor_any_strict(path: str,
                           map_location: str = "cpu",
                           prefer_dtype: Optional[torch.dtype] = None,
                           prefer_ndim: Optional[int] = None,
                           expect_shape: Optional[Tuple[int,...]] = None) -> Tuple[torch.Tensor, str]:
    is_jit_archive = False
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
            is_jit_archive = ("constants.pkl" in names) or any(n.startswith("code/") for n in names)
    except: is_jit_archive = False

    if is_jit_archive:
        m = torch.jit.load(path, map_location=map_location)
        out = _extract_from_module(m, prefer_dtype, prefer_ndim, expect_shape)
        if out is not None:
            t,how = out; return t, f"torchscript:{how}"
        raise RuntimeError("TorchScript 模块中未能自动抽取到 Tensor。")

    obj = torch.load(path, map_location=map_location, weights_only=False)
    # torch.load 可能返回 Module
    try:
        from torch.jit._script import RecursiveScriptModule
        if isinstance(obj, RecursiveScriptModule):
            out = _extract_from_module(obj, prefer_dtype, prefer_ndim, expect_shape)
            if out is not None:
                t,how = out; return t, f"torch.load:{how}"
    except: pass
    if hasattr(obj, "state_dict") or hasattr(obj, "named_buffers") or hasattr(obj, "named_parameters"):
        out = _extract_from_module(obj, prefer_dtype, prefer_ndim, expect_shape)
        if out is not None:
            t,how = out; return t, f"torch.load:{how}"

    if isinstance(obj, torch.Tensor):
        if expect_shape is None or tuple(obj.shape)==tuple(expect_shape):
            return obj, "pt:tensor"
    pool: Dict[str, torch.Tensor] = {}
    _flatten_tensors(obj, pool)
    if expect_shape is not None:
        pool = {k:v for k,v in pool.items() if tuple(v.shape)==tuple(expect_shape)}
    if pool:
        k,t = pick_best_tensor(pool, prefer_dtype, prefer_ndim)
        return t, f"dict:{k}"
    raise RuntimeError(f"读取到非 Tensor 对象: {type(obj)}")

def safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not torch.any(finite): return float('nan')
    a = a[finite]; b = b[finite]
    num = (a*b).sum().item()
    da = (a*a).sum().item()
    db = (b*b).sum().item()
    den = math.sqrt(max(da,1e-30)*max(db,1e-30))
    if den==0: return float('nan')
    c = num/den
    return 1.0 if c>1.0 else (-1.0 if c<-1.0 else c)

def linfit_stats(a: torch.Tensor, b: torch.Tensor):
    # 最小二乘 y ≈ kx + b，返回 k, b, R^2
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not torch.any(finite):
        return float('nan'), float('nan'), float('nan')
    x = a[finite].float(); y = b[finite].float()
    n = x.numel()
    sx = x.sum(); sy = y.sum()
    sxx = (x*x).sum(); syy = (y*y).sum(); sxy = (x*y).sum()
    den = n*sxx - sx*sx
    if abs(den) < 1e-30:
        return float('nan'), float('nan'), float('nan')
    k = (n*sxy - sx*sy) / den
    b0 = (sy - k*sx) / n
    # R^2
    yhat = k*x + b0
    ss_res = ((y - yhat)**2).sum()
    ss_tot = ((y - y.mean())**2).sum().clamp_min(1e-30)
    r2 = 1.0 - (ss_res/ss_tot)
    return float(k.item()), float(b0.item()), float(r2.item())

def summarize_cols(A, B, out_csv, head):
    # A,B: (BN, D), float32
    BN, D = A.shape
    cos_list=[]; valid_list=[]; k_list=[]; b_list=[]; r2_list=[]
    for d in range(D):
        c = safe_cosine(A[:,d], B[:,d])
        finite = torch.isfinite(A[:,d]) & torch.isfinite(B[:,d])
        cos_list.append(c)
        valid_list.append(int(finite.sum().item()))
        k,b0,r2 = linfit_stats(A[:,d], B[:,d])
        k_list.append(k); b_list.append(b0); r2_list.append(r2)
        if (d+1)%8==0:
            print(f"[progress] cols processed: {d+1}/{D}")
    cos = torch.tensor(cos_list, dtype=torch.float32)
    mask = torch.isfinite(cos)
    cos_f = cos[mask]
    q = torch.quantile(cos_f, torch.tensor([0.1,0.5,0.9]))
    print("====== 逐列余弦（head 级）======")
    print(f"总列数: {D}")
    print(f"cos mean: {cos_f.mean().item():.6f}, min: {cos_f.min().item():.6f}, "
          f"median: {q[1].item():.6f}, p10: {q[0].item():.6f}, p90: {q[2].item():.6f}, max: {cos_f.max().item():.6f}")
    # 最差 10 列
    values, idx = torch.topk(cos, k=min(10, D), largest=False)
    print("------ 最差的 10 列 ------")
    for r in range(values.numel()):
        d = int(idx[r].item()); cval = float(values[r].item()) if math.isfinite(values[r].item()) else float('nan')
        print(f"col {d:4d}: cos={cval:.6f}, valid={valid_list[d]}, k={k_list[d]:.4g}, b={b_list[d]:.4g}, R2={r2_list[d]:.4f}")
    # CSV
    df = pd.DataFrame({"col": list(range(D)),
                       "cosine": cos_list,
                       "valid": valid_list,
                       "slope_k": k_list,
                       "intercept_b": b_list,
                       "R2": r2_list})
    df.to_csv(out_csv or f"col_cosine_head{head}.csv", index=False, float_format="%.8f")
    print(f"[csv] per-col cosine written to {out_csv or f'col_cosine_head{head}.csv'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt1", default="/home/tmp/newv/vbefore.pt")
    ap.add_argument("--pt2", default="/home/tmp/newv/vafter_first.pt")
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--prefer-ndim", type=int, default=4)
    ap.add_argument("--prefer-dtype", type=str, default="float16")
    ap.add_argument("--out", type=str, default="/home/tmp/newv/out.csv")
    ap.add_argument("--try-swap", action="store_true", help="同时尝试把 t2 的 n/d 互换再比一次")
    args = ap.parse_args()

    dtype_map = {"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32,"float":torch.float32}
    prefer_dtype = dtype_map.get(args.prefer_dtype.lower(), None)

    for p in (args.pt1, args.pt2):
        st = os.stat(p)
        print(f"[file] {p}  mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}  size={st.st_size}")

    # 先读 t1（基准），用其 shape 作为 t2 的 expect_shape
    t1, s1 = load_tensor_any_strict(args.pt1, map_location="cpu", prefer_dtype=prefer_dtype, prefer_ndim=args.prefer_ndim)
    print(f"[raw] t1(src={s1}): dtype={t1.dtype}, shape={tuple(t1.shape)}")
    if t1.ndim != 4:
        raise RuntimeError("t1 必须是 4 维 (B,H,N,D)")
    expect_shape = tuple(t1.shape)

    t2, s2 = load_tensor_any_strict(args.pt2, map_location="cpu", prefer_dtype=prefer_dtype, prefer_ndim=args.prefer_ndim, expect_shape=expect_shape)
    print(f"[raw] t2(src={s2}): dtype={t2.dtype}, shape={tuple(t2.shape)}")

    if tuple(t2.shape) != expect_shape:
        raise RuntimeError(f"两个张量形状不一致: {t1.shape} vs {t2.shape}")

    B,H,N,D = t1.shape
    head = args.head
    if not (0 <= head < H):
        raise RuntimeError(f"head 超界: {head} (H={H})")

    A = t1[:, head, :, :].reshape(-1, N, D).reshape(-1, D).to(torch.float32).contiguous()
    Bm = t2[:, head, :, :].reshape(-1, N, D).reshape(-1, D).to(torch.float32).contiguous()
    summarize_cols(A, Bm, args.out, head)

    if args.try_swap:
        print("\n[try-swap] 尝试把 t2 的最后两维互换（怀疑 (n,d) 轴搞反）后再比一次：")
        t2s = t2.permute(0,1,3,2).contiguous()  # (B,H,D,N)
        if t2s.shape == t1.shape:
            Bm2 = t2s[:, head, :, :].reshape(-1, N, D).reshape(-1, D).to(torch.float32).contiguous()
            summarize_cols(A, Bm2, (args.out or f"col_cosine_head{head}.csv").replace(".csv","_swap.csv"), head)
        else:
            print("[try-swap] 互换后形状不匹配，跳过。")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
