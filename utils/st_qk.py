# utils/st_qk.py
import os, sys, zipfile, argparse
import torch
import pandas as pd

def is_zip_file(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return True if zf.namelist() else False
    except zipfile.BadZipFile:
        return False

def pick_best_tensor(obj_dict, prefer_dtype=None, prefer_ndim=None):
    """
    在若干候选里挑“最像目标”的张量：
    - 优先满足 prefer_dtype（例如 torch.int32 / torch.float32）
    - 优先满足 prefer_ndim（例如 4 维）
    - 再按 numel 最大
    返回 (key, tensor)；若没有合适返回 (None, None)
    """
    best = (None, None)
    best_score = (-1, -1, -1)  # (dtype_ok, ndim_ok, numel)
    for k, v in obj_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        dtype_ok = 1 if (prefer_dtype is None or v.dtype == prefer_dtype) else 0
        ndim_ok  = 1 if (prefer_ndim  is None or v.ndim == prefer_ndim) else 0
        score = (dtype_ok, ndim_ok, int(v.numel()))
        if score > best_score:
            best = (k, v)
            best_score = score
    return best

def load_tensor_any(path, map_location="cpu", prefer_dtype=None, prefer_ndim=None):
    """
    读 pt，自动兼容：
    - 纯 Tensor（优先 weights_only=True）
    - TorchScript 存档（jit.load 后从 state_dict / buffers / params / forward() 自动找张量）
    - 回退 weights_only=False（仅限可信）
    返回 (tensor, how)；how 说明使用的路径
    """
    # 路径检查
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    # 1) 尝试作为纯 Tensor（2.6+ 默认值已经适配）
    try:
        t = torch.load(path, map_location=map_location, weights_only=True)
        if isinstance(t, torch.Tensor):
            return t, "weights_only=True"
        # 不是张量，继续尝试
    except RuntimeError as e:
        # 典型：TorchScript archive 会在这里抛错
        if "TorchScript" not in str(e) and "archive" not in str(e):
            # 不是 TorchScript 的典型错误，继续往下兜底
            pass

    # 2) TorchScript 存档路径
    if is_zip_file(path):
        # 2.1 读成 ScriptModule
        m = torch.jit.load(path, map_location=map_location)

        # 2.2 从 state_dict / buffers / params 里选最匹配的张量
        pool = {}
        try:
            pool.update({f"sd:{k}": v for k, v in m.state_dict().items()})
        except Exception:
            pass
        try:
            pool.update({f"buf:{k}": v for k, v in m.named_buffers()})
        except Exception:
            pass
        try:
            pool.update({f"par:{k}": p.data for k, p in m.named_parameters()})
        except Exception:
            pass

        if pool:
            k, t = pick_best_tensor(pool, prefer_dtype=prefer_dtype, prefer_ndim=prefer_ndim)
            if t is not None:
                return t, f"jit.load->({k})"

        # 2.3 尝试 forward()（若可无参调用）
        try:
            res = m()
            if isinstance(res, dict):
                k, t = pick_best_tensor(res, prefer_dtype=prefer_dtype, prefer_ndim=prefer_ndim)
                if t is not None:
                    return t, "jit.load->forward()->dict"
            if isinstance(res, torch.Tensor):
                return res, "jit.load->forward()->tensor"
        except Exception:
            pass

        raise RuntimeError("TorchScript 存档中未找到合适的张量；请确认该文件确实包含目标数据。")

    # 3) 最后回退：weights_only=False（⚠️仅限可信文件）
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj, "weights_only=False"
    # 如果是 dict，挑一个
    if isinstance(obj, dict):
        k, t = pick_best_tensor(obj, prefer_dtype=prefer_dtype, prefer_ndim=prefer_ndim)
        if t is not None:
            return t, f"weights_only=False->dict({k})"
    raise RuntimeError(f"读取到非 Tensor 对象: {type(obj)}，且无法自动提取。")

def save_head_csv(tensor: torch.Tensor, n: int, csv_path: str):
    flat = tensor.flatten().to("cpu")[:n]
    pd.DataFrame(flat.numpy(), columns=["values"]).to_csv(csv_path, index=False)
    print(f"[OK] 已写出前 {len(flat)} 个元素到 {csv_path}")

def main():
    ap = argparse.ArgumentParser(description="Load tensor (Tensor/TorchScript) and dump first N elems to CSV")
    ap.add_argument("--pt",   default="/home/tmp/qkroc/qk_int32_first.pt", help="要读取的 .pt 路径")
    ap.add_argument("--csv",  default="/home/tmp/qkroc/qkernel_first.csv", help="要写出的 CSV 文件")
    ap.add_argument("-n", "--num", type=int, default=1024, help="写出的前 N 个元素")
    # logits 偏好：float32 + 4D；若是 probs 同理
    ap.add_argument("--prefer", choices=["int32","float32","none"], default="float32")
    args = ap.parse_args()

    prefer_dtype = {"int32": torch.int32, "float32": torch.float32, "none": None}[args.prefer]

    t, how = load_tensor_any(args.pt, map_location="cpu", prefer_dtype=prefer_dtype, prefer_ndim=4)
    print(f"[INFO] 加载成功 via {how}: shape={tuple(t.shape)}, dtype={t.dtype}")
    save_head_csv(t, args.num, args.csv)

if __name__ == "__main__":
    main()
