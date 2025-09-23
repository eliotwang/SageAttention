import torch
import sageattention._fused as fused
from .triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton

head = 16
batch = 4
headdim = 128
seq_len = 256

SEED1 = 20250902
g1 = torch.Generator(device="cpu").manual_seed(SEED1)
SEED2 = 20250911
g2 = torch.Generator(device="cpu").manual_seed(SEED2)
q = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, generator=g1, device="cpu").to("cuda").contiguous()
k = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, generator=g2, device="cpu").to("cuda").contiguous()

q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout="HND", BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)

v_transposed_permutted = torch.empty(batch, head, headdim, seq_len, dtype=torch.bfloat16, device="cuda")

scale_max: float = 448.0

_tensor_layout = 1

fused.transpose_pad_permute_cuda(v, v_transposed_permutted, _tensor_layout)
v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fnuz, device=v.device)
v_scale = torch.empty((batch, head, headdim), dtype=torch.float32, device=v.device)
vm = torch.empty((batch, head, headdim), dtype=torch.float32, device=v.device)
fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, seq_len, scale_max, _tensor_layout)

import os, sys
import torch.distributed as dist

SAVE_PATH = "/home/v_files/v_fp8.pt"
MARKER    = SAVE_PATH + ".first"

def _is_rank0():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

def _save_once(tensor, path=SAVE_PATH, marker=MARKER):
        # 仅 rank0 保存
    if not _is_rank0():
        return
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return  # 已保存
    else:
        with os.fdopen(fd, "w") as f:
            f.write("saved\n")

    tmp = path + ".tmp"
    torch.save(tensor.detach().cpu(), tmp)
    os.replace(tmp, path)
    print(f"[INFO] v_trans saved to: {path}")
    sys.stdout.flush()
_save_once(v_fp8)