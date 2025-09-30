import numpy as np
import torch

# 文件路径（按你实际文件改）
hip_bytes_path = "hip_out.npy"        # HIP 写出的 raw bytes (dtype np.uint8)
hip_scale_path = "hip_scale.npy"
pt_bytes_path  = "pt_v_fp8_bytes.pt"  # PyTorch 原始 v_fp8 bytes 存成 tensor 或 numpy
pt_input_path  = "pt_input.npy"       # 原始 float 输入 (用于 recompute scale)

# load
hip_raw = np.load(hip_bytes_path).astype(np.uint8)
hip_scale = np.load(hip_scale_path).astype(np.float32)

try:
    pt_raw_t = torch.load(pt_bytes_path)   # 可能是 torch tensor
    pt_raw = pt_raw_t.view(torch.uint8).cpu().numpy().astype(np.uint8)
except Exception as e:
    print("无法加载 pt raw bytes:", e)
    pt_raw = None

# print first bytes
def dump_head(arr, name, n=128):
    flat = arr.flatten()
    print(f"--- {name} first {n} bytes (as uint8) ---")
    print(flat[:n].tolist())

dump_head(hip_raw, "HIP")
if pt_raw is not None:
    dump_head(pt_raw, "PT")

# decode HIP raw with pytorch decoder
hip_torch_view = torch.from_numpy(hip_raw).view(torch.float8_e4m3fnuz)
hip_decoded = hip_torch_view.to(torch.float32).cpu().numpy()

print("--- HIP decoded first 32 floats (torch decode) ---")
print(hip_decoded.flatten()[:32].tolist())

if pt_raw is not None:
    pt_torch_view = torch.from_numpy(pt_raw).view(torch.float8_e4m3fnuz)
    pt_decoded = pt_torch_view.to(torch.float32).cpu().numpy()
    print("--- PT decoded first 32 floats (torch decode) ---")
    print(pt_decoded.flatten()[:32].tolist())

# compare decoded values if both present
if pt_raw is not None:
    eq_bytes = np.array_equal(pt_raw.flatten(), hip_raw.flatten())
    print("Raw bytes equal? ", eq_bytes)
    if not eq_bytes:
        # show elementwise comparison for first 64
        print("first 64 bytes PT vs HIP:")
        for i in range(min(64, pt_raw.size, hip_raw.size)):
            print(i, int(pt_raw.flatten()[i]), int(hip_raw.flatten()[i]))
# compare scales if pt_input exists
if os.path.exists(pt_input_path):
    pt_input = torch.from_numpy(np.load(pt_input_path)).to(torch.float32)
    # pytorch used FP8_MAX you observed (try two candidates)
    for FP8 in (224.0, 240.0, 448.0):
        max_vals = pt_input.abs().amax(dim=-1).cpu().numpy()
        scale_pt = (max_vals / FP8).clip(min=1e-8)
        print(f"FP8 {FP8} sample torch scale (first 8):", scale_pt.flatten()[:8].tolist())
    if hip_scale is not None:
        print("HIP scale sample (first 8):", hip_scale.flatten()[:8].tolist())
        # show ratio with candidate FP8=224 (torch used 224)
        max_vals = pt_input.abs().amax(dim=-1).cpu().numpy()
        scale_224 = (max_vals / 224.0).clip(min=1e-8)
        print("ratio torch(224)/hip:", (scale_224.flatten()[:8] / (hip_scale.flatten()[:8] + 1e-12)).tolist())
