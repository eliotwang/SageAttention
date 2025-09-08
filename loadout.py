import os
import torch
import numpy as np


data = torch.load("/home/v_files/quantized_v.pt", map_location="cpu")
v_fp8 = data["v_fp8"].to(torch.float32).cpu().numpy()   # float8 -> float32，便于保存
v_scale = data["v_scale"].cpu().numpy()

print(f"v_fp8 shape: {v_fp8.shape}, dtype: {v_fp8.dtype}")
print(f"v_scale shape: {v_scale.shape}, dtype: {v_scale.dtype}")

# ---- v_fp8：导出前 1000 个值（展平为 1D）----
v_fp8_sample = v_fp8.reshape(-1)[:3000]
np.savetxt("/home/v_files/v_fp8_sample.csv", v_fp8_sample, fmt="%.6f", delimiter=",")

# ---- v_scale：去掉 batch 维，保存 2D 的前 10 行 ----
# 你的 v_scale 是 (1, 30, 64)，squeeze 掉 batch 维 -> (30, 64)
v_scale_2d = np.squeeze(v_scale, axis=0) if v_scale.ndim == 3 and v_scale.shape[0] == 1 else v_scale

# 只导出前 10 行（按需调整）
rows_to_save = min(10, v_scale_2d.shape[0]) if v_scale_2d.ndim == 2 else 1
to_save = v_scale_2d[:rows_to_save] if v_scale_2d.ndim == 2 else v_scale_2d.reshape(1, -1)

np.savetxt("/home/v_files/v_scale_sample.csv", to_save, fmt="%.6f", delimiter=",")

print("[INFO] 导出完成：")
print("  /home/v_files/v_fp8_sample.csv")
print("  /home/v_files/v_scale_sample.csv")
