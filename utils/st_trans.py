import torch
import pandas as pd

# 参数
PT_FILE = "/home/tmp/triton/qk.pt"
CSV_FILE = "/home/tmp/triton/qk.csv"
NUM_VALUES = 2048  # 导出多少个数

# 读取 .pt 文件
tensor = torch.load(PT_FILE)

# print(type(tensor))
# if isinstance(tensor, dict):
#     print(tensor.keys())
# print("o:", type(tensor["o"]), tensor["o"].dtype, tensor["o"].shape)
# # print("v_scale:", type(tensor["v_scale"]), tensor["v_scale"].dtype, tensor["v_scale"].shape)

output = tensor["o"]

# 展平为一维
flat_tensor = output.flatten()

# 将 bfloat16 转换成 float32，便于导出
flat_tensor = flat_tensor.to(torch.float32)

# 取前 N 个
num_to_save = min(NUM_VALUES, flat_tensor.numel())
data_to_save = flat_tensor[:num_to_save].cpu().numpy()

# 保存到 CSV
df = pd.DataFrame(data_to_save, columns=["value"])
df.to_csv(CSV_FILE, index=False)

print(f"前 {num_to_save} 个数已保存到 {CSV_FILE}")
