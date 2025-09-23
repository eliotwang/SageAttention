import torch
import pandas as pd

# 参数
PT_FILE = "/home/tmp/triton/o.pt"
CSV_FILE = "/home/tmp/triton/o_last.csv"
NUM_VALUES = 10000  # 导出多少个数

# 读取 .pt 文件
tensor = torch.load(PT_FILE)

output = tensor["o"]

# 展平为一维
flat_tensor = output.flatten()

# 将 bfloat16 转换成 float32，便于导出
flat_tensor = flat_tensor.to(torch.float32)

# 从后往前取 N 个
num_to_save = min(NUM_VALUES, flat_tensor.numel())
# 先取末尾 num_to_save 个，再反转顺序
data_to_save = torch.flip(flat_tensor[-num_to_save:], dims=[0]).cpu().numpy()

# 保存到 CSV
df = pd.DataFrame(data_to_save, columns=["value"])
df.to_csv(CSV_FILE, index=False)

print(f"从后往前的 {num_to_save} 个数已保存到 {CSV_FILE}")
