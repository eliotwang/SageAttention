import torch
import pandas as pd

PT_FILE = "/home/tmp/rightv/v_scale.pt"
CSV_FILE = "/home/tmp/rightv/v_scale.csv"
NUM_VALUES = 12000  # 导出多少个数

obj = torch.load(PT_FILE, map_location="cpu")

# 选择要导出的 Tensor
output = None
if torch.is_tensor(obj):
    output = obj
elif isinstance(obj, dict):
    # 优先找常见键名
    for k in ("o", "output", "v", "value", "data"):
        if k in obj and torch.is_tensor(obj[k]):
            output = obj[k]
            break
    # 若没命中，退而求其次：找字典里的第一个 tensor
    if output is None:
        for v in obj.values():
            if torch.is_tensor(v):
                output = v
                break
elif hasattr(obj, "o") and torch.is_tensor(getattr(obj, "o")):
    output = getattr(obj, "o")

if output is None:
    raise TypeError(f"文件里没有找到可导出的 Tensor（类型: {type(obj)}）。")

# 展平并转为 float32（无论是 bfloat16 / float16 / float8 都先转成 float32 便于导出）
flat_tensor = output.reshape(-1).to(torch.float32)

# 取前 N 个
num_to_save = min(NUM_VALUES, flat_tensor.numel())
data_to_save = flat_tensor[:num_to_save].numpy()

# 保存到 CSV
df = pd.DataFrame(data_to_save, columns=["value"])
df.to_csv(CSV_FILE, index=False)

print(f"源 dtype: {output.dtype}, 形状: {tuple(output.shape)}")
print(f"前 {num_to_save} 个数已保存到 {CSV_FILE}")
