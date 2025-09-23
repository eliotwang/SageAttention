import torch,os
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from sageattention import sageattn
import torch.nn.functional as F
import argparse

os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="THUDM/CogVideoX-2b", help='Model path')
parser.add_argument('--compile', action='store_true', help='Compile the model')
parser.add_argument('--attention_type', type=str, default='sage', choices=['sdpa', 'sage', 'fa3', 'fa3_fp8'], help='Attention type')
args = parser.parse_args()

if args.attention_type == 'sage':
    F.scaled_dot_product_attention = sageattn
elif args.attention_type == 'fa3':
    from sageattention.fa3_wrapper import fa3
    F.scaled_dot_product_attention = fa3
elif args.attention_type == 'fa3_fp8':
    from sageattention.fa3_wrapper import fa3_fp8
    F.scaled_dot_product_attention = fa3_fp8

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
).to("cuda")

if args.compile:
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

pipe.enable_model_cpu_offload() 
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

import time
with torch.no_grad():
    start = time.time()
    video = pipe(
        prompt=prompt,
        guidance_scale=1.0,          # 关掉 CFG，避免把 batch 翻倍
        use_dynamic_cfg=False,       # （需要 CFG 时再开 True）
        num_inference_steps=36,      # 先降到 36 步
        num_frames=33,               # 默认是 48/49；先降到 33（需被 VAE 时间因子整除）
        height=384, width=640,       # 默认 480x720；先降一档
        num_videos_per_prompt=1,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]
    end = time.time()     
    print(f"[INFO] Inference took {end - start:.2f} seconds")

export_to_video(video, f"f_thread_{args.attention_type}.mp4", fps=8)
