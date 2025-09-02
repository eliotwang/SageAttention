# import os, torch
# os.environ["LD_LIBRARY_PATH"] = os.path.join(torch.__path__[0], "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# import importlib.util
# from pathlib import Path

# so_path = Path("/home/SageAttention/build/lib.linux-x86_64-cpython-312/sageattention/_qattn_rocm.cpython-312-x86_64-linux-gnu.so")

# spec = importlib.util.spec_from_file_location("_qattn_rocm", so_path)
# q_attn = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(q_attn)

import sageattention._qattn_rocm as q_attn

# 调用测试
q_attn.simple_gemm(1024,128,1024)
