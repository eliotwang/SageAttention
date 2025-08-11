import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


OFFLOAD_ARCHS = os.environ.get("ROCM_OFFLOAD_ARCHS", "gfx942").split(",")


SRC_DIR = "csrc/qattn/rocm"
sources = [
    f"{SRC_DIR}/py_bind_rocm.cpp",   
    # f"{SRC_DIR}/simple_rocgemm.hip", 
    # f"{SRC_DIR}/i8_pw_acc32.hip",    
    f"{SRC_DIR}/kernel_test.hip",    
]


include_dirs = [
    "third_party/rocwmma/library/include",
    "/opt/rocm-6.4.0/include",
    "/opt/rocm-6.4.0/include/hip",
]

define_macros = [
    ("__HIP_PLATFORM_AMD__", "1"),
    ("USE_ROCM", "1"),
    ("HIPBLAS_V2", None),
    # 下面两个按你日志里保持一致（避免 half 运算符冲突）
    ("CUDA_HAS_FP16", "1"),
    ("__HIP_NO_HALF_OPERATORS__", "1"),
    ("__HIP_NO_HALF_CONVERSIONS__", "1"),
]

cxx_flags = [
    "-O3", "-std=c++17",
    "-fopenmp", "-lgomp",
    "-fPIC",
    "-DENABLE_BF16",              
]

hipcc_flags = [
    "-O3", "-std=c++17",
    "-fPIC",
    "-fno-gpu-rdc",
]
for arch in OFFLOAD_ARCHS:
    hipcc_flags += [f"--offload-arch={arch}"]

ext = CUDAExtension(
    name="sageattention._qattn_rocm",
    sources=sources,
    include_dirs=include_dirs,
    define_macros=define_macros,
    extra_compile_args={
        "cxx": cxx_flags,
        "nvcc": hipcc_flags,
        "hipcc": hipcc_flags,
    },
)

setup(
    name="sageattention",
    version="2.2.0",
    packages=["sageattention"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
