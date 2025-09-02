# -*- coding: utf-8 -*-

"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""

import os
import sys
import subprocess
import threading
import warnings
from packaging.version import parse, Version

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


# ========= clean 快速通道（修复 NameError）=========
# 运行 `python setup.py clean` 时，直接使用 setuptools 的 clean，
# 不进入后续 CUDA/ROCm 探测与版本检查逻辑，避免 nvcc_cuda_version 未定义。
if "clean" in sys.argv:
    setup(name="sageattention", packages=find_packages())
    raise SystemExit(0)
# =================================================


HAS_SM80 = False
HAS_SM86 = False
HAS_SM89 = False
HAS_SM90 = False
HAS_SM120 = False

# Supported NVIDIA GPU architectures.
SUPPORTED_ARCHS = {"8.0", "8.6", "8.9", "9.0", "12.0"}

# Compiler flags.
CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    "--threads=8",
    "-Xptxas=-v",
    "-diag-suppress=174",  # suppress the specific warning
]

# C++ ABI 与 PyTorch 一致
ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc."""
    nvcc_output = subprocess.check_output([os.path.join(cuda_dir, "bin", "nvcc"), "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


# 只有在 CUDA 存在时才做 CUDA 的探测和校验（修复 NameError 的第二部分）
compute_capabilities = set()
nvcc_cuda_version = None

if CUDA_HOME is not None and torch.cuda.is_available():
    # 收集算力
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
            continue
        compute_capabilities.add(f"{major}.{minor}")

    if not compute_capabilities:
        warnings.warn("No suitable GPUs (cc >= 8.0) found; CUDA extensions will not be built.")
    else:
        print(f"Detect GPUs with compute capabilities: {compute_capabilities}")

    # NVCC 版本
    nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
    print(f"Detected NVCC CUDA version: {nvcc_cuda_version}")

    # 版本校验
    if nvcc_cuda_version < Version("12.0"):
        raise RuntimeError("CUDA 12.0 or higher is required to build the package.")
    if nvcc_cuda_version < Version("12.4") and any(cc.startswith("8.9") for cc in compute_capabilities):
        raise RuntimeError("CUDA 12.4 or higher is required for compute capability 8.9.")
    if nvcc_cuda_version < Version("12.3") and any(cc.startswith("9.0") for cc in compute_capabilities):
        raise RuntimeError("CUDA 12.3 or higher is required for compute capability 9.0.")
    if nvcc_cuda_version < Version("12.8") and any(cc.startswith("12.0") for cc in compute_capabilities):
        raise RuntimeError("CUDA 12.8 or higher is required for compute capability 12.0.")

    # 目标算力加入 NVCC flags
    for capability in compute_capabilities:
        if capability.startswith("8.0"):
            HAS_SM80 = True
            num = "80"
        elif capability.startswith("8.6"):
            HAS_SM86 = True
            num = "86"
        elif capability.startswith("8.9"):
            HAS_SM89 = True
            num = "89"
        elif capability.startswith("9.0"):
            HAS_SM90 = True
            # 需要 sm90a 以使用 wgmma 指令
            num = "90a"
        elif capability.startswith("12.0"):
            HAS_SM120 = True
            num = "120"
        else:
            continue

        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if capability.endswith("+PTX"):
            NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]
else:
    # 没有 CUDA_HOME 或 torch.cuda 不可用：不做任何 CUDA 相关校验与构建
    print("CUDA not detected (either CUDA_HOME is None or torch.cuda is not available). "
          "CUDA extensions will be skipped.")


# 这里按照你原本的结构，根据 HAS_SM** 条件追加各自的扩展
ext_modules = []

if HAS_SM80 or HAS_SM86:
    qattn_sm80 = CUDAExtension(
        name="sageattention._qattn_sm80",
        sources=[
            "csrc/qattn/pybind_sm80.cpp",
            "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(qattn_sm80)

if HAS_SM89 or HAS_SM120:
    qattn_sm89 = CUDAExtension(
        name="sageattention._qattn_sm89",
        sources=[
            "csrc/qattn/pybind_sm89.cpp",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_attn_inst_buf.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu",
            "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(qattn_sm89)

if HAS_SM90:
    qattn_sm90 = CUDAExtension(
        name="sageattention._qattn_sm90",
        sources=[
            "csrc/qattn/pybind_sm90.cpp",
            "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
        extra_link_args=["-lcuda"],
    )
    ext_modules.append(qattn_sm90)

for var in ("HCC_AMDGPU_TARGET", "AMDGPU_TARGETS", "HIPCC_COMPILE_FLAGS_APPEND",
            "HIP_TARGETS", "ROCM_TARGET_LST"):
    if os.environ.get(var):
        print(f"setup.py: ignoring env {var}={os.environ[var]}")
        del os.environ[var]
from torch.utils.cpp_extension import CppExtension

# ROCm extension (build with hipcc)
from torch.utils.cpp_extension import CUDAExtension  # 用它来编 HIP
IS_ROCM = getattr(torch.version, "hip", None) is not None
ROCM_HOME = os.environ.get("ROCM_HOME", "/opt/rocm-6.3.0")
TORCH_LIB_DIR = os.path.join(torch.__path__[0], "lib")
ROCM_LIB_DIRS = [os.path.join(ROCM_HOME, "lib"), os.path.join(ROCM_HOME, "lib64")]

rocm_offload_arch = ["--offload-arch=gfx942"]  # 按你的目标卡改

rocm_cxx_flags = [
    "-O3", "-fPIC", "-std=c++17",
    "-DUSE_ROCM=1",
    "-D__HIP_PLATFORM_AMD__=1",
    "-D__HIP_NO_HALF_OPERATORS__=1",
    "-D__HIP_NO_HALF_CONVERSIONS__=1",
    "-DHIPBLAS_V2",
    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
]

# hipcc 的参数用 'nvcc' key 传入（在 ROCm 下会映射到 hipcc）
rocm_hipcc_flags = [
    "-O3", "-g", "-ggdb", "-std=c++17",
] + rocm_offload_arch

if IS_ROCM:
    qattn_rocm = CUDAExtension(
        name="sageattention._qattn_rocm",
        sources=[
            "csrc/qattn/rocm/pybind_gfx942.cpp",
            # "csrc/qattn/rocm/simple_ge1d.hip",
            "csrc/qattn/rocm/qk_gemm.hip",
            "csrc/qattn/rocm/sv_gemm.hip",
            "csrc/qattn/rocm/gfx942.hip",
            # "csrc/qattn/rocm/qk2douter.hip",
            # "csrc/qattn/rocm/sv2d.hip",
            # "csrc/qattn/rocm/gfx942_2d.hip",
        ],
        include_dirs=[
            "third_party/rocwmma/library/include",
            os.path.join(ROCM_HOME, "include"),
            os.path.join(ROCM_HOME, "include", "hip"),
        ],
        extra_compile_args={"cxx": rocm_cxx_flags, "nvcc": rocm_hipcc_flags},
        # 这三行确保无需 LD_LIBRARY_PATH 也能找到依赖
        libraries=["amdhip64", "hiprtc", "rocblas", "hipblas"],   # 视你代码实际用到的库增减
        library_dirs=ROCM_LIB_DIRS + [TORCH_LIB_DIR],
        runtime_library_dirs=ROCM_LIB_DIRS + [TORCH_LIB_DIR],     # <— 关键
    )
    ext_modules.append(qattn_rocm)

    fused_extension = CUDAExtension(
    name="sageattention._fused",
    sources=["csrc/fused/pybind_hip.cpp", "csrc/fused/fused.hip"],
    include_dirs=[
            "third_party/rocwmma/library/include",
            os.path.join(ROCM_HOME, "include"),
            os.path.join(ROCM_HOME, "include", "hip"),
        ],
    extra_compile_args={"cxx": rocm_cxx_flags, "nvcc": rocm_hipcc_flags},
    # 这三行确保无需 LD_LIBRARY_PATH 也能找到依赖
    libraries=["amdhip64", "hiprtc", "rocblas", "hipblas"],   # 视你代码实际用到的库增减
    library_dirs=ROCM_LIB_DIRS + [TORCH_LIB_DIR],
    runtime_library_dirs=ROCM_LIB_DIRS + [TORCH_LIB_DIR],
    )
    ext_modules.append(fused_extension)

# ====== 保持你原本的并行编译与隔离输出目录的做法 ======
parallel = None
if 'EXT_PARALLEL' in os.environ:
    try:
        parallel = int(os.getenv('EXT_PARALLEL'))
    finally:
        pass


class BuildExtensionSeparateDir(BuildExtension):
    build_extension_patch_lock = threading.Lock()
    thread_ext_name_map = {}

    def finalize_options(self):
        if parallel is not None:
            self.parallel = parallel
        super().finalize_options()

    def build_extension(self, ext):
        with self.build_extension_patch_lock:
            if not getattr(self.compiler, "_compile_separate_output_dir", False):
                compile_orig = self.compiler.compile

                def compile_new(*args, **kwargs):
                    return compile_orig(*args, **{
                        **kwargs,
                        "output_dir": os.path.join(
                            kwargs["output_dir"],
                            self.thread_ext_name_map[threading.current_thread().ident]),
                    })
                self.compiler.compile = compile_new
                self.compiler._compile_separate_output_dir = True
        self.thread_ext_name_map[threading.current_thread().ident] = ext.name
        objects = super().build_extension(ext)
        return objects
# ====================================================


setup(
    name='sageattention',
    version='2.2.0',
    author='SageAttention team',
    license='Apache 2.0 License',
    description='Accurate and efficient plug-and-play low-bit attention.',
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/thu-ml/SageAttention',
    packages=find_packages(),
    python_requires='>=3.9',
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtensionSeparateDir} if ext_modules else {},
)
