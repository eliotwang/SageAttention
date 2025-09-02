# sageattention/__init__.py
from importlib import import_module
import torch as _torch

__all__ = []

# ROCm: 导出便捷别名 qattn
if getattr(_torch.version, "hip", None):
    try:
        from . import _qattn_rocm as qattn
        __all__.append("qattn")
    except Exception:
        # 构建未完成时允许跳过
        pass
else:
    # CUDA 环境下保持兼容（任选其一可用的 qattn）
    for _name in ("_qattn_sm89", "_qattn_sm90", "_qattn_sm80"):
        try:
            qattn = import_module(f".{_name}", __name__)
            __all__.append("qattn")
            break
        except Exception:
            continue

def __getattr__(name):
    # 懒加载旧接口
    if name in ("sageattn", "sageattn_varlen"):
        mod = import_module(".core", __name__)
        return getattr(mod, name)

    if name == "_fused":
        # 1) 先尝试常规的包内相对导入
        try:
            return import_module("._fused", __name__)
        except ModuleNotFoundError:
            # 2) 尝试把 build/lib.* 下的包目录临时加入 sys.path
            import os, sys, glob
            _pkg_dir = os.path.dirname(__file__)           # .../sageattention
            _root = os.path.dirname(_pkg_dir)              # 项目根目录
            # 兼容多种构建标签：lib.linux-...-cpython-...
            for _cand in sorted(glob.glob(os.path.join(_root, "build", "lib.*"))):
                _p = os.path.join(_cand, "sageattention")
                if os.path.isdir(_p) and _p not in sys.path:
                    sys.path.insert(0, _p)
            try:
                return import_module("._fused", __name__)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "sageattention._fused not found. "
                    "解决办法：\n"
                    "  1) 在项目根运行 `python setup.py build_ext --inplace`，"
                    "     确保 ./sageattention/ 里生成 _fused*.so；或\n"
                    "  2) 运行 `pip install --no-binary :all: .` 并从非源码目录运行脚本；或\n"
                    "  3) 临时 `export PYTHONPATH=./build/lib.*:$PYTHONPATH` 后再运行。"
                ) from e

    raise AttributeError(f"module {__name__} has no attribute {name}")

try:
    # prefer ROCm kernel if available
    from . import _qattn_rocm as _qattn_backend
    _impl = getattr(_qattn_backend, "sageattn", None) or getattr(_qattn_backend, "forward", None)
except Exception:
    _qattn_backend, _impl = None, None

if _impl is None:
    try:
        # fall back to python core implementation
        from .core import sageattn as _impl
    except Exception:
        _impl = None

def _stub(name):
    def _f(*args, **kwargs):
        raise RuntimeError(f"{name} is not available on this build; no fallback found.")
    return _f

def _export(alias, target):
    globals()[alias] = target
    try:
        __all__.append(alias)
    except Exception:
        pass

# 2) export the names diffusers tries to import
_needed = [
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda_sm90",
    "sageattn_qk_int8_pv_fp16_triton",
    # 你可以按需要再加其它被上游 import 的名字
]

for _name in _needed:
    _export(_name, _impl if _impl is not None else _stub(_name))