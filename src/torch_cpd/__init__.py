"""torch-cpd: non-rigid Coherent Point Drift on CPU or GPU with PyTorch."""

from torch_cpd.deformable import (
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_CUDA,
    GPU_WORTHWHILE_POINTS,
    CPDResult,
    apply_deformation,
    cpd_deformable,
    cuda_available,
    describe_devices,
    resolve_device,
    torch_available,
)

__version__ = "0.1.0"

__all__ = [
    "CPDResult",
    "DEVICE_AUTO",
    "DEVICE_CPU",
    "DEVICE_CUDA",
    "GPU_WORTHWHILE_POINTS",
    "__version__",
    "apply_deformation",
    "cpd_deformable",
    "cuda_available",
    "describe_devices",
    "resolve_device",
    "torch_available",
]
