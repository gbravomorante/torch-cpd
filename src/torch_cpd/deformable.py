"""Non-rigid Coherent Point Drift in PyTorch.

Coherent Point Drift (Myronenko & Song, 2010) registers one point cloud onto
another by treating the source points as the centroids of a Gaussian mixture
and moving them coherently towards the target. The deformable variant has two
expensive steps per iteration, both of which a GPU handles well:

* the E-step, an ``(N x M)`` distance matrix between the two clouds, and
* the M-step, a dense ``(M x M)`` linear solve.

At 3,000 points per cloud that is nine million distances and a solve of about
2.7e10 operations per iteration. On a CPU that dominates the runtime; on a GPU
it takes a fraction of it.

The same code runs on CPU and CUDA, so the two paths can be checked against
each other and against the reference NumPy implementation, ``pycpd``. The test
suite does both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"

#: Below this many points, transfer and kernel-launch overhead is comparable to
#: the arithmetic and the GPU stops paying for itself. Measured on an
#: RTX 4080 SUPER: at 500 points GPU 0.9 s vs CPU 1.1 s; at 3,000 points 2.0 s vs 50.6 s.
GPU_WORTHWHILE_POINTS = 400

__all__ = [
    "CPDResult",
    "DEVICE_AUTO",
    "DEVICE_CPU",
    "DEVICE_CUDA",
    "GPU_WORTHWHILE_POINTS",
    "apply_deformation",
    "cpd_deformable",
    "cuda_available",
    "describe_devices",
    "resolve_device",
    "torch_available",
]


@dataclass
class CPDResult:
    """A fitted deformation, with everything needed to move *any* point.

    ``weights`` and ``source`` together define a smooth displacement field, so
    points that were not part of the registered cloud (anatomical landmarks,
    for example) can be carried through it with :func:`apply_deformation`.
    """

    transformed: np.ndarray  #: ``(M, D)`` the deformed source cloud
    weights: np.ndarray  #: ``(M, D)`` the coefficient matrix ``W``
    source: np.ndarray  #: ``(M, D)`` the control points ``W`` refers to
    beta: float  #: Gaussian kernel width used
    sigma2: float  #: final residual variance
    iterations: int  #: iterations actually run
    device: str  #: ``"cpu"`` or ``"cuda"``
    converged: bool  #: whether the tolerance was reached


def torch_available() -> bool:
    """Whether PyTorch can be imported."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def cuda_available() -> bool:
    """Whether PyTorch reports a usable CUDA device. Never raises."""
    if not torch_available():
        return False
    import torch

    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken driver must not crash the caller
        return False


def describe_devices() -> str:
    """A one-line, human-readable statement of what will actually be used."""
    if not torch_available():
        return "PyTorch is not installed."
    import torch

    if not cuda_available():
        return f"PyTorch {torch.__version__} with no usable CUDA device: running on the CPU."
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"GPU available: {name} ({total:.1f} GB), CUDA {torch.version.cuda}."


def resolve_device(requested: str = DEVICE_AUTO, n_points: int = 0) -> str:
    """Choose the device to run on.

    ``"auto"`` means: the GPU whenever one is usable *and* the problem is large
    enough to be worth the transfer. An unavailable GPU always degrades to the
    CPU rather than raising, so a missing driver slows a batch down instead of
    stopping it.
    """
    requested = (requested or DEVICE_AUTO).lower()
    if requested == DEVICE_CPU:
        return DEVICE_CPU
    if requested == DEVICE_CUDA:
        return DEVICE_CUDA if cuda_available() else DEVICE_CPU
    if not cuda_available():
        return DEVICE_CPU
    return DEVICE_CUDA if n_points >= GPU_WORTHWHILE_POINTS else DEVICE_CPU


def _gaussian_kernel(points_a, points_b, beta: float):
    """``exp(-||a - b||^2 / (2 beta^2))`` without a difference tensor.

    ``torch.cdist`` is used instead of the expansion ``a^2 - 2ab + b^2``, which
    loses precision in float32 when coordinates are large, as scanner
    coordinates often are (hundreds of millimetres from the origin).
    """
    import torch

    distances = torch.cdist(points_a, points_b)
    return torch.exp(-(distances**2) / (2.0 * beta * beta))


def cpd_deformable(
    source: np.ndarray,
    target: np.ndarray,
    beta: float,
    alpha: float = 1.0,
    max_iterations: int = 150,
    tolerance: float = 1e-5,
    device: str = DEVICE_AUTO,
    dtype: str = "float32",
) -> CPDResult:
    """Deform ``source`` onto ``target`` with non-rigid CPD.

    Parameters
    ----------
    source, target:
        Point clouds of shape ``(M, D)`` and ``(N, D)``. They need not have the
        same number of points.
    beta:
        Width of the Gaussian smoothness kernel, in the coordinates' own units.
        Larger values give smoother, more global deformations.
    alpha:
        Regularisation weight. Larger values give stiffer deformations.
    max_iterations, tolerance:
        Stop after ``max_iterations`` or when the residual variance changes by
        less than ``tolerance``.
    device:
        ``"auto"``, ``"cpu"`` or ``"cuda"``. See :func:`resolve_device`.
    dtype:
        ``"float32"`` (default; consumer GPUs run float64 at a fraction of the
        speed) or ``"float64"``.
    """
    if not torch_available():
        raise ValueError("cpd_deformable needs PyTorch: pip install torch")
    import torch

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"Both clouds must be (n_points, n_dim); got {source.shape} and {target.shape}."
        )
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"The clouds have different dimensionality: {source.shape[1]} and {target.shape[1]}."
        )
    if beta <= 0:
        raise ValueError(f"beta must be positive; got {beta}.")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive; got {alpha}.")

    chosen = resolve_device(device, n_points=max(len(source), len(target)))
    torch_device = torch.device(chosen)
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32

    Y = torch.as_tensor(source, dtype=torch_dtype, device=torch_device)  # (M, D)
    X = torch.as_tensor(target, dtype=torch_dtype, device=torch_device)  # (N, D)
    M, D = Y.shape
    N = X.shape[0]

    G = _gaussian_kernel(Y, Y, beta)  # (M, M), fixed across iterations
    W = torch.zeros((M, D), dtype=torch_dtype, device=torch_device)
    T = Y.clone()

    # Initial isotropic variance: mean squared distance over all pairs.
    sigma2 = (torch.cdist(X, Y) ** 2).sum() / (D * M * N)
    eye = torch.eye(M, dtype=torch_dtype, device=torch_device)

    converged = False
    used = 0
    for used in range(1, int(max_iterations) + 1):  # noqa: B007 - read after the loop
        # E-step: responsibility of each source point for each target point.
        P = torch.exp(-(torch.cdist(X, T) ** 2) / (2.0 * sigma2))  # (N, M)
        # With no uniform-noise weight this is a plain normalisation; the
        # epsilon guards a target point far from every source point.
        P = P / (P.sum(dim=1, keepdim=True) + 1e-10)

        Pt1 = P.sum(dim=0)  # (M,)
        PX = P.T @ X  # (M, D)
        Np = Pt1.sum()

        # M-step: solve (diag(Pt1) G + alpha sigma2 I) W = PX - diag(Pt1) Y.
        left = torch.diag(Pt1) @ G + alpha * sigma2 * eye
        right = PX - torch.diag(Pt1) @ Y
        W = torch.linalg.solve(left, right)
        T_new = Y + G @ W

        # Variance update and convergence on its change.
        trace_xpx = (X * X).sum(dim=1) @ P.sum(dim=1)
        trace_tpt = (T_new * T_new).sum(dim=1) @ Pt1
        trace_ptx = (T_new * PX).sum()
        sigma2_new = torch.clamp((trace_xpx - 2.0 * trace_ptx + trace_tpt) / (Np * D), min=1e-12)

        change = torch.abs(sigma2_new - sigma2)
        T, sigma2 = T_new, sigma2_new
        if float(change) < tolerance:
            converged = True
            break

    return CPDResult(
        transformed=T.detach().cpu().numpy().astype(float),
        weights=W.detach().cpu().numpy().astype(float),
        source=source,
        beta=float(beta),
        sigma2=float(sigma2),
        iterations=used,
        device=chosen,
        converged=converged,
    )


def apply_deformation(
    points: np.ndarray, result: CPDResult, device: str = DEVICE_AUTO
) -> np.ndarray:
    """Carry arbitrary points through a fitted deformation.

    The field is defined everywhere as ``x + G(x, Y) W``, so the points need
    not belong to the registered cloud. This out-of-sample extension is what
    makes CPD usable for transferring landmarks between shapes.
    """
    points = np.asarray(points, dtype=float)
    if not torch_available():
        squared = ((points[:, None, :] - result.source[None, :, :]) ** 2).sum(axis=2)
        kernel = np.exp(-squared / (2.0 * result.beta**2))
        return points + kernel @ result.weights

    import torch

    chosen = resolve_device(device, n_points=len(result.source))
    torch_device = torch.device(chosen)
    dtype = torch.float32

    P = torch.as_tensor(points, dtype=dtype, device=torch_device)
    Y = torch.as_tensor(result.source, dtype=dtype, device=torch_device)
    W = torch.as_tensor(result.weights, dtype=dtype, device=torch_device)
    kernel = _gaussian_kernel(P, Y, result.beta)
    return (P + kernel @ W).detach().cpu().numpy().astype(float)
