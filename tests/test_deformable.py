"""Tests for torch_cpd.

The point is not that the GPU path runs, but that it computes the same thing
as the CPU path and as the reference implementation. A fast wrong answer is
worse than a slow right one, so the equivalence check against pycpd is the
test that matters here.
"""

import numpy as np
import pytest

from torch_cpd import (
    DEVICE_CPU,
    DEVICE_CUDA,
    apply_deformation,
    cpd_deformable,
    cuda_available,
    describe_devices,
    resolve_device,
    torch_available,
)

requires_torch = pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
requires_cuda = pytest.mark.skipif(not cuda_available(), reason="no usable CUDA device")


def _bent_pair(n=300, seed=0, bend=0.08):
    """A source cloud and a smoothly deformed copy - the situation CPD is for."""
    rng = np.random.default_rng(seed)
    source = rng.uniform(-1.0, 1.0, size=(n, 3)) * np.array([4.0, 3.0, 1.0])
    target = source.copy()
    u = source[:, 0] / 4.0
    target[:, 1] += bend * 4.0 * u**2
    target[:, 2] += bend * 2.0 * u
    target += rng.normal(scale=0.01, size=target.shape)
    return source, target


# --- device selection, no GPU needed --------------------------------------


def test_describe_devices_always_says_something_true():
    text = describe_devices()
    assert text
    if not torch_available():
        assert "not installed" in text
    elif not cuda_available():
        assert "CPU" in text
    else:
        assert "GPU available" in text


def test_cpu_is_honoured_even_when_a_gpu_exists():
    assert resolve_device(DEVICE_CPU, n_points=100000) == DEVICE_CPU


def test_a_requested_gpu_degrades_to_cpu_when_unavailable():
    """A missing GPU must slow a batch down, never stop it."""
    expected = DEVICE_CUDA if cuda_available() else DEVICE_CPU
    assert resolve_device(DEVICE_CUDA, n_points=5000) == expected


def test_auto_keeps_small_problems_on_the_cpu():
    assert resolve_device("auto", n_points=10) == DEVICE_CPU


@pytest.mark.skipif(not cuda_available(), reason="no usable CUDA device")
def test_auto_uses_the_gpu_for_big_problems():
    assert resolve_device("auto", n_points=5000) == DEVICE_CUDA


# --- correctness -----------------------------------------------------------


@requires_torch
def test_rejects_bad_arguments():
    source, target = _bent_pair(n=50)
    with pytest.raises(ValueError, match="positive"):
        cpd_deformable(source, target, beta=0.0)
    with pytest.raises(ValueError, match="positive"):
        cpd_deformable(source, target, beta=1.0, alpha=0.0)
    with pytest.raises(ValueError, match="different dimensionality"):
        cpd_deformable(source, target[:, :2], beta=1.0)
    with pytest.raises(ValueError, match=r"\(n_points, n_dim\)"):
        cpd_deformable(np.zeros(5), target, beta=1.0)


@requires_torch
def test_deformation_moves_the_source_toward_the_target():
    source, target = _bent_pair(n=250, bend=0.12)
    before = np.linalg.norm(source - target, axis=1).mean()

    result = cpd_deformable(
        source, target, beta=2.0, alpha=1.0, max_iterations=100, device=DEVICE_CPU
    )
    after = np.linalg.norm(result.transformed - target, axis=1).mean()
    assert after < 0.5 * before, f"{before:.4f} -> {after:.4f}"


@requires_torch
def test_the_out_of_sample_extension_matches_the_fitted_points():
    """Landmarks are not sampled points, so the field must be re-evaluable at
    arbitrary positions - and must agree at the positions it was fitted on."""
    source, target = _bent_pair(n=200)
    result = cpd_deformable(source, target, beta=2.0, max_iterations=60, device=DEVICE_CPU)
    reapplied = apply_deformation(source, result, device=DEVICE_CPU)
    assert np.allclose(reapplied, result.transformed, atol=1e-4)


@requires_torch
def test_a_stiffer_model_fits_less_closely():
    """Regularisation trades fit for smoothness: a much stiffer model must leave a larger
    residual against the target than a soft one."""
    source, target = _bent_pair(n=200, bend=0.15)

    def residual(alpha):
        result = cpd_deformable(
            source, target, beta=2.0, alpha=alpha, max_iterations=80, device=DEVICE_CPU
        )
        return np.linalg.norm(result.transformed - target, axis=1).mean()

    assert residual(1000.0) > 1.5 * residual(0.1)


@requires_torch
def test_identical_clouds_are_barely_moved():
    source, _t = _bent_pair(n=150)
    result = cpd_deformable(source, source.copy(), beta=2.0, max_iterations=50, device=DEVICE_CPU)
    scale = float(np.linalg.norm(source.max(0) - source.min(0)))
    assert np.linalg.norm(result.transformed - source, axis=1).max() < 0.02 * scale


# --- the check that actually matters --------------------------------------


@requires_cuda
def test_gpu_and_cpu_agree():
    """Same algorithm, same answer - the device may only change the speed."""
    source, target = _bent_pair(n=400, seed=3, bend=0.1)
    kwargs = dict(beta=2.0, alpha=1.0, max_iterations=80, tolerance=1e-9)

    on_cpu = cpd_deformable(source, target, device=DEVICE_CPU, **kwargs)
    on_gpu = cpd_deformable(source, target, device=DEVICE_CUDA, **kwargs)

    assert on_cpu.device == DEVICE_CPU and on_gpu.device == DEVICE_CUDA
    scale = float(np.linalg.norm(source.max(0) - source.min(0)))
    difference = np.linalg.norm(on_cpu.transformed - on_gpu.transformed, axis=1).max()
    assert difference < 1e-3 * scale, f"CPU and GPU differ by {difference:.6f}"


@requires_torch
def test_this_implementation_agrees_with_pycpd():
    """The reference implementation is pycpd, so this must reproduce it.

    Not bit-identical - the convergence check differs slightly - but the two
    must place points in the same place to well within the noise of the data.
    """
    pycpd = pytest.importorskip("pycpd")

    source, target = _bent_pair(n=250, seed=5, bend=0.1)
    beta, alpha, iterations = 2.0, 1.0, 120

    reference = pycpd.DeformableRegistration(
        X=target, Y=source, beta=beta, alpha=alpha, max_iterations=iterations
    )
    reference.register()

    ours = cpd_deformable(
        source,
        target,
        beta=beta,
        alpha=alpha,
        max_iterations=iterations,
        tolerance=0.0,
        device=DEVICE_CPU,
        dtype="float64",
    )

    scale = float(np.linalg.norm(source.max(0) - source.min(0)))
    difference = np.linalg.norm(reference.TY - ours.transformed, axis=1).mean()
    assert difference < 0.01 * scale, (
        f"mean disagreement with pycpd {difference:.5f} on a {scale:.2f}-unit object"
    )


@requires_torch
def test_result_records_what_it_did():
    source, target = _bent_pair(n=120)
    result = cpd_deformable(source, target, beta=2.0, max_iterations=40, device=DEVICE_CPU)
    assert result.iterations <= 40
    assert result.beta == 2.0
    assert result.sigma2 > 0
    assert result.weights.shape == source.shape
    assert result.source.shape == source.shape
    assert result.device in (DEVICE_CPU, DEVICE_CUDA)
