"""Time torch-cpd on CPU and GPU against the reference pycpd implementation.

    python benchmarks/benchmark.py --sizes 500 1000 3000 --iterations 150

Each run deforms a random 3-D cloud onto a smoothly bent copy of itself with a
fixed number of iterations (tolerance 0), so every backend does the same work.
pycpd is skipped automatically if it is not installed, and CUDA if no GPU is
available.
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import numpy as np

from torch_cpd import cpd_deformable, cuda_available


def bent_pair(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source = rng.uniform(-1.0, 1.0, size=(n, 3)) * np.array([4.0, 3.0, 1.0])
    u = source[:, 0] / 4.0
    target = source + np.column_stack([np.zeros(n), 0.4 * u**2, 0.2 * u])
    return source, target + rng.normal(scale=0.01, size=target.shape)


def timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 3000])
    parser.add_argument("--iterations", type=int, default=150)
    args = parser.parse_args()

    try:
        import pycpd
    except ImportError:
        pycpd = None

    kwargs = dict(beta=2.0, alpha=1.0, max_iterations=args.iterations, tolerance=0.0)
    print(f"{'points':>7}  {'pycpd':>9}  {'torch cpu':>9}  {'torch cuda':>10}")
    for n in args.sizes:
        source, target = bent_pair(n)
        row = [f"{n:>7}"]
        if pycpd is not None:
            reg = pycpd.DeformableRegistration(
                X=target,
                Y=source,
                beta=2.0,
                alpha=1.0,
                max_iterations=args.iterations,
                tolerance=0.0,
            )
            row.append(f"{timed(reg.register):>8.1f}s")
        else:
            row.append(f"{'-':>9}")
        run = partial(cpd_deformable, source, target, **kwargs)
        row.append(f"{timed(partial(run, device='cpu')):>8.1f}s")
        if cuda_available():
            cpd_deformable(source[:50], target[:50], device="cuda", beta=2.0, max_iterations=2)
            row.append(f"{timed(partial(run, device='cuda')):>9.1f}s")
        else:
            row.append(f"{'-':>10}")
        print("  ".join(row))


if __name__ == "__main__":
    main()
