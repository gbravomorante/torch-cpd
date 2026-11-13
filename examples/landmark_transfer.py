"""Transfer landmarks from one shape to another with torch-cpd.

A synthetic 2-D outline is bent and stretched into a second shape. CPD
registers the two outlines, and the fitted displacement field then carries a
set of landmarks - points that were never part of the registration - onto the
target. Run it to reproduce ``docs/img/landmark_transfer.png``.

    python examples/landmark_transfer.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from torch_cpd import apply_deformation, cpd_deformable


def outline(n: int = 400) -> np.ndarray:
    """A smooth, bone-like closed outline: two lobes joined by a waist."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = 1.0 + 0.35 * np.cos(2 * t) - 0.12 * np.cos(4 * t)
    return np.column_stack([1.8 * r * np.cos(t), r * np.sin(t)])


def deform(points: np.ndarray) -> np.ndarray:
    """A smooth, non-affine change of shape: bending, tapering and a shear."""
    x, y = points[:, 0], points[:, 1]
    return np.column_stack(
        [
            1.1 * x + 0.15 * y,
            0.85 * y + 0.18 * x**2 - 0.25 + 0.08 * np.sin(2.0 * x),
        ]
    )


def main() -> None:
    rng = np.random.default_rng(7)
    source = outline()
    target = deform(source) + rng.normal(scale=0.01, size=source.shape)
    target = target[rng.permutation(len(target))[:320]]  # unequal, unordered

    # Six landmarks, chosen on the source outline but held out of the fit.
    landmarks = outline(12)[::2] * 0.98
    truth = deform(landmarks)

    result = cpd_deformable(source, target, beta=1.0, alpha=0.5, max_iterations=200)
    moved = apply_deformation(landmarks, result)

    error = np.linalg.norm(moved - truth, axis=1)
    span = np.ptp(target, axis=0).max()
    print(f"device={result.device}  iterations={result.iterations}  converged={result.converged}")
    print(
        f"landmark error: mean {error.mean():.3f}, max {error.max():.3f} "
        f"({100 * error.mean() / span:.1f}% of the shape's length)"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), sharex=True, sharey=True)
    grey, blue, orange = "#9aa0a6", "#1a73e8", "#e8710a"
    panels = [
        ("1  Source and target", source, landmarks),
        ("2  Registered source", result.transformed, None),
        ("3  Landmarks carried through the field", result.transformed, moved),
    ]
    for ax, (title, cloud, marks) in zip(axes, panels, strict=True):
        ax.scatter(*target.T, s=4, c=grey, label="target")
        ax.scatter(*cloud.T, s=4, c=blue, label="source")
        if marks is not None:
            ax.scatter(
                *marks.T,
                s=60,
                c=orange,
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
                label="landmarks",
            )
        if marks is moved:
            ax.scatter(
                *truth.T,
                s=150,
                facecolors="none",
                edgecolors="#202124",
                linewidths=1.0,
                zorder=4,
                label="true position",
            )
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        markerscale=1.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = Path(__file__).resolve().parents[1] / "docs" / "img" / "landmark_transfer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
