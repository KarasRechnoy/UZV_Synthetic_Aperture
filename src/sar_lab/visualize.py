"""Стандартные плоты SAR-изображения (README §5)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def to_db(image: np.ndarray, floor_db: float = -50.0) -> np.ndarray:
    """Амплитуда комплексного изображения в дБ, нормированная на пик."""
    mag = np.abs(image)
    peak = mag.max()
    if peak <= 0:
        return np.full(mag.shape, floor_db)
    db = 20.0 * np.log10(np.maximum(mag / peak, 1e-12))
    return np.clip(db, floor_db, 0.0)


def plot_sar_image(
    image: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    ground_truth: dict[str, tuple[float, float]] | None = None,
    dynamic_range_db: float = 28.0,
    title: str = "SAR backprojection",
):
    """Изображение в дБ-шкале. `ground_truth` опционален (можно без координат)."""
    db = to_db(image, floor_db=-dynamic_range_db)
    fig, ax = plt.subplots(figsize=(9, 7))
    extent = [
        x_grid[0] * 1e3, x_grid[-1] * 1e3,
        y_grid[0] * 1e3, y_grid[-1] * 1e3,
    ]
    im = ax.imshow(
        db, origin="lower", aspect="auto", extent=extent,
        cmap="inferno", vmin=-dynamic_range_db, vmax=0.0,
    )
    if ground_truth:
        for oid, (gx, gy) in ground_truth.items():
            ax.plot(gx * 1e3, gy * 1e3, "c+", markersize=12, markeredgewidth=2)
            ax.annotate(oid, (gx * 1e3, gy * 1e3), color="cyan",
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("азимут x, мм")
    ax.set_ylabel("дальность y, мм")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="дБ (отн. пика)")
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
