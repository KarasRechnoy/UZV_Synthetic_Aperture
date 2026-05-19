"""Автофокус по резкости изображения (README §2.7).

Метрики качества фокусировки и одномерная оптимизация скорости звука `c`
методом Брента (без перебора по сетке, без знания координат целей).
"""

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize_scalar


def image_entropy(image: np.ndarray) -> float:
    """Энтропия Шеннона нормированной интенсивности. МЕНЬШЕ = резче.

    `p = |I|² / Σ|I|²`,  `H = −Σ p·ln p`. Сфокусированная сцена концентрирует
    энергию в точках → низкая энтропия.
    """
    i2 = np.abs(image).astype(np.float64) ** 2
    total = i2.sum()
    if total <= 0:
        return 0.0
    p = (i2 / total).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def image_contrast(image: np.ndarray) -> float:
    """Контраст интенсивности `std(|I|²)/mean(|I|²)`. БОЛЬШЕ = резче."""
    i2 = np.abs(image).astype(np.float64) ** 2
    m = i2.mean()
    return float(i2.std() / m) if m > 0 else 0.0


def autofocus_scalar(
    make_image: Callable[[float], np.ndarray],
    bounds: tuple[float, float],
    *,
    metric: str = "entropy",
    xatol: float = 0.05,
) -> tuple[float, np.ndarray, list[tuple[float, float]]]:
    """Подобрать скалярный параметр (обычно `c`) по резкости изображения.

    `make_image(value) -> комплексное изображение`. Для `metric="entropy"`
    минимизируется энтропия, для `"contrast"` — максимизируется контраст
    (минимизируется −contrast). Возвращает `(best_value, best_image, history)`.
    """
    history: list[tuple[float, float]] = []
    cache: dict[float, np.ndarray] = {}

    def cost(v: float) -> float:
        img = make_image(float(v))
        cache[float(v)] = img
        if metric == "entropy":
            c = image_entropy(img)
        elif metric == "contrast":
            c = -image_contrast(img)
        else:
            raise ValueError(f"unknown metric {metric!r}")
        history.append((float(v), c))
        return c

    res = minimize_scalar(
        cost, bounds=bounds, method="bounded",
        options={"xatol": xatol},
    )
    best_v = float(res.x)
    # ближайшее вычисленное изображение к оптимуму
    best_key = min(cache, key=lambda k: abs(k - best_v))
    return best_v, cache[best_key], history
