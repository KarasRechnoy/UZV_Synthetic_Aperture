"""Оконные функции для апертуры и оси дальности (README §2.6)."""

import numpy as np
from scipy.signal import windows as _w


def aperture_window(name: str | None, n: int) -> np.ndarray:
    """Окно длины `n`.

    Поддерживаются: None / "none" / "rect" (прямоугольное), "hamming",
    "hann", "blackman", "kaiser" (β=6), "taylor" (−35 дБ).
    """
    if n <= 0:
        raise ValueError("window length must be positive")
    if name is None:
        return np.ones(n)

    key = name.lower()
    if key in ("none", "rect", "rectangular", "boxcar"):
        return np.ones(n)
    if key == "hamming":
        return _w.hamming(n, sym=False)
    if key == "hann":
        return _w.hann(n, sym=False)
    if key == "blackman":
        return _w.blackman(n, sym=False)
    if key == "kaiser":
        return _w.kaiser(n, beta=6.0, sym=False)
    if key == "taylor":
        return _w.taylor(n, nbar=4, sll=35, sym=False)
    raise ValueError(f"unknown window: {name!r}")
