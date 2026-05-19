"""Предобработка сырых данных: вычитание фона (README §4.1, шаг 3)."""

import numpy as np


def subtract_background(scene: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """`S_clean = S_scene - S_empty`.

    Возвращает float64. Если число пингов/отсчётов отличается, обрезается до
    общего минимума (съёмки empty и scene могут различаться на ±1 пинг).
    """
    if scene.ndim != 2 or empty.ndim != 2:
        raise ValueError("scene and empty must be 2-D (n_pings, n_samples)")

    n_p = min(scene.shape[0], empty.shape[0])
    n_s = min(scene.shape[1], empty.shape[1])
    s = scene[:n_p, :n_s].astype(np.float64)
    e = empty[:n_p, :n_s].astype(np.float64)
    return s - e
