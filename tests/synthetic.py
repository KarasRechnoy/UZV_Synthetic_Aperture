"""Синтетический прямой модель: одна точечная цель (README §2.1).

Используется тестами для проверки гиперболической структуры raw-данных и
корректности фокусировки.
"""

import numpy as np

from sar_lab import config


def point_target_raw(
    x0: float,
    y0: float,
    *,
    x_ant: np.ndarray,
    n_samples: int,
    sound_speed: float = config.DEFAULT_SOUND_SPEED_MPS,
    fs: float = config.SAMPLING_FREQUENCY_HZ,
    f0: float = config.CARRIER_FREQUENCY_HZ,
    pulse_duration: float = config.PULSE_DURATION_S,
    amplitude: float = 1000.0,
    monostatic: bool = True,
) -> np.ndarray:
    """Сырая матрица `S[p, k]` для одной точечной цели (монастатика).

    `s_p(t) = A · sin(2π f₀ (t − τ_p))` на окне `[τ_p, τ_p + τ_pulse]`,
    где `τ_p = 2·R_p / c`, `R_p = sqrt((x_p − x0)² + y0²)`.
    """
    x_ant = np.asarray(x_ant, dtype=np.float64)
    n_pings = x_ant.size
    s = np.zeros((n_pings, n_samples), dtype=np.float64)
    n_pulse = int(round(pulse_duration * fs))
    t_pulse = np.arange(n_pulse) / fs

    for p, xp in enumerate(x_ant):
        r = np.hypot(xp - x0, y0)
        tau = 2.0 * r / sound_speed
        k0 = int(round(tau * fs))
        if k0 < 0 or k0 + n_pulse > n_samples:
            continue
        # Точная дробная задержка в фазе несущей.
        frac_delay = tau - k0 / fs
        wave = amplitude * np.sin(2.0 * np.pi * f0 * (t_pulse - frac_delay))
        s[p, k0 : k0 + n_pulse] += wave

    return s
