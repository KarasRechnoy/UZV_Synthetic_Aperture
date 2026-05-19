"""Range compression: согласованный фильтр + аналитический сигнал (README §2.3)."""

import numpy as np
from scipy.signal import hilbert

from . import config


def make_cw_pulse(
    fs: float = config.SAMPLING_FREQUENCY_HZ,
    duration: float = config.PULSE_DURATION_S,
    f0: float = config.CARRIER_FREQUENCY_HZ,
) -> np.ndarray:
    """Опорный CW-импульс `p(t) = sin(2π f₀ t)` на окне [0, τ_p]."""
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    return np.sin(2.0 * np.pi * f0 * t)


def make_chirp_pulse(
    f_lo: float,
    f_hi: float,
    fs: float = config.SAMPLING_FREQUENCY_HZ,
    duration: float = 1e-3,
) -> np.ndarray:
    """Линейный ЛЧМ-импульс от `f_lo` до `f_hi` на окне [0, duration].

    Для экспериментов 38–42 кГц и 39–41 кГц (config.txt: длительность 1 мс).
    """
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    k = (f_hi - f_lo) / duration
    phase = 2.0 * np.pi * (f_lo * t + 0.5 * k * t**2)
    return np.sin(phase)


def compress(
    s: np.ndarray,
    pulse: np.ndarray,
    fs: float = config.SAMPLING_FREQUENCY_HZ,
) -> np.ndarray:
    """Согласованная фильтрация каждой строки в частотной области.

    `S_rc[p, k] = IFFT_k{ FFT_k{S[p,:]} · conj(FFT_k{p}) }` (README §2.3).

    `s` — (n_pings, n_samples) или 1-D. Возвращает действительный массив той же
    формы. Пик отклика точечной цели оказывается в `k ≈ τ_p · f_s`.
    """
    s = np.asarray(s, dtype=np.float64)
    one_d = s.ndim == 1
    if one_d:
        s = s[None, :]
    if s.ndim != 2:
        raise ValueError("s must be 1-D or 2-D")

    n = s.shape[1]
    if pulse.size > n:
        raise ValueError("pulse longer than ping window")

    pulse_padded = np.zeros(n, dtype=np.float64)
    pulse_padded[: pulse.size] = pulse

    S = np.fft.rfft(s, axis=1)
    P = np.fft.rfft(pulse_padded)
    s_rc = np.fft.irfft(S * np.conj(P), n=n, axis=1)

    return s_rc[0] if one_d else s_rc


def analytic_signal(s_rc: np.ndarray) -> np.ndarray:
    """Преобразование Гильберта по оси τ → комплексный сигнал (README §2.3).

    Несущая остаётся на f₀ (НЕ сдвигается в baseband); компенсация фазы
    несущей выполняется в backprojection.
    """
    s_an = hilbert(np.asarray(s_rc, dtype=np.float64), axis=-1)
    return s_an.astype(np.complex64)
