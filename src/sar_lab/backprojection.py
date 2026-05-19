"""Time-domain backprojection

Стенд работает в near-field, поэтому используется точная
гиперболическая (сферически-волновая) задержка без параболической
аппроксимации.
"""

import numpy as np

from . import config
from .windows import aperture_window


def _slant_range(
    x_ant: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    bistatic: bool,
    dx_tx: float,
    dx_rx: float,
    dy_rx: float,
    dz_rx: float,
) -> np.ndarray:
    """Полный путь Tx→цель→Rx для одной позиции каретки.

    Возвращает 2-D массив (n_y, n_x) — двусторонний путь в метрах.
    """
    xg = x_grid[None, :]
    yg = y_grid[:, None]

    if not bistatic:
        r = np.sqrt((x_ant - xg) ** 2 + yg ** 2)
        return 2.0 * r

    r_tx = np.sqrt((x_ant + dx_tx - xg) ** 2 + yg ** 2)
    r_rx = np.sqrt((x_ant + dx_rx - xg) ** 2 + (yg - dy_rx) ** 2 + dz_rx ** 2)
    return r_tx + r_rx


def backproject(
    s_an: np.ndarray,
    x_ant: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    sound_speed: float = config.DEFAULT_SOUND_SPEED_MPS,
    fs: float = config.SAMPLING_FREQUENCY_HZ,
    time_offset: float = 0.0,
    bistatic: bool = False,
    dx_tx: float = 0.0,
    dx_rx: float = 0.0,
    dy_rx: float = config.RX_OFFSET_RANGE_M,
    dz_rx: float = config.RX_OFFSET_VERTICAL_M,
    window: str | None = config.DEFAULT_APERTURE_WINDOW,
) -> np.ndarray:
    """Когерентная backprojection аналитического сигнала на сетку (x, y).

    Параметры
    ---------
    s_an : (n_pings, n_samples) complex
        Аналитический сигнал после range compression.
    x_ant : (n_pings,)
        Позиция каретки (азимут, м) в каждом пинге.
    x_grid, y_grid : 1-D, м
        Узлы выходной сетки. Изображение имеет форму (len(y_grid), len(x_grid)).
    bistatic : bool
        False → монастатическое приближение `τ = 2R/c`.
        True  → полная Tx/Rx-геометрия со сдвигами `dx_tx, dx_rx, dy_rx, dz_rx`.

    Фаза несущей 40 кГц уже содержится в passband-аналитическом сигнале по
    быстрому времени (Гильберт не сдвигает в baseband, README §2.3), поэтому
    выборка `S_an` в гипотетический момент задержки автоматически даёт нужную
    фазу — дополнительная компенсация `exp(j·2π f₀·τ)` НЕ применяется
    (README §2.4: `γ̂ = Σ w · S_an[p, k_p]`).

    Возвращает комплексное изображение `γ̂` формы (n_y, n_x), complex64.
    """
    s_an = np.asarray(s_an)
    if s_an.ndim != 2:
        raise ValueError("s_an must be 2-D (n_pings, n_samples)")
    x_ant = np.asarray(x_ant, dtype=np.float64)
    if x_ant.shape[0] != s_an.shape[0]:
        raise ValueError("len(x_ant) must equal n_pings")

    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    n_pings, n_samples = s_an.shape
    sample_idx = np.arange(n_samples)

    w = aperture_window(window, n_pings)

    image = np.zeros((y_grid.size, x_grid.size), dtype=np.complex128)

    for p in range(n_pings):
        if w[p] == 0.0:
            continue
        r_total = _slant_range(
            x_ant[p], x_grid, y_grid,
            bistatic=bistatic, dx_tx=dx_tx, dx_rx=dx_rx, dy_rx=dy_rx, dz_rx=dz_rx,
        )
        # Эхо реальной цели лежит на τ = 2R/c + t0 (постоянная задержка
        # тракта/триггера). Без t0 кривизна гиперболы не совпадает -> смаз.
        tau = r_total / sound_speed + time_offset
        k = tau * fs  # дробный индекс отсчёта

        flat_k = k.ravel()
        valid = (flat_k >= 0.0) & (flat_k <= n_samples - 1)

        row = s_an[p]
        contrib = np.zeros(flat_k.size, dtype=np.complex128)
        # Линейная интерполяция по дробному индексу (отдельно Re/Im).
        kv = flat_k[valid]
        re = np.interp(kv, sample_idx, row.real)
        im = np.interp(kv, sample_idx, row.imag)
        contrib[valid] = re + 1j * im
        image += w[p] * contrib.reshape(image.shape)

    return image.astype(np.complex64)
