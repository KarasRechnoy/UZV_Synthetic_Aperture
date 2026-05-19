import numpy as np

from sar_lab import config
from sar_lab.backprojection import backproject
from sar_lab.range_compression import analytic_signal, compress, make_cw_pulse
from synthetic import point_target_raw


def _focused_image(x0, y0):
    dx = config.dx_from_motor_steps(92)
    n_pings = 161
    x_ant = (np.arange(n_pings) - n_pings // 2) * dx  # центр апертуры в 0

    n_samples = 9000
    s = point_target_raw(x0, y0, x_ant=x_ant, n_samples=n_samples)
    s_an = analytic_signal(compress(s, make_cw_pulse()))

    x_grid = np.arange(-0.10, 0.10 + 1e-9, 0.002)
    y_grid = np.arange(0.50, 0.70 + 1e-9, 0.005)
    img = backproject(s_an, x_ant, x_grid, y_grid, bistatic=False)
    return img, x_grid, y_grid


def test_backprojection_single_peak_at_target():
    x0, y0 = 0.0, 0.60
    img, x_grid, y_grid = _focused_image(x0, y0)

    mag = np.abs(img)
    jy, jx = np.unravel_index(np.argmax(mag), mag.shape)
    x_peak, y_peak = x_grid[jx], y_grid[jy]

    # Азимут локализуется тонко; дальность — в пределах CW-разрешения (~21 мм).
    assert abs(x_peak - x0) <= 0.004
    assert abs(y_peak - y0) <= 0.020


def test_backprojection_offset_target():
    x0, y0 = 0.04, 0.60
    img, x_grid, y_grid = _focused_image(x0, y0)
    mag = np.abs(img)
    jy, jx = np.unravel_index(np.argmax(mag), mag.shape)
    assert abs(x_grid[jx] - x0) <= 0.005
    assert abs(y_grid[jy] - y0) <= 0.020


def test_backprojection_peak_dominates_background():
    img, x_grid, y_grid = _focused_image(0.0, 0.60)
    mag = np.abs(img)
    peak = mag.max()
    # Энергия сфокусирована: пик заметно выше медианного уровня по сцене.
    assert peak > 10.0 * np.median(mag)


def test_backprojection_azimuth_irw_matches_theory():
    """IRW по азимуту ≈ λR/(2·L_SA), уширенный окном Хэмминга (~1.47×)."""
    dx = config.dx_from_motor_steps(92)
    n_pings = 161
    x_ant = (np.arange(n_pings) - n_pings // 2) * dx
    s = point_target_raw(0.0, 0.60, x_ant=x_ant, n_samples=9000)
    s_an = analytic_signal(compress(s, make_cw_pulse()))

    x_grid = np.arange(-0.05, 0.05 + 1e-9, 0.0005)
    y_grid = np.arange(0.55, 0.65 + 1e-9, 0.002)
    img = np.abs(backproject(s_an, x_ant, x_grid, y_grid, bistatic=False))

    jy = int(np.argmax(img.max(axis=1)))
    prof = img[jy]
    above = x_grid[prof >= prof.max() / np.sqrt(2.0)]
    irw = above.max() - above.min()

    L_sa = n_pings * dx
    theory = config.WAVELENGTH_M * 0.60 / (2.0 * L_sa)  # ≈ 7.5 мм
    assert theory <= irw <= 2.5 * theory


def test_backprojection_output_shape_and_dtype():
    img, x_grid, y_grid = _focused_image(0.0, 0.60)
    assert img.shape == (y_grid.size, x_grid.size)
    assert img.dtype == np.complex64
