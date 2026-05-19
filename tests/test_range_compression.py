import numpy as np

from sar_lab import config
from sar_lab.range_compression import analytic_signal, compress, make_cw_pulse
from synthetic import point_target_raw


def _expected_cell(r, c=config.DEFAULT_SOUND_SPEED_MPS, fs=config.SAMPLING_FREQUENCY_HZ):
    return int(round(2.0 * r / c * fs))


def test_make_cw_pulse_length():
    pulse = make_cw_pulse()
    assert pulse.size == int(round(config.PULSE_DURATION_S * config.SAMPLING_FREQUENCY_HZ))
    assert np.abs(pulse).max() <= 1.0 + 1e-9


def test_range_compression_peak_at_target_cell():
    y0 = 0.6
    x_ant = np.array([0.0])  # одна позиция, цель прямо перед антенной
    n_samples = 9000
    s = point_target_raw(0.0, y0, x_ant=x_ant, n_samples=n_samples)

    s_rc = compress(s, make_cw_pulse())
    assert s_rc.shape == s.shape

    peak = int(np.argmax(np.abs(s_rc[0])))
    expected = _expected_cell(y0)
    # CW matched filter: пик в ячейке задержки в пределах ~разрешения.
    assert abs(peak - expected) <= 5


def test_analytic_signal_is_complex_envelope():
    y0 = 0.6
    s = point_target_raw(0.0, y0, x_ant=np.array([0.0]), n_samples=9000)
    s_rc = compress(s, make_cw_pulse())
    s_an = analytic_signal(s_rc)

    assert s_an.dtype == np.complex64
    assert s_an.shape == s_rc.shape
    # Огибающая аналитического сигнала повторяет пик сжатого отклика.
    assert abs(int(np.argmax(np.abs(s_an[0]))) - _expected_cell(y0)) <= 5
    # Действительная часть совпадает с входом (свойство преобразования Гильберта).
    np.testing.assert_allclose(s_an[0].real, s_rc[0].astype(np.float32), rtol=0, atol=1e-2)


def test_compress_1d_input():
    s = point_target_raw(0.0, 0.6, x_ant=np.array([0.0]), n_samples=9000)
    out = compress(s[0], make_cw_pulse())
    assert out.ndim == 1
    assert out.shape == (9000,)
