import numpy as np

from sar_lab import config


def test_sound_speed_from_temperature():
    assert config.sound_speed_from_temperature(20.0) == 331.3 + 0.6 * 20.0
    # При 20°C ≈ 343.3 м/с
    assert abs(config.sound_speed_from_temperature(20.0) - 343.3) < 0.1


def test_wavelength_from_temperature():
    c = config.sound_speed_from_temperature(22.0)
    assert np.isclose(config.wavelength_from_temperature(22.0), c / config.CARRIER_FREQUENCY_HZ)


def test_dx_from_motor_steps():
    dx = config.dx_from_motor_steps(92)
    # ≈ 2.14 мм (README §1.1)
    assert np.isclose(dx, 92 * 0.0233e-3)
    assert 2.0e-3 < dx < 2.3e-3


def test_max_recordable_range():
    r = config.max_recordable_range_m()
    # README §2.2: ≈ 4.29 м при c=343
    assert np.isclose(r, 343.0 * 50000 / (2 * 2e6))
    assert 4.0 < r < 4.5
