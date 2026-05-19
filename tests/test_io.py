import numpy as np
import pytest

from sar_lab import config, io


def test_load_raw_roundtrip(tmp_path):
    data = np.random.randint(-2048, 2048, size=(10, 100)).astype(np.int16)
    path = tmp_path / "raw.npy"
    io.save_raw(path, data)
    loaded = io.load_raw(path)
    assert loaded.dtype == np.int16
    assert loaded.shape == (10, 100)
    np.testing.assert_array_equal(loaded, data)


def test_load_raw_rejects_1d(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros(50, dtype=np.int16))
    with pytest.raises(ValueError):
        io.load_raw(path)


def _write_gt(tmp_path, sound_speed):
    text = f"""
scene_id: 99
conditions:
  temperature_celsius: 22.0
  sound_speed_mps: {sound_speed}
acquisition:
  carriage_step_motor_steps: 92
  dx_mm_calibrated: null
objects:
  - id: corner_calib
    x_mm: -200
    y_mm: 600
  - id: foil
    x_center_mm: -100
    y_center_mm: 1100
"""
    p = tmp_path / "scene_99.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_ground_truth_sound_speed_from_temperature(tmp_path):
    gt = io.load_ground_truth(_write_gt(tmp_path, "null"))
    expected_c = config.sound_speed_from_temperature(22.0)
    assert np.isclose(gt["conditions"]["sound_speed_mps"], expected_c)
    assert np.isclose(gt["derived"]["wavelength_m"], expected_c / config.CARRIER_FREQUENCY_HZ)
    assert np.isclose(gt["derived"]["dx_m"], config.dx_from_motor_steps(92))


def test_ground_truth_explicit_sound_speed(tmp_path):
    gt = io.load_ground_truth(_write_gt(tmp_path, "345.0"))
    assert gt["conditions"]["sound_speed_mps"] == 345.0


def test_object_positions_m(tmp_path):
    gt = io.load_ground_truth(_write_gt(tmp_path, "null"))
    pos = io.object_positions_m(gt)
    assert np.allclose(pos["corner_calib"], (-0.200, 0.600))
    assert np.allclose(pos["foil"], (-0.100, 1.100))
