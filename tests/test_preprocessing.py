import numpy as np

from sar_lab.preprocessing import subtract_background


def test_subtract_background_basic():
    scene = np.full((5, 8), 10, dtype=np.int16)
    empty = np.full((5, 8), 3, dtype=np.int16)
    out = subtract_background(scene, empty)
    assert out.dtype == np.float64
    assert np.all(out == 7.0)


def test_subtract_background_trims_mismatch():
    scene = np.ones((6, 10), dtype=np.int16)
    empty = np.ones((5, 9), dtype=np.int16) * 2
    out = subtract_background(scene, empty)
    assert out.shape == (5, 9)
    assert np.all(out == -1.0)


def test_subtract_background_removes_stationary_clutter():
    rng = np.random.default_rng(0)
    clutter = rng.normal(0, 50, size=(20, 200))
    target = np.zeros((20, 200))
    target[:, 100] = 500.0
    scene = clutter + target
    empty = clutter.copy()
    out = subtract_background(scene, empty)
    np.testing.assert_allclose(out, target, atol=1e-9)
