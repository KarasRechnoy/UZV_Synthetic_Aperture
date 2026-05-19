import numpy as np

from sar_lab import config
from sar_lab.autofocus import autofocus_scalar, image_contrast, image_entropy
from sar_lab.backprojection import backproject
from sar_lab.range_compression import analytic_signal, compress, make_cw_pulse
from synthetic import point_target_raw


def test_entropy_lower_for_focused():
    """Сфокусированное (точка) изображение — энтропия ниже, контраст выше."""
    focused = np.zeros((40, 40), complex)
    focused[20, 20] = 1.0
    diffuse = np.ones((40, 40), complex)
    assert image_entropy(focused) < image_entropy(diffuse)
    assert image_contrast(focused) > image_contrast(diffuse)


def _point_scene(c_true=343.0):
    dx = config.dx_from_motor_steps(92)
    n = 201
    x_ant = (np.arange(n) - (n - 1) / 2.0) * dx
    # три точки на разной дальности -> заметная чувствительность к c
    s = np.zeros((n, 12000))
    for (x0, y0) in [(-0.10, 0.70), (0.0, 1.20), (0.12, 1.80)]:
        s = s + point_target_raw(x0, y0, x_ant=x_ant, n_samples=12000,
                                 sound_speed=c_true, pulse_duration=1e-3)
    s_an = analytic_signal(compress(s, make_cw_pulse(duration=1e-3)))
    x_g = np.arange(-0.25, 0.25 + 1e-9, 0.004)
    y_g = np.arange(0.55, 1.95 + 1e-9, 0.006)
    return s_an, x_ant, x_g, y_g


def test_mismatched_sound_speed_is_less_focused():
    """Грубо неверная c → выше энтропия / ниже контраст, чем при истинной."""
    s_an, x_ant, x_g, y_g = _point_scene(343.0)

    def img(c):
        return backproject(s_an, x_ant, x_g, y_g, sound_speed=c,
                           window="hamming")

    good = img(343.0)
    bad = img(310.0)
    assert image_entropy(good) < image_entropy(bad)
    assert image_contrast(good) > image_contrast(bad)


def test_autofocus_runs_and_improves_metric():
    """Автофокус возвращает c в пределах и не хуже стартовой энтропии."""
    s_an, x_ant, x_g, y_g = _point_scene(343.0)

    def make_img(c):
        return backproject(s_an, x_ant, x_g, y_g, sound_speed=c,
                           window="hamming")

    c_hat, img, hist = autofocus_scalar(make_img, (320.0, 366.0),
                                        metric="entropy", xatol=0.5)
    assert 320.0 <= c_hat <= 366.0
    assert image_entropy(img) <= image_entropy(make_img(320.0)) + 1e-9
    assert img.shape == (y_g.size, x_g.size)
