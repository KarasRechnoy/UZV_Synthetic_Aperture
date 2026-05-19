# -*- coding: utf-8 -*-
"""Полный SAR-конвейер: npy -> сфокусированное изображение.

Этапы: загрузка -> вычитание фона -> range compression -> аналитический
сигнал -> (опц.) энтропийный автофокус c -> backprojection (моно/бистатика,
наклон антенны) -> калибровка дальности по t0 -> визуализация.

Координаты целей не требуются. Примеры:

  # ЛЧМ 38-42 кГц, автофокус, бистатика
  python scripts/run_pipeline.py --scene data/raw/scene.npy --empty data/raw/empty.npy \
      --pulse chirp --band 38e3 42e3 --dur 1e-3 --autofocus --bistatic \
      --t0-ms 2.4 --out results/scene

  # CW 40 кГц / 25 мкс, наклонная антенна 30 градусов
  python scripts/run_pipeline.py --scene data/raw/scene_a30.npy --empty data/raw/empty_a30.npy \
      --pulse cw --f0 40e3 --dur 25e-6 --autofocus --bistatic --tilt-deg 30 \
      --t0-ms 2.4 --out results/a30
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# консоль Windows бывает в cp1251 — выводим лог в UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sar_lab import config, io
from sar_lab.autofocus import autofocus_scalar, image_entropy
from sar_lab.backprojection import backproject
from sar_lab.preprocessing import subtract_background
from sar_lab.range_compression import (
    analytic_signal, compress, make_chirp_pulse, make_cw_pulse,
)
from sar_lab.visualize import plot_sar_image, save_figure


def parse_args():
    ap = argparse.ArgumentParser(description="SAR backprojection pipeline")
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--empty", type=Path, default=None,
                    help="фон для вычитания (без него — сырая сцена)")
    # зондирующий импульс
    ap.add_argument("--pulse", choices=["cw", "chirp"], default="cw")
    ap.add_argument("--f0", type=float, default=config.CARRIER_FREQUENCY_HZ)
    ap.add_argument("--band", type=float, nargs=2, default=(38e3, 42e3),
                    help="для chirp: f_lo f_hi")
    ap.add_argument("--dur", type=float, default=1e-3, help="длительность, с")
    # геометрия
    ap.add_argument("--dx-mm", type=float,
                    default=config.MOTOR_STEPS_PER_PING_DEFAULT * config.MM_PER_MOTOR_STEP,
                    help="шаг каретки, мм/пинг")
    ap.add_argument("--c", type=float, default=config.DEFAULT_SOUND_SPEED_MPS,
                    help="скорость звука, м/с (старт для автофокуса)")
    ap.add_argument("--autofocus", action="store_true",
                    help="подобрать c по минимуму энтропии (метод Брента)")
    ap.add_argument("--c-bounds", type=float, nargs=2, default=(335.0, 355.0))
    ap.add_argument("--t0-ms", type=float, default=0.0,
                    help="постоянная задержка тракта, мс (калибровка дальности)")
    ap.add_argument("--bistatic", action="store_true")
    ap.add_argument("--tilt-deg", type=float, default=0.0,
                    help="наклон антенны: поворот смещения Rx в плоскости y-z")
    # сетка изображения
    ap.add_argument("--rmin", type=float, default=0.3)
    ap.add_argument("--rmax", type=float, default=3.6)
    ap.add_argument("--xspan", type=float, default=1.8)
    ap.add_argument("--dr", type=float, default=0.004)
    ap.add_argument("--dx-pix", type=float, default=0.004)
    ap.add_argument("--window", default="taylor")
    ap.add_argument("--dyn-db", type=float, default=28.0)
    ap.add_argument("--out", type=Path, default=Path("results/run"))
    return ap.parse_args()


def rx_offset(tilt_deg: float) -> dict:
    """Смещение Rx с учётом наклона антенны (поворот в плоскости дальность-высота)."""
    th = np.radians(tilt_deg)
    dy0, dz0 = config.RX_OFFSET_RANGE_M, config.RX_OFFSET_VERTICAL_M
    return dict(
        dx_tx=0.0, dx_rx=0.0,
        dy_rx=dy0 * np.cos(th) - dz0 * np.sin(th),
        dz_rx=dy0 * np.sin(th) + dz0 * np.cos(th),
    )


def main():
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    scene = io.load_raw(a.scene)
    if a.empty is not None:
        s = subtract_background(scene, io.load_raw(a.empty))
        print(f"фон вычтен: {a.empty.name}")
    else:
        s = scene.astype(np.float64)
        print("без фона — сырая сцена (остаётся стационарный клаттер)")

    if a.pulse == "cw":
        pulse = make_cw_pulse(duration=a.dur, f0=a.f0)
        wf = f"CW {a.f0/1e3:.0f}кГц {a.dur*1e6:.0f}мкс"
    else:
        pulse = make_chirp_pulse(a.band[0], a.band[1], duration=a.dur)
        wf = f"ЛЧМ {a.band[0]/1e3:.0f}-{a.band[1]/1e3:.0f}кГц {a.dur*1e3:.0f}мс"
    s_an = analytic_signal(compress(s, pulse))
    n = s_an.shape[0]
    dx = a.dx_mm * 1e-3
    x_ant = (np.arange(n) - (n - 1) / 2.0) * dx
    t0 = a.t0_ms * 1e-3
    bist = rx_offset(a.tilt_deg) if a.bistatic else {}
    print(f"импульс: {wf} | dx={a.dx_mm:.3f} мм | L_SA={(n-1)*dx:.3f} м "
          f"| bistatic={a.bistatic} tilt={a.tilt_deg}°")

    x_g = np.arange(-a.xspan / 2, a.xspan / 2 + 1e-9, a.dx_pix)
    y_g = np.arange(a.rmin, a.rmax + 1e-9, a.dr)

    c = a.c
    if a.autofocus:
        # автофокус на прореженном грубом изображении (быстро, без координат)
        sa = s_an[::2]
        npg = sa.shape[0]
        xc = (np.arange(npg) - (npg - 1) / 2.0) * dx * 2
        yc = np.arange(a.rmin, a.rmax, 0.02)
        xcg = np.arange(-a.xspan / 2, a.xspan / 2, 0.02)

        def coarse(cc):
            return backproject(sa, xc, xcg, yc, sound_speed=cc, time_offset=t0,
                               bistatic=a.bistatic, window=a.window, **bist)

        c, _, hist = autofocus_scalar(coarse, tuple(a.c_bounds),
                                      metric="entropy", xatol=0.1)
        print(f"автофокус: c* = {c:.2f} м/с ({len(hist)} вычислений)")

    print(f"backprojection: сетка {y_g.size}x{x_g.size}, c={c:.2f}, t0={a.t0_ms} мс")
    img = backproject(s_an, x_ant, x_g, y_g, sound_speed=c, time_offset=t0,
                      bistatic=a.bistatic, window=a.window, **bist)
    np.save(a.out / "image.npy", img)

    mag = np.abs(img)
    work = mag.copy()
    print("ярчайшие отклики (x, R):")
    for _ in range(5):
        j, i = np.unravel_index(int(work.argmax()), work.shape)
        print(f"  x={x_g[i]*1e3:+7.1f} мм   R={y_g[j]*1e3:7.1f} мм   "
              f"|pk|={mag[j, i]:.2e}")
        work[np.ix_(np.abs(y_g - y_g[j]) < 0.25,
                    np.abs(x_g - x_g[i]) < 0.25)] = 0.0
    print(f"энтропия изображения: {image_entropy(img):.3f}")

    fig = plot_sar_image(img, x_g, y_g, dynamic_range_db=a.dyn_db,
                         title=f"{a.scene.stem} [{wf}] c={c:.1f} t0={a.t0_ms}мс")
    save_figure(fig, a.out / "image.png")
    print(f"сохранено: {a.out/'image.png'}, {a.out/'image.npy'}")


if __name__ == "__main__":
    main()
