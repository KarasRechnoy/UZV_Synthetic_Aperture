"""Загрузка/сохранение сырых данных и парсинг ground truth (README §6, §8)."""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import config

PathLike = str | Path


def load_raw(path: PathLike) -> np.ndarray:
    """Загрузить сырую матрицу `S[p, k]` (int16) из .npy.

    Возвращает 2-D массив формы (n_pings, n_samples_per_ping).
    """
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"raw data must be 2-D (n_pings, n_samples), got {arr.shape}")
    if arr.dtype != np.int16:
        # Допускаем другие целые типы, но фиксируем int16-семантику стенда.
        arr = arr.astype(np.int16)
    return arr


def save_raw(path: PathLike, data: np.ndarray) -> None:
    np.save(path, np.ascontiguousarray(data, dtype=np.int16))


def load_ground_truth(path: PathLike) -> dict[str, Any]:
    """Распарсить ground-truth YAML (README §6).

    Дополнительно:
    - если `conditions.sound_speed_mps` is null → считается из температуры;
    - добавляется `derived` с `sound_speed_mps`, `wavelength_m`, `dx_m`.
    """
    with open(path, encoding="utf-8") as fh:
        gt: dict[str, Any] = yaml.safe_load(fh)

    conditions = gt.get("conditions", {})
    t_celsius = conditions.get("temperature_celsius")
    c = conditions.get("sound_speed_mps")
    if c is None:
        if t_celsius is None:
            c = config.DEFAULT_SOUND_SPEED_MPS
        else:
            c = config.sound_speed_from_temperature(float(t_celsius))
        conditions["sound_speed_mps"] = c
    gt["conditions"] = conditions

    acq = gt.get("acquisition", {})
    steps = acq.get("carriage_step_motor_steps", config.MOTOR_STEPS_PER_PING_DEFAULT)
    dx_cal = acq.get("dx_mm_calibrated")
    if dx_cal is not None:
        dx_m = float(dx_cal) * 1e-3
    else:
        dx_m = config.dx_from_motor_steps(int(steps))

    gt["derived"] = {
        "sound_speed_mps": float(c),
        "wavelength_m": float(c) / config.CARRIER_FREQUENCY_HZ,
        "dx_m": dx_m,
    }
    return gt


def object_positions_m(gt: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Координаты (x, y) объектов сцены в метрах.

    Для точечных объектов берётся (x_mm, y_mm), для протяжённых — центр
    (x_center_mm, y_center_mm).
    """
    out: dict[str, tuple[float, float]] = {}
    for obj in gt.get("objects", []):
        x_mm = obj.get("x_mm", obj.get("x_center_mm"))
        y_mm = obj.get("y_mm", obj.get("y_center_mm"))
        if x_mm is None or y_mm is None:
            continue
        out[obj["id"]] = (float(x_mm) * 1e-3, float(y_mm) * 1e-3)
    return out
