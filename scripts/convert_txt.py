# -*- coding: utf-8 -*-
"""Конвертация сырого txt стенда в .npy.

Формат входа: одна строка = один пинг, ровно N целых, разделённых пробелами.
Быстрый C-парсер pandas; результат — матрица int16 формы (n_pings, N).

Пример:
    python scripts/convert_txt.py raw/180520data1.txt data/raw/scene.npy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_SAMPLES_PER_PING = 50000


def convert(src: Path, dst: Path, n_samples: int = N_SAMPLES_PER_PING) -> None:
    df = pd.read_csv(src, sep=r"\s+", header=None, dtype=np.int32, engine="c")
    if df.shape[1] != n_samples:
        raise ValueError(f"{src.name}: {df.shape[1]} отсчётов/пинг != {n_samples}")
    arr = df.to_numpy()
    lo, hi = int(arr.min()), int(arr.max())
    if lo < np.iinfo(np.int16).min or hi > np.iinfo(np.int16).max:
        raise ValueError(f"{src.name}: диапазон [{lo}, {hi}] вне int16")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr.astype(np.int16))
    print(f"{src.name} -> {dst}  shape={arr.shape}  range=[{lo}, {hi}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="txt -> npy")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES_PER_PING)
    a = ap.parse_args()
    convert(a.src, a.dst, a.n_samples)
