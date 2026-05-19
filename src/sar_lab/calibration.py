"""Калибровка дальности/скорости и азимутальной фокусировки (README §2.7).

Не требует азимутальных координат: дальность калибруется по известной
дистанции хотя бы до одного объекта, а шаг каретки `dx` — по максимуму
резкости отклика (однопараметрический скан).
"""

import numpy as np

from . import config


def range_axis(
    n_samples: int,
    sound_speed: float = config.DEFAULT_SOUND_SPEED_MPS,
    fs: float = config.SAMPLING_FREQUENCY_HZ,
) -> np.ndarray:
    """Дальность (м, односторонняя) для каждого отсчёта быстрого времени.

    `R = c · k / (2 f_s)` (двусторонний путь).
    """
    return sound_speed * np.arange(n_samples) / (2.0 * fs)


def sample_of_range(
    r_m: float,
    sound_speed: float = config.DEFAULT_SOUND_SPEED_MPS,
    fs: float = config.SAMPLING_FREQUENCY_HZ,
) -> float:
    """Индекс отсчёта (дробный) для дальности `r_m`."""
    return 2.0 * r_m * fs / sound_speed


def estimate_sound_speed(
    sample_index: float,
    true_range_m: float,
    fs: float = config.SAMPLING_FREQUENCY_HZ,
) -> float:
    """Скорость звука по известной дальности и индексу пика: `c = 2 R f_s / k`."""
    if sample_index <= 0:
        raise ValueError("sample_index must be positive")
    return 2.0 * true_range_m * fs / sample_index


def range_profile(envelope_2d: np.ndarray, reducer: str = "max") -> np.ndarray:
    """Свёртка (n_pings, n_samples) → (n_samples,) огибающая по быстрому времени.

    `max` — пик по апертуре (объект ярок лишь в части пингов);
    `mean` — средняя энергия.
    """
    if reducer == "max":
        return envelope_2d.max(axis=0)
    if reducer == "mean":
        return envelope_2d.mean(axis=0)
    raise ValueError(f"unknown reducer {reducer!r}")


def find_range_peaks(
    profile: np.ndarray,
    sound_speed: float,
    fs: float = config.SAMPLING_FREQUENCY_HZ,
    *,
    k_min: int = 1,
    n_peaks: int = 3,
    min_separation_samples: int = 400,
) -> list[tuple[int, float, float]]:
    """`n_peaks` локальных максимумов профиля.

    Возвращает список `(k, R_m, value)`, отсортированный по убыванию value,
    с защитой от слипания пиков (`min_separation_samples`).
    """
    p = profile.copy().astype(np.float64)
    p[:k_min] = 0.0
    out: list[tuple[int, float, float]] = []
    for _ in range(n_peaks):
        k = int(np.argmax(p))
        if p[k] <= 0:
            break
        r = sound_speed * k / (2.0 * fs)
        out.append((k, r, float(profile[k])))
        lo = max(0, k - min_separation_samples)
        hi = min(p.size, k + min_separation_samples)
        p[lo:hi] = 0.0
    return out


def focus_metric(image: np.ndarray) -> float:
    """Резкость изображения = peak / median(|image|).

    Используется как целевая функция при подборе `dx` (без координат).
    Чем выше — тем лучше сфокусировано.
    """
    mag = np.abs(image)
    med = np.median(mag)
    return float(mag.max() / med) if med > 0 else float(mag.max())


def image_sharpness(image: np.ndarray) -> float:
    """Резкость по концентрации энергии: `mean(|I|^4) / mean(|I|^2)^2`.

    Куртозис-подобная мера: чем сильнее энергия собрана в точки (фокус),
    тем выше значение. В отличие от peak/median, устойчива к одиночным
    артефактам и не требует координат — годится для автофокуса по
    произвольной сцене.
    """
    i2 = np.abs(image).astype(np.float64) ** 2
    denom = i2.mean() ** 2
    return float((i2**2).mean() / denom) if denom > 0 else 0.0
