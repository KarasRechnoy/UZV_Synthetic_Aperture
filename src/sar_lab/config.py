"""Глобальные константы стенда (README §7).

Все величины в СИ (метры, секунды, герцы), если не указано иное.
Координаты в ground-truth YAML заданы в мм и переводятся в метры в io.py.
"""

# --- Физика ---
DEFAULT_SOUND_SPEED_MPS = 343.0       # при 20°C
CARRIER_FREQUENCY_HZ = 40e3
WAVELENGTH_M = DEFAULT_SOUND_SPEED_MPS / CARRIER_FREQUENCY_HZ   # ≈ 8.575 мм

# --- АЦП и импульс ---
SAMPLING_FREQUENCY_HZ = 2e6
PULSE_DURATION_S = 125e-6
N_SAMPLES_PER_PING = 50000

# --- Геометрия антенн ---
TX_ELEMENT_SIZE_M = 0.0066            # размер одного Tx-элемента
RX_OFFSET_VERTICAL_M = 0.050          # сдвиг Rx-палки по высоте над Tx (Δz_Rx)
RX_OFFSET_RANGE_M = 0.008             # сдвиг Rx-палки по дальности (Δy_Rx)

# --- Шаг каретки ---
MOTOR_STEPS_PER_PING_DEFAULT = 92
MM_PER_MOTOR_STEP = 0.0233            # ~ λ/2 / 184

# --- Обработка ---
DEFAULT_RANGE_MIN_M = 0.30            # обрезаем direct path
DEFAULT_RANGE_MAX_M = 2.00
DEFAULT_APERTURE_WINDOW = "hamming"
DEFAULT_RANGE_WINDOW = "hamming"


def sound_speed_from_temperature(t_celsius: float) -> float:
    """Скорость звука в воздухе от температуры, README §1.4."""
    return 331.3 + 0.6 * t_celsius


def wavelength_from_temperature(
    t_celsius: float, f0: float = CARRIER_FREQUENCY_HZ
) -> float:
    return sound_speed_from_temperature(t_celsius) / f0


def dx_from_motor_steps(
    steps_per_ping: int = MOTOR_STEPS_PER_PING_DEFAULT,
    mm_per_step: float = MM_PER_MOTOR_STEP,
) -> float:
    """Номинальный шаг каретки между пингами в метрах."""
    return steps_per_ping * mm_per_step * 1e-3


def max_recordable_range_m(
    sound_speed: float = DEFAULT_SOUND_SPEED_MPS,
    n_samples: int = N_SAMPLES_PER_PING,
    fs: float = SAMPLING_FREQUENCY_HZ,
) -> float:
    """Максимальная дальность в окне записи (двусторонний путь), README §2.2."""
    return sound_speed * n_samples / (2.0 * fs)
