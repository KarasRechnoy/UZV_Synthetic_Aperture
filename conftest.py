"""Делает src/ и tests/ импортируемыми без установки пакета."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("src", "tests"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
