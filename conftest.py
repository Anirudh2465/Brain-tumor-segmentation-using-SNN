"""
conftest.py — Pytest global configuration.

Sets PYTHONPATH so tests can import spiking_useg without editable install.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path regardless of how pytest is invoked
SRC = str(Path(__file__).parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
