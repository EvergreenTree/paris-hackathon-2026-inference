#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server.app import app, main  # noqa: E402

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
