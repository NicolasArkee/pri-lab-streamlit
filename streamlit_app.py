from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
  sys.path.insert(0, str(SRC_PATH))

from pri_lab.dashboard_app import main


if __name__ == "__main__":
  main()
