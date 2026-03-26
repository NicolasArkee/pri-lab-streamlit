from __future__ import annotations

import os
import subprocess
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
  sys.path.insert(0, str(SRC_PATH))

# On Streamlit Cloud, LFS files are pointer stubs after clone.
# Pull real data if needed.
_lfs_marker = PROJECT_ROOT / "artifacts" / "lbc" / "pages.parquet"
if _lfs_marker.exists() and _lfs_marker.stat().st_size < 1024:
  subprocess.run(["git", "lfs", "install"], cwd=PROJECT_ROOT, check=False)
  subprocess.run(["git", "lfs", "pull"], cwd=PROJECT_ROOT, check=False)

# Default workspace to lbc if available
if not os.getenv("PRI_LAB_WORKSPACE"):
  lbc_ws = PROJECT_ROOT / "artifacts" / "lbc"
  if (lbc_ws / "pages.parquet").exists():
    os.environ["PRI_LAB_WORKSPACE"] = str(lbc_ws)

from pri_lab.dashboard_app import main

if __name__ == "__main__":
  main()
