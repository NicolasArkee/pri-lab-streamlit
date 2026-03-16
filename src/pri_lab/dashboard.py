from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def default_workspace_path() -> Path:
  return (Path(__file__).resolve().parents[2] / "artifacts" / "default").resolve()


def resolve_workspace_path(workspace: Path | None) -> Path:
  if workspace is not None:
    return workspace.expanduser().resolve()

  workspace_from_env = os.getenv("PRI_LAB_WORKSPACE")
  if workspace_from_env:
    return Path(workspace_from_env).expanduser().resolve()

  return default_workspace_path()


def build_streamlit_command(
  workspace: Path,
  host: str,
  port: int,
) -> list[str]:
  app_path = Path(__file__).with_name("dashboard_app.py")
  return [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    str(app_path),
    "--server.address",
    host,
    "--server.port",
    str(port),
    "--server.headless",
    "true",
    "--browser.gatherUsageStats",
    "false",
    "--",
    "--workspace",
    str(workspace),
  ]


def launch_dashboard(
  workspace: Path | None = None,
  host: str = "127.0.0.1",
  port: int = 8501,
) -> None:
  resolved_workspace = resolve_workspace_path(workspace)
  command = build_streamlit_command(
    workspace=resolved_workspace,
    host=host,
    port=port,
  )
  subprocess.run(command, check=False)
