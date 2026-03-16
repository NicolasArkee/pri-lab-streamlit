from __future__ import annotations

from pathlib import Path

from pri_lab.dashboard import build_streamlit_command, resolve_workspace_path


def test_resolve_workspace_path_prefers_explicit_path(tmp_path: Path) -> None:
  explicit_workspace = tmp_path / "workspace"
  resolved = resolve_workspace_path(explicit_workspace)
  assert resolved == explicit_workspace.resolve()


def test_build_streamlit_command_contains_workspace_and_server_flags(tmp_path: Path) -> None:
  workspace = tmp_path / "workspace"
  command = build_streamlit_command(
    workspace=workspace,
    host="0.0.0.0",
    port=8899,
  )
  command_text = " ".join(command)

  assert "--server.address" in command_text
  assert "0.0.0.0" in command_text
  assert "--server.port" in command_text
  assert "8899" in command_text
  assert "--workspace" in command_text
  assert str(workspace) in command_text
