from __future__ import annotations

import json
from pathlib import Path

from pri_lab.pipeline import (
  BuildEdgesOptions,
  ComputePriOptions,
  build_edges,
  compute_pri,
  export_workspace_report,
  model_anchor_scenarios,
  prepare_dashboard_data,
  prepare_outlinks_dataset,
  prepare_pages,
)


OUTLINKS_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "outlinks_small.csv"
ARBO_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "arbo_small.json"


def test_export_workspace_report_exports_csv_and_synthesis(tmp_path: Path) -> None:
  workspace = tmp_path / "workspace"
  workspace.mkdir(parents=True, exist_ok=True)

  pages_path = workspace / "pages.parquet"
  edges_path = workspace / "edges.parquet"
  anchors_path = workspace / "anchor_candidates.parquet"
  pri_path = workspace / "pri_scores.parquet"
  page_segments_path = workspace / "page_segments.parquet"
  url_metrics_path = workspace / "url_metrics.parquet"
  segment_metrics_path = workspace / "segment_metrics.parquet"
  scenarios_dir = workspace / "scenarios"

  prepare_outlinks_dataset(
    input_csv_path=OUTLINKS_FIXTURE_PATH,
    output_pages_path=pages_path,
    output_edges_path=edges_path,
    output_anchor_candidates_path=anchors_path,
    max_out_links_per_page=25,
  )
  compute_pri(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    output_pri_path=pri_path,
    options=ComputePriOptions(damping=0.85, include_block_types=None, use_weights=True),
  )
  prepare_dashboard_data(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    input_pri_path=pri_path,
    output_page_segments_path=page_segments_path,
    output_url_metrics_path=url_metrics_path,
    output_segment_metrics_path=segment_metrics_path,
    input_anchor_candidates_path=anchors_path,
  )
  model_anchor_scenarios(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    output_dir=scenarios_dir,
    input_anchor_candidates_path=anchors_path,
    damping=0.85,
  )

  output_dir = tmp_path / "exports"
  summary = export_workspace_report(
    workspace=workspace,
    output_dir=output_dir,
    include_scenarios=True,
  )

  assert summary["exported_csv_count"] >= 7
  assert Path(summary["synthesis_report_json"]).exists()
  assert Path(summary["synthesis_report_markdown"]).exists()
  assert (output_dir / "csv" / "pages.csv").exists()
  assert (output_dir / "csv" / "edges.csv").exists()
  assert (output_dir / "csv" / "pri_scores.csv").exists()
  assert (output_dir / "csv" / "anchor_candidates.csv").exists()
  assert (output_dir / "csv" / "page_segments.csv").exists()
  assert (output_dir / "csv" / "url_metrics.csv").exists()
  assert (output_dir / "csv" / "segment_metrics.csv").exists()
  assert (output_dir / "csv" / "scenarios" / "scenario_pri_comparison.csv").exists()

  report_json = json.loads(Path(summary["synthesis_report_json"]).read_text(encoding="utf-8"))
  report_md = Path(summary["synthesis_report_markdown"]).read_text(encoding="utf-8")
  assert report_json["kpis"]["page_count"] > 0
  assert report_json["kpis"]["edge_count"] > 0
  assert len(report_json["top_pri_pages"]) > 0
  assert "Rapport de synthèse PRi Lab" in report_md
  assert "Top pages PRi" in report_md


def test_export_workspace_report_exports_required_artifacts_when_optional_are_missing(tmp_path: Path) -> None:
  workspace = tmp_path / "workspace"
  workspace.mkdir(parents=True, exist_ok=True)

  pages_path = workspace / "pages.parquet"
  edges_path = workspace / "edges.parquet"
  pri_path = workspace / "pri_scores.parquet"

  prepare_pages(
    input_json_path=ARBO_FIXTURE_PATH,
    output_pages_path=pages_path,
  )
  build_edges(
    input_pages_path=pages_path,
    output_edges_path=edges_path,
    options=BuildEdgesOptions(cluster_peer_k=2, max_out_links_per_page=10),
  )
  compute_pri(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    output_pri_path=pri_path,
    options=ComputePriOptions(
      damping=0.85,
      include_block_types=["hierarchy_up", "hierarchy_down", "cluster_peer"],
      use_weights=True,
    ),
  )

  output_dir = tmp_path / "exports"
  summary = export_workspace_report(
    workspace=workspace,
    output_dir=output_dir,
    include_scenarios=False,
  )

  assert summary["exported_csv_count"] == 3
  assert (output_dir / "csv" / "pages.csv").exists()
  assert (output_dir / "csv" / "edges.csv").exists()
  assert (output_dir / "csv" / "pri_scores.csv").exists()
  missing_optional = summary["missing_optional_artifacts"]
  assert any(str(path).endswith("/anchor_candidates.parquet") for path in missing_optional)
  assert any(str(path).endswith("/url_metrics.parquet") for path in missing_optional)
  assert any(str(path).endswith("/segment_metrics.parquet") for path in missing_optional)
