from __future__ import annotations

from pathlib import Path

import polars as pl

from pri_lab.pipeline import (
  BuildEdgesOptions,
  build_edges,
  model_anchor_scenarios,
  prepare_anchor_dataset,
  prepare_pages,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "arbo_small.json"


def test_prepare_anchor_dataset_builds_all_anchor_types(tmp_path: Path) -> None:
  pages_path = tmp_path / "pages.parquet"
  edges_path = tmp_path / "edges.parquet"
  anchors_path = tmp_path / "anchor_candidates.parquet"

  prepare_pages(FIXTURE_PATH, pages_path)
  build_edges(
    input_pages_path=pages_path,
    output_edges_path=edges_path,
    options=BuildEdgesOptions(cluster_peer_k=2, max_out_links_per_page=10),
  )
  summary = prepare_anchor_dataset(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    output_anchor_candidates_path=anchors_path,
  )

  edges_count = pl.read_parquet(edges_path).height
  assert anchors_path.exists()
  assert summary["candidate_count"] == edges_count * 4

  candidates_df = pl.read_parquet(anchors_path)
  assert sorted(candidates_df.get_column("anchor_type").unique().to_list()) == [
    "contextual",
    "exact",
    "generic",
    "partial",
  ]


def test_model_anchor_scenarios_exports_three_scenarios_and_comparison(tmp_path: Path) -> None:
  pages_path = tmp_path / "pages.parquet"
  edges_path = tmp_path / "edges.parquet"
  output_dir = tmp_path / "scenarios"

  prepare_pages(FIXTURE_PATH, pages_path)
  build_edges(
    input_pages_path=pages_path,
    output_edges_path=edges_path,
    options=BuildEdgesOptions(cluster_peer_k=2, max_out_links_per_page=10),
  )

  summary = model_anchor_scenarios(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    output_dir=output_dir,
    damping=0.85,
  )

  assert len(summary["scenarios"]) == 3
  comparison_path = Path(summary["scenario_comparison_parquet"])
  assert comparison_path.exists()

  comparison_df = pl.read_parquet(comparison_path)
  assert "pri_scenario_1_exact_focus" in comparison_df.columns
  assert "pri_scenario_2_balanced_mix" in comparison_df.columns
  assert "pri_scenario_3_diversity_first" in comparison_df.columns

  averages = {
    scenario["scenario"]: scenario["avg_anchor_diversity_score"]
    for scenario in summary["scenarios"]
  }
  assert averages["scenario_3_diversity_first"] >= averages["scenario_1_exact_focus"]
