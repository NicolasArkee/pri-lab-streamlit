from __future__ import annotations

from pathlib import Path

import polars as pl

from pri_lab.pipeline import model_anchor_scenarios, prepare_outlinks_dataset


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "outlinks_small.csv"


def test_prepare_outlinks_dataset_builds_pages_edges_and_anchors(tmp_path: Path) -> None:
  pages_path = tmp_path / "pages.parquet"
  edges_path = tmp_path / "edges.parquet"
  anchors_path = tmp_path / "anchor_candidates.parquet"

  summary = prepare_outlinks_dataset(
    input_csv_path=FIXTURE_PATH,
    output_pages_path=pages_path,
    output_edges_path=edges_path,
    output_anchor_candidates_path=anchors_path,
    max_out_links_per_page=20,
  )

  assert summary["source_host"] == "example.com"
  assert summary["page_count"] == 4
  assert summary["edge_count"] == 5
  assert summary["anchor_candidate_count"] == 5

  pages_df = pl.read_parquet(pages_path)
  edges_df = pl.read_parquet(edges_path)
  anchors_df = pl.read_parquet(anchors_path)

  assert sorted(pages_df.get_column("path").to_list()) == ["/", "/cars", "/cars/suv", "/contact"]
  assert edges_df.columns == ["source_id", "target_id", "block_type", "rule_weight"]
  assert anchors_df.columns == [
    "source_id",
    "target_id",
    "block_type",
    "rule_weight",
    "anchor_type",
    "anchor_text",
    "anchor_token_count",
  ]
  assert sorted(anchors_df.get_column("anchor_type").unique().to_list()) == ["contextual", "exact", "partial"]


def test_model_anchor_scenarios_accepts_external_anchor_candidates(tmp_path: Path) -> None:
  pages_path = tmp_path / "pages.parquet"
  edges_path = tmp_path / "edges.parquet"
  anchors_path = tmp_path / "anchor_candidates.parquet"
  scenarios_dir = tmp_path / "scenarios"

  prepare_outlinks_dataset(
    input_csv_path=FIXTURE_PATH,
    output_pages_path=pages_path,
    output_edges_path=edges_path,
    output_anchor_candidates_path=anchors_path,
    max_out_links_per_page=20,
  )

  summary = model_anchor_scenarios(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    output_dir=scenarios_dir,
    input_anchor_candidates_path=anchors_path,
    damping=0.85,
  )

  assert len(summary["scenarios"]) == 3
  assert summary["prepare_anchor_dataset"]["source"] == "provided"
  assert Path(summary["scenario_comparison_parquet"]).exists()
