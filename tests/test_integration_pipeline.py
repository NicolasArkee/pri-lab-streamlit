from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from pri_lab.pipeline import (
  BuildEdgesOptions,
  ComputePriOptions,
  build_edges,
  compute_pri,
  prepare_dashboard_data,
  prepare_pages,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "arbo_small.json"


def test_full_pipeline_produces_consistent_outputs(tmp_path: Path) -> None:
  pages_path = tmp_path / "pages.parquet"
  edges_path = tmp_path / "edges.parquet"
  pri_path = tmp_path / "pri.parquet"
  pri_path_second = tmp_path / "pri_second.parquet"
  pri_path_filtered = tmp_path / "pri_filtered.parquet"
  page_segments_path = tmp_path / "page_segments.parquet"
  url_metrics_path = tmp_path / "url_metrics.parquet"
  segment_metrics_path = tmp_path / "segment_metrics.parquet"

  prepare_pages(
    input_json_path=FIXTURE_PATH,
    output_pages_path=pages_path,
  )
  build_edges(
    input_pages_path=pages_path,
    output_edges_path=edges_path,
    options=BuildEdgesOptions(
      cluster_peer_k=2,
      max_out_links_per_page=6,
    ),
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

  compute_pri(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    output_pri_path=pri_path_second,
    options=ComputePriOptions(
      damping=0.85,
      include_block_types=["hierarchy_up", "hierarchy_down", "cluster_peer"],
      use_weights=True,
    ),
  )

  compute_pri(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    output_pri_path=pri_path_filtered,
    options=ComputePriOptions(
      damping=0.85,
      include_block_types=["hierarchy_up"],
      use_weights=True,
    ),
  )

  prepare_dashboard_data(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    input_pri_path=pri_path,
    output_page_segments_path=page_segments_path,
    output_url_metrics_path=url_metrics_path,
    output_segment_metrics_path=segment_metrics_path,
    input_anchor_candidates_path=None,
  )

  pages_df = pl.read_parquet(pages_path)
  edges_df = pl.read_parquet(edges_path)
  pri_df = pl.read_parquet(pri_path).sort("page_id")
  pri_second_df = pl.read_parquet(pri_path_second).sort("page_id")
  pri_filtered_df = pl.read_parquet(pri_path_filtered).sort("page_id")
  page_segments_df = pl.read_parquet(page_segments_path)
  url_metrics_df = pl.read_parquet(url_metrics_path)
  segment_metrics_df = pl.read_parquet(segment_metrics_path)

  page_ids = set(pages_df.get_column("page_id").to_list())
  assert set(edges_df.get_column("source_id").to_list()).issubset(page_ids)
  assert set(edges_df.get_column("target_id").to_list()).issubset(page_ids)

  assert "cheirank_score" in pri_df.columns
  assert "cheirank_rank" in pri_df.columns
  assert "juice_potential_score" in pri_df.columns
  assert "is_low_power" in pri_df.columns
  assert "can_give_juice" in pri_df.columns
  assert math.isclose(pri_df.get_column("pri_score").sum(), 1.0, rel_tol=1e-8, abs_tol=1e-8)
  assert math.isclose(pri_df.get_column("cheirank_score").sum(), 1.0, rel_tol=1e-8, abs_tol=1e-8)
  assert pri_df.to_dicts() == pri_second_df.to_dicts()
  assert pri_df.get_column("pri_score").to_list() != pri_filtered_df.get_column("pri_score").to_list()
  assert page_segments_df.filter(pl.col("level") == 0).height == pages_df.height
  assert set(url_metrics_df.get_column("page_id").to_list()) == page_ids
  assert segment_metrics_df.height > 0
