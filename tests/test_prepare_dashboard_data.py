from __future__ import annotations

from pathlib import Path

import polars as pl

from pri_lab.pipeline import (
  BuildEdgesOptions,
  ComputePriOptions,
  build_edges,
  compute_pri,
  prepare_dashboard_data,
  prepare_outlinks_dataset,
  prepare_pages,
)


OUTLINKS_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "outlinks_small.csv"
ARBO_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "arbo_small.json"


def test_prepare_dashboard_data_generates_expected_url_segments_and_metrics(tmp_path: Path) -> None:
  pages_path = tmp_path / "pages.parquet"
  edges_path = tmp_path / "edges.parquet"
  anchors_path = tmp_path / "anchor_candidates.parquet"
  pri_path = tmp_path / "pri_scores.parquet"
  page_segments_path = tmp_path / "page_segments.parquet"
  url_metrics_path = tmp_path / "url_metrics.parquet"
  segment_metrics_path = tmp_path / "segment_metrics.parquet"

  page_segments_path_second = tmp_path / "page_segments_second.parquet"
  url_metrics_path_second = tmp_path / "url_metrics_second.parquet"
  segment_metrics_path_second = tmp_path / "segment_metrics_second.parquet"

  prepare_outlinks_dataset(
    input_csv_path=OUTLINKS_FIXTURE_PATH,
    output_pages_path=pages_path,
    output_edges_path=edges_path,
    output_anchor_candidates_path=anchors_path,
    max_out_links_per_page=20,
  )
  compute_pri(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    output_pri_path=pri_path,
    options=ComputePriOptions(
      damping=0.85,
      include_block_types=None,
      use_weights=True,
    ),
  )

  summary = prepare_dashboard_data(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    input_pri_path=pri_path,
    output_page_segments_path=page_segments_path,
    output_url_metrics_path=url_metrics_path,
    output_segment_metrics_path=segment_metrics_path,
    input_anchor_candidates_path=anchors_path,
  )
  prepare_dashboard_data(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    input_pri_path=pri_path,
    output_page_segments_path=page_segments_path_second,
    output_url_metrics_path=url_metrics_path_second,
    output_segment_metrics_path=segment_metrics_path_second,
    input_anchor_candidates_path=anchors_path,
  )

  assert summary["anchor_candidates_used"] is True

  page_segments_df = pl.read_parquet(page_segments_path)
  url_metrics_df = pl.read_parquet(url_metrics_path)
  segment_metrics_df = pl.read_parquet(segment_metrics_path)
  pages_df = pl.read_parquet(pages_path)

  assert page_segments_df.columns == [
    "page_id",
    "level",
    "segment",
    "segment_path",
    "parent_segment_path",
    "depth",
    "is_terminal",
  ]
  assert url_metrics_df.columns == [
    "page_id",
    "path",
    "depth",
    "pri_score",
    "rank",
    "cheirank_score",
    "cheirank_rank",
    "in_degree",
    "out_degree",
    "can_give_juice",
    "is_low_power",
    "incoming_links",
    "outgoing_links",
    "incoming_weight",
    "outgoing_weight",
    "unique_out_targets",
    "unique_in_sources",
    "unique_anchor_texts_out",
    "unique_anchor_types_out",
    "lexical_diversity_out",
  ]
  assert segment_metrics_df.columns == [
    "level",
    "segment_path",
    "segment",
    "page_count",
    "pri_sum",
    "pri_mean",
    "cheirank_sum",
    "cheirank_mean",
    "donor_ratio",
    "low_power_ratio",
    "avg_in_degree",
    "avg_out_degree",
    "avg_lexical_diversity_out",
  ]

  root_path_page_id = int(pages_df.filter(pl.col("path") == "/").select("page_id").item())
  root_segments_df = (
    page_segments_df
    .filter(pl.col("page_id") == root_path_page_id)
    .sort("level")
  )
  assert root_segments_df.height == 1
  assert root_segments_df.get_column("segment_path").to_list() == ["/"]
  assert root_segments_df.get_column("is_terminal").to_list() == [True]

  cars_suv_page_id = int(pages_df.filter(pl.col("path") == "/cars/suv").select("page_id").item())
  cars_suv_segments_df = (
    page_segments_df
    .filter(pl.col("page_id") == cars_suv_page_id)
    .sort("level")
  )
  assert cars_suv_segments_df.get_column("level").to_list() == [0, 1, 2]
  assert cars_suv_segments_df.get_column("segment").to_list() == ["root", "cars", "suv"]
  assert cars_suv_segments_df.get_column("segment_path").to_list() == ["/", "/cars", "/cars/suv"]
  assert cars_suv_segments_df.get_column("is_terminal").to_list() == [False, False, True]

  assert pl.read_parquet(page_segments_path).to_dicts() == pl.read_parquet(page_segments_path_second).to_dicts()
  assert pl.read_parquet(url_metrics_path).to_dicts() == pl.read_parquet(url_metrics_path_second).to_dicts()
  assert pl.read_parquet(segment_metrics_path).to_dicts() == pl.read_parquet(segment_metrics_path_second).to_dicts()


def test_prepare_dashboard_data_handles_missing_anchor_dataset(tmp_path: Path) -> None:
  pages_path = tmp_path / "pages.parquet"
  edges_path = tmp_path / "edges.parquet"
  pri_path = tmp_path / "pri_scores.parquet"
  page_segments_path = tmp_path / "page_segments.parquet"
  url_metrics_path = tmp_path / "url_metrics.parquet"
  segment_metrics_path = tmp_path / "segment_metrics.parquet"

  prepare_pages(input_json_path=ARBO_FIXTURE_PATH, output_pages_path=pages_path)
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

  summary = prepare_dashboard_data(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    input_pri_path=pri_path,
    output_page_segments_path=page_segments_path,
    output_url_metrics_path=url_metrics_path,
    output_segment_metrics_path=segment_metrics_path,
    input_anchor_candidates_path=None,
  )

  assert summary["anchor_candidates_used"] is False
  page_segments_df = pl.read_parquet(page_segments_path)
  url_metrics_df = pl.read_parquet(url_metrics_path)
  pages_df = pl.read_parquet(pages_path)

  root_count = int(page_segments_df.filter(pl.col("level") == 0).height)
  assert root_count == pages_df.height
  assert int(url_metrics_df.select(pl.max("unique_anchor_texts_out")).item()) == 0
  assert int(url_metrics_df.select(pl.max("unique_anchor_types_out")).item()) == 0
