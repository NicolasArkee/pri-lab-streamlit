from __future__ import annotations

from pathlib import Path

import polars as pl

from pri_lab.pipeline import ComputePriOptions, compute_pri
from pri_lab.synthetic import generate_synthetic_edges, generate_synthetic_pages


def test_synthetic_smoke_pipeline_runs(tmp_path: Path) -> None:
  pages_path = tmp_path / "synthetic_pages.parquet"
  edges_path = tmp_path / "synthetic_edges.parquet"
  pri_path = tmp_path / "synthetic_pri.parquet"

  pages_summary = generate_synthetic_pages(
    output_pages_path=pages_path,
    node_count=5_000,
    cluster_count=50,
  )
  edges_summary = generate_synthetic_edges(
    output_edges_path=edges_path,
    node_count=5_000,
    target_edge_count=30_000,
  )

  pri_summary = compute_pri(
    input_pages_path=pages_path,
    input_edges_path=edges_path,
    output_pri_path=pri_path,
    options=ComputePriOptions(
      damping=0.85,
      include_block_types=["synthetic"],
      use_weights=True,
    ),
  )

  pri_df = pl.read_parquet(pri_path)
  assert pages_summary["node_count"] == 5_000
  assert edges_summary["edge_count"] > 0
  assert pri_summary["node_count"] == 5_000
  assert pri_df.height == 5_000
