from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class SyntheticBenchmarkOptions:
  node_count: int
  target_edge_count: int
  cluster_count: int


def generate_synthetic_pages(
  output_pages_path: Path,
  node_count: int,
  cluster_count: int,
) -> dict[str, object]:
  output_pages_path.parent.mkdir(parents=True, exist_ok=True)

  pages_df = (
    pl.DataFrame(
      {
        "page_id": pl.int_range(1, node_count + 1, eager=True).cast(pl.Int32),
      },
    )
    .with_columns(
      pl.format("synthetic/{}/", pl.col("page_id")).alias("path"),
      pl.lit(None, dtype=pl.Utf8).alias("parent_path"),
      pl.lit(2, dtype=pl.Int16).alias("depth"),
      pl.lit("synthetic").alias("section"),
      pl.format(
        "synthetic:cluster-{}",
        ((pl.col("page_id") - 1) % max(cluster_count, 1)).cast(pl.Int32),
      ).alias("cluster_thematique"),
      pl.lit(True).alias("is_leaf"),
    )
    .select("page_id", "path", "parent_path", "depth", "section", "cluster_thematique", "is_leaf")
  )
  pages_df.write_parquet(output_pages_path, compression="zstd")

  return {
    "output_pages_parquet": str(output_pages_path),
    "node_count": pages_df.height,
    "cluster_count": max(cluster_count, 1),
  }


def generate_synthetic_edges(
  output_edges_path: Path,
  node_count: int,
  target_edge_count: int,
) -> dict[str, object]:
  output_edges_path.parent.mkdir(parents=True, exist_ok=True)

  if node_count <= 0 or target_edge_count <= 0:
    empty_df = pl.DataFrame(
      schema={
        "source_id": pl.Int32,
        "target_id": pl.Int32,
        "block_type": pl.Utf8,
        "rule_weight": pl.Float32,
      },
    )
    empty_df.write_parquet(output_edges_path, compression="zstd")
    return {
      "output_edges_parquet": str(output_edges_path),
      "edge_count": 0,
      "edges_per_node": 0,
    }

  edges_per_node = math.ceil(target_edge_count / node_count)
  source_df = pl.DataFrame({"source_id": pl.int_range(1, node_count + 1, eager=True).cast(pl.Int32)})
  step_df = pl.DataFrame({"step": pl.int_range(1, edges_per_node + 1, eager=True).cast(pl.Int32)})

  edges_lf = (
    source_df.lazy()
    .join(step_df.lazy(), how="cross")
    .with_columns(
      (
        ((pl.col("source_id") + (pl.col("step") * 17) - 1) % node_count) + 1
      ).cast(pl.Int32).alias("target_id"),
      pl.lit("synthetic").alias("block_type"),
      pl.lit(1.0, dtype=pl.Float32).alias("rule_weight"),
    )
    .filter(pl.col("source_id") != pl.col("target_id"))
    .select("source_id", "target_id", "block_type", "rule_weight")
    .head(target_edge_count)
  )
  edges_lf.sink_parquet(output_edges_path, compression="zstd")

  edge_count = int(pl.scan_parquet(output_edges_path).select(pl.len()).collect().item())
  return {
    "output_edges_parquet": str(output_edges_path),
    "edge_count": edge_count,
    "edges_per_node": edges_per_node,
  }
