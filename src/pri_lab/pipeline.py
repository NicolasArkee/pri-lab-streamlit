from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any

import igraph as ig
import polars as pl


EDGE_SCHEMA: dict[str, pl.DataType] = {
  "source_id": pl.Int32,
  "target_id": pl.Int32,
  "block_type": pl.Utf8,
  "rule_weight": pl.Float32,
}

EDGE_INTERNAL_SCHEMA: dict[str, pl.DataType] = {
  **EDGE_SCHEMA,
  "priority": pl.Int8,
}

BLOCK_PRIORITY: dict[str, int] = {
  "hierarchy_up": 0,
  "hierarchy_down": 1,
  "cluster_peer": 2,
}

ANCHOR_TYPES = ["exact", "partial", "contextual", "generic"]
GENERIC_ANCHOR_TEXTS = [
  "cliquez ici",
  "click here",
  "en savoir plus",
  "lire la suite",
  "voir plus",
  "learn more",
  "read more",
  "découvrir",
  "decouvrir",
  "ici",
]

ANCHOR_SCENARIOS: dict[str, dict[str, object]] = {
  "scenario_1_exact_focus": {
    "distribution": {
      "exact": 0.75,
      "partial": 0.15,
      "contextual": 0.07,
      "generic": 0.03,
    },
    "diversity_alpha": 0.05,
  },
  "scenario_2_balanced_mix": {
    "distribution": {
      "exact": 0.40,
      "partial": 0.30,
      "contextual": 0.20,
      "generic": 0.10,
    },
    "diversity_alpha": 0.18,
  },
  "scenario_3_diversity_first": {
    "distribution": {
      "exact": 0.20,
      "partial": 0.30,
      "contextual": 0.30,
      "generic": 0.20,
    },
    "diversity_alpha": 0.35,
  },
}


@dataclass(frozen=True)
class BuildEdgesOptions:
  enable_hierarchy_up: bool = True
  enable_hierarchy_down: bool = True
  enable_cluster_peer: bool = True
  cluster_peer_k: int = 8
  max_out_links_per_page: int = 40
  weight_hierarchy_up: float = 1.0
  weight_hierarchy_down: float = 0.7
  weight_cluster_peer: float = 0.4


@dataclass(frozen=True)
class ComputePriOptions:
  damping: float = 0.85
  include_block_types: list[str] | None = None
  use_weights: bool = True


def prepare_pages(
  input_json_path: Path,
  output_pages_path: Path,
  compression: str = "zstd",
) -> dict[str, object]:
  with input_json_path.open("r", encoding="utf-8") as file:
    raw_tree = json.load(file)

  rows = flatten_arborescence(raw_tree)
  pages_df = (
    pl.DataFrame(rows)
    .sort("path")
    .with_row_index(name="page_id", offset=1)
    .with_columns(
      pl.col("page_id").cast(pl.Int32),
      pl.col("depth").cast(pl.Int16),
      pl.col("is_leaf").cast(pl.Boolean),
    )
    .select(
      "page_id",
      "path",
      "parent_path",
      "depth",
      "section",
      "cluster_thematique",
      "is_leaf",
    )
  )

  output_pages_path.parent.mkdir(parents=True, exist_ok=True)
  pages_df.write_parquet(output_pages_path, compression=compression)

  section_counts_df = pages_df.group_by("section").len().sort("len", descending=True)
  top_sections = [
    {"section": row["section"], "count": int(row["len"])}
    for row in section_counts_df.head(10).to_dicts()
  ]

  return {
    "input_json": str(input_json_path),
    "output_pages_parquet": str(output_pages_path),
    "page_count": pages_df.height,
    "leaf_count": int(pages_df.select(pl.sum("is_leaf")).item()),
    "cluster_count": pages_df.select(pl.col("cluster_thematique").n_unique()).item(),
    "top_sections": top_sections,
  }


def prepare_outlinks_dataset(
  input_csv_path: Path,
  output_pages_path: Path,
  output_edges_path: Path,
  output_anchor_candidates_path: Path,
  source_host: str | None = None,
  max_out_links_per_page: int = 120,
  compression: str = "zstd",
) -> dict[str, object]:
  raw_lf = pl.scan_csv(
    str(input_csv_path),
    infer_schema_length=2_048,
    truncate_ragged_lines=True,
    ignore_errors=True,
  ).select(
    "Type",
    "From",
    "To",
    "Anchor Text",
    "Alt Text",
    "Follow",
    "Status Code",
    "Link Position",
  )

  normalized_lf = raw_lf.with_columns(
    pl.col("Type").fill_null("").str.strip_chars().str.to_lowercase().alias("link_type"),
    _url_host_expr(pl.col("From")).alias("from_host"),
    _url_host_expr(pl.col("To")).alias("to_host"),
    _url_path_expr(pl.col("From")).alias("from_path"),
    _url_path_expr(pl.col("To")).alias("to_path"),
    _normalize_block_type_expr(pl.col("Link Position")).alias("block_type"),
    _parse_boolean_expr(pl.col("Follow")).alias("is_follow"),
    pl.col("Status Code").cast(pl.Int32, strict=False).alias("status_code"),
    _coalesce_anchor_text_expr(
      anchor_text_expr=pl.col("Anchor Text"),
      alt_text_expr=pl.col("Alt Text"),
      target_path_expr=_url_path_expr(pl.col("To")),
    ).alias("anchor_text"),
  )

  selected_host = source_host.strip().lower() if source_host else None
  if selected_host is None:
    host_counts_df = (
      normalized_lf
      .filter(pl.col("from_host").is_not_null())
      .group_by("from_host")
      .len()
      .sort("len", descending=True)
      .collect()
    )
    if host_counts_df.height == 0:
      raise ValueError("Unable to infer source host from outlinks CSV.")
    selected_host = str(host_counts_df.row(0)[0])

  filtered_lf = (
    normalized_lf
    .filter(
      (pl.col("link_type") == "hyperlink")
      & (pl.col("from_host") == selected_host)
      & (pl.col("to_host") == selected_host)
      & pl.col("is_follow")
      & (pl.col("status_code") == 200)
      & pl.col("from_path").is_not_null()
      & pl.col("to_path").is_not_null()
      & (pl.col("from_path") != pl.col("to_path")),
    )
    .select("from_path", "to_path", "block_type", "anchor_text")
  )

  source_paths_lf = filtered_lf.select(pl.col("from_path").alias("path")).unique()
  pages_lf = (
    pl.concat(
      [
        source_paths_lf,
        filtered_lf.select(pl.col("to_path").alias("path")).unique(),
      ],
      how="vertical_relaxed",
    )
    .unique()
    .sort("path")
    .with_row_index(name="page_id", offset=1)
    .with_columns(
      pl.col("page_id").cast(pl.Int32),
      _parent_path_expr(pl.col("path")).alias("parent_path"),
      _depth_expr(pl.col("path")).alias("depth"),
      _section_expr(pl.col("path")).alias("section"),
      _cluster_expr(pl.col("path")).alias("cluster_thematique"),
    )
    .join(
      source_paths_lf.with_columns(pl.lit(True).alias("has_out")),
      on="path",
      how="left",
    )
    .with_columns(pl.col("has_out").fill_null(False).not_().alias("is_leaf"))
    .select(
      "page_id",
      "path",
      "parent_path",
      pl.col("depth").cast(pl.Int16),
      "section",
      "cluster_thematique",
      pl.col("is_leaf").cast(pl.Boolean),
    )
  )
  pages_df = pages_lf.collect()

  page_lookup_lf = pages_df.lazy().select(
    pl.col("path"),
    pl.col("page_id").cast(pl.Int32),
  )
  edges_with_ids_lf = (
    filtered_lf
    .join(
      page_lookup_lf.select(
        pl.col("path").alias("from_path"),
        pl.col("page_id").alias("source_id"),
      ),
      on="from_path",
      how="inner",
    )
    .join(
      page_lookup_lf.select(
        pl.col("path").alias("to_path"),
        pl.col("page_id").alias("target_id"),
      ),
      on="to_path",
      how="inner",
    )
    .with_columns(_block_rule_weight_expr(pl.col("block_type")).alias("base_weight"))
    .select(
      pl.col("source_id").cast(pl.Int32),
      pl.col("target_id").cast(pl.Int32),
      "block_type",
      pl.col("base_weight").cast(pl.Float32),
      "anchor_text",
      "to_path",
    )
  )

  edges_lf = (
    edges_with_ids_lf
    .group_by(["source_id", "target_id", "block_type"])
    .agg(pl.sum("base_weight").cast(pl.Float32).alias("rule_weight"))
    .sort(["source_id", "rule_weight", "target_id"], descending=[False, True, False])
  )
  if max_out_links_per_page > 0:
    edges_lf = edges_lf.group_by("source_id", maintain_order=True).head(max_out_links_per_page)
  edges_lf = edges_lf.select("source_id", "target_id", "block_type", "rule_weight")

  anchor_candidates_lf = (
    edges_with_ids_lf
    .with_columns(
      _infer_anchor_type_expr(
        anchor_text_expr=pl.col("anchor_text"),
        target_path_expr=pl.col("to_path"),
      ).alias("anchor_type"),
      pl.col("anchor_text").str.split(" ").list.len().cast(pl.Int16).alias("anchor_token_count"),
    )
    .group_by(["source_id", "target_id", "block_type", "anchor_type", "anchor_text"])
    .agg(
      pl.len().alias("anchor_instances"),
      pl.mean("base_weight").cast(pl.Float32).alias("rule_weight"),
      pl.first("anchor_token_count").cast(pl.Int16).alias("anchor_token_count"),
    )
    .sort(
      [
        "source_id",
        "target_id",
        "block_type",
        "anchor_type",
        "anchor_instances",
        "anchor_text",
      ],
      descending=[False, False, False, False, True, False],
    )
    .group_by(["source_id", "target_id", "block_type", "anchor_type"], maintain_order=True)
    .head(1)
    .join(
      edges_lf.select("source_id", "target_id", "block_type"),
      on=["source_id", "target_id", "block_type"],
      how="inner",
    )
    .select(
      pl.col("source_id").cast(pl.Int32),
      pl.col("target_id").cast(pl.Int32),
      "block_type",
      pl.col("rule_weight").cast(pl.Float32),
      "anchor_type",
      "anchor_text",
      pl.col("anchor_token_count").cast(pl.Int16),
    )
  )

  output_pages_path.parent.mkdir(parents=True, exist_ok=True)
  output_edges_path.parent.mkdir(parents=True, exist_ok=True)
  output_anchor_candidates_path.parent.mkdir(parents=True, exist_ok=True)
  pages_df.write_parquet(output_pages_path, compression=compression)
  edges_lf.sink_parquet(output_edges_path, compression=compression)
  anchor_candidates_lf.sink_parquet(output_anchor_candidates_path, compression=compression)

  total_row_count = int(raw_lf.select(pl.len()).collect().item())
  filtered_row_count = int(filtered_lf.select(pl.len()).collect().item())
  edge_count = int(pl.scan_parquet(output_edges_path).select(pl.len()).collect().item())
  candidate_count = int(pl.scan_parquet(output_anchor_candidates_path).select(pl.len()).collect().item())
  block_counts_df = (
    pl.scan_parquet(output_edges_path)
    .group_by("block_type")
    .len()
    .sort("len", descending=True)
    .collect()
  )
  anchor_type_counts_df = (
    pl.scan_parquet(output_anchor_candidates_path)
    .group_by("anchor_type")
    .len()
    .sort("len", descending=True)
    .collect()
  )

  return {
    "input_csv": str(input_csv_path),
    "source_host": selected_host,
    "raw_row_count": total_row_count,
    "filtered_row_count": filtered_row_count,
    "output_pages_parquet": str(output_pages_path),
    "output_edges_parquet": str(output_edges_path),
    "output_anchor_candidates_parquet": str(output_anchor_candidates_path),
    "page_count": pages_df.height,
    "edge_count": edge_count,
    "anchor_candidate_count": candidate_count,
    "max_out_links_per_page": max_out_links_per_page,
    "block_counts": [
      {"block_type": row["block_type"], "count": int(row["len"])}
      for row in block_counts_df.to_dicts()
    ],
    "anchor_type_counts": [
      {"anchor_type": row["anchor_type"], "count": int(row["len"])}
      for row in anchor_type_counts_df.to_dicts()
    ],
  }


def flatten_arborescence(tree: dict[str, Any]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []

  def visit(node: dict[str, Any], parent_path: str | None) -> None:
    for raw_key, child in node.items():
      normalized_key = _normalize_key(raw_key)
      path = f"{parent_path}{normalized_key}" if parent_path else normalized_key
      segments = _path_segments(path)
      rows.append(
        {
          "path": path,
          "parent_path": parent_path,
          "depth": len(segments),
          "section": segments[0] if segments else "root",
          "cluster_thematique": infer_cluster_thematique(path),
          "is_leaf": not isinstance(child, dict) or len(child) == 0,
        },
      )
      if isinstance(child, dict) and child:
        visit(child, path)

  visit(tree, parent_path=None)
  return rows


def infer_cluster_thematique(path: str) -> str:
  segments = _path_segments(path)
  if not segments:
    return "root:__root__"

  section = segments[0]
  anchor = segments[1] if len(segments) > 1 else "__root__"
  return f"{section}:{anchor}"


def build_edges(
  input_pages_path: Path,
  output_edges_path: Path,
  options: BuildEdgesOptions,
  compression: str = "zstd",
) -> dict[str, object]:
  pages_lf = pl.scan_parquet(input_pages_path).select(
    pl.col("page_id").cast(pl.Int32),
    pl.col("path"),
    pl.col("parent_path"),
    pl.col("cluster_thematique"),
  )
  final_edges_lf = build_edges_lazyframe(pages_lf, options)

  output_edges_path.parent.mkdir(parents=True, exist_ok=True)
  final_edges_lf.sink_parquet(output_edges_path, compression=compression)

  edges_scan = pl.scan_parquet(output_edges_path)
  edge_count = int(edges_scan.select(pl.len()).collect().item())
  block_counts_df = edges_scan.group_by("block_type").len().sort("len", descending=True).collect()
  max_out_degree = edges_scan.group_by("source_id").len().select(pl.max("len")).collect().item() if edge_count > 0 else 0

  return {
    "input_pages_parquet": str(input_pages_path),
    "output_edges_parquet": str(output_edges_path),
    "edge_count": edge_count,
    "max_out_degree_after_cap": int(max_out_degree),
    "block_counts": [
      {"block_type": row["block_type"], "count": int(row["len"])}
      for row in block_counts_df.to_dicts()
    ],
    "options": {
      "enable_hierarchy_up": options.enable_hierarchy_up,
      "enable_hierarchy_down": options.enable_hierarchy_down,
      "enable_cluster_peer": options.enable_cluster_peer,
      "cluster_peer_k": options.cluster_peer_k,
      "max_out_links_per_page": options.max_out_links_per_page,
      "weight_hierarchy_up": options.weight_hierarchy_up,
      "weight_hierarchy_down": options.weight_hierarchy_down,
      "weight_cluster_peer": options.weight_cluster_peer,
    },
  }


def build_edges_lazyframe(
  pages_lf: pl.LazyFrame,
  options: BuildEdgesOptions,
) -> pl.LazyFrame:
  edge_frames: list[pl.LazyFrame] = []

  parent_child_lf = _parent_child_pairs(pages_lf)

  if options.enable_hierarchy_up:
    edge_frames.append(
      parent_child_lf.select(
        pl.col("child_id").alias("source_id"),
        pl.col("parent_id").alias("target_id"),
        pl.lit("hierarchy_up").alias("block_type"),
        pl.lit(options.weight_hierarchy_up, dtype=pl.Float32).alias("rule_weight"),
        pl.lit(BLOCK_PRIORITY["hierarchy_up"], dtype=pl.Int8).alias("priority"),
      ),
    )

  if options.enable_hierarchy_down:
    edge_frames.append(
      parent_child_lf.select(
        pl.col("parent_id").alias("source_id"),
        pl.col("child_id").alias("target_id"),
        pl.lit("hierarchy_down").alias("block_type"),
        pl.lit(options.weight_hierarchy_down, dtype=pl.Float32).alias("rule_weight"),
        pl.lit(BLOCK_PRIORITY["hierarchy_down"], dtype=pl.Int8).alias("priority"),
      ),
    )

  if options.enable_cluster_peer and options.cluster_peer_k > 0:
    edge_frames.append(_build_cluster_peer_edges(pages_lf, options))

  raw_edges_lf = (
    pl.concat(edge_frames, how="vertical_relaxed")
    if edge_frames
    else _empty_edges_lazyframe(include_priority=True)
  )

  dedup_lf = (
    raw_edges_lf
    .filter(pl.col("source_id") != pl.col("target_id"))
    .group_by(["source_id", "target_id"])
    .agg(
      pl.sum("rule_weight").cast(pl.Float32).alias("rule_weight"),
      pl.min("priority").cast(pl.Int8).alias("priority"),
      pl.col("block_type").sort_by([pl.col("priority"), pl.col("block_type")]).first().alias("block_type"),
    )
  )

  ordered_lf = dedup_lf.sort(
    by=["source_id", "priority", "rule_weight", "target_id"],
    descending=[False, False, True, False],
  )

  capped_lf = (
    ordered_lf.group_by("source_id", maintain_order=True).head(options.max_out_links_per_page)
    if options.max_out_links_per_page > 0
    else ordered_lf
  )

  return capped_lf.select(
    pl.col("source_id").cast(pl.Int32),
    pl.col("target_id").cast(pl.Int32),
    pl.col("block_type"),
    pl.col("rule_weight").cast(pl.Float32),
  )


def compute_pri(
  input_pages_path: Path,
  input_edges_path: Path,
  output_pri_path: Path,
  options: ComputePriOptions,
  compression: str = "zstd",
) -> dict[str, object]:
  page_ids_df = pl.scan_parquet(input_pages_path).select("page_id").sort("page_id").collect()
  page_ids = page_ids_df.get_column("page_id").cast(pl.Int32).to_list()
  node_count = len(page_ids)

  edges_lf = pl.scan_parquet(input_edges_path).select("source_id", "target_id", "block_type", "rule_weight")
  if options.include_block_types:
    edges_lf = edges_lf.filter(pl.col("block_type").is_in(options.include_block_types))

  edges_df = edges_lf.collect()
  edge_count = edges_df.height

  invalid_edges = 0
  if node_count > 0 and edge_count > 0:
    invalid_edges = int(
      edges_df.filter(
        (pl.col("source_id") < 1)
        | (pl.col("source_id") > node_count)
        | (pl.col("target_id") < 1)
        | (pl.col("target_id") > node_count),
      ).height,
    )
  if invalid_edges > 0:
    raise ValueError(f"Found {invalid_edges} edges with source/target IDs outside page_id range.")

  if node_count == 0:
    pri_scores: list[float] = []
    cheirank_scores: list[float] = []
    out_degree: list[int] = []
    in_degree: list[int] = []
  elif edge_count == 0:
    pri_scores = [1.0 / node_count] * node_count
    cheirank_scores = [1.0 / node_count] * node_count
    out_degree = [0] * node_count
    in_degree = [0] * node_count
  else:
    source_idx = (edges_df.get_column("source_id").cast(pl.Int64) - 1).to_list()
    target_idx = (edges_df.get_column("target_id").cast(pl.Int64) - 1).to_list()
    edge_weights = edges_df.get_column("rule_weight").cast(pl.Float64).to_list() if options.use_weights else None

    graph = ig.Graph(n=node_count, directed=True)
    graph.add_edges(zip(source_idx, target_idx))

    if options.use_weights:
      graph.es["weight"] = edge_weights
      weights = graph.es["weight"]
    else:
      weights = None

    pri_scores = graph.pagerank(directed=True, damping=options.damping, weights=weights)
    out_degree = graph.outdegree()
    in_degree = graph.indegree()

    reversed_graph = ig.Graph(n=node_count, directed=True)
    reversed_graph.add_edges(zip(target_idx, source_idx))
    if options.use_weights:
      reversed_graph.es["weight"] = edge_weights
      reversed_weights = reversed_graph.es["weight"]
    else:
      reversed_weights = None
    cheirank_scores = reversed_graph.pagerank(
      directed=True,
      damping=options.damping,
      weights=reversed_weights,
    )

  results_df = pl.DataFrame(
    {
      "page_id": page_ids,
      "pri_score": pri_scores,
      "cheirank_score": cheirank_scores,
      "out_degree": out_degree,
      "in_degree": in_degree,
    },
    schema={
      "page_id": pl.Int32,
      "pri_score": pl.Float64,
      "cheirank_score": pl.Float64,
      "out_degree": pl.Int32,
      "in_degree": pl.Int32,
    },
  )

  rank_df = (
    results_df
    .select("page_id", "pri_score")
    .sort("pri_score", descending=True)
    .with_row_index(name="rank", offset=1)
    .select("page_id", pl.col("rank").cast(pl.Int32))
  )
  cheirank_rank_df = (
    results_df
    .select("page_id", "cheirank_score")
    .sort("cheirank_score", descending=True)
    .with_row_index(name="cheirank_rank", offset=1)
    .select("page_id", pl.col("cheirank_rank").cast(pl.Int32))
  )

  juice_potential_df = (
    results_df
    .select(
      "page_id",
      (
        pl.col("cheirank_score")
        * (pl.col("out_degree").cast(pl.Float64) + 1).log()
      ).alias("juice_potential_score"),
    )
  )
  results_df = (
    results_df
    .join(rank_df, on="page_id", how="left")
    .join(cheirank_rank_df, on="page_id", how="left")
    .join(juice_potential_df, on="page_id", how="left")
    .with_columns(pl.col("juice_potential_score").fill_null(0.0).cast(pl.Float64))
    .select(
      "page_id",
      "pri_score",
      "rank",
      "cheirank_score",
      "cheirank_rank",
      "juice_potential_score",
      "out_degree",
      "in_degree",
    )
    .sort("page_id")
  )

  low_power_threshold = (
    float(results_df.select(pl.col("pri_score").quantile(0.10, interpolation="nearest")).item())
    if node_count > 0
    else 0.0
  )
  donor_threshold = (
    float(results_df.select(pl.col("juice_potential_score").quantile(0.90, interpolation="nearest")).item())
    if node_count > 0
    else 0.0
  )
  results_df = results_df.with_columns(
    (pl.col("pri_score") <= low_power_threshold).alias("is_low_power"),
    (pl.col("juice_potential_score") >= donor_threshold).alias("can_give_juice"),
  )

  output_pri_path.parent.mkdir(parents=True, exist_ok=True)
  results_df.write_parquet(output_pri_path, compression=compression)

  pri_sum = float(results_df.select(pl.sum("pri_score")).item()) if node_count > 0 else 0.0
  cheirank_sum = float(results_df.select(pl.sum("cheirank_score")).item()) if node_count > 0 else 0.0
  top_pages = results_df.sort("pri_score", descending=True).head(10).to_dicts()
  top_cheirank_pages = results_df.sort("cheirank_score", descending=True).head(10).to_dicts()
  top_juice_donors = results_df.sort("juice_potential_score", descending=True).head(10).to_dicts()
  low_power_page_count = int(results_df.filter(pl.col("is_low_power")).height) if node_count > 0 else 0
  donor_page_count = int(results_df.filter(pl.col("can_give_juice")).height) if node_count > 0 else 0

  return {
    "input_pages_parquet": str(input_pages_path),
    "input_edges_parquet": str(input_edges_path),
    "output_pri_parquet": str(output_pri_path),
    "node_count": node_count,
    "edge_count": edge_count,
    "included_block_types": options.include_block_types or [],
    "use_weights": options.use_weights,
    "damping": options.damping,
    "pri_sum": pri_sum,
    "cheirank_sum": cheirank_sum,
    "low_power_threshold": low_power_threshold,
    "donor_threshold": donor_threshold,
    "low_power_page_count": low_power_page_count,
    "donor_page_count": donor_page_count,
    "top_pages": top_pages,
    "top_cheirank_pages": top_cheirank_pages,
    "top_juice_donors": top_juice_donors,
  }


def prepare_dashboard_data(
  input_pages_path: Path,
  input_edges_path: Path,
  input_pri_path: Path,
  output_page_segments_path: Path,
  output_url_metrics_path: Path,
  output_segment_metrics_path: Path,
  input_anchor_candidates_path: Path | None = None,
  compression: str = "zstd",
) -> dict[str, object]:
  pages_df = pl.read_parquet(input_pages_path).select(
    pl.col("page_id").cast(pl.Int32),
    pl.col("path").cast(pl.Utf8),
    pl.col("depth").cast(pl.Int16),
  )
  pri_df = _normalize_pri_for_dashboard(pl.read_parquet(input_pri_path))
  edges_lf = pl.scan_parquet(input_edges_path).select(
    pl.col("source_id").cast(pl.Int32),
    pl.col("target_id").cast(pl.Int32),
    pl.col("block_type"),
    pl.col("rule_weight").cast(pl.Float32),
  )
  out_stats_lf = (
    edges_lf
    .group_by("source_id")
    .agg(
      pl.len().cast(pl.Int32).alias("outgoing_links"),
      pl.sum("rule_weight").cast(pl.Float64).alias("outgoing_weight"),
      pl.col("target_id").n_unique().cast(pl.Int32).alias("unique_out_targets"),
    )
    .select(
      pl.col("source_id").alias("page_id"),
      "outgoing_links",
      "outgoing_weight",
      "unique_out_targets",
    )
  )
  in_stats_lf = (
    edges_lf
    .group_by("target_id")
    .agg(
      pl.len().cast(pl.Int32).alias("incoming_links"),
      pl.sum("rule_weight").cast(pl.Float64).alias("incoming_weight"),
      pl.col("source_id").n_unique().cast(pl.Int32).alias("unique_in_sources"),
    )
    .select(
      pl.col("target_id").alias("page_id"),
      "incoming_links",
      "incoming_weight",
      "unique_in_sources",
    )
  )

  anchor_candidates_exists = (
    input_anchor_candidates_path is not None
    and input_anchor_candidates_path.exists()
  )
  if anchor_candidates_exists and input_anchor_candidates_path is not None:
    anchor_stats_lf = (
      pl.scan_parquet(input_anchor_candidates_path)
      .select(
        pl.col("source_id").cast(pl.Int32),
        pl.col("anchor_text"),
        pl.col("anchor_type"),
      )
      .group_by("source_id")
      .agg(
        pl.col("anchor_text").n_unique().cast(pl.Int32).alias("unique_anchor_texts_out"),
        pl.col("anchor_type").n_unique().cast(pl.Int32).alias("unique_anchor_types_out"),
      )
      .select(
        pl.col("source_id").alias("page_id"),
        "unique_anchor_texts_out",
        "unique_anchor_types_out",
      )
    )
  else:
    anchor_stats_lf = pl.DataFrame(
      schema={
        "page_id": pl.Int32,
        "unique_anchor_texts_out": pl.Int32,
        "unique_anchor_types_out": pl.Int32,
      },
    ).lazy()

  url_metrics_df = (
    pages_df.lazy()
    .join(pri_df.lazy(), on="page_id", how="left")
    .join(out_stats_lf, on="page_id", how="left")
    .join(in_stats_lf, on="page_id", how="left")
    .join(anchor_stats_lf, on="page_id", how="left")
    .with_columns(
      pl.col("incoming_links").fill_null(0).cast(pl.Int32),
      pl.col("outgoing_links").fill_null(0).cast(pl.Int32),
      pl.col("incoming_weight").fill_null(0.0).cast(pl.Float64),
      pl.col("outgoing_weight").fill_null(0.0).cast(pl.Float64),
      pl.col("unique_out_targets").fill_null(0).cast(pl.Int32),
      pl.col("unique_in_sources").fill_null(0).cast(pl.Int32),
      pl.col("unique_anchor_texts_out").fill_null(0).cast(pl.Int32),
      pl.col("unique_anchor_types_out").fill_null(0).cast(pl.Int32),
    )
    .with_columns(
      (
        pl.col("unique_anchor_texts_out")
        / pl.col("outgoing_links").clip(lower_bound=1)
      ).cast(pl.Float64).alias("lexical_diversity_out"),
    )
    .select(
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
    )
    .sort("page_id")
    .collect()
  )

  page_segments_df = _build_page_segments_df(
    url_metrics_df.select("page_id", "path", "depth"),
  )
  segment_metrics_df = (
    page_segments_df.lazy()
    .join(
      url_metrics_df.lazy().select(
        "page_id",
        "pri_score",
        "cheirank_score",
        "can_give_juice",
        "is_low_power",
        "in_degree",
        "out_degree",
        "lexical_diversity_out",
      ),
      on="page_id",
      how="left",
    )
    .group_by(["level", "segment_path", "segment"])
    .agg(
      pl.len().cast(pl.Int32).alias("page_count"),
      pl.sum("pri_score").cast(pl.Float64).alias("pri_sum"),
      pl.mean("pri_score").cast(pl.Float64).alias("pri_mean"),
      pl.sum("cheirank_score").cast(pl.Float64).alias("cheirank_sum"),
      pl.mean("cheirank_score").cast(pl.Float64).alias("cheirank_mean"),
      pl.col("can_give_juice").cast(pl.Float64).mean().cast(pl.Float64).alias("donor_ratio"),
      pl.col("is_low_power").cast(pl.Float64).mean().cast(pl.Float64).alias("low_power_ratio"),
      pl.mean("in_degree").cast(pl.Float64).alias("avg_in_degree"),
      pl.mean("out_degree").cast(pl.Float64).alias("avg_out_degree"),
      pl.mean("lexical_diversity_out").cast(pl.Float64).alias("avg_lexical_diversity_out"),
    )
    .sort(["level", "page_count", "segment_path"], descending=[False, True, False])
    .collect()
  )

  output_page_segments_path.parent.mkdir(parents=True, exist_ok=True)
  output_url_metrics_path.parent.mkdir(parents=True, exist_ok=True)
  output_segment_metrics_path.parent.mkdir(parents=True, exist_ok=True)
  page_segments_df.write_parquet(output_page_segments_path, compression=compression)
  url_metrics_df.write_parquet(output_url_metrics_path, compression=compression)
  segment_metrics_df.write_parquet(output_segment_metrics_path, compression=compression)

  top_segments_df = (
    segment_metrics_df
    .filter(pl.col("level") == 1)
    .sort("page_count", descending=True)
    .head(10)
  )

  return {
    "input_pages_parquet": str(input_pages_path),
    "input_edges_parquet": str(input_edges_path),
    "input_pri_parquet": str(input_pri_path),
    "input_anchor_candidates_parquet": str(input_anchor_candidates_path) if input_anchor_candidates_path else None,
    "output_page_segments_parquet": str(output_page_segments_path),
    "output_url_metrics_parquet": str(output_url_metrics_path),
    "output_segment_metrics_parquet": str(output_segment_metrics_path),
    "anchor_candidates_used": anchor_candidates_exists,
    "page_count": url_metrics_df.height,
    "page_segment_rows": page_segments_df.height,
    "segment_count": segment_metrics_df.height,
    "top_level_1_segments": [
      {
        "segment_path": row["segment_path"],
        "page_count": int(row["page_count"]),
        "pri_sum": float(row["pri_sum"]),
      }
      for row in top_segments_df.to_dicts()
    ],
  }


def prepare_anchor_dataset(
  input_pages_path: Path,
  input_edges_path: Path,
  output_anchor_candidates_path: Path,
  compression: str = "zstd",
) -> dict[str, object]:
  pages_lf = pl.scan_parquet(input_pages_path).select(
    pl.col("page_id").cast(pl.Int32),
    pl.col("path"),
    pl.col("parent_path"),
  )
  edges_lf = pl.scan_parquet(input_edges_path).select(
    pl.col("source_id").cast(pl.Int32),
    pl.col("target_id").cast(pl.Int32),
    pl.col("block_type"),
    pl.col("rule_weight").cast(pl.Float32),
  )

  target_pages_lf = pages_lf.select(
    pl.col("page_id").alias("target_id"),
    pl.col("path").alias("target_path"),
    pl.col("parent_path").alias("target_parent_path"),
  )

  base_lf = (
    edges_lf
    .join(target_pages_lf, on="target_id", how="left")
    .with_columns(
      _label_expr_from_path(pl.col("target_path")).alias("target_label"),
      _label_expr_from_path(pl.col("target_parent_path")).alias("target_parent_label"),
    )
    .select(
      "source_id",
      "target_id",
      "block_type",
      "rule_weight",
      "target_label",
      "target_parent_label",
    )
  )

  exact_lf = base_lf.select(
    "source_id",
    "target_id",
    "block_type",
    "rule_weight",
    pl.lit("exact").alias("anchor_type"),
    pl.col("target_label").alias("anchor_text"),
  )
  partial_lf = base_lf.select(
    "source_id",
    "target_id",
    "block_type",
    "rule_weight",
    pl.lit("partial").alias("anchor_type"),
    pl.when(pl.col("target_parent_label").is_not_null() & (pl.col("target_parent_label") != pl.col("target_label")))
    .then(pl.concat_str([pl.col("target_label"), pl.lit(" "), pl.col("target_parent_label")]))
    .otherwise(pl.col("target_label"))
    .alias("anchor_text"),
  )
  contextual_lf = base_lf.select(
    "source_id",
    "target_id",
    "block_type",
    "rule_weight",
    pl.lit("contextual").alias("anchor_type"),
    pl.concat_str([pl.lit("Voir "), pl.col("target_label")]).alias("anchor_text"),
  )
  generic_lf = base_lf.select(
    "source_id",
    "target_id",
    "block_type",
    "rule_weight",
    pl.lit("generic").alias("anchor_type"),
    pl.lit("En savoir plus").alias("anchor_text"),
  )

  candidates_lf = (
    pl.concat([exact_lf, partial_lf, contextual_lf, generic_lf], how="vertical_relaxed")
    .with_columns(
      pl.col("anchor_text").fill_null("Découvrir"),
      pl.col("anchor_text").str.split(" ").list.len().cast(pl.Int16).alias("anchor_token_count"),
    )
    .select(
      pl.col("source_id").cast(pl.Int32),
      pl.col("target_id").cast(pl.Int32),
      "block_type",
      pl.col("rule_weight").cast(pl.Float32),
      "anchor_type",
      "anchor_text",
      "anchor_token_count",
    )
  )

  output_anchor_candidates_path.parent.mkdir(parents=True, exist_ok=True)
  candidates_lf.sink_parquet(output_anchor_candidates_path, compression=compression)

  candidates_scan = pl.scan_parquet(output_anchor_candidates_path)
  candidate_count = int(candidates_scan.select(pl.len()).collect().item())
  anchor_type_counts_df = candidates_scan.group_by("anchor_type").len().sort("len", descending=True).collect()

  return {
    "input_pages_parquet": str(input_pages_path),
    "input_edges_parquet": str(input_edges_path),
    "output_anchor_candidates_parquet": str(output_anchor_candidates_path),
    "candidate_count": candidate_count,
    "anchor_type_counts": [
      {"anchor_type": row["anchor_type"], "count": int(row["len"])}
      for row in anchor_type_counts_df.to_dicts()
    ],
  }


def model_anchor_scenarios(
  input_pages_path: Path,
  input_edges_path: Path,
  output_dir: Path,
  input_anchor_candidates_path: Path | None = None,
  damping: float = 0.85,
  compression: str = "zstd",
) -> dict[str, object]:
  output_dir.mkdir(parents=True, exist_ok=True)
  if input_anchor_candidates_path is None:
    candidates_path = output_dir / "anchor_candidates.parquet"
    prepare_summary = prepare_anchor_dataset(
      input_pages_path=input_pages_path,
      input_edges_path=input_edges_path,
      output_anchor_candidates_path=candidates_path,
      compression=compression,
    )
  else:
    candidates_path = input_anchor_candidates_path
    if not candidates_path.exists():
      raise ValueError(f"Anchor candidates parquet not found: {candidates_path}")
    candidates_scan = pl.scan_parquet(candidates_path)
    required_columns = {"source_id", "target_id", "block_type", "rule_weight", "anchor_type", "anchor_text"}
    missing_columns = required_columns.difference(set(candidates_scan.collect_schema().names()))
    if missing_columns:
      raise ValueError(
        f"Anchor candidates parquet misses required columns: {sorted(missing_columns)}",
      )
    anchor_type_counts_df = candidates_scan.group_by("anchor_type").len().sort("len", descending=True).collect()
    prepare_summary = {
      "input_pages_parquet": str(input_pages_path),
      "input_edges_parquet": str(input_edges_path),
      "output_anchor_candidates_parquet": str(candidates_path),
      "candidate_count": int(candidates_scan.select(pl.len()).collect().item()),
      "anchor_type_counts": [
        {"anchor_type": row["anchor_type"], "count": int(row["len"])}
        for row in anchor_type_counts_df.to_dicts()
      ],
      "source": "provided",
    }

  candidates_df = pl.read_parquet(candidates_path)
  base_edges_df = (
    candidates_df
    .select("source_id", "target_id", "block_type", "rule_weight")
    .unique()
    .sort(["source_id", "target_id", "block_type"])
  )

  scenario_summaries: list[dict[str, object]] = []
  scenario_pri_tables: list[tuple[str, pl.DataFrame]] = []

  for scenario_name, scenario_settings in ANCHOR_SCENARIOS.items():
    distribution = scenario_settings["distribution"]
    diversity_alpha = float(scenario_settings["diversity_alpha"])

    scenario_edges_df = _build_edges_for_anchor_scenario(
      base_edges_df=base_edges_df,
      candidates_df=candidates_df,
      distribution=distribution,
      diversity_alpha=diversity_alpha,
    )

    scenario_edges_path = output_dir / f"{scenario_name}_edges.parquet"
    scenario_pri_path = output_dir / f"{scenario_name}_pri.parquet"
    scenario_edges_df.write_parquet(scenario_edges_path, compression=compression)

    pri_summary = compute_pri(
      input_pages_path=input_pages_path,
      input_edges_path=scenario_edges_path,
      output_pri_path=scenario_pri_path,
      options=ComputePriOptions(
        damping=damping,
        include_block_types=None,
        use_weights=True,
      ),
      compression=compression,
    )
    scenario_pri_df = pl.read_parquet(scenario_pri_path).select(
      "page_id",
      pl.col("pri_score").alias(f"pri_{scenario_name}"),
    )
    scenario_pri_tables.append((scenario_name, scenario_pri_df))

    anchor_mix_df = (
      scenario_edges_df
      .group_by("anchor_type")
      .len()
      .sort("len", descending=True)
    )
    scenario_summaries.append(
      {
        "scenario": scenario_name,
        "scenario_edges_parquet": str(scenario_edges_path),
        "scenario_pri_parquet": str(scenario_pri_path),
        "edge_count": scenario_edges_df.height,
        "avg_anchor_diversity_score": float(scenario_edges_df.select(pl.mean("anchor_diversity_score")).item()),
        "anchor_mix": [
          {"anchor_type": row["anchor_type"], "count": int(row["len"])}
          for row in anchor_mix_df.to_dicts()
        ],
        "pri_sum": pri_summary["pri_sum"],
      },
    )

  comparison_df = _build_scenario_comparison(scenario_pri_tables)
  comparison_path = output_dir / "scenario_pri_comparison.parquet"
  comparison_df.write_parquet(comparison_path, compression=compression)

  return {
    "input_pages_parquet": str(input_pages_path),
    "input_edges_parquet": str(input_edges_path),
    "output_dir": str(output_dir),
    "anchor_candidates_parquet": str(candidates_path),
    "scenario_comparison_parquet": str(comparison_path),
    "prepare_anchor_dataset": prepare_summary,
    "scenarios": scenario_summaries,
  }


def append_experiment_log(log_path: Path, record: dict[str, object]) -> None:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  record_df = pl.DataFrame([record])
  if log_path.exists():
    existing_df = pl.read_parquet(log_path)
    pl.concat([existing_df, record_df], how="vertical_relaxed").write_parquet(
      log_path,
      compression="zstd",
    )
    return

  record_df.write_parquet(log_path, compression="zstd")


def export_workspace_report(
  workspace: Path,
  output_dir: Path,
  include_scenarios: bool = True,
) -> dict[str, object]:
  resolved_workspace = workspace.resolve()
  resolved_output_dir = output_dir.resolve()
  csv_dir = resolved_output_dir / "csv"
  csv_dir.mkdir(parents=True, exist_ok=True)

  base_artifacts = [
    {
      "name": "pages",
      "path": resolved_workspace / "pages.parquet",
      "required": True,
      "csv_path": csv_dir / "pages.csv",
    },
    {
      "name": "edges",
      "path": resolved_workspace / "edges.parquet",
      "required": True,
      "csv_path": csv_dir / "edges.csv",
    },
    {
      "name": "pri_scores",
      "path": resolved_workspace / "pri_scores.parquet",
      "required": True,
      "csv_path": csv_dir / "pri_scores.csv",
    },
    {
      "name": "anchor_candidates",
      "path": resolved_workspace / "anchor_candidates.parquet",
      "required": False,
      "csv_path": csv_dir / "anchor_candidates.csv",
    },
    {
      "name": "page_segments",
      "path": resolved_workspace / "page_segments.parquet",
      "required": False,
      "csv_path": csv_dir / "page_segments.csv",
    },
    {
      "name": "url_metrics",
      "path": resolved_workspace / "url_metrics.parquet",
      "required": False,
      "csv_path": csv_dir / "url_metrics.csv",
    },
    {
      "name": "segment_metrics",
      "path": resolved_workspace / "segment_metrics.parquet",
      "required": False,
      "csv_path": csv_dir / "segment_metrics.csv",
    },
  ]

  scenario_artifacts: list[dict[str, Any]] = []
  scenarios_dir = resolved_workspace / "scenarios"
  if include_scenarios and scenarios_dir.exists():
    for scenario_path in sorted(scenarios_dir.glob("*.parquet")):
      scenario_artifacts.append(
        {
          "name": f"scenarios/{scenario_path.stem}",
          "path": scenario_path,
          "required": False,
          "csv_path": csv_dir / "scenarios" / f"{scenario_path.stem}.csv",
        },
      )

  artifact_specs = [*base_artifacts, *scenario_artifacts]
  missing_required = [
    str(spec["path"])
    for spec in artifact_specs
    if spec["required"] and not Path(spec["path"]).exists()
  ]
  if missing_required:
    raise ValueError(
      "Missing required artifacts for export: "
      + ", ".join(missing_required),
    )

  exported_files: list[dict[str, object]] = []
  missing_optional_artifacts: list[str] = []

  for spec in artifact_specs:
    source_path = Path(spec["path"])
    if not source_path.exists():
      if not spec["required"]:
        missing_optional_artifacts.append(str(source_path))
      continue

    csv_path = Path(spec["csv_path"])
    row_count = _parquet_row_count(source_path)
    _parquet_to_csv(source_path, csv_path)
    exported_files.append(
      {
        "name": str(spec["name"]),
        "source_parquet": str(source_path),
        "output_csv": str(csv_path),
        "row_count": row_count,
      },
    )

  summary_payload = _build_export_summary_payload(
    workspace=resolved_workspace,
    output_dir=resolved_output_dir,
    exported_files=exported_files,
    missing_optional_artifacts=missing_optional_artifacts,
  )
  report_json_path = resolved_output_dir / "synthesis_report.json"
  report_markdown_path = resolved_output_dir / "synthesis_report.md"

  resolved_output_dir.mkdir(parents=True, exist_ok=True)
  with report_json_path.open("w", encoding="utf-8") as file:
    json.dump(summary_payload, file, indent=2, ensure_ascii=False)
  with report_markdown_path.open("w", encoding="utf-8") as file:
    file.write(_build_export_summary_markdown(summary_payload))

  return {
    "workspace": str(resolved_workspace),
    "output_dir": str(resolved_output_dir),
    "exported_csv_count": len(exported_files),
    "exported_csv_files": exported_files,
    "missing_optional_artifacts": missing_optional_artifacts,
    "synthesis_report_json": str(report_json_path),
    "synthesis_report_markdown": str(report_markdown_path),
    "kpis": summary_payload["kpis"],
  }


def _parent_child_pairs(pages_lf: pl.LazyFrame) -> pl.LazyFrame:
  path_lookup_lf = pages_lf.select(
    pl.col("path").alias("lookup_path"),
    pl.col("page_id").alias("lookup_page_id"),
  )
  return (
    pages_lf
    .filter(pl.col("parent_path").is_not_null())
    .join(
      path_lookup_lf,
      left_on="parent_path",
      right_on="lookup_path",
      how="inner",
    )
    .select(
      pl.col("page_id").alias("child_id"),
      pl.col("lookup_page_id").alias("parent_id"),
    )
  )


def _build_cluster_peer_edges(
  pages_lf: pl.LazyFrame,
  options: BuildEdgesOptions,
) -> pl.LazyFrame:
  ranked_lf = (
    pages_lf
    .select("page_id", "cluster_thematique")
    .sort(["cluster_thematique", "page_id"])
    .with_columns(
      pl.int_range(pl.len()).over("cluster_thematique").alias("cluster_rank"),
      pl.len().over("cluster_thematique").alias("cluster_size"),
    )
  )

  lookup_lf = ranked_lf.select(
    "cluster_thematique",
    pl.col("cluster_rank").alias("target_rank"),
    pl.col("page_id").alias("target_id"),
  )

  step_frames: list[pl.LazyFrame] = []
  for step in range(1, options.cluster_peer_k + 1):
    step_frames.append(
      ranked_lf
      .with_columns((pl.col("cluster_rank") + step).alias("target_rank"))
      .filter(pl.col("target_rank") < pl.col("cluster_size"))
      .join(
        lookup_lf,
        on=["cluster_thematique", "target_rank"],
        how="inner",
      )
      .select(
        pl.col("page_id").alias("source_id"),
        pl.col("target_id").alias("target_id"),
        pl.lit("cluster_peer").alias("block_type"),
        pl.lit(options.weight_cluster_peer, dtype=pl.Float32).alias("rule_weight"),
        pl.lit(BLOCK_PRIORITY["cluster_peer"], dtype=pl.Int8).alias("priority"),
      ),
    )

  return (
    pl.concat(step_frames, how="vertical_relaxed")
    if step_frames
    else _empty_edges_lazyframe(include_priority=True)
  )


def _label_expr_from_path(path_expr: pl.Expr) -> pl.Expr:
  cleaned = path_expr.fill_null("").str.strip_suffix("/")
  segments = cleaned.str.split("/")
  last_segment = segments.list.last()
  previous_segment = segments.list.get(-2, null_on_oob=True)
  slug = (
    pl.when(last_segment == "_meta")
    .then(previous_segment)
    .otherwise(last_segment)
    .fill_null("destination")
  )
  return (
    slug
    .str.replace_all("-", " ")
    .str.replace_all("_", " ")
    .str.to_titlecase()
  )


def _url_host_expr(url_expr: pl.Expr) -> pl.Expr:
  return (
    url_expr
    .fill_null("")
    .str.strip_chars()
    .str.replace_all(r"[?#].*$", "")
    .str.extract(r"(?i)^https?://([^/:?#]+)", 1)
    .str.to_lowercase()
    .str.replace(r"^www\.", "")
  )


def _url_path_expr(url_expr: pl.Expr) -> pl.Expr:
  raw_path = (
    url_expr
    .fill_null("")
    .str.strip_chars()
    .str.replace_all(r"[?#].*$", "")
    .str.extract(r"(?i)^https?://[^/:?#]+([^?#]*)$", 1)
    .fill_null("")
    .str.replace_all(r"/{2,}", "/")
    .str.replace(r"/+$", "")
  )
  return pl.when(raw_path == "").then(pl.lit("/")).otherwise(raw_path)


def _parse_boolean_expr(value_expr: pl.Expr) -> pl.Expr:
  return (
    value_expr
    .fill_null("")
    .str.strip_chars()
    .str.to_lowercase()
    .is_in(["true", "1", "yes", "y"])
  )


def _normalize_block_type_expr(block_type_expr: pl.Expr) -> pl.Expr:
  normalized = (
    block_type_expr
    .fill_null("")
    .str.strip_chars()
    .str.to_lowercase()
    .str.replace_all(r"[^a-z0-9]+", "_")
    .str.strip_chars("_")
  )
  return pl.when(normalized == "").then(pl.lit("unknown")).otherwise(normalized)


def _block_rule_weight_expr(block_type_expr: pl.Expr) -> pl.Expr:
  return (
    pl.when(block_type_expr == "content")
    .then(pl.lit(1.0))
    .when(block_type_expr == "navigation")
    .then(pl.lit(0.9))
    .when(block_type_expr == "footer")
    .then(pl.lit(0.45))
    .when(block_type_expr == "head")
    .then(pl.lit(0.25))
    .otherwise(pl.lit(0.6))
    .cast(pl.Float32)
  )


def _coalesce_anchor_text_expr(
  anchor_text_expr: pl.Expr,
  alt_text_expr: pl.Expr,
  target_path_expr: pl.Expr,
) -> pl.Expr:
  normalized_anchor = (
    anchor_text_expr
    .fill_null("")
    .str.strip_chars()
  )
  normalized_alt = (
    alt_text_expr
    .fill_null("")
    .str.strip_chars()
  )
  target_label = _label_expr_from_path(target_path_expr).fill_null("En savoir plus")
  return (
    pl.when(normalized_anchor != "")
    .then(normalized_anchor)
    .when(normalized_alt != "")
    .then(normalized_alt)
    .otherwise(target_label)
  )


def _normalize_anchor_lexical_expr(anchor_text_expr: pl.Expr) -> pl.Expr:
  return (
    anchor_text_expr
    .fill_null("")
    .str.to_lowercase()
    .str.replace_all(r"[^\p{L}\p{N}]+", " ")
    .str.replace_all(r"\s+", " ")
    .str.strip_chars()
  )


def _infer_anchor_type_expr(
  anchor_text_expr: pl.Expr,
  target_path_expr: pl.Expr,
) -> pl.Expr:
  normalized_anchor = _normalize_anchor_lexical_expr(anchor_text_expr)
  normalized_label = _normalize_anchor_lexical_expr(_label_expr_from_path(target_path_expr))
  partial_match = (
    (
      normalized_anchor.str.starts_with(normalized_label)
      | normalized_anchor.str.ends_with(normalized_label)
      | normalized_label.str.starts_with(normalized_anchor)
    )
    & (normalized_anchor.str.len_chars() >= 3)
    & (normalized_label.str.len_chars() >= 3)
  )
  return (
    pl.when(normalized_anchor.is_in(GENERIC_ANCHOR_TEXTS) | (normalized_anchor == ""))
    .then(pl.lit("generic"))
    .when(normalized_anchor == normalized_label)
    .then(pl.lit("exact"))
    .when(partial_match)
    .then(pl.lit("partial"))
    .otherwise(pl.lit("contextual"))
  )


def _parent_path_expr(path_expr: pl.Expr) -> pl.Expr:
  parent_candidate = path_expr.str.replace(r"/[^/]+$", "")
  return (
    pl.when(path_expr == "/")
    .then(pl.lit(None, dtype=pl.Utf8))
    .when(parent_candidate == "")
    .then(pl.lit("/"))
    .otherwise(parent_candidate)
  )


def _depth_expr(path_expr: pl.Expr) -> pl.Expr:
  return (
    pl.when(path_expr == "/")
    .then(pl.lit(0))
    .otherwise(path_expr.str.strip_prefix("/").str.split("/").list.len())
    .cast(pl.Int16)
  )


def _section_expr(path_expr: pl.Expr) -> pl.Expr:
  return (
    pl.when(path_expr == "/")
    .then(pl.lit("root"))
    .otherwise(path_expr.str.strip_prefix("/").str.split("/").list.first().fill_null("root"))
  )


def _cluster_expr(path_expr: pl.Expr) -> pl.Expr:
  segments = path_expr.str.strip_prefix("/").str.split("/")
  section = pl.when(path_expr == "/").then(pl.lit("root")).otherwise(segments.list.first().fill_null("root"))
  cluster_anchor = segments.list.get(1, null_on_oob=True).fill_null("__root__")
  return pl.concat_str([section, pl.lit(":"), cluster_anchor])


def _build_edges_for_anchor_scenario(
  base_edges_df: pl.DataFrame,
  candidates_df: pl.DataFrame,
  distribution: dict[str, float],
  diversity_alpha: float,
) -> pl.DataFrame:
  selected_edges_df = (
    base_edges_df
    .with_columns(
      _scenario_selector_expr().alias("anchor_selector"),
    )
    .with_columns(
      _anchor_type_selection_expr(distribution).alias("anchor_type"),
    )
    .drop("anchor_selector")
    .join(
      candidates_df.select("source_id", "target_id", "block_type", "anchor_type", "anchor_text"),
      on=["source_id", "target_id", "block_type", "anchor_type"],
      how="left",
    )
    .with_columns(pl.col("anchor_text").fill_null("En savoir plus"))
  )

  lexical_diversity_df = (
    selected_edges_df
    .group_by("source_id")
    .agg(
      pl.len().alias("out_degree"),
      pl.col("anchor_text").n_unique().alias("unique_anchors"),
    )
    .with_columns(
      (
        pl.col("unique_anchors")
        / pl.col("out_degree").clip(lower_bound=1)
      ).cast(pl.Float32).alias("lexical_diversity_ratio"),
    )
    .select("source_id", "lexical_diversity_ratio")
  )

  type_entropy_df = (
    selected_edges_df
    .group_by(["source_id", "anchor_type"])
    .len()
    .with_columns(
      (
        pl.col("len")
        / pl.col("len").sum().over("source_id")
      ).alias("type_probability"),
    )
    .with_columns(
      (
        -pl.col("type_probability")
        * pl.col("type_probability").log()
      ).alias("entropy_component"),
    )
    .group_by("source_id")
    .agg(pl.sum("entropy_component").alias("anchor_type_entropy"))
    .with_columns(
      (
        pl.col("anchor_type_entropy")
        / math.log(len(ANCHOR_TYPES))
      ).cast(pl.Float32).alias("anchor_type_entropy_norm"),
    )
    .select("source_id", "anchor_type_entropy_norm")
  )

  diversity_df = (
    lexical_diversity_df
    .join(type_entropy_df, on="source_id", how="left")
    .with_columns(
      pl.col("anchor_type_entropy_norm").fill_null(0.0),
      (
        (pl.col("lexical_diversity_ratio") * 0.5)
        + (pl.col("anchor_type_entropy_norm") * 0.5)
      ).cast(pl.Float32).alias("anchor_diversity_score"),
    )
    .select("source_id", "anchor_diversity_score")
  )

  return (
    selected_edges_df
    .join(diversity_df, on="source_id", how="left")
    .with_columns(
      pl.col("anchor_diversity_score").fill_null(0.0),
      (
        pl.col("rule_weight")
        * (1 + (pl.col("anchor_diversity_score") * diversity_alpha))
      ).cast(pl.Float32).alias("rule_weight"),
    )
    .select(
      pl.col("source_id").cast(pl.Int32),
      pl.col("target_id").cast(pl.Int32),
      "block_type",
      pl.col("rule_weight").cast(pl.Float32),
      "anchor_type",
      "anchor_text",
      pl.col("anchor_diversity_score").cast(pl.Float32),
    )
  )


def _build_scenario_comparison(
  scenario_pri_tables: list[tuple[str, pl.DataFrame]],
) -> pl.DataFrame:
  if not scenario_pri_tables:
    return pl.DataFrame(schema={"page_id": pl.Int32})

  baseline_name, baseline_df = scenario_pri_tables[0]
  merged_df = baseline_df
  for _, scenario_df in scenario_pri_tables[1:]:
    merged_df = merged_df.join(scenario_df, on="page_id", how="inner")

  baseline_col = f"pri_{baseline_name}"
  for scenario_name, _ in scenario_pri_tables[1:]:
    scenario_col = f"pri_{scenario_name}"
    merged_df = merged_df.with_columns(
      (pl.col(scenario_col) - pl.col(baseline_col)).alias(f"delta_{scenario_name}_vs_{baseline_name}"),
    )
  return merged_df.sort("page_id")


def _normalize_pri_for_dashboard(pri_df: pl.DataFrame) -> pl.DataFrame:
  schema_columns = set(pri_df.columns)
  if "page_id" not in schema_columns or "pri_score" not in schema_columns:
    raise ValueError("`pri_scores.parquet` must contain at least `page_id` and `pri_score`.")

  normalized_df = pri_df.with_columns(
    pl.col("page_id").cast(pl.Int32),
    pl.col("pri_score").cast(pl.Float64),
  )
  if "rank" not in schema_columns:
    normalized_df = normalized_df.with_columns(
      pl.col("pri_score").rank(method="ordinal", descending=True).cast(pl.Int32).alias("rank"),
    )
  else:
    normalized_df = normalized_df.with_columns(pl.col("rank").cast(pl.Int32))

  if "cheirank_score" not in schema_columns:
    normalized_df = normalized_df.with_columns(pl.col("pri_score").alias("cheirank_score"))
  else:
    normalized_df = normalized_df.with_columns(pl.col("cheirank_score").cast(pl.Float64))

  if "cheirank_rank" not in schema_columns:
    normalized_df = normalized_df.with_columns(
      pl.col("cheirank_score").rank(method="ordinal", descending=True).cast(pl.Int32).alias("cheirank_rank"),
    )
  else:
    normalized_df = normalized_df.with_columns(pl.col("cheirank_rank").cast(pl.Int32))

  if "out_degree" not in schema_columns:
    normalized_df = normalized_df.with_columns(pl.lit(0).cast(pl.Int32).alias("out_degree"))
  else:
    normalized_df = normalized_df.with_columns(pl.col("out_degree").cast(pl.Int32))

  if "in_degree" not in schema_columns:
    normalized_df = normalized_df.with_columns(pl.lit(0).cast(pl.Int32).alias("in_degree"))
  else:
    normalized_df = normalized_df.with_columns(pl.col("in_degree").cast(pl.Int32))

  if "juice_potential_score" not in schema_columns:
    normalized_df = normalized_df.with_columns(
      (
        pl.col("cheirank_score")
        * (pl.col("out_degree").cast(pl.Float64) + 1).log()
      ).alias("juice_potential_score"),
    )
  else:
    normalized_df = normalized_df.with_columns(pl.col("juice_potential_score").cast(pl.Float64))

  low_power_threshold = (
    float(normalized_df.select(pl.col("pri_score").quantile(0.10, interpolation="nearest")).item())
    if normalized_df.height > 0
    else 0.0
  )
  donor_threshold = (
    float(normalized_df.select(pl.col("juice_potential_score").quantile(0.90, interpolation="nearest")).item())
    if normalized_df.height > 0
    else 0.0
  )

  if "is_low_power" not in schema_columns:
    normalized_df = normalized_df.with_columns((pl.col("pri_score") <= low_power_threshold).alias("is_low_power"))
  else:
    normalized_df = normalized_df.with_columns(pl.col("is_low_power").cast(pl.Boolean))

  if "can_give_juice" not in schema_columns:
    normalized_df = normalized_df.with_columns((pl.col("juice_potential_score") >= donor_threshold).alias("can_give_juice"))
  else:
    normalized_df = normalized_df.with_columns(pl.col("can_give_juice").cast(pl.Boolean))

  return (
    normalized_df
    .select(
      "page_id",
      "pri_score",
      "rank",
      "cheirank_score",
      "cheirank_rank",
      "juice_potential_score",
      "out_degree",
      "in_degree",
      "is_low_power",
      "can_give_juice",
    )
    .sort("page_id")
  )


def _build_page_segments_df(
  pages_df: pl.DataFrame,
) -> pl.DataFrame:
  rows: list[dict[str, Any]] = []
  for row in pages_df.iter_rows(named=True):
    page_id = int(row["page_id"])
    depth = int(row["depth"])
    raw_path = str(row["path"])
    normalized_path = _normalize_path_for_segments(raw_path)
    segments = _path_segments(normalized_path)

    rows.append(
      {
        "page_id": page_id,
        "level": 0,
        "segment": "root",
        "segment_path": "/",
        "parent_segment_path": None,
        "depth": depth,
        "is_terminal": len(segments) == 0,
      },
    )

    for index, segment in enumerate(segments, start=1):
      segment_path = "/" + "/".join(segments[:index])
      parent_segment_path = "/" if index == 1 else "/" + "/".join(segments[:index - 1])
      rows.append(
        {
          "page_id": page_id,
          "level": index,
          "segment": segment,
          "segment_path": segment_path,
          "parent_segment_path": parent_segment_path,
          "depth": depth,
          "is_terminal": index == len(segments),
        },
      )

  if not rows:
    return pl.DataFrame(
      schema={
        "page_id": pl.Int32,
        "level": pl.Int8,
        "segment": pl.Utf8,
        "segment_path": pl.Utf8,
        "parent_segment_path": pl.Utf8,
        "depth": pl.Int16,
        "is_terminal": pl.Boolean,
      },
    )

  return (
    pl.DataFrame(rows)
    .with_columns(
      pl.col("page_id").cast(pl.Int32),
      pl.col("level").cast(pl.Int8),
      pl.col("segment").cast(pl.Utf8),
      pl.col("segment_path").cast(pl.Utf8),
      pl.col("parent_segment_path").cast(pl.Utf8),
      pl.col("depth").cast(pl.Int16),
      pl.col("is_terminal").cast(pl.Boolean),
    )
    .sort(["page_id", "level"])
  )


def _normalize_path_for_segments(path: str) -> str:
  cleaned = path.strip().split("#", 1)[0].split("?", 1)[0]
  if cleaned == "":
    return "/"
  if not cleaned.startswith("/"):
    cleaned = f"/{cleaned}"
  while "//" in cleaned:
    cleaned = cleaned.replace("//", "/")
  if cleaned != "/" and cleaned.endswith("/"):
    cleaned = cleaned.rstrip("/")
  return cleaned if cleaned else "/"


def _scenario_selector_expr() -> pl.Expr:
  return (
    (
      pl.concat_str(
        [
          pl.col("source_id").cast(pl.Utf8),
          pl.lit(":"),
          pl.col("target_id").cast(pl.Utf8),
          pl.lit(":"),
          pl.col("block_type"),
        ],
      )
      .hash(seed=42)
      % 10_000
    )
    / 10_000
  )


def _anchor_type_selection_expr(distribution: dict[str, float]) -> pl.Expr:
  exact_threshold = float(distribution["exact"])
  partial_threshold = exact_threshold + float(distribution["partial"])
  contextual_threshold = partial_threshold + float(distribution["contextual"])
  return (
    pl.when(pl.col("anchor_selector") < exact_threshold)
    .then(pl.lit("exact"))
    .when(pl.col("anchor_selector") < partial_threshold)
    .then(pl.lit("partial"))
    .when(pl.col("anchor_selector") < contextual_threshold)
    .then(pl.lit("contextual"))
    .otherwise(pl.lit("generic"))
  )


def _normalize_key(key: str) -> str:
  if key == "_meta":
    return key
  if key.endswith("/"):
    return key
  return f"{key}/"


def _path_segments(path: str) -> list[str]:
  return [segment for segment in path.strip("/").split("/") if segment]


def _empty_edges_lazyframe(include_priority: bool) -> pl.LazyFrame:
  schema = EDGE_INTERNAL_SCHEMA if include_priority else EDGE_SCHEMA
  return pl.DataFrame(schema=schema).lazy()


def _parquet_row_count(path: Path) -> int:
  return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _parquet_to_csv(input_path: Path, output_path: Path) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  pl.scan_parquet(input_path).sink_csv(output_path)


def _build_export_summary_payload(
  workspace: Path,
  output_dir: Path,
  exported_files: list[dict[str, object]],
  missing_optional_artifacts: list[str],
) -> dict[str, object]:
  pages_path = workspace / "pages.parquet"
  edges_path = workspace / "edges.parquet"
  pri_path = workspace / "pri_scores.parquet"
  anchors_path = workspace / "anchor_candidates.parquet"
  segment_metrics_path = workspace / "segment_metrics.parquet"
  run_metrics_path = workspace / "run_metrics.json"

  kpis: dict[str, object] = {}

  if pages_path.exists():
    kpis["page_count"] = _parquet_row_count(pages_path)
  if edges_path.exists():
    kpis["edge_count"] = _parquet_row_count(edges_path)
    block_mix_df = (
      pl.scan_parquet(edges_path)
      .group_by("block_type")
      .len()
      .sort("len", descending=True)
      .head(15)
      .collect()
    )
    kpis["block_type_mix"] = [
      {"block_type": row["block_type"], "count": int(row["len"])}
      for row in block_mix_df.to_dicts()
    ]

  top_pri_pages: list[dict[str, object]] = []
  if pri_path.exists():
    pri_scan = pl.scan_parquet(pri_path)
    pri_schema = set(pri_scan.collect_schema().names())
    pri_exprs = [
      pl.sum("pri_score").cast(pl.Float64).alias("pri_sum"),
      pl.len().cast(pl.Int32).alias("node_count"),
    ]
    if "cheirank_score" in pri_schema:
      pri_exprs.append(pl.sum("cheirank_score").cast(pl.Float64).alias("cheirank_sum"))
    if "can_give_juice" in pri_schema:
      pri_exprs.append(pl.col("can_give_juice").cast(pl.Int32).sum().cast(pl.Int32).alias("donor_count"))
    if "is_low_power" in pri_schema:
      pri_exprs.append(pl.col("is_low_power").cast(pl.Int32).sum().cast(pl.Int32).alias("low_power_count"))
    pri_stats = pri_scan.select(pri_exprs).collect().row(0, named=True)
    kpis.update({key: value for key, value in pri_stats.items() if value is not None})

    top_columns = ["page_id", "pri_score"]
    for optional_column in ["rank", "cheirank_score", "cheirank_rank", "can_give_juice", "is_low_power"]:
      if optional_column in pri_schema:
        top_columns.append(optional_column)
    top_pri_df = pri_scan.select(top_columns).sort("pri_score", descending=True).head(20).collect()
    if pages_path.exists():
      top_pri_df = top_pri_df.join(
        pl.scan_parquet(pages_path).select("page_id", "path").collect(),
        on="page_id",
        how="left",
      )
      ordered_columns = ["page_id", "path", *[column for column in top_pri_df.columns if column not in {"page_id", "path"}]]
      top_pri_df = top_pri_df.select(ordered_columns)
    top_pri_pages = top_pri_df.to_dicts()

  anchor_mix: list[dict[str, object]] = []
  if anchors_path.exists():
    kpis["anchor_candidate_count"] = _parquet_row_count(anchors_path)
    anchor_mix_df = (
      pl.scan_parquet(anchors_path)
      .group_by("anchor_type")
      .len()
      .sort("len", descending=True)
      .head(15)
      .collect()
    )
    anchor_mix = [
      {"anchor_type": row["anchor_type"], "count": int(row["len"])}
      for row in anchor_mix_df.to_dicts()
    ]
    kpis["anchor_type_mix"] = anchor_mix

  top_segments: list[dict[str, object]] = []
  if segment_metrics_path.exists():
    top_segments_df = (
      pl.scan_parquet(segment_metrics_path)
      .filter(pl.col("level") == 1)
      .sort("page_count", descending=True)
      .head(20)
      .collect()
    )
    top_segments = top_segments_df.to_dicts()

  run_metrics: dict[str, object] | None = None
  if run_metrics_path.exists():
    with run_metrics_path.open("r", encoding="utf-8") as file:
      run_metrics = json.load(file)

  return {
    "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    "workspace": str(workspace),
    "output_dir": str(output_dir),
    "exported_csv_count": len(exported_files),
    "exported_csv_files": exported_files,
    "missing_optional_artifacts": missing_optional_artifacts,
    "kpis": kpis,
    "top_pri_pages": top_pri_pages,
    "top_level_segments": top_segments,
    "run_metrics": run_metrics,
  }


def _build_export_summary_markdown(summary_payload: dict[str, Any]) -> str:
  lines = [
    "# Rapport de synthèse PRi Lab",
    "",
    f"- Généré le: `{summary_payload['generated_at']}`",
    f"- Workspace: `{summary_payload['workspace']}`",
    f"- Dossier export: `{summary_payload['output_dir']}`",
    f"- Fichiers CSV exportés: `{summary_payload['exported_csv_count']}`",
    "",
    "## Fichiers CSV",
    "",
    "| Artefact | Lignes | CSV |",
    "| --- | ---: | --- |",
  ]

  for exported_file in summary_payload["exported_csv_files"]:
    lines.append(
      f"| `{exported_file['name']}` | {exported_file['row_count']:,} | `{exported_file['output_csv']}` |",
    )

  if summary_payload["missing_optional_artifacts"]:
    lines.extend(
      [
        "",
        "## Artefacts optionnels absents",
        "",
      ],
    )
    for artifact_path in summary_payload["missing_optional_artifacts"]:
      lines.append(f"- `{artifact_path}`")

  lines.extend(
    [
      "",
      "## KPIs",
      "",
      "```json",
      json.dumps(summary_payload["kpis"], indent=2, ensure_ascii=False),
      "```",
      "",
      "## Top pages PRi",
      "",
      "```json",
      json.dumps(summary_payload["top_pri_pages"][:20], indent=2, ensure_ascii=False),
      "```",
      "",
      "## Top segments niveau 1",
      "",
      "```json",
      json.dumps(summary_payload["top_level_segments"][:20], indent=2, ensure_ascii=False),
      "```",
    ],
  )

  if summary_payload["run_metrics"] is not None:
    lines.extend(
      [
        "",
        "## Run metrics",
        "",
        "```json",
        json.dumps(summary_payload["run_metrics"], indent=2, ensure_ascii=False),
        "```",
      ],
    )

  return "\n".join(lines) + "\n"
