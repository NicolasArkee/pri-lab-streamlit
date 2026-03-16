from __future__ import annotations

import math

import polars as pl

from pri_lab.pipeline import BuildEdgesOptions, build_edges_lazyframe


def test_build_edges_respects_global_cap_and_is_deterministic() -> None:
  pages_df = pl.DataFrame(
    {
      "page_id": [1, 2, 3, 4, 5],
      "path": [
        "catalog/",
        "catalog/a/",
        "catalog/b/",
        "catalog/c/",
        "services/x/",
      ],
      "parent_path": [
        None,
        "catalog/",
        "catalog/",
        "catalog/",
        None,
      ],
      "cluster_thematique": [
        "catalog:__root__",
        "catalog:a",
        "catalog:b",
        "catalog:c",
        "services:x",
      ],
    },
    schema={
      "page_id": pl.Int32,
      "path": pl.Utf8,
      "parent_path": pl.Utf8,
      "cluster_thematique": pl.Utf8,
    },
  )

  options = BuildEdgesOptions(
    cluster_peer_k=3,
    max_out_links_per_page=2,
  )

  edges_a = build_edges_lazyframe(pages_df.lazy(), options).sort(["source_id", "target_id"]).collect()
  edges_b = build_edges_lazyframe(pages_df.lazy(), options).sort(["source_id", "target_id"]).collect()

  assert edges_a.to_dicts() == edges_b.to_dicts()
  max_out = edges_a.group_by("source_id").len().select(pl.max("len")).item()
  assert max_out <= 2


def test_dedup_aggregates_weights_and_keeps_high_priority_block_type() -> None:
  pages_df = pl.DataFrame(
    {
      "page_id": [1, 2],
      "path": ["catalog/", "catalog/a/"],
      "parent_path": [None, "catalog/"],
      "cluster_thematique": ["catalog:a", "catalog:a"],
    },
    schema={
      "page_id": pl.Int32,
      "path": pl.Utf8,
      "parent_path": pl.Utf8,
      "cluster_thematique": pl.Utf8,
    },
  )

  options = BuildEdgesOptions(
    enable_hierarchy_up=False,
    enable_hierarchy_down=True,
    enable_cluster_peer=True,
    cluster_peer_k=1,
    max_out_links_per_page=10,
    weight_hierarchy_down=0.7,
    weight_cluster_peer=0.4,
  )
  edges_df = build_edges_lazyframe(pages_df.lazy(), options).collect()

  duplicate_edge = edges_df.filter(
    (pl.col("source_id") == 1) & (pl.col("target_id") == 2),
  )
  assert duplicate_edge.height == 1
  assert duplicate_edge.get_column("block_type").item() == "hierarchy_down"
  assert math.isclose(duplicate_edge.get_column("rule_weight").item(), 1.1, rel_tol=1e-6, abs_tol=1e-6)


def test_cap_priority_keeps_hierarchy_up_before_cluster_peer() -> None:
  pages_df = pl.DataFrame(
    {
      "page_id": [1, 2, 3],
      "path": ["catalog/", "catalog/a/", "catalog/b/"],
      "parent_path": [None, "catalog/", "catalog/"],
      "cluster_thematique": ["catalog:root", "catalog:root", "catalog:root"],
    },
    schema={
      "page_id": pl.Int32,
      "path": pl.Utf8,
      "parent_path": pl.Utf8,
      "cluster_thematique": pl.Utf8,
    },
  )
  options = BuildEdgesOptions(
    enable_hierarchy_up=True,
    enable_hierarchy_down=False,
    enable_cluster_peer=True,
    cluster_peer_k=1,
    max_out_links_per_page=1,
  )

  edges_df = build_edges_lazyframe(pages_df.lazy(), options).collect()
  source_two = edges_df.filter(pl.col("source_id") == 2).to_dicts()

  assert source_two == [
    {
      "source_id": 2,
      "target_id": 1,
      "block_type": "hierarchy_up",
      "rule_weight": 1.0,
    },
  ]
