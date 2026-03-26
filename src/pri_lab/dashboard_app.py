from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import polars as pl
import streamlit as st

from pri_lab.dashboard import default_workspace_path
from pri_lab.lbc_generator import CATEGORY_TO_VERTICALE
from pri_lab.pipeline import prepare_dashboard_data


URL_METRICS_REQUIRED_COLUMNS = {
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
}

PAGE_SEGMENTS_REQUIRED_COLUMNS = {
  "page_id",
  "level",
  "segment",
  "segment_path",
  "parent_segment_path",
  "depth",
  "is_terminal",
}

SEGMENT_METRICS_REQUIRED_COLUMNS = {
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
}

MAX_SCATTER_POINTS = 50_000


def main() -> None:
  st.set_page_config(
    page_title="PRi Lab Dashboard",
    page_icon="📊",
    layout="wide",
  )
  workspace = _resolve_workspace_from_args()

  st.title("PRi Lab Visual Dashboard v2")
  st.caption("Analyse granulaire du maillage interne, segments URL, ancres, PRi et CheiRank.")

  # Workspace selector: auto-discover available workspaces
  artifacts_root = Path(__file__).resolve().parents[2] / "artifacts"
  available_workspaces = sorted([d.name for d in artifacts_root.iterdir() if d.is_dir() and (d / "pages.parquet").exists()]) if artifacts_root.exists() else []
  st.sidebar.header("Workspace")
  if available_workspaces and len(available_workspaces) > 1:
    chosen = st.sidebar.selectbox("Dataset", available_workspaces, index=available_workspaces.index(workspace.name) if workspace.name in available_workspaces else 0)
    workspace = (artifacts_root / chosen).resolve()
  st.sidebar.code(str(workspace))

  pages_path = workspace / "pages.parquet"
  edges_path = workspace / "edges.parquet"
  pri_path = workspace / "pri_scores.parquet"
  anchors_path = workspace / "anchor_candidates.parquet"
  metrics_path = workspace / "run_metrics.json"
  scenarios_dir = workspace / "scenarios"
  comparison_path = scenarios_dir / "scenario_pri_comparison.parquet"

  page_segments_path = workspace / "page_segments.parquet"
  url_metrics_path = workspace / "url_metrics.parquet"
  segment_metrics_path = workspace / "segment_metrics.parquet"

  required_paths = [pages_path, edges_path, pri_path]
  missing_required = [path for path in required_paths if not path.exists()]
  if missing_required:
    st.error("Artefacts de base manquants. Lance `pri-lab run-experiment` ou `pri-lab run-outlinks-analysis`.")
    st.code("\n".join(str(path) for path in missing_required))
    st.stop()

  pages_df = _load_parquet(pages_path)
  edges_df = _load_parquet(edges_path)
  pri_df = _load_parquet(pri_path)
  anchors_df = _load_parquet(anchors_path) if anchors_path.exists() else None
  metrics_payload = _load_json(metrics_path) if metrics_path.exists() else None

  dashboard_artifacts_ready = (
    page_segments_path.exists()
    and url_metrics_path.exists()
    and segment_metrics_path.exists()
  )
  if dashboard_artifacts_ready:
    page_segments_df = _load_parquet(page_segments_path)
    url_metrics_df = _load_parquet(url_metrics_path)
    segment_metrics_df = _load_parquet(segment_metrics_path)
    dashboard_artifacts_ready = (
      URL_METRICS_REQUIRED_COLUMNS.issubset(set(url_metrics_df.columns))
      and PAGE_SEGMENTS_REQUIRED_COLUMNS.issubset(set(page_segments_df.columns))
      and SEGMENT_METRICS_REQUIRED_COLUMNS.issubset(set(segment_metrics_df.columns))
    )
    if dashboard_artifacts_ready:
      url_metrics_df = _enrich_with_template_verticale(url_metrics_df, pages_df)
  else:
    page_segments_df = None
    url_metrics_df = None
    segment_metrics_df = None

  if not dashboard_artifacts_ready:
    st.warning(
      "Datasets dashboard manquants/obsolètes (`page_segments.parquet`, `url_metrics.parquet`, `segment_metrics.parquet`).",
    )
    _render_dashboard_data_cta(
      workspace=workspace,
      pages_path=pages_path,
      edges_path=edges_path,
      pri_path=pri_path,
      anchors_path=anchors_path if anchors_path.exists() else None,
      page_segments_path=page_segments_path,
      url_metrics_path=url_metrics_path,
      segment_metrics_path=segment_metrics_path,
    )
    url_metrics_df = _build_light_url_metrics(
      pages_df=pages_df,
      edges_df=edges_df,
      pri_df=pri_df,
      anchors_df=anchors_df,
    )
    url_metrics_df = _enrich_with_template_verticale(url_metrics_df, pages_df)
    page_segments_df = None
    segment_metrics_df = None

  all_block_types = sorted(edges_df.get_column("block_type").unique().to_list())
  all_anchor_types = (
    sorted(anchors_df.get_column("anchor_type").unique().to_list())
    if anchors_df is not None and "anchor_type" in anchors_df.columns
    else []
  )
  filters = _render_sidebar_filters(
    url_metrics_df=url_metrics_df,
    segment_metrics_df=segment_metrics_df,
    all_block_types=all_block_types,
    all_anchor_types=all_anchor_types,
  )

  filtered_url_df = _apply_global_filters(
    url_metrics_df=url_metrics_df,
    page_segments_df=page_segments_df,
    edges_df=edges_df,
    anchors_df=anchors_df,
    filters=filters,
    all_block_types=all_block_types,
    all_anchor_types=all_anchor_types,
  )
  filtered_page_ids_df = filtered_url_df.select("page_id").unique()
  filtered_edges_df = _filter_edges(
    edges_df=edges_df,
    filtered_page_ids_df=filtered_page_ids_df,
    selected_block_types=filters["block_types"],
    all_block_types=all_block_types,
  )
  filtered_anchors_df = _filter_anchors(
    anchors_df=anchors_df,
    filtered_page_ids_df=filtered_page_ids_df,
    selected_block_types=filters["block_types"],
    selected_anchor_types=filters["anchor_types"],
    all_block_types=all_block_types,
    all_anchor_types=all_anchor_types,
  )

  selected_page_id = _sync_selected_page_id(filtered_url_df)
  _render_export_sidebar(
    workspace=workspace,
    filtered_url_df=filtered_url_df,
    filtered_edges_df=filtered_edges_df,
    filtered_anchors_df=filtered_anchors_df,
    selected_page_id=selected_page_id,
  )

  # Load R&D audit data for new tabs
  # data/ lives at project root (pri-lab-streamlit/data/)
  data_dir = Path(__file__).resolve().parent.parent.parent / "data"
  audit_templates_data = _load_json(data_dir / "audit_templates_diff.json") if (data_dir / "audit_templates_diff.json").exists() else None
  audit_maillage_data = _load_json(data_dir / "audit_maillage_interne.json") if (data_dir / "audit_maillage_interne.json").exists() else None
  maillage_model_data = _load_json(data_dir / "maillage_model.json") if (data_dir / "maillage_model.json").exists() else None

  tabs = st.tabs([
    "Vue globale",
    "Opportunités",
    "Verticales",
    "Maillage Section",
    "Audit Technique",
    "Segments URL",
    "URL Explorer",
    "Ancres",
    "CheiRank & jus",
    "Scénarios",
  ])

  with tabs[0]:
    _render_overview_tab(
      url_metrics_df=url_metrics_df,
      filtered_url_df=filtered_url_df,
      filtered_edges_df=filtered_edges_df,
      metrics_payload=metrics_payload,
    )

  with tabs[1]:
    _render_opportunities_tab(filtered_url_df=filtered_url_df, filtered_edges_df=filtered_edges_df)

  with tabs[2]:
    _render_verticales_tab(filtered_url_df=filtered_url_df, filtered_edges_df=filtered_edges_df)

  with tabs[3]:
    _render_maillage_section_tab(
      audit_maillage_data=audit_maillage_data,
      maillage_model_data=maillage_model_data,
      filtered_edges_df=filtered_edges_df,
      filtered_url_df=filtered_url_df,
    )

  with tabs[4]:
    _render_audit_technique_tab(
      audit_templates_data=audit_templates_data,
      audit_maillage_data=audit_maillage_data,
    )

  with tabs[5]:
    _render_segments_tab(
      segment_metrics_df=segment_metrics_df,
      filters=filters,
    )

  with tabs[6]:
    _render_url_explorer_tab(
      filtered_url_df=filtered_url_df,
      pages_df=pages_df,
      edges_path=edges_path,
      anchors_path=anchors_path if anchors_path.exists() else None,
      selected_page_id=selected_page_id,
      selected_block_types=filters["block_types"],
      selected_anchor_types=filters["anchor_types"],
      all_block_types=all_block_types,
      all_anchor_types=all_anchor_types,
    )

  with tabs[7]:
    _render_anchors_tab(
      filtered_anchors_df=filtered_anchors_df,
      selected_page_id=selected_page_id,
      selected_block_types=filters["block_types"],
      selected_anchor_types=filters["anchor_types"],
      all_block_types=all_block_types,
      all_anchor_types=all_anchor_types,
    )

  with tabs[8]:
    _render_cheirank_tab(
      filtered_url_df=filtered_url_df,
      selected_page_id=selected_page_id,
    )

  with tabs[9]:
    _render_scenarios_tab(
      scenarios_dir=scenarios_dir,
      comparison_path=comparison_path,
      pages_df=pages_df,
      filtered_page_ids_df=filtered_page_ids_df,
      selected_page_id=selected_page_id,
    )


@st.cache_data(show_spinner=False)
def _load_parquet(path: Path) -> pl.DataFrame:
  return pl.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as file:
    return json.load(file)


def _resolve_workspace_from_args() -> Path:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--workspace", type=str, default=None)
  args, _ = parser.parse_known_args(sys.argv[1:])
  if args.workspace:
    return Path(args.workspace).expanduser().resolve()

  workspace_from_env = os.getenv("PRI_LAB_WORKSPACE")
  if workspace_from_env:
    return Path(workspace_from_env).expanduser().resolve()
  return default_workspace_path()


def _render_dashboard_data_cta(
  workspace: Path,
  pages_path: Path,
  edges_path: Path,
  pri_path: Path,
  anchors_path: Path | None,
  page_segments_path: Path,
  url_metrics_path: Path,
  segment_metrics_path: Path,
) -> None:
  st.info(
    "Clique sur le bouton pour générer les datasets du dashboard, ou lance en CLI:\n"
    f"`pri-lab prepare-dashboard-data --workspace {workspace}`",
  )
  if st.button("Générer les datasets dashboard", type="primary"):
    with st.spinner("Génération des datasets dashboard en cours..."):
      prepare_dashboard_data(
        input_pages_path=pages_path,
        input_edges_path=edges_path,
        input_pri_path=pri_path,
        output_page_segments_path=page_segments_path,
        output_url_metrics_path=url_metrics_path,
        output_segment_metrics_path=segment_metrics_path,
        input_anchor_candidates_path=anchors_path,
      )
    st.success("Datasets dashboard générés.")
    st.rerun()


def _build_light_url_metrics(
  pages_df: pl.DataFrame,
  edges_df: pl.DataFrame,
  pri_df: pl.DataFrame,
  anchors_df: pl.DataFrame | None,
) -> pl.DataFrame:
  normalized_pri_df = _ensure_pri_columns(pri_df)
  out_stats_df = (
    edges_df
    .group_by("source_id")
    .agg(
      pl.len().cast(pl.Int32).alias("outgoing_links"),
      pl.sum("rule_weight").cast(pl.Float64).alias("outgoing_weight"),
      pl.col("target_id").n_unique().cast(pl.Int32).alias("unique_out_targets"),
    )
    .rename({"source_id": "page_id"})
  )
  in_stats_df = (
    edges_df
    .group_by("target_id")
    .agg(
      pl.len().cast(pl.Int32).alias("incoming_links"),
      pl.sum("rule_weight").cast(pl.Float64).alias("incoming_weight"),
      pl.col("source_id").n_unique().cast(pl.Int32).alias("unique_in_sources"),
    )
    .rename({"target_id": "page_id"})
  )
  if anchors_df is None:
    anchor_stats_df = pl.DataFrame(
      schema={
        "page_id": pl.Int32,
        "unique_anchor_texts_out": pl.Int32,
        "unique_anchor_types_out": pl.Int32,
      },
    )
  else:
    anchor_stats_df = (
      anchors_df
      .group_by("source_id")
      .agg(
        pl.col("anchor_text").n_unique().cast(pl.Int32).alias("unique_anchor_texts_out"),
        pl.col("anchor_type").n_unique().cast(pl.Int32).alias("unique_anchor_types_out"),
      )
      .rename({"source_id": "page_id"})
    )

  return (
    pages_df
    .select(
      pl.col("page_id").cast(pl.Int32),
      pl.col("path").cast(pl.Utf8),
      pl.col("depth").cast(pl.Int16),
    )
    .join(normalized_pri_df, on="page_id", how="left")
    .join(out_stats_df, on="page_id", how="left")
    .join(in_stats_df, on="page_id", how="left")
    .join(anchor_stats_df, on="page_id", how="left")
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
    .sort("pri_score", descending=True)
  )


def _enrich_with_template_verticale(
  url_metrics_df: pl.DataFrame,
  pages_df: pl.DataFrame,
) -> pl.DataFrame:
  """Join template, category, verticale from pages into url_metrics."""
  if "template" not in pages_df.columns or "cluster_thematique" not in pages_df.columns:
    return url_metrics_df
  enrichment_df = (
    pages_df
    .select("page_id", "template", "cluster_thematique")
    .with_columns(
      pl.col("cluster_thematique").str.extract(r":(.+)$", 1).alias("category"),
    )
    .with_columns(
      pl.col("category").replace_strict(CATEGORY_TO_VERTICALE, default="_autre_").alias("verticale"),
    )
    .select(
      pl.col("page_id").cast(pl.Int32),
      "template",
      "category",
      "verticale",
    )
  )
  return url_metrics_df.join(enrichment_df, on="page_id", how="left")


def _ensure_pri_columns(pri_df: pl.DataFrame) -> pl.DataFrame:
  schema_columns = set(pri_df.columns)
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

  return normalized_df.select(
    "page_id",
    "pri_score",
    "rank",
    "cheirank_score",
    "cheirank_rank",
    "out_degree",
    "in_degree",
    "can_give_juice",
    "is_low_power",
  )


def _render_sidebar_filters(
  url_metrics_df: pl.DataFrame,
  segment_metrics_df: pl.DataFrame | None,
  all_block_types: list[str],
  all_anchor_types: list[str],
) -> dict[str, Any]:
  st.sidebar.header("Filtres globaux")

  min_depth = int(url_metrics_df.select(pl.min("depth")).item()) if url_metrics_df.height > 0 else 0
  max_depth = int(url_metrics_df.select(pl.max("depth")).item()) if url_metrics_df.height > 0 else 0
  depth_range = st.sidebar.slider(
    "Profondeur URL",
    min_value=min_depth,
    max_value=max_depth,
    value=(min_depth, max_depth),
    step=1,
  )

  if segment_metrics_df is not None and segment_metrics_df.height > 0:
    levels = sorted(segment_metrics_df.get_column("level").unique().to_list())
    if st.session_state.get("segment_level_filter") not in levels:
      st.session_state["segment_level_filter"] = levels[0]
    segment_level = st.sidebar.selectbox(
      "Niveau segment",
      options=levels,
      key="segment_level_filter",
    )
    level_segments_df = (
      segment_metrics_df
      .filter(pl.col("level") == segment_level)
      .sort("page_count", descending=True)
      .select("segment_path")
    )
    segment_options = level_segments_df.get_column("segment_path").to_list()
    segment_select_options = ["(all)", *segment_options]
    if st.session_state.get("segment_path_filter") not in segment_select_options:
      st.session_state["segment_path_filter"] = "(all)"
    segment_selected = st.sidebar.selectbox(
      "Segment URL",
      options=segment_select_options,
      key="segment_path_filter",
    )
    segment_path = segment_selected if segment_selected != "(all)" else None
  else:
    st.sidebar.caption("Segmentation indisponible tant que `prepare-dashboard-data` n'a pas été exécuté.")
    segment_level = 0
    segment_path = None

  block_types = st.sidebar.multiselect(
    "Block types",
    options=all_block_types,
    default=all_block_types,
  )
  anchor_types = st.sidebar.multiselect(
    "Anchor types",
    options=all_anchor_types,
    default=all_anchor_types,
    disabled=not all_anchor_types,
  )
  url_search = st.sidebar.text_input("Recherche URL (contains)").strip()

  templates: list[str] = []
  verticales: list[str] = []
  if "template" in url_metrics_df.columns:
    all_templates = sorted(url_metrics_df.get_column("template").drop_nulls().unique().to_list())
    templates = st.sidebar.multiselect("Templates", options=all_templates, default=all_templates)
  if "verticale" in url_metrics_df.columns:
    all_verticales = sorted(url_metrics_df.get_column("verticale").drop_nulls().unique().to_list())
    verticales = st.sidebar.multiselect("Verticales", options=all_verticales, default=all_verticales)

  return {
    "depth_min": depth_range[0],
    "depth_max": depth_range[1],
    "segment_level": segment_level,
    "segment_path": segment_path,
    "block_types": block_types,
    "anchor_types": anchor_types,
    "url_search": url_search,
    "templates": templates,
    "verticales": verticales,
  }


def _apply_global_filters(
  url_metrics_df: pl.DataFrame,
  page_segments_df: pl.DataFrame | None,
  edges_df: pl.DataFrame,
  anchors_df: pl.DataFrame | None,
  filters: dict[str, Any],
  all_block_types: list[str],
  all_anchor_types: list[str],
) -> pl.DataFrame:
  filtered_df = url_metrics_df

  if page_segments_df is not None and filters["segment_path"] is not None:
    page_ids_df = (
      page_segments_df
      .filter(
        (pl.col("level") == filters["segment_level"])
        & (pl.col("segment_path") == filters["segment_path"]),
      )
      .select("page_id")
      .unique()
    )
    filtered_df = filtered_df.join(page_ids_df, on="page_id", how="inner")

  filtered_df = filtered_df.filter(
    (pl.col("depth") >= filters["depth_min"])
    & (pl.col("depth") <= filters["depth_max"]),
  )

  if filters["url_search"]:
    filtered_df = filtered_df.filter(
      pl.col("path").str.to_lowercase().str.contains(filters["url_search"].lower(), literal=True),
    )

  if "template" in filtered_df.columns and filters.get("templates"):
    all_tpl = filtered_df.get_column("template").drop_nulls().unique().to_list()
    if len(filters["templates"]) < len(all_tpl):
      filtered_df = filtered_df.filter(pl.col("template").is_in(filters["templates"]))

  if "verticale" in filtered_df.columns and filters.get("verticales"):
    all_vert = filtered_df.get_column("verticale").drop_nulls().unique().to_list()
    if len(filters["verticales"]) < len(all_vert):
      filtered_df = filtered_df.filter(pl.col("verticale").is_in(filters["verticales"]))

  if all_block_types:
    if not filters["block_types"]:
      return filtered_df.head(0)
    if len(filters["block_types"]) < len(all_block_types):
      allowed_source_df = (
        edges_df
        .filter(pl.col("block_type").is_in(filters["block_types"]))
        .select(pl.col("source_id").alias("page_id"))
        .unique()
      )
      filtered_df = filtered_df.join(allowed_source_df, on="page_id", how="inner")

  if anchors_df is not None and all_anchor_types:
    if not filters["anchor_types"]:
      return filtered_df.head(0)
    if len(filters["anchor_types"]) < len(all_anchor_types):
      allowed_anchor_source_df = (
        anchors_df
        .filter(pl.col("anchor_type").is_in(filters["anchor_types"]))
        .select(pl.col("source_id").alias("page_id"))
        .unique()
      )
      filtered_df = filtered_df.join(allowed_anchor_source_df, on="page_id", how="inner")

  return filtered_df.sort("pri_score", descending=True)


def _filter_edges(
  edges_df: pl.DataFrame,
  filtered_page_ids_df: pl.DataFrame,
  selected_block_types: list[str],
  all_block_types: list[str],
) -> pl.DataFrame:
  if filtered_page_ids_df.height == 0:
    return edges_df.head(0)

  filtered_edges_df = edges_df
  if all_block_types and len(selected_block_types) < len(all_block_types):
    filtered_edges_df = filtered_edges_df.filter(pl.col("block_type").is_in(selected_block_types))
  return filtered_edges_df.join(
    filtered_page_ids_df.select(pl.col("page_id").alias("source_id")),
    on="source_id",
    how="inner",
  )


def _filter_anchors(
  anchors_df: pl.DataFrame | None,
  filtered_page_ids_df: pl.DataFrame,
  selected_block_types: list[str],
  selected_anchor_types: list[str],
  all_block_types: list[str],
  all_anchor_types: list[str],
) -> pl.DataFrame | None:
  if anchors_df is None or filtered_page_ids_df.height == 0:
    return None

  filtered_df = anchors_df
  if all_block_types and len(selected_block_types) < len(all_block_types):
    filtered_df = filtered_df.filter(pl.col("block_type").is_in(selected_block_types))
  if all_anchor_types and len(selected_anchor_types) < len(all_anchor_types):
    filtered_df = filtered_df.filter(pl.col("anchor_type").is_in(selected_anchor_types))
  return filtered_df.join(
    filtered_page_ids_df.select(pl.col("page_id").alias("source_id")),
    on="source_id",
    how="inner",
  )


def _sync_selected_page_id(filtered_url_df: pl.DataFrame) -> int | None:
  selected_page_id = st.session_state.get("selected_page_id")
  if selected_page_id is not None:
    in_filtered = filtered_url_df.filter(pl.col("page_id") == int(selected_page_id)).height > 0
    if in_filtered:
      return int(selected_page_id)

  if filtered_url_df.height == 0:
    st.session_state["selected_page_id"] = None
    return None

  new_selected_page_id = int(filtered_url_df.get_column("page_id").item(0))
  st.session_state["selected_page_id"] = new_selected_page_id
  return new_selected_page_id


def _render_export_sidebar(
  workspace: Path,
  filtered_url_df: pl.DataFrame,
  filtered_edges_df: pl.DataFrame,
  filtered_anchors_df: pl.DataFrame | None,
  selected_page_id: int | None,
) -> None:
  st.sidebar.header("Exports")
  st.sidebar.caption("Exports des données filtrées (CSV) et rapport de synthèse local.")

  st.sidebar.download_button(
    "Exporter URL metrics (CSV)",
    data=_to_csv_bytes(filtered_url_df),
    file_name="url_metrics_filtered.csv",
    mime="text/csv",
    use_container_width=True,
  )
  st.sidebar.download_button(
    "Exporter edges filtrées (CSV)",
    data=_to_csv_bytes(filtered_edges_df),
    file_name="edges_filtered.csv",
    mime="text/csv",
    use_container_width=True,
  )
  if filtered_anchors_df is not None:
    st.sidebar.download_button(
      "Exporter ancres filtrées (CSV)",
      data=_to_csv_bytes(filtered_anchors_df),
      file_name="anchors_filtered.csv",
      mime="text/csv",
      use_container_width=True,
    )

  report_markdown = _build_filtered_summary_markdown(
    workspace=workspace,
    filtered_url_df=filtered_url_df,
    filtered_edges_df=filtered_edges_df,
    filtered_anchors_df=filtered_anchors_df,
    selected_page_id=selected_page_id,
  )
  st.sidebar.download_button(
    "Exporter rapport synthèse (MD)",
    data=report_markdown.encode("utf-8"),
    file_name="dashboard_filtered_summary.md",
    mime="text/markdown",
    use_container_width=True,
  )


def _to_csv_bytes(frame: pl.DataFrame) -> bytes:
  return frame.write_csv().encode("utf-8")


def _build_filtered_summary_markdown(
  workspace: Path,
  filtered_url_df: pl.DataFrame,
  filtered_edges_df: pl.DataFrame,
  filtered_anchors_df: pl.DataFrame | None,
  selected_page_id: int | None,
) -> str:
  pri_sum = float(filtered_url_df.select(pl.sum("pri_score")).item()) if filtered_url_df.height > 0 else 0.0
  cheirank_sum = (
    float(filtered_url_df.select(pl.sum("cheirank_score")).item())
    if filtered_url_df.height > 0
    else 0.0
  )
  donor_count = int(filtered_url_df.filter(pl.col("can_give_juice")).height) if filtered_url_df.height > 0 else 0
  low_power_count = int(filtered_url_df.filter(pl.col("is_low_power")).height) if filtered_url_df.height > 0 else 0

  block_mix = (
    filtered_edges_df
    .group_by("block_type")
    .len()
    .sort("len", descending=True)
    .to_dicts()
    if filtered_edges_df.height > 0
    else []
  )
  anchor_mix = (
    filtered_anchors_df
    .group_by("anchor_type")
    .len()
    .sort("len", descending=True)
    .to_dicts()
    if filtered_anchors_df is not None and filtered_anchors_df.height > 0
    else []
  )
  top_pages = (
    filtered_url_df
    .sort("pri_score", descending=True)
    .head(20)
    .select("page_id", "path", "pri_score", "rank", "cheirank_score", "cheirank_rank")
    .to_dicts()
  )

  selected_page = None
  if selected_page_id is not None:
    selected_page_df = filtered_url_df.filter(pl.col("page_id") == int(selected_page_id)).head(1)
    if selected_page_df.height > 0:
      selected_page = selected_page_df.to_dicts()[0]

  lines = [
    "# Rapport de synthèse dashboard PRi Lab",
    "",
    f"- Workspace: `{workspace}`",
    f"- Pages filtrées: `{filtered_url_df.height:,}`",
    f"- Arêtes filtrées: `{filtered_edges_df.height:,}`",
    f"- Somme PRi filtrée: `{pri_sum:.6f}`",
    f"- Somme CheiRank filtrée: `{cheirank_sum:.6f}`",
    f"- Pages donneuses: `{donor_count:,}`",
    f"- Pages faible puissance: `{low_power_count:,}`",
    "",
    "## Mix block_type",
    "",
    "```json",
    json.dumps(block_mix, indent=2, ensure_ascii=False),
    "```",
  ]

  if filtered_anchors_df is not None:
    lines.extend(
      [
        "",
        "## Mix anchor_type",
        "",
        "```json",
        json.dumps(anchor_mix, indent=2, ensure_ascii=False),
        "```",
      ],
    )

  lines.extend(
    [
      "",
      "## Top pages PRi (filtré)",
      "",
      "```json",
      json.dumps(top_pages, indent=2, ensure_ascii=False),
      "```",
    ],
  )

  if selected_page is not None:
    lines.extend(
      [
        "",
        "## URL sélectionnée",
        "",
        "```json",
        json.dumps(selected_page, indent=2, ensure_ascii=False),
        "```",
      ],
    )

  return "\n".join(lines) + "\n"


def _render_overview_tab(
  url_metrics_df: pl.DataFrame,
  filtered_url_df: pl.DataFrame,
  filtered_edges_df: pl.DataFrame,
  metrics_payload: dict[str, Any] | None,
) -> None:
  pri_sum_filtered = float(filtered_url_df.select(pl.sum("pri_score")).item()) if filtered_url_df.height > 0 else 0.0
  avg_out_degree_filtered = (
    float(filtered_url_df.select(pl.mean("out_degree")).item())
    if filtered_url_df.height > 0
    else 0.0
  )

  columns = st.columns(6)
  columns[0].metric("Pages filtrées", f"{filtered_url_df.height:,}")
  columns[1].metric("Pages total", f"{url_metrics_df.height:,}")
  columns[2].metric("Arêtes filtrées", f"{filtered_edges_df.height:,}")
  columns[3].metric("Somme PRi filtrée", f"{pri_sum_filtered:.6f}")
  columns[4].metric("Out-degree moyen filtré", f"{avg_out_degree_filtered:.2f}")
  columns[5].metric("URLs donneuses (filtré)", f"{int(filtered_url_df.filter(pl.col('can_give_juice')).height):,}")

  if filtered_edges_df.height > 0:
    block_mix_df = filtered_edges_df.group_by("block_type").len().sort("len", descending=True)
    st.caption("Répartition des blocs de liens (périmètre filtré)")
    st.bar_chart(
      block_mix_df.rename({"len": "count"}).to_pandas().set_index("block_type"),
      use_container_width=True,
    )

  if "template" in filtered_url_df.columns and filtered_url_df.height > 0:
    template_stats_df = (
      filtered_url_df
      .group_by("template")
      .agg(
        pl.len().alias("pages"),
        pl.sum("pri_score").alias("pri_sum"),
        pl.mean("pri_score").alias("pri_mean"),
      )
      .sort("pri_sum", descending=True)
    )
    col_a, col_b = st.columns(2)
    with col_a:
      st.caption("PRi total par template")
      st.bar_chart(template_stats_df.to_pandas().set_index("template")[["pri_sum"]], use_container_width=True)
    with col_b:
      st.caption("Nombre de pages par template")
      st.bar_chart(template_stats_df.to_pandas().set_index("template")[["pages"]], use_container_width=True)

  top_n = st.slider("Top pages PRi", min_value=10, max_value=200, value=25, step=5)
  top_cols = ["rank", "page_id", "path", "depth", "pri_score", "cheirank_score", "in_degree", "out_degree", "can_give_juice", "is_low_power"]
  if "template" in filtered_url_df.columns:
    top_cols.insert(4, "template")
  top_df = (
    filtered_url_df
    .sort("pri_score", descending=True)
    .head(top_n)
    .select([c for c in top_cols if c in filtered_url_df.columns])
  )
  st.dataframe(top_df.to_pandas(), use_container_width=True, hide_index=True)

  _render_depth_analysis_section(filtered_url_df=filtered_url_df)

  # ── Pareto du trafic (si données réelles) ──
  real_df = _load_real_metrics(filtered_url_df)
  if real_df is not None:
    clicks_df = real_df.filter(pl.col("real_clicks").is_not_null() & (pl.col("real_clicks") > 0)).sort("real_clicks", descending=True)
    if clicks_df.height > 0:
      st.subheader("Pareto du trafic (données GSC réelles)")
      total_clicks = float(clicks_df.select(pl.sum("real_clicks")).item())
      cumulative = clicks_df.with_columns(
        (pl.col("real_clicks").cum_sum() / total_clicks * 100).alias("cumul_clicks_pct"),
        (pl.int_range(1, pl.len() + 1) / clicks_df.height * 100).alias("pages_pct"),
      ).select("pages_pct", "cumul_clicks_pct")
      try:
        import plotly.express as px
        fig = px.line(cumulative.to_pandas(), x="pages_pct", y="cumul_clicks_pct",
                      labels={"pages_pct": "% des pages (triées par clicks)", "cumul_clicks_pct": "% cumulé des clicks"})
        fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.5, annotation_text="50% clicks")
        fig.add_hline(y=80, line_dash="dash", line_color="orange", opacity=0.5, annotation_text="80% clicks")
        fig.update_layout(height=350, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
      except ImportError:
        pass
      # Key stats
      pages_50pct = cumulative.filter(pl.col("cumul_clicks_pct") >= 50).head(1)
      pages_80pct = cumulative.filter(pl.col("cumul_clicks_pct") >= 80).head(1)
      p_cols = st.columns(3)
      p_cols[0].metric("Pages avec clicks", f"{clicks_df.height:,}")
      if pages_50pct.height > 0:
        p_cols[1].metric("50% du trafic", f"Top {pages_50pct['pages_pct'].item():.1f}% des pages")
      if pages_80pct.height > 0:
        p_cols[2].metric("80% du trafic", f"Top {pages_80pct['pages_pct'].item():.1f}% des pages")

  if metrics_payload:
    st.caption("Dernier run pipeline")
    st.json(
      {
        "started_at": metrics_payload.get("started_at"),
        "finished_at": metrics_payload.get("finished_at"),
        "duration_seconds": metrics_payload.get("duration_seconds"),
        "peak_memory_mb": metrics_payload.get("peak_memory_mb"),
      },
      expanded=False,
    )


def _render_depth_analysis_section(
  filtered_url_df: pl.DataFrame,
) -> None:
  if filtered_url_df.height == 0:
    st.info("Aucune donnée de profondeur sur le périmètre filtré.")
    return

  min_depth = int(filtered_url_df.select(pl.min("depth")).item())
  max_depth = int(filtered_url_df.select(pl.max("depth")).item())
  median_depth = float(filtered_url_df.select(pl.median("depth")).item())
  p90_depth = float(filtered_url_df.select(pl.col("depth").quantile(0.90, interpolation="nearest")).item())
  deep_page_count = int(filtered_url_df.filter(pl.col("depth") >= 3).height)

  st.subheader("État des lieux de la profondeur")
  kpis = st.columns(5)
  kpis[0].metric("Profondeur min", str(min_depth))
  kpis[1].metric("Profondeur max", str(max_depth))
  kpis[2].metric("Profondeur médiane", f"{median_depth:.1f}")
  kpis[3].metric("Profondeur P90", f"{p90_depth:.1f}")
  kpis[4].metric("Pages profondeur ≥ 3", f"{deep_page_count:,}")

  depth_stats_df = (
    filtered_url_df
    .group_by("depth")
    .agg(
      pl.len().cast(pl.Int32).alias("page_count"),
      pl.sum("pri_score").cast(pl.Float64).alias("pri_sum"),
      pl.mean("pri_score").cast(pl.Float64).alias("pri_mean"),
      pl.col("pri_score").quantile(0.10, interpolation="nearest").cast(pl.Float64).alias("pri_p10"),
      pl.col("pri_score").quantile(0.50, interpolation="nearest").cast(pl.Float64).alias("pri_p50"),
      pl.col("pri_score").quantile(0.90, interpolation="nearest").cast(pl.Float64).alias("pri_p90"),
      pl.col("can_give_juice").cast(pl.Float64).mean().cast(pl.Float64).alias("donor_ratio"),
      pl.col("is_low_power").cast(pl.Float64).mean().cast(pl.Float64).alias("low_power_ratio"),
    )
    .sort("depth")
  )

  pri_total = float(depth_stats_df.select(pl.sum("pri_sum")).item())
  depth_stats_df = depth_stats_df.with_columns(
    (
      pl.col("pri_sum")
      / (pl.lit(pri_total) if pri_total > 0 else pl.lit(1.0))
    ).cast(pl.Float64).alias("pri_share"),
  )

  st.caption("Distribution PRi par profondeur (P10 / médiane / P90)")
  pri_distribution_df = depth_stats_df.select("depth", "pri_p10", "pri_p50", "pri_p90", "pri_mean")
  st.line_chart(
    pri_distribution_df.to_pandas().set_index("depth"),
    use_container_width=True,
  )

  st.caption("Part de PRi par profondeur")
  st.bar_chart(
    depth_stats_df.select("depth", "pri_share").to_pandas().set_index("depth"),
    use_container_width=True,
  )

  st.caption("Tableau profondeur (pages + PRi + ratios)")
  depth_overview_df = (
    depth_stats_df
    .with_columns(
      (pl.col("pri_share") * 100).cast(pl.Float64).alias("pri_share_pct"),
      (pl.col("donor_ratio") * 100).cast(pl.Float64).alias("donor_ratio_pct"),
      (pl.col("low_power_ratio") * 100).cast(pl.Float64).alias("low_power_ratio_pct"),
    )
    .select(
      "depth",
      "page_count",
      "pri_sum",
      "pri_share_pct",
      "pri_mean",
      "pri_p10",
      "pri_p50",
      "pri_p90",
      "donor_ratio_pct",
      "low_power_ratio_pct",
    )
    .sort("depth")
  )
  st.dataframe(
    depth_overview_df.to_pandas(),
    use_container_width=True,
    hide_index=True,
  )


def _render_opportunities_tab(
  filtered_url_df: pl.DataFrame,
  filtered_edges_df: pl.DataFrame,
) -> None:
  """Onglet Opportunités & Bottlenecks — le tab le plus actionnable."""
  if filtered_url_df.height == 0:
    st.warning("Aucune page dans le périmètre filtré.")
    return

  # ── Santé globale du maillage ──
  total = filtered_url_df.height
  orphan_count = int(filtered_url_df.filter(pl.col("in_degree") <= 2).height)
  low_power_count = int(filtered_url_df.filter(pl.col("is_low_power")).height)
  donor_count = int(filtered_url_df.filter(pl.col("can_give_juice")).height)
  median_out = float(filtered_url_df.select(pl.median("out_degree")).item()) if total > 0 else 0
  underused_donors = int(filtered_url_df.filter(pl.col("can_give_juice") & (pl.col("out_degree") < median_out)).height)

  # Score santé (0-100)
  orphan_pct = orphan_count / max(total, 1)
  low_power_pct = low_power_count / max(total, 1)
  health_score = max(0, min(100, int(100 * (1 - orphan_pct * 2 - low_power_pct))))

  cols = st.columns(6)
  cols[0].metric("Score santé maillage", f"{health_score}/100")
  cols[1].metric("Pages orphelines (in<=2)", f"{orphan_count:,}", delta=f"{orphan_pct*100:.1f}%", delta_color="inverse")
  cols[2].metric("Pages low power", f"{low_power_count:,}", delta=f"{low_power_pct*100:.1f}%", delta_color="inverse")
  cols[3].metric("Donneuses de jus", f"{donor_count:,}")
  cols[4].metric("Donneuses sous-utilisées", f"{underused_donors:,}", delta_color="inverse")
  cols[5].metric("Pages total", f"{total:,}")

  # ── Bottlenecks ──
  st.subheader("Bottlenecks — Pages qui bloquent le jus")
  st.caption("Pages avec beaucoup de liens entrants mais peu de sortants : le jus entre mais ne circule pas.")
  p75_in = float(filtered_url_df.select(pl.col("in_degree").quantile(0.75)).item()) if total > 0 else 0
  p25_out = float(filtered_url_df.select(pl.col("out_degree").quantile(0.25)).item()) if total > 0 else 0
  bottlenecks = (
    filtered_url_df
    .filter((pl.col("in_degree") > p75_in) & (pl.col("out_degree") < max(p25_out, 5)))
    .sort("in_degree", descending=True)
    .head(30)
  )
  display_cols = [c for c in ["path", "template", "category", "in_degree", "out_degree", "pri_score", "rank"] if c in bottlenecks.columns]
  if bottlenecks.height > 0:
    st.dataframe(bottlenecks.select(display_cols).to_pandas(), use_container_width=True, hide_index=True)
  else:
    st.success("Aucun bottleneck détecté.")

  # ── Pages orphelines ──
  st.subheader("Pages orphelines — Quasi-isolées du maillage")
  st.caption("Pages avec 2 liens entrants ou moins (hors homepage et /dc/).")
  exclude_tpl = ["homepage", "dc"]
  orphans = (
    filtered_url_df
    .filter((pl.col("in_degree") <= 2) & (~pl.col("template").is_in(exclude_tpl) if "template" in filtered_url_df.columns else pl.lit(True)))
    .sort("pri_score", descending=True)
  )
  if "category" in orphans.columns:
    orphan_by_cat = orphans.group_by("category").len().sort("len", descending=True).head(15)
    st.caption("Orphelines par catégorie (top 15)")
    st.bar_chart(orphan_by_cat.to_pandas().set_index("category").rename(columns={"len": "pages_orphelines"}), use_container_width=True)

  st.dataframe(orphans.head(30).select(display_cols).to_pandas(), use_container_width=True, hide_index=True)
  st.caption(f"Total orphelines : {orphans.height:,} pages ({orphans.height / max(total, 1) * 100:.1f}%)")

  # ── Donneuses sous-utilisées ──
  st.subheader("Donneuses sous-utilisées — Potentiel de jus non exploité")
  st.caption("Pages à fort PRi (can_give_juice) mais avec peu de liens sortants.")
  underused = (
    filtered_url_df
    .filter(pl.col("can_give_juice") & (pl.col("out_degree") < median_out))
    .sort("pri_score", descending=True)
    .head(30)
  )
  if underused.height > 0:
    st.dataframe(underused.select(display_cols).to_pandas(), use_container_width=True, hide_index=True)

  # ── Corrélation PRi vs clicks réels ──
  real_df = _load_real_metrics(filtered_url_df)
  if real_df is not None:
    st.subheader("Corrélation PRi modèle vs Clicks GSC réels")
    st.caption("Croisement du score PRi (maillage interne) avec les vrais clicks Google Search Console.")

    # Join on path
    corr_df = (
      filtered_url_df
      .select("page_id", "path", "pri_score", "in_degree", "template")
      .join(
        real_df.select("path", "real_clicks", "botify_pr").filter(pl.col("real_clicks").is_not_null()),
        on="path",
        how="inner",
      )
    )

    if corr_df.height > 0:
      # Quadrant metrics
      med_pri = float(corr_df.select(pl.median("pri_score")).item())
      med_clicks = float(corr_df.select(pl.median("real_clicks")).item())
      q_cols = st.columns(4)
      aligned = corr_df.filter((pl.col("pri_score") >= med_pri) & (pl.col("real_clicks") >= med_clicks)).height
      over_linked = corr_df.filter((pl.col("pri_score") >= med_pri) & (pl.col("real_clicks") < med_clicks)).height
      under_linked = corr_df.filter((pl.col("pri_score") < med_pri) & (pl.col("real_clicks") >= med_clicks)).height
      weak = corr_df.filter((pl.col("pri_score") < med_pri) & (pl.col("real_clicks") < med_clicks)).height
      q_cols[0].metric("Alignées (fort PRi + clicks)", f"{aligned:,}")
      q_cols[1].metric("Sur-maillées (PRi ok, clicks faibles)", f"{over_linked:,}")
      q_cols[2].metric("SOUS-MAILLÉES (clicks ok, PRi faible)", f"{under_linked:,}", delta="Opportunité!", delta_color="normal")
      q_cols[3].metric("Faibles (ni PRi ni clicks)", f"{weak:,}")

      # Scatter
      scatter_df = corr_df
      if scatter_df.height > 30_000:
        scatter_df = scatter_df.sample(n=30_000, seed=42)
      try:
        import plotly.express as px
        fig = px.scatter(
          scatter_df.to_pandas(),
          x="pri_score",
          y="real_clicks",
          color="template" if "template" in scatter_df.columns else None,
          hover_data=["path", "in_degree"],
          opacity=0.5,
          log_y=True,
          labels={"pri_score": "PRi (modèle maillage)", "real_clicks": "Clicks GSC (réel)"},
        )
        # Add quadrant lines
        fig.add_hline(y=med_clicks, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=med_pri, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(height=500, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
      except ImportError:
        st.info("Installez plotly pour le scatter interactif.")

      # Top sous-maillées
      st.caption("Top 20 pages sous-maillées — Fort trafic réel, faible PRi (opportunités de liens)")
      under_linked_df = (
        corr_df
        .filter((pl.col("pri_score") < med_pri) & (pl.col("real_clicks") >= med_clicks))
        .sort("real_clicks", descending=True)
        .head(20)
      )
      ul_cols = [c for c in ["path", "template", "real_clicks", "pri_score", "in_degree", "botify_pr"] if c in under_linked_df.columns]
      st.dataframe(under_linked_df.select(ul_cols).to_pandas(), use_container_width=True, hide_index=True)
    else:
      st.info("Aucune correspondance entre les pages générées et le CSV réel (URLs synthétiques).")


def _render_maillage_section_tab(
  audit_maillage_data: dict[str, Any] | None,
  maillage_model_data: dict[str, Any] | None,
  filtered_edges_df: pl.DataFrame,
  filtered_url_df: pl.DataFrame,
) -> None:
  """Onglet Maillage par Section — analyse par section HTML."""
  if audit_maillage_data is None and maillage_model_data is None:
    st.info("Données d'audit maillage non disponibles. Placez `audit_maillage_interne.json` et `maillage_model.json` dans le dossier `data/`.")
    return

  # ── Linking Matrix (section × destination) ──
  if audit_maillage_data and "linking_matrix_section_x_destination" in audit_maillage_data:
    st.subheader("Matrice de liens : Section HTML → Destination")
    st.caption("Nombre moyen de liens par section HTML vers chaque type de destination (données crawl réel).")
    matrix = audit_maillage_data["linking_matrix_section_x_destination"]
    # Build DataFrame
    rows = []
    for section, dests in matrix.items():
      if isinstance(dests, dict):
        for dest, val in dests.items():
          rows.append({"section": section, "destination": dest, "avg_links": float(val)})
    if rows:
      matrix_df = pl.DataFrame(rows)
      # Pivot for heatmap
      try:
        import plotly.express as px
        pivot_pd = matrix_df.to_pandas().pivot(index="section", columns="destination", values="avg_links").fillna(0)
        fig = px.imshow(
          pivot_pd,
          color_continuous_scale="Blues",
          text_auto=".0f",
          aspect="auto",
          labels={"color": "Avg liens"},
        )
        fig.update_layout(height=350, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
      except ImportError:
        st.dataframe(matrix_df.to_pandas(), use_container_width=True, hide_index=True)

  # ── Stats par template (depuis linking_rules_per_template) ──
  rules_data = audit_maillage_data.get("linking_rules_per_template", {}) if audit_maillage_data else {}
  if rules_data:
    st.subheader("Profil de liens par template")
    tpl_names = list(rules_data.keys())

    selected_tpl = st.selectbox("Template", tpl_names)
    if selected_tpl and selected_tpl in rules_data:
      tpl = rules_data[selected_tpl]
      t_cols = st.columns(2)
      t_cols[0].metric("Pages crawlées", tpl.get("file_count", "?"))
      sections = tpl.get("sections", {})
      t_cols[1].metric("Sections HTML", str(len(sections)))

      # Per-section breakdown
      if isinstance(sections, dict):
        section_rows = []
        for sec_name, sec_data in sections.items():
          avg_total = sec_data.get("avg_total_links", 0)
          dests = sec_data.get("avg_links_by_destination", {})
          for dest, val in dests.items():
            section_rows.append({"section": sec_name, "destination": dest, "avg_links": float(val)})
          if not dests:
            section_rows.append({"section": sec_name, "destination": "(total)", "avg_links": float(avg_total)})
        if section_rows:
          sec_df = pl.DataFrame(section_rows)
          # Bar chart section → total links
          sec_totals = sec_df.group_by("section").agg(pl.sum("avg_links").alias("total")).sort("total", descending=True)
          st.bar_chart(sec_totals.to_pandas().set_index("section"), use_container_width=True)
          # Detail table
          st.dataframe(sec_df.sort(["section", "avg_links"], descending=[False, True]).to_pandas(), use_container_width=True, hide_index=True)

  # ── Heatmap complète tous templates × sections ──
  if rules_data:
    st.subheader("Heatmap — Liens moyens par section (tous templates)")
    all_section_rows = []
    for tpl_name, tpl_data in rules_data.items():
      sections = tpl_data.get("sections", {})
      if isinstance(sections, dict):
        for sec_name, sec_data in sections.items():
          avg_total = sec_data.get("avg_total_links", 0)
          all_section_rows.append({"template": tpl_name, "section": sec_name, "avg_links": float(avg_total)})
    if all_section_rows:
      heat_df = pl.DataFrame(all_section_rows)
      try:
        import plotly.express as px
        pivot_pd = heat_df.to_pandas().pivot(index="template", columns="section", values="avg_links").fillna(0)
        fig = px.imshow(
          pivot_pd,
          color_continuous_scale="YlOrRd",
          text_auto=".0f",
          aspect="auto",
          labels={"color": "Avg liens"},
        )
        fig.update_layout(height=400, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
      except ImportError:
        st.dataframe(heat_df.to_pandas(), use_container_width=True, hide_index=True)

  # ── Maillage anomalies ──
  if audit_maillage_data and "anomalies" in audit_maillage_data:
    st.subheader("Anomalies de maillage")
    anomalies = audit_maillage_data["anomalies"]
    if anomalies:
      anom_df = pl.DataFrame(anomalies)
      st.dataframe(anom_df.to_pandas(), use_container_width=True, hide_index=True)
    else:
      st.success("Aucune anomalie de maillage détectée.")

  # ── Fuite PageRank /ad/ ──
  st.subheader("Fuite PageRank — Pages /ad/")
  st.caption("Les pages /ad/ (annonces) représentent 22M de pages mais ne renvoient que 5-6 liens internes. C'est la plus grande fuite de jus du site.")
  ad_stats_cols = st.columns(3)
  ad_stats_cols[0].metric("Pages /ad/ estimées", "22.3M")
  ad_stats_cols[1].metric("Liens retour /ad/ → listing", "5-6 / page")
  ad_stats_cols[2].metric("Impact", "Massif — 22M × 5 = 110M liens de faible qualité")

  if audit_maillage_data and "linking_rules_per_template" in audit_maillage_data:
    ad_rules = audit_maillage_data["linking_rules_per_template"].get("/ad/", {})
    ad_sections = ad_rules.get("sections", {})
    if isinstance(ad_sections, dict):
      mc = ad_sections.get("main_content", {})
      dests = mc.get("avg_links_by_destination", {})
      st.caption(f"Destinations main_content /ad/ : {dests}")
      total_out = sum(sec.get("avg_total_links", 0) for sec in ad_sections.values() if isinstance(sec, dict))
      st.caption(f"Total liens sortants moyen /ad/ : {total_out:.1f}")
      st.info("**Recommandation** : Augmenter les liens /ad/ → /cl/ (catégorie×géo) de 5.5 à 15. Impact estimé : +40% de PRi pour les pages /cl/ à fort trafic.")


def _render_audit_technique_tab(
  audit_templates_data: dict[str, Any] | None,
  audit_maillage_data: dict[str, Any] | None,
) -> None:
  """Onglet Audit Technique — anomalies, checklist schema, roadmap."""
  if audit_templates_data is None:
    st.info("Données d'audit templates non disponibles. Placez `audit_templates_diff.json` dans le dossier `data/`.")
    return

  # ── Résumé anomalies ──
  summary = audit_templates_data.get("anomalies_summary", {})
  total_anom = summary.get("total", 0)
  high = summary.get("HIGH", 0)
  medium = summary.get("MEDIUM", 0)
  low = summary.get("LOW", 0)

  cols = st.columns(4)
  cols[0].metric("Total anomalies", f"{total_anom}")
  cols[1].metric("HIGH", f"{high}", delta_color="inverse")
  cols[2].metric("MEDIUM", f"{medium}")
  cols[3].metric("LOW", f"{low}")

  # ── Table des anomalies ──
  st.subheader("Détail des anomalies")
  anomalies = audit_templates_data.get("anomalies", [])
  if anomalies:
    anom_df = pl.DataFrame(anomalies)
    # Filter by severity
    sev_filter = st.multiselect("Filtrer par sévérité", options=["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
    if sev_filter:
      anom_df = anom_df.filter(pl.col("severity").is_in(sev_filter))
    st.dataframe(anom_df.to_pandas(), use_container_width=True, hide_index=True)

    # By template breakdown
    if "template" in anom_df.columns:
      by_tpl = anom_df.group_by("template").len().sort("len", descending=True)
      st.caption("Anomalies par template")
      st.bar_chart(by_tpl.to_pandas().set_index("template").rename(columns={"len": "anomalies"}), use_container_width=True)

  # ── Checklist JSON-LD schema ──
  st.subheader("Checklist Schema JSON-LD par template")
  json_ld_diff = audit_templates_data.get("json_ld_diff", {})
  if json_ld_diff:
    schema_rows = []
    for tpl, data in json_ld_diff.items():
      schemas = data.get("schemas_found", {})
      has_schema = data.get("has_schema", False)
      schema_rows.append({
        "template": tpl,
        "has_schema": has_schema,
        "BreadcrumbList": "BreadcrumbList" in schemas,
        "CollectionPage": "CollectionPage" in schemas,
        "Product": "Product" in schemas or "Vehicle" in schemas,
        "RealEstateListing": "RealEstateListing" in schemas,
        "Article": "Article" in schemas,
        "schemas_found": ", ".join(schemas.keys()) if schemas else "Aucun",
      })
    schema_df = pl.DataFrame(schema_rows)
    st.dataframe(schema_df.to_pandas(), use_container_width=True, hide_index=True)

  # ── Roadmap Impact × Effort ──
  st.subheader("Roadmap — Impact × Effort")
  recommendations = [
    {"action": "Canonicals absolus /ad/", "impact": 10, "effort": 1, "pages": 22_300_000, "sprint": "Sprint 1"},
    {"action": "Article schema /guide/", "impact": 4, "effort": 1, "pages": 48, "sprint": "Sprint 1"},
    {"action": "Liens internes /dc/", "impact": 3, "effort": 1, "pages": 9, "sprint": "Sprint 1"},
    {"action": "BreadcrumbList 6 templates", "impact": 7, "effort": 2, "pages": 1_740_000, "sprint": "Sprint 2"},
    {"action": "RealEstateListing /ad/ immo", "impact": 9, "effort": 3, "pages": 500_000, "sprint": "Sprint 2"},
    {"action": "H1 dynamiques /cl/", "impact": 6, "effort": 2, "pages": 1_300_000, "sprint": "Sprint 2"},
    {"action": "Renforcer liens /ad/ → /cl/", "impact": 9, "effort": 4, "pages": 22_300_000, "sprint": "Sprint 3"},
    {"action": "Hreflang (multilingue)", "impact": 5, "effort": 5, "pages": 24_000_000, "sprint": "Sprint 3"},
  ]
  reco_df = pl.DataFrame(recommendations)
  st.dataframe(reco_df.to_pandas(), use_container_width=True, hide_index=True)

  try:
    import plotly.express as px
    fig = px.scatter(
      reco_df.to_pandas(),
      x="effort",
      y="impact",
      size="pages",
      color="sprint",
      text="action",
      size_max=50,
      labels={"effort": "Effort (1=trivial, 5=lourd)", "impact": "Impact SEO (1-10)"},
      color_discrete_map={"Sprint 1": "#2ecc71", "Sprint 2": "#f39c12", "Sprint 3": "#e74c3c"},
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(height=450, margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
  except ImportError:
    pass

  # Maillage anomalies (from audit_maillage_interne.json)
  if audit_maillage_data and "anomalies" in audit_maillage_data:
    st.subheader("Anomalies de maillage interne")
    mail_anom = audit_maillage_data["anomalies"]
    if mail_anom:
      st.dataframe(pl.DataFrame(mail_anom).to_pandas(), use_container_width=True, hide_index=True)


def _render_verticales_tab(
  filtered_url_df: pl.DataFrame,
  filtered_edges_df: pl.DataFrame,
) -> None:
  """Render ultra-granular Verticales tab with sub-tabs per verticale."""
  if "verticale" not in filtered_url_df.columns or "category" not in filtered_url_df.columns:
    st.info("Colonnes verticale/catégorie non disponibles. Utilisez le workspace LBC (`artifacts/lbc`).")
    return
  if filtered_url_df.height == 0:
    st.warning("Aucune page dans le périmètre filtré.")
    return

  # Load real metrics if available
  real_df = _load_real_metrics(filtered_url_df)

  # ── Global KPIs ──
  n_vert = int(filtered_url_df.get_column("verticale").drop_nulls().n_unique())
  n_cats = int(filtered_url_df.get_column("category").drop_nulls().n_unique())
  total_pri = float(filtered_url_df.select(pl.sum("pri_score")).item())
  kpi_cols = st.columns(5)
  kpi_cols[0].metric("Verticales", f"{n_vert}")
  kpi_cols[1].metric("Catégories", f"{n_cats}")
  kpi_cols[2].metric("Pages (modèle)", f"{filtered_url_df.height:,}")
  kpi_cols[3].metric("Pages réelles (CSV)", f"{real_df.height:,}" if real_df is not None else "—")
  total_clicks = int(real_df.select(pl.sum("real_clicks")).item()) if real_df is not None else 0
  kpi_cols[4].metric("Clicks GSC total", f"{total_clicks:,}" if total_clicks > 0 else "—")

  # ── Treemap ──
  treemap_df = (
    filtered_url_df
    .filter(pl.col("verticale").is_not_null() & pl.col("category").is_not_null())
    .group_by(["verticale", "category", "template"])
    .agg(pl.sum("pri_score").alias("pri_sum"), pl.len().alias("page_count"))
    .filter(pl.col("pri_sum") > 0)
  )
  if treemap_df.height > 0:
    try:
      import plotly.express as px
      fig = px.treemap(
        treemap_df.to_pandas(),
        path=["verticale", "category", "template"],
        values="pri_sum",
        color="pri_sum",
        color_continuous_scale="Blues",
      )
      fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=480)
      st.plotly_chart(fig, use_container_width=True)
    except ImportError:
      st.warning("Installez plotly pour le treemap.")

  # ── Sub-tabs per verticale ──
  st.divider()
  verticales = (
    filtered_url_df
    .filter(pl.col("verticale").is_not_null())
    .group_by("verticale")
    .agg(pl.sum("pri_score").alias("pri_sum"))
    .sort("pri_sum", descending=True)
    .get_column("verticale")
    .to_list()
  )
  if not verticales:
    return

  sub_tabs = st.tabs(verticales)
  for i, vert_name in enumerate(verticales):
    with sub_tabs[i]:
      _render_single_verticale(vert_name, filtered_url_df, filtered_edges_df, real_df, total_pri)


@st.cache_data(show_spinner=False)
def _load_real_metrics_cached(parquet_path: str) -> pl.DataFrame:
  return pl.read_parquet(parquet_path)


def _load_real_metrics(filtered_url_df: pl.DataFrame) -> pl.DataFrame | None:
  """Load real_metrics_1m.parquet if it exists in the workspace."""
  workspace = _resolve_workspace_from_args()
  real_path = workspace / "real_metrics_1m.parquet"
  if not real_path.exists():
    return None
  try:
    return _load_real_metrics_cached(str(real_path))
  except Exception:
    return None


def _render_single_verticale(
  verticale: str,
  full_url_df: pl.DataFrame,
  full_edges_df: pl.DataFrame,
  real_df: pl.DataFrame | None,
  total_pri: float,
) -> None:
  """Render a single verticale sub-tab with deep granularity."""
  vert_df = full_url_df.filter(pl.col("verticale") == verticale)
  if vert_df.height == 0:
    st.info("Aucune page pour cette verticale.")
    return

  vert_pri = float(vert_df.select(pl.sum("pri_score")).item())

  # ── A. KPIs ──
  cols = st.columns(7)
  cols[0].metric("Pages", f"{vert_df.height:,}")
  cols[1].metric("PRi total", f"{vert_pri:.6f}")
  cols[2].metric("% du PRi global", f"{vert_pri / max(total_pri, 1e-12) * 100:.1f}%")
  cols[3].metric("PRi moyen", f"{vert_pri / max(vert_df.height, 1):.8f}")
  avg_in = float(vert_df.select(pl.mean("in_degree")).item()) if vert_df.height > 0 else 0
  cols[4].metric("Avg in-degree", f"{avg_in:.0f}")
  cols[5].metric("Donneuses", f"{vert_df.filter(pl.col('can_give_juice')).height:,}")
  cols[6].metric("Low power", f"{vert_df.filter(pl.col('is_low_power')).height:,}")

  # Real clicks KPI if available
  if real_df is not None:
    vert_real = _real_for_verticale(real_df, verticale)
    if vert_real is not None and vert_real.height > 0:
      real_clicks = int(vert_real.select(pl.sum("real_clicks")).item()) if vert_real.select(pl.sum("real_clicks")).item() is not None else 0
      real_pages = vert_real.height
      avg_clicks = real_clicks / max(real_pages, 1)
      rc = st.columns(4)
      rc[0].metric("Pages réelles (CSV)", f"{real_pages:,}")
      rc[1].metric("Clicks GSC", f"{real_clicks:,}")
      rc[2].metric("Clicks/page", f"{avg_clicks:.0f}")
      rc[3].metric("Avg Botify PR", f"{vert_real.select(pl.mean('botify_pr')).item():.1f}" if vert_real.select(pl.mean('botify_pr')).item() is not None else "—")

  # ── B. Table catégories ──
  st.subheader("Catégories")
  cat_table = _verticale_category_table(vert_df, real_df, verticale)
  st.dataframe(cat_table.to_pandas(), use_container_width=True, hide_index=True)

  # ── C+D. Top/Bottom maillage ──
  st.subheader("Pages les plus / moins maillées")
  col_l, col_r = st.columns(2)
  with col_l:
    st.caption("Top 20 — Plus maillées (in_degree desc)")
    st.dataframe(_top_n_pages(vert_df, "in_degree", 20, desc=True).to_pandas(), use_container_width=True, hide_index=True)
  with col_r:
    st.caption("Top 20 — Moins maillées (in_degree asc, >= 1)")
    st.dataframe(_top_n_pages(vert_df, "in_degree", 20, desc=False, min_val=1).to_pandas(), use_container_width=True, hide_index=True)

  # ── E+F. Top/Bottom PRi ──
  st.subheader("PRi le plus fort / le plus faible")
  col_l2, col_r2 = st.columns(2)
  with col_l2:
    st.caption("Top 20 — PRi le plus fort")
    st.dataframe(_top_n_pages(vert_df, "pri_score", 20, desc=True).to_pandas(), use_container_width=True, hide_index=True)
  with col_r2:
    st.caption("Top 20 — Low power (PRi le plus faible)")
    low_df = vert_df.filter(pl.col("is_low_power"))
    if low_df.height > 0:
      st.dataframe(_top_n_pages(low_df, "pri_score", 20, desc=False).to_pandas(), use_container_width=True, hide_index=True)
    else:
      st.info("Aucune page low power dans cette verticale.")

  # ── G. Block types ──
  vert_page_ids = vert_df.select("page_id")
  vert_edges = full_edges_df.join(
    vert_page_ids.select(pl.col("page_id").alias("source_id")),
    on="source_id",
    how="inner",
  )
  if vert_edges.height > 0:
    st.subheader("Répartition des block types")
    block_mix = vert_edges.group_by("block_type").len().sort("len", descending=True)
    st.bar_chart(
      block_mix.rename({"len": "arêtes"}).to_pandas().set_index("block_type"),
      use_container_width=True,
    )

  # ── H. Pages /ck/ (keywords) ──
  ck_df = vert_df.filter(pl.col("template") == "ck") if "template" in vert_df.columns else pl.DataFrame()
  if ck_df.height > 0:
    st.subheader(f"Pages /ck/ (keywords) — {ck_df.height:,} pages")
    col_ck_l, col_ck_r = st.columns(2)
    with col_ck_l:
      st.caption("Top 10 /ck/ par PRi")
      ck_cols = [c for c in ["path", "pri_score", "in_degree", "out_degree"] if c in ck_df.columns]
      st.dataframe(ck_df.sort("pri_score", descending=True).head(10).select(ck_cols).to_pandas(), use_container_width=True, hide_index=True)
    with col_ck_r:
      orphan_ck = ck_df.filter(pl.col("in_degree") <= 2)
      st.caption(f"/ck/ orphelines (in_degree <= 2) : {orphan_ck.height:,} / {ck_df.height:,}")
      if orphan_ck.height > 0:
        st.dataframe(orphan_ck.sort("pri_score", descending=False).head(10).select(ck_cols).to_pandas(), use_container_width=True, hide_index=True)

  # ── I. Drill-down catégorie ──
  _render_category_drilldown(vert_df, full_edges_df, real_df, verticale)


def _real_for_verticale(
  real_df: pl.DataFrame,
  verticale: str,
) -> pl.DataFrame | None:
  """Filter real metrics to a verticale using botify_verticale mapping."""
  botify_map: dict[str, list[str]] = {
    "_vehicules_": ["Motors"],
    "_immobilier_": ["RealEstate"],
    "_mode_": ["Mode"],
    "_maison_jardin_": ["MaisonJardin"],
    "_electronique_": ["Autres"],  # Botify lumps electronics into Autres
    "_loisirs_": ["Autres"],
    "_famille_": ["Autres"],
    "_services_": ["Services", "Jobs"],
    "_autre_": ["Autres", "Vacances", "MatPro"],
  }
  botify_values = botify_map.get(verticale)
  if botify_values is None or "botify_verticale" not in real_df.columns:
    return None
  return real_df.filter(pl.col("botify_verticale").is_in(botify_values))


def _verticale_category_table(
  vert_df: pl.DataFrame,
  real_df: pl.DataFrame | None,
  verticale: str,
) -> pl.DataFrame:
  """Build category-level summary table for a verticale."""
  cat_stats = (
    vert_df
    .filter(pl.col("category").is_not_null())
    .group_by("category")
    .agg(
      pl.col("template").mode().first().alias("template_dom"),
      pl.len().alias("pages"),
      pl.sum("pri_score").alias("pri_sum"),
      pl.mean("pri_score").alias("pri_mean"),
      pl.mean("in_degree").cast(pl.Float64).alias("avg_in"),
      pl.mean("out_degree").cast(pl.Float64).alias("avg_out"),
      (pl.col("is_low_power").cast(pl.Float64).mean() * 100).alias("low_power_%"),
      (pl.col("can_give_juice").cast(pl.Float64).mean() * 100).alias("donor_%"),
      (pl.col("in_degree") <= 2).cast(pl.Int64).sum().alias("orphelines"),
    )
    .sort("pri_sum", descending=True)
  )
  # Join real click data by category if available
  if real_df is not None:
    vert_real = _real_for_verticale(real_df, verticale)
    if vert_real is not None and vert_real.height > 0:
      real_cat = (
        vert_real
        .filter(pl.col("category").is_not_null())
        .group_by("category")
        .agg(
          pl.len().alias("pages_reelles"),
          pl.sum("real_clicks").alias("clicks_gsc"),
          pl.mean("real_clicks").alias("clicks_avg"),
          pl.mean("real_inlinks").alias("real_inlinks_avg"),
        )
      )
      cat_stats = cat_stats.join(real_cat, on="category", how="left")
  return cat_stats


def _top_n_pages(
  df: pl.DataFrame,
  sort_col: str,
  n: int,
  desc: bool = True,
  min_val: int | float | None = None,
) -> pl.DataFrame:
  """Return top N pages sorted by sort_col."""
  base_cols = ["path", "template", "category", "pri_score", "rank", "in_degree", "out_degree"]
  cols = [sort_col] + [c for c in base_cols if c != sort_col]
  # Deduplicate column names (handles case where df already has duped columns)
  seen: set[str] = set()
  available = []
  for c in cols:
    if c in df.columns and c not in seen:
      available.append(c)
      seen.add(c)
  result = df
  if min_val is not None:
    result = result.filter(pl.col(sort_col) >= min_val)
  return result.sort(sort_col, descending=desc).head(n).select(available)


def _render_category_drilldown(
  vert_df: pl.DataFrame,
  full_edges_df: pl.DataFrame,
  real_df: pl.DataFrame | None,
  verticale: str,
) -> None:
  """Render drill-down into a specific category within a verticale."""
  st.subheader("Drill-down par catégorie")
  cats = (
    vert_df
    .filter(pl.col("category").is_not_null())
    .group_by("category")
    .agg(pl.sum("pri_score").alias("ps"))
    .sort("ps", descending=True)
  )
  cat_options = cats.get_column("category").to_list()
  if not cat_options:
    return

  sel = st.selectbox("Catégorie", cat_options, key=f"catdd_{verticale}")
  if not sel:
    return

  cat_df = vert_df.filter(pl.col("category") == sel)
  cat_pri = float(cat_df.select(pl.sum("pri_score")).item())

  # Mini KPIs
  c1, c2, c3, c4, c5 = st.columns(5)
  c1.metric("Pages", f"{cat_df.height:,}")
  c2.metric("PRi sum", f"{cat_pri:.6f}")
  avg_in = float(cat_df.select(pl.mean("in_degree")).item()) if cat_df.height else 0
  c3.metric("Avg in°", f"{avg_in:.0f}")
  avg_out = float(cat_df.select(pl.mean("out_degree")).item()) if cat_df.height else 0
  c4.metric("Avg out°", f"{avg_out:.0f}")
  c5.metric("Low power", f"{cat_df.filter(pl.col('is_low_power')).height:,}")

  # Real click data for this category
  if real_df is not None:
    vert_real = _real_for_verticale(real_df, verticale)
    if vert_real is not None:
      cat_real = vert_real.filter(pl.col("category") == sel)
      if cat_real.height > 0:
        rc = st.columns(3)
        total_clicks = cat_real.select(pl.sum("real_clicks")).item()
        rc[0].metric("Pages réelles (CSV)", f"{cat_real.height:,}")
        rc[1].metric("Clicks GSC", f"{int(total_clicks):,}" if total_clicks is not None else "—")
        rc[2].metric("Avg Botify PR", f"{cat_real.select(pl.mean('botify_pr')).item():.1f}" if cat_real.select(pl.mean('botify_pr')).item() is not None else "—")

  # Top/Bottom tables
  col_l, col_r = st.columns(2)
  with col_l:
    st.caption(f"Plus maillées — {sel}")
    st.dataframe(_top_n_pages(cat_df, "in_degree", 15, desc=True).to_pandas(), use_container_width=True, hide_index=True)
  with col_r:
    st.caption(f"Moins maillées — {sel}")
    st.dataframe(_top_n_pages(cat_df, "in_degree", 15, desc=False, min_val=1).to_pandas(), use_container_width=True, hide_index=True)

  # Template distribution
  if "template" in cat_df.columns and cat_df.height > 0:
    tpl_dist = cat_df.group_by("template").agg(pl.len().alias("pages"), pl.sum("pri_score").alias("pri_sum")).sort("pages", descending=True)
    col_a, col_b = st.columns(2)
    with col_a:
      st.caption(f"Templates — {sel}")
      st.bar_chart(tpl_dist.to_pandas().set_index("template")[["pages"]], use_container_width=True)
    with col_b:
      st.caption(f"PRi par template — {sel}")
      st.bar_chart(tpl_dist.to_pandas().set_index("template")[["pri_sum"]], use_container_width=True)

  # Scatter PRi vs in_degree
  if cat_df.height > 0:
    scatter_df = cat_df.select("page_id", "pri_score", "in_degree", "out_degree", "template")
    if scatter_df.height > 50_000:
      scatter_df = scatter_df.sample(n=50_000, seed=42)
    try:
      import plotly.express as px
      fig = px.scatter(
        scatter_df.to_pandas(),
        x="in_degree",
        y="pri_score",
        color="template",
        hover_data=["page_id", "out_degree"],
        opacity=0.5,
      )
      fig.update_layout(height=400, margin=dict(t=10, l=10, r=10, b=10))
      st.plotly_chart(fig, use_container_width=True)
    except ImportError:
      pass


def _render_segments_tab(
  segment_metrics_df: pl.DataFrame | None,
  filters: dict[str, Any],
) -> None:
  st.subheader("Segments URL")
  if segment_metrics_df is None:
    st.info("Exécute `pri-lab prepare-dashboard-data` pour activer l'analyse segmentaire complète.")
    return

  selected_level = int(filters["segment_level"])
  level_df = (
    segment_metrics_df
    .filter(pl.col("level") == selected_level)
    .sort("page_count", descending=True)
  )
  if level_df.height == 0:
    st.info("Aucun segment sur ce niveau.")
    return

  st.caption(f"Niveau {selected_level}: {level_df.height:,} segments")
  top_chart_df = level_df.head(30).select("segment_path", "page_count")
  st.bar_chart(top_chart_df.to_pandas().set_index("segment_path"), use_container_width=True)

  segment_table_df = level_df.select(
    "level",
    "segment_path",
    "segment",
    "page_count",
    "pri_sum",
    "pri_mean",
    "cheirank_mean",
    "donor_ratio",
    "low_power_ratio",
    "avg_in_degree",
    "avg_out_degree",
  )
  state = st.dataframe(
    segment_table_df.to_pandas(),
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="segments_table",
  )
  rows = _extract_selected_rows(state)
  if rows:
    selected_row = segment_table_df.row(rows[0], named=True)
    selected_segment = str(selected_row["segment_path"])
    changed = st.session_state.get("segment_path_filter") != selected_segment
    st.session_state["segment_level_filter"] = int(selected_row["level"])
    st.session_state["segment_path_filter"] = selected_segment
    if changed:
      st.rerun()


def _render_url_explorer_tab(
  filtered_url_df: pl.DataFrame,
  pages_df: pl.DataFrame,
  edges_path: Path,
  anchors_path: Path | None,
  selected_page_id: int | None,
  selected_block_types: list[str],
  selected_anchor_types: list[str],
  all_block_types: list[str],
  all_anchor_types: list[str],
) -> None:
  st.subheader("URL Explorer")
  if filtered_url_df.height == 0:
    st.info("Aucune URL dans le périmètre filtré.")
    return

  display_limit = st.slider("Nombre max de lignes URL", min_value=100, max_value=5000, value=1000, step=100)
  explorer_cols = [
    "page_id", "path", "depth", "pri_score", "rank",
    "cheirank_score", "cheirank_rank", "in_degree", "out_degree",
    "incoming_links", "outgoing_links", "can_give_juice", "is_low_power",
  ]
  for extra_col in ("template", "verticale", "category"):
    if extra_col in filtered_url_df.columns:
      explorer_cols.insert(3, extra_col)
  table_df = (
    filtered_url_df
    .head(display_limit)
    .select([c for c in explorer_cols if c in filtered_url_df.columns])
  )
  state = st.dataframe(
    table_df.to_pandas(),
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="url_explorer_table",
  )
  rows = _extract_selected_rows(state)
  if rows:
    selected_page_id = int(table_df.row(rows[0], named=True)["page_id"])
    st.session_state["selected_page_id"] = selected_page_id

  selected_page_id = st.session_state.get("selected_page_id")
  top_n = st.slider("Top N pour détails (voisins/ancres)", min_value=20, max_value=500, value=100, step=20)
  _render_url_detail_panel(
    selected_page_id=selected_page_id,
    filtered_url_df=filtered_url_df,
    pages_df=pages_df,
    edges_path=edges_path,
    anchors_path=anchors_path,
    selected_block_types=selected_block_types,
    selected_anchor_types=selected_anchor_types,
    all_block_types=all_block_types,
    all_anchor_types=all_anchor_types,
    top_n=top_n,
  )


def _render_url_detail_panel(
  selected_page_id: int | None,
  filtered_url_df: pl.DataFrame,
  pages_df: pl.DataFrame,
  edges_path: Path,
  anchors_path: Path | None,
  selected_block_types: list[str],
  selected_anchor_types: list[str],
  all_block_types: list[str],
  all_anchor_types: list[str],
  top_n: int,
) -> None:
  if selected_page_id is None:
    st.info("Sélectionne une URL pour afficher son détail.")
    return

  selected_df = filtered_url_df.filter(pl.col("page_id") == int(selected_page_id))
  if selected_df.height == 0:
    st.warning("URL sélectionnée hors périmètre filtré.")
    return

  selected_row = selected_df.row(0, named=True)
  st.markdown(f"### URL sélectionnée: `{selected_row['path']}`")

  metrics = st.columns(8)
  metrics[0].metric("Page ID", str(selected_row["page_id"]))
  metrics[1].metric("PRi", f"{float(selected_row['pri_score']):.8f}")
  metrics[2].metric("Rank PRi", str(int(selected_row["rank"])))
  metrics[3].metric("CheiRank", f"{float(selected_row['cheirank_score']):.8f}")
  metrics[4].metric("Rank CheiRank", str(int(selected_row["cheirank_rank"])))
  metrics[5].metric("In-degree", str(int(selected_row["in_degree"])))
  metrics[6].metric("Out-degree", str(int(selected_row["out_degree"])))
  metrics[7].metric("Lexical diversity out", f"{float(selected_row['lexical_diversity_out']):.3f}")

  page_edges_df = _load_page_edges_detail(
    edges_path=edges_path,
    page_id=int(selected_page_id),
    selected_block_types=selected_block_types,
    all_block_types=all_block_types,
  )
  out_edges_df = page_edges_df.filter(pl.col("source_id") == int(selected_page_id))
  in_edges_df = page_edges_df.filter(pl.col("target_id") == int(selected_page_id))

  path_lookup_df = pages_df.select("page_id", "path")
  out_neighbors_df = (
    out_edges_df
    .join(
      path_lookup_df.select(
        pl.col("page_id").alias("target_id"),
        pl.col("path").alias("target_path"),
      ),
      on="target_id",
      how="left",
    )
    .sort("rule_weight", descending=True)
    .head(top_n)
    .select("target_id", "target_path", "block_type", "rule_weight")
  )
  in_neighbors_df = (
    in_edges_df
    .join(
      path_lookup_df.select(
        pl.col("page_id").alias("source_id"),
        pl.col("path").alias("source_path"),
      ),
      on="source_id",
      how="left",
    )
    .sort("rule_weight", descending=True)
    .head(top_n)
    .select("source_id", "source_path", "block_type", "rule_weight")
  )

  neighbor_columns = st.columns(2)
  with neighbor_columns[0]:
    st.caption("Top voisins sortants")
    st.dataframe(out_neighbors_df.to_pandas(), use_container_width=True, hide_index=True)
  with neighbor_columns[1]:
    st.caption("Top voisins entrants")
    st.dataframe(in_neighbors_df.to_pandas(), use_container_width=True, hide_index=True)

  block_mix_columns = st.columns(2)
  with block_mix_columns[0]:
    out_mix_df = out_edges_df.group_by("block_type").len().sort("len", descending=True)
    st.caption("Mix blocs sortants")
    if out_mix_df.height > 0:
      st.bar_chart(out_mix_df.rename({"len": "count"}).to_pandas().set_index("block_type"), use_container_width=True)
    else:
      st.info("Aucun bloc sortant.")
  with block_mix_columns[1]:
    in_mix_df = in_edges_df.group_by("block_type").len().sort("len", descending=True)
    st.caption("Mix blocs entrants")
    if in_mix_df.height > 0:
      st.bar_chart(in_mix_df.rename({"len": "count"}).to_pandas().set_index("block_type"), use_container_width=True)
    else:
      st.info("Aucun bloc entrant.")

  if anchors_path is None:
    st.info("Aucun dataset d'ancres disponible pour le drill-down ancres.")
    return

  page_anchors_df = _load_page_anchors_detail(
    anchors_path=anchors_path,
    page_id=int(selected_page_id),
    selected_block_types=selected_block_types,
    selected_anchor_types=selected_anchor_types,
    all_block_types=all_block_types,
    all_anchor_types=all_anchor_types,
  )
  out_anchors_df = page_anchors_df.filter(pl.col("source_id") == int(selected_page_id))
  in_anchors_df = page_anchors_df.filter(pl.col("target_id") == int(selected_page_id))

  anchor_columns = st.columns(2)
  with anchor_columns[0]:
    st.caption("Ancres sortantes")
    out_anchor_text_df = (
      out_anchors_df
      .group_by(["anchor_text", "anchor_type"])
      .len()
      .sort("len", descending=True)
      .head(top_n)
      .rename({"len": "count"})
    )
    if out_anchor_text_df.height > 0:
      st.dataframe(out_anchor_text_df.to_pandas(), use_container_width=True, hide_index=True)
    else:
      st.info("Aucune ancre sortante.")
  with anchor_columns[1]:
    st.caption("Ancres entrantes")
    in_anchor_text_df = (
      in_anchors_df
      .group_by(["anchor_text", "anchor_type"])
      .len()
      .sort("len", descending=True)
      .head(top_n)
      .rename({"len": "count"})
    )
    if in_anchor_text_df.height > 0:
      st.dataframe(in_anchor_text_df.to_pandas(), use_container_width=True, hide_index=True)
    else:
      st.info("Aucune ancre entrante.")


@st.cache_data(show_spinner=False)
def _load_page_edges_detail(
  edges_path: Path,
  page_id: int,
  selected_block_types: list[str],
  all_block_types: list[str],
) -> pl.DataFrame:
  edges_lf = pl.scan_parquet(edges_path).select("source_id", "target_id", "block_type", "rule_weight")
  if all_block_types and len(selected_block_types) < len(all_block_types):
    edges_lf = edges_lf.filter(pl.col("block_type").is_in(selected_block_types))
  return edges_lf.filter(
    (pl.col("source_id") == page_id)
    | (pl.col("target_id") == page_id),
  ).collect()


@st.cache_data(show_spinner=False)
def _load_page_anchors_detail(
  anchors_path: Path,
  page_id: int,
  selected_block_types: list[str],
  selected_anchor_types: list[str],
  all_block_types: list[str],
  all_anchor_types: list[str],
) -> pl.DataFrame:
  anchors_lf = pl.scan_parquet(anchors_path).select(
    "source_id",
    "target_id",
    "block_type",
    "anchor_type",
    "anchor_text",
    "rule_weight",
  )
  if all_block_types and len(selected_block_types) < len(all_block_types):
    anchors_lf = anchors_lf.filter(pl.col("block_type").is_in(selected_block_types))
  if all_anchor_types and len(selected_anchor_types) < len(all_anchor_types):
    anchors_lf = anchors_lf.filter(pl.col("anchor_type").is_in(selected_anchor_types))
  return anchors_lf.filter(
    (pl.col("source_id") == page_id)
    | (pl.col("target_id") == page_id),
  ).collect()


def _render_anchors_tab(
  filtered_anchors_df: pl.DataFrame | None,
  selected_page_id: int | None,
  selected_block_types: list[str],
  selected_anchor_types: list[str],
  all_block_types: list[str],
  all_anchor_types: list[str],
) -> None:
  st.subheader("Ancres")
  if filtered_anchors_df is None:
    st.info("Aucun dataset d'ancres disponible.")
    return

  if filtered_anchors_df.height == 0:
    st.info("Aucune ancre dans le périmètre filtré.")
    return

  anchor_type_df = filtered_anchors_df.group_by("anchor_type").len().sort("len", descending=True)
  st.caption("Distribution des types d'ancres")
  st.bar_chart(
    anchor_type_df.rename({"len": "count"}).to_pandas().set_index("anchor_type"),
    use_container_width=True,
  )

  top_anchor_texts_df = (
    filtered_anchors_df
    .group_by("anchor_text")
    .len()
    .sort("len", descending=True)
    .head(30)
    .rename({"len": "count"})
  )
  st.caption("Top textes d'ancre")
  st.dataframe(top_anchor_texts_df.to_pandas(), use_container_width=True, hide_index=True)

  source_diversity_df = (
    filtered_anchors_df
    .group_by("source_id")
    .agg(
      pl.len().alias("out_links"),
      pl.col("anchor_text").n_unique().alias("unique_anchor_texts"),
      pl.col("anchor_type").n_unique().alias("unique_anchor_types"),
    )
    .with_columns(
      (
        pl.col("unique_anchor_texts")
        / pl.col("out_links").clip(lower_bound=1)
      ).alias("lexical_diversity_ratio"),
    )
  )
  diversity_stats = source_diversity_df.select(
    pl.mean("lexical_diversity_ratio").alias("mean_lexical_diversity_ratio"),
    pl.median("lexical_diversity_ratio").alias("median_lexical_diversity_ratio"),
    pl.mean("unique_anchor_types").alias("mean_unique_anchor_types"),
  ).to_dicts()[0]
  st.caption("Qualité diversité ancres (agrégée)")
  st.json(diversity_stats, expanded=False)

  if selected_page_id is None:
    st.info("Sélectionne une URL dans `URL Explorer` pour voir ses ancres entrantes/sortantes.")
    return

  selected_page_anchors_df = filtered_anchors_df.filter(
    (pl.col("source_id") == int(selected_page_id))
    | (pl.col("target_id") == int(selected_page_id)),
  )
  if selected_page_anchors_df.height == 0:
    st.info("Aucune ancre pour l'URL sélectionnée dans le périmètre filtré.")
    return

  st.caption(f"Focus URL sélectionnée (page_id={selected_page_id})")
  out_focus_df = (
    selected_page_anchors_df
    .filter(pl.col("source_id") == int(selected_page_id))
    .group_by(["anchor_text", "anchor_type", "block_type"])
    .len()
    .sort("len", descending=True)
    .head(100)
    .rename({"len": "count"})
  )
  in_focus_df = (
    selected_page_anchors_df
    .filter(pl.col("target_id") == int(selected_page_id))
    .group_by(["anchor_text", "anchor_type", "block_type"])
    .len()
    .sort("len", descending=True)
    .head(100)
    .rename({"len": "count"})
  )
  columns = st.columns(2)
  with columns[0]:
    st.caption("Ancres sortantes URL sélectionnée")
    st.dataframe(out_focus_df.to_pandas(), use_container_width=True, hide_index=True)
  with columns[1]:
    st.caption("Ancres entrantes URL sélectionnée")
    st.dataframe(in_focus_df.to_pandas(), use_container_width=True, hide_index=True)

  if all_block_types and len(selected_block_types) < len(all_block_types):
    st.caption(f"Filtre block_type actif: {', '.join(selected_block_types)}")
  if all_anchor_types and len(selected_anchor_types) < len(all_anchor_types):
    st.caption(f"Filtre anchor_type actif: {', '.join(selected_anchor_types)}")


def _render_cheirank_tab(
  filtered_url_df: pl.DataFrame,
  selected_page_id: int | None,
) -> None:
  st.subheader("CheiRank & puissance de jus")
  if filtered_url_df.height == 0:
    st.info("Aucune URL dans le périmètre filtré.")
    return

  cheirank_sum = float(filtered_url_df.select(pl.sum("cheirank_score")).item())
  donor_count = int(filtered_url_df.filter(pl.col("can_give_juice")).height)
  low_power_count = int(filtered_url_df.filter(pl.col("is_low_power")).height)

  columns = st.columns(4)
  columns[0].metric("Somme CheiRank", f"{cheirank_sum:.6f}")
  columns[1].metric("Pages donneuses de jus", f"{donor_count:,}")
  columns[2].metric("Pages faible puissance", f"{low_power_count:,}")
  columns[3].metric("Ratio low power", f"{(low_power_count / filtered_url_df.height * 100):.2f}%")

  top_n = st.slider("Top pages CheiRank", min_value=10, max_value=200, value=25, step=5)
  top_cheirank_df = (
    filtered_url_df
    .sort("cheirank_score", descending=True)
    .head(top_n)
    .select(
      "cheirank_rank",
      "page_id",
      "path",
      "cheirank_score",
      "pri_score",
      "in_degree",
      "out_degree",
      "can_give_juice",
      "is_low_power",
    )
  )
  st.dataframe(top_cheirank_df.to_pandas(), use_container_width=True, hide_index=True)

  donor_pages_df = (
    filtered_url_df
    .filter(pl.col("can_give_juice"))
    .sort("cheirank_score", descending=True)
    .head(30)
    .select("page_id", "path", "cheirank_score", "pri_score", "out_degree", "in_degree")
  )
  low_power_pages_df = (
    filtered_url_df
    .filter(pl.col("is_low_power"))
    .sort("pri_score", descending=False)
    .head(30)
    .select("page_id", "path", "pri_score", "cheirank_score", "in_degree", "out_degree")
  )
  split_columns = st.columns(2)
  with split_columns[0]:
    st.caption("Pages pouvant donner du jus")
    st.dataframe(donor_pages_df.to_pandas(), use_container_width=True, hide_index=True)
  with split_columns[1]:
    st.caption("Pages avec peu de puissance")
    st.dataframe(low_power_pages_df.to_pandas(), use_container_width=True, hide_index=True)

  scatter_df = (
    filtered_url_df
    .with_columns(
      pl.when(pl.col("can_give_juice"))
      .then(pl.lit("donor"))
      .when(pl.col("is_low_power"))
      .then(pl.lit("low_power"))
      .otherwise(pl.lit("normal"))
      .alias("segment"),
    )
    .select("page_id", "pri_score", "cheirank_score", "out_degree", "segment")
  )
  scatter_df = _downsample_for_scatter(scatter_df, max_points=MAX_SCATTER_POINTS)
  st.caption(f"Carte PRi vs CheiRank (échantillon déterministe max {MAX_SCATTER_POINTS:,} points)")
  st.scatter_chart(
    scatter_df.to_pandas(),
    x="pri_score",
    y="cheirank_score",
    color="segment",
    size="out_degree",
    use_container_width=True,
  )

  if selected_page_id is not None:
    selected_df = filtered_url_df.filter(pl.col("page_id") == int(selected_page_id))
    if selected_df.height > 0:
      st.caption("URL sélectionnée (mise en évidence textuelle)")
      st.dataframe(
        selected_df.select(
          "page_id",
          "path",
          "pri_score",
          "rank",
          "cheirank_score",
          "cheirank_rank",
          "in_degree",
          "out_degree",
          "can_give_juice",
          "is_low_power",
        ).to_pandas(),
        use_container_width=True,
        hide_index=True,
      )


def _render_scenarios_tab(
  scenarios_dir: Path,
  comparison_path: Path,
  pages_df: pl.DataFrame,
  filtered_page_ids_df: pl.DataFrame,
  selected_page_id: int | None,
) -> None:
  st.subheader("Scénarios")
  if not scenarios_dir.exists():
    st.info("Aucun scénario trouvé. Lance `pri-lab model-anchor-scenarios`.")
    return

  scenario_edges_paths = sorted(scenarios_dir.glob("scenario_*_edges.parquet"))
  scenario_pri_paths = sorted(scenarios_dir.glob("scenario_*_pri.parquet"))
  if not scenario_edges_paths or not scenario_pri_paths:
    st.info("Scénarios incomplets. Lance `pri-lab model-anchor-scenarios`.")
    return

  scenario_rows: list[dict[str, Any]] = []
  scenario_anchor_mix_rows: list[dict[str, Any]] = []

  for scenario_edges_path in scenario_edges_paths:
    scenario_name = scenario_edges_path.name.replace("_edges.parquet", "")
    scenario_edges_df = _load_parquet(scenario_edges_path)
    avg_diversity = (
      float(scenario_edges_df.select(pl.mean("anchor_diversity_score")).item())
      if "anchor_diversity_score" in scenario_edges_df.columns and scenario_edges_df.height > 0
      else 0.0
    )
    avg_weight = float(scenario_edges_df.select(pl.mean("rule_weight")).item()) if scenario_edges_df.height > 0 else 0.0
    scenario_rows.append(
      {
        "scenario": scenario_name,
        "edge_count": scenario_edges_df.height,
        "avg_anchor_diversity_score": avg_diversity,
        "avg_rule_weight": avg_weight,
      },
    )
    mix_df = scenario_edges_df.group_by("anchor_type").len()
    for row in mix_df.to_dicts():
      scenario_anchor_mix_rows.append(
        {
          "scenario": scenario_name,
          "anchor_type": row["anchor_type"],
          "count": int(row["len"]),
        },
      )

  scenario_summary_df = pl.DataFrame(scenario_rows).sort("scenario")
  st.dataframe(scenario_summary_df.to_pandas(), use_container_width=True, hide_index=True)

  scenario_mix_df = pl.DataFrame(scenario_anchor_mix_rows)
  scenario_mix_pivot_df = (
    scenario_mix_df
    .pivot(on="anchor_type", index="scenario", values="count", aggregate_function="sum")
    .fill_null(0)
    .sort("scenario")
  )
  st.caption("Mix d'ancres par scénario")
  st.bar_chart(
    scenario_mix_pivot_df.to_pandas().set_index("scenario"),
    use_container_width=True,
  )

  scenario_options = scenario_summary_df.get_column("scenario").to_list()
  selected_scenario = st.selectbox(
    "Inspecter un scénario",
    options=scenario_options,
    index=0,
  )
  selected_pri_path = scenarios_dir / f"{selected_scenario}_pri.parquet"
  if selected_pri_path.exists():
    selected_pri_df = _load_parquet(selected_pri_path)
    scoped_pri_df = selected_pri_df
    if filtered_page_ids_df.height > 0:
      scoped_pri_df = scoped_pri_df.join(filtered_page_ids_df, on="page_id", how="inner")
    selected_top_df = (
      scoped_pri_df
      .sort("pri_score", descending=True)
      .head(20)
      .join(pages_df.select("page_id", "path"), on="page_id", how="left")
      .select("rank", "page_id", "path", "pri_score", "in_degree", "out_degree")
    )
    st.caption(f"Top pages - {selected_scenario} (périmètre filtré)")
    st.dataframe(selected_top_df.to_pandas(), use_container_width=True, hide_index=True)

  if comparison_path.exists():
    comparison_df = _load_parquet(comparison_path)
    delta_columns = [column for column in comparison_df.columns if column.startswith("delta_")]
    if delta_columns:
      selected_delta = st.selectbox(
        "Delta PRi à analyser",
        options=delta_columns,
        index=0,
      )
      delta_df = (
        comparison_df
        .select("page_id", pl.col(selected_delta).alias("delta_pri"))
        .join(pages_df.select("page_id", "path"), on="page_id", how="left")
      )
      if filtered_page_ids_df.height > 0:
        delta_df = delta_df.join(filtered_page_ids_df, on="page_id", how="inner")
      gainers_df = delta_df.sort("delta_pri", descending=True).head(15)
      losers_df = delta_df.sort("delta_pri", descending=False).head(15)
      columns = st.columns(2)
      with columns[0]:
        st.caption("Top gainers")
        st.dataframe(gainers_df.to_pandas(), use_container_width=True, hide_index=True)
      with columns[1]:
        st.caption("Top losers")
        st.dataframe(losers_df.to_pandas(), use_container_width=True, hide_index=True)

      delta_stats = delta_df.select(
        pl.min("delta_pri").alias("min_delta_pri"),
        pl.mean("delta_pri").alias("mean_delta_pri"),
        pl.median("delta_pri").alias("median_delta_pri"),
        pl.max("delta_pri").alias("max_delta_pri"),
      ).to_dicts()[0]
      st.caption("Statistiques delta")
      st.json(delta_stats, expanded=False)

      if selected_page_id is not None:
        selected_page_delta_df = (
          comparison_df
          .filter(pl.col("page_id") == int(selected_page_id))
          .join(pages_df.select("page_id", "path"), on="page_id", how="left")
        )
        if selected_page_delta_df.height > 0:
          ordered_columns = [
            "page_id",
            "path",
            *[column for column in selected_page_delta_df.columns if column.startswith("pri_")],
            *[column for column in selected_page_delta_df.columns if column.startswith("delta_")],
          ]
          st.caption("URL sélectionnée - impact scénarios")
          st.dataframe(
            selected_page_delta_df.select(ordered_columns).to_pandas(),
            use_container_width=True,
            hide_index=True,
          )


def _extract_selected_rows(selection_state: Any) -> list[int]:
  if selection_state is None:
    return []
  if hasattr(selection_state, "selection") and hasattr(selection_state.selection, "rows"):
    return [int(index) for index in selection_state.selection.rows]
  if isinstance(selection_state, dict):
    rows = selection_state.get("selection", {}).get("rows", [])
    return [int(index) for index in rows]
  return []


def _downsample_for_scatter(
  frame: pl.DataFrame,
  max_points: int,
) -> pl.DataFrame:
  if frame.height <= max_points:
    return frame
  step = max(1, math.ceil(frame.height / max_points))
  return (
    frame
    .with_row_index(name="row_idx")
    .filter((pl.col("row_idx") % step) == 0)
    .drop("row_idx")
    .head(max_points)
  )


if __name__ == "__main__":
  main()
