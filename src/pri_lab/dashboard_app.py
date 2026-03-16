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

  st.sidebar.header("Workspace")
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

  tabs = st.tabs(["Vue globale", "Segments URL", "URL Explorer", "Ancres", "CheiRank & jus", "Scénarios"])

  with tabs[0]:
    _render_overview_tab(
      url_metrics_df=url_metrics_df,
      filtered_url_df=filtered_url_df,
      filtered_edges_df=filtered_edges_df,
      metrics_payload=metrics_payload,
    )

  with tabs[1]:
    _render_segments_tab(
      segment_metrics_df=segment_metrics_df,
      filters=filters,
    )

  with tabs[2]:
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

  with tabs[3]:
    _render_anchors_tab(
      filtered_anchors_df=filtered_anchors_df,
      selected_page_id=selected_page_id,
      selected_block_types=filters["block_types"],
      selected_anchor_types=filters["anchor_types"],
      all_block_types=all_block_types,
      all_anchor_types=all_anchor_types,
    )

  with tabs[4]:
    _render_cheirank_tab(
      filtered_url_df=filtered_url_df,
      selected_page_id=selected_page_id,
    )

  with tabs[5]:
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

  return {
    "depth_min": depth_range[0],
    "depth_max": depth_range[1],
    "segment_level": segment_level,
    "segment_path": segment_path,
    "block_types": block_types,
    "anchor_types": anchor_types,
    "url_search": url_search,
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

  top_n = st.slider("Top pages PRi", min_value=10, max_value=200, value=25, step=5)
  top_df = (
    filtered_url_df
    .sort("pri_score", descending=True)
    .head(top_n)
    .select(
      "rank",
      "page_id",
      "path",
      "depth",
      "pri_score",
      "cheirank_score",
      "in_degree",
      "out_degree",
      "can_give_juice",
      "is_low_power",
    )
  )
  st.dataframe(top_df.to_pandas(), use_container_width=True, hide_index=True)

  _render_depth_analysis_section(filtered_url_df=filtered_url_df)

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
  table_df = (
    filtered_url_df
    .head(display_limit)
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
      "incoming_links",
      "outgoing_links",
      "can_give_juice",
      "is_low_power",
    )
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
