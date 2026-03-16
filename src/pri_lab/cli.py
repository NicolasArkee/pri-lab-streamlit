from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from pri_lab.config import default_config_path, load_experiment_config
from pri_lab.dashboard import launch_dashboard
from pri_lab.metrics import MetricsRecorder
from pri_lab.pipeline import (
  BuildEdgesOptions,
  ComputePriOptions,
  append_experiment_log,
  build_edges,
  compute_pri,
  export_workspace_report,
  model_anchor_scenarios,
  prepare_anchor_dataset,
  prepare_dashboard_data,
  prepare_outlinks_dataset,
  prepare_pages,
)
from pri_lab.synthetic import generate_synthetic_edges, generate_synthetic_pages


def main() -> None:
  parser = _build_parser()
  args = parser.parse_args()

  command_handlers = {
    "prepare-pages": _handle_prepare_pages,
    "prepare-outlinks": _handle_prepare_outlinks,
    "build-edges": _handle_build_edges,
    "compute-pri": _handle_compute_pri,
    "prepare-anchor-dataset": _handle_prepare_anchor_dataset,
    "prepare-dashboard-data": _handle_prepare_dashboard_data,
    "export-report": _handle_export_report,
    "model-anchor-scenarios": _handle_model_anchor_scenarios,
    "dashboard": _handle_dashboard,
    "run-experiment": _handle_run_experiment,
    "run-outlinks-analysis": _handle_run_outlinks_analysis,
    "benchmark-synthetic": _handle_benchmark_synthetic,
  }

  handler = command_handlers[args.command]
  try:
    payload = handler(args)
  except Exception as error:
    failure_payload = {
      "status": "error",
      "command": args.command,
      "error": str(error),
    }
    print(json.dumps(failure_payload, indent=2, ensure_ascii=False))
    raise

  print(json.dumps(payload, indent=2, ensure_ascii=False))


def _handle_prepare_pages(args: argparse.Namespace) -> dict[str, object]:
  recorder = MetricsRecorder(command="prepare-pages")
  with recorder.stage("prepare-pages") as details:
    details.update(
      prepare_pages(
        input_json_path=args.input_json.resolve(),
        output_pages_path=args.output_pages.resolve(),
      ),
    )
  payload = recorder.finish(extra={"status": "ok"})
  _write_metrics_if_needed(payload, args.metrics_json)
  return payload


def _handle_prepare_outlinks(args: argparse.Namespace) -> dict[str, object]:
  recorder = MetricsRecorder(command="prepare-outlinks")
  with recorder.stage("prepare-outlinks") as details:
    details.update(
      prepare_outlinks_dataset(
        input_csv_path=args.input_csv.resolve(),
        output_pages_path=args.output_pages.resolve(),
        output_edges_path=args.output_edges.resolve(),
        output_anchor_candidates_path=args.output_anchor_candidates.resolve(),
        source_host=args.source_host,
        max_out_links_per_page=args.max_out_links_per_page,
      ),
    )
  payload = recorder.finish(extra={"status": "ok"})
  _write_metrics_if_needed(payload, args.metrics_json)
  return payload


def _handle_build_edges(args: argparse.Namespace) -> dict[str, object]:
  recorder = MetricsRecorder(command="build-edges")
  options = BuildEdgesOptions(
    enable_hierarchy_up=not args.disable_hierarchy_up,
    enable_hierarchy_down=not args.disable_hierarchy_down,
    enable_cluster_peer=not args.disable_cluster_peer,
    cluster_peer_k=args.cluster_peer_k,
    max_out_links_per_page=args.max_out_links_per_page,
    weight_hierarchy_up=args.weight_hierarchy_up,
    weight_hierarchy_down=args.weight_hierarchy_down,
    weight_cluster_peer=args.weight_cluster_peer,
  )
  with recorder.stage("build-edges") as details:
    details.update(
      build_edges(
        input_pages_path=args.input_pages.resolve(),
        output_edges_path=args.output_edges.resolve(),
        options=options,
      ),
    )
  payload = recorder.finish(extra={"status": "ok"})
  _write_metrics_if_needed(payload, args.metrics_json)
  return payload


def _handle_compute_pri(args: argparse.Namespace) -> dict[str, object]:
  recorder = MetricsRecorder(command="compute-pri")
  options = ComputePriOptions(
    damping=args.damping,
    include_block_types=args.include_block_type or None,
    use_weights=not args.unweighted,
  )
  with recorder.stage("compute-pri") as details:
    details.update(
      compute_pri(
        input_pages_path=args.input_pages.resolve(),
        input_edges_path=args.input_edges.resolve(),
        output_pri_path=args.output_pri.resolve(),
        options=options,
      ),
    )
  payload = recorder.finish(extra={"status": "ok"})
  _write_metrics_if_needed(payload, args.metrics_json)
  return payload


def _handle_prepare_anchor_dataset(args: argparse.Namespace) -> dict[str, object]:
  recorder = MetricsRecorder(command="prepare-anchor-dataset")
  with recorder.stage("prepare-anchor-dataset") as details:
    details.update(
      prepare_anchor_dataset(
        input_pages_path=args.input_pages.resolve(),
        input_edges_path=args.input_edges.resolve(),
        output_anchor_candidates_path=args.output_anchor_candidates.resolve(),
      ),
    )
  payload = recorder.finish(extra={"status": "ok"})
  _write_metrics_if_needed(payload, args.metrics_json)
  return payload


def _handle_prepare_dashboard_data(args: argparse.Namespace) -> dict[str, object]:
  workspace = args.workspace.resolve()
  pages_path = workspace / "pages.parquet"
  edges_path = workspace / "edges.parquet"
  pri_path = workspace / "pri_scores.parquet"
  anchors_path = workspace / "anchor_candidates.parquet"
  page_segments_path = workspace / "page_segments.parquet"
  url_metrics_path = workspace / "url_metrics.parquet"
  segment_metrics_path = workspace / "segment_metrics.parquet"

  recorder = MetricsRecorder(command="prepare-dashboard-data")
  with recorder.stage("prepare-dashboard-data") as details:
    details.update(
      prepare_dashboard_data(
        input_pages_path=pages_path,
        input_edges_path=edges_path,
        input_pri_path=pri_path,
        output_page_segments_path=page_segments_path,
        output_url_metrics_path=url_metrics_path,
        output_segment_metrics_path=segment_metrics_path,
        input_anchor_candidates_path=anchors_path if anchors_path.exists() else None,
      ),
    )
  payload = recorder.finish(
    extra={
      "status": "ok",
      "workspace": str(workspace),
      "page_segments_parquet": str(page_segments_path),
      "url_metrics_parquet": str(url_metrics_path),
      "segment_metrics_parquet": str(segment_metrics_path),
    },
  )
  _write_metrics_if_needed(payload, args.metrics_json)
  return payload


def _handle_export_report(args: argparse.Namespace) -> dict[str, object]:
  workspace = args.workspace.resolve()
  if args.output_dir is not None:
    output_dir = args.output_dir.resolve()
  else:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (workspace / "exports" / timestamp).resolve()

  recorder = MetricsRecorder(command="export-report")
  with recorder.stage("export-report") as details:
    details.update(
      export_workspace_report(
        workspace=workspace,
        output_dir=output_dir,
        include_scenarios=not args.skip_scenarios,
      ),
    )
  payload = recorder.finish(
    extra={
      "status": "ok",
      "workspace": str(workspace),
      "output_dir": str(output_dir),
      "include_scenarios": not args.skip_scenarios,
    },
  )
  _write_metrics_if_needed(payload, args.metrics_json)
  return payload


def _handle_model_anchor_scenarios(args: argparse.Namespace) -> dict[str, object]:
  recorder = MetricsRecorder(command="model-anchor-scenarios")
  with recorder.stage("model-anchor-scenarios") as details:
    details.update(
      model_anchor_scenarios(
        input_pages_path=args.input_pages.resolve(),
        input_edges_path=args.input_edges.resolve(),
        output_dir=args.output_dir.resolve(),
        input_anchor_candidates_path=(
          args.input_anchor_candidates.resolve()
          if args.input_anchor_candidates is not None
          else None
        ),
        damping=args.damping,
      ),
    )
  payload = recorder.finish(extra={"status": "ok"})
  _write_metrics_if_needed(payload, args.metrics_json)
  return payload


def _handle_dashboard(args: argparse.Namespace) -> dict[str, object]:
  launch_dashboard(
    workspace=args.workspace.resolve() if args.workspace else None,
    host=args.host,
    port=args.port,
  )
  return {
    "status": "ok",
    "command": "dashboard",
    "workspace": str(args.workspace.resolve() if args.workspace else ""),
    "host": args.host,
    "port": args.port,
  }


def _handle_run_experiment(args: argparse.Namespace) -> dict[str, object]:
  config = load_experiment_config(
    config_path=args.config.resolve(),
    workspace_override=args.workspace.resolve() if args.workspace else None,
  )
  config.paths.workspace.mkdir(parents=True, exist_ok=True)

  run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
  recorder = MetricsRecorder(command="run-experiment")

  with recorder.stage("prepare-pages") as details:
    details.update(
      prepare_pages(
        input_json_path=config.paths.input_json,
        output_pages_path=config.paths.pages_parquet,
      ),
    )

  with recorder.stage("build-edges") as details:
    details.update(
      build_edges(
        input_pages_path=config.paths.pages_parquet,
        output_edges_path=config.paths.edges_parquet,
        options=BuildEdgesOptions(
          enable_hierarchy_up=config.rules.enable_hierarchy_up,
          enable_hierarchy_down=config.rules.enable_hierarchy_down,
          enable_cluster_peer=config.rules.enable_cluster_peer,
          cluster_peer_k=config.rules.cluster_peer_k,
          max_out_links_per_page=config.rules.max_out_links_per_page,
          weight_hierarchy_up=config.rules.weight_hierarchy_up,
          weight_hierarchy_down=config.rules.weight_hierarchy_down,
          weight_cluster_peer=config.rules.weight_cluster_peer,
        ),
      ),
    )

  with recorder.stage("compute-pri") as details:
    details.update(
      compute_pri(
        input_pages_path=config.paths.pages_parquet,
        input_edges_path=config.paths.edges_parquet,
        output_pri_path=config.paths.pri_scores_parquet,
        options=ComputePriOptions(
          damping=config.pagerank.damping,
          include_block_types=config.pagerank.included_block_types,
          use_weights=config.pagerank.use_weights,
        ),
      ),
    )

  page_segments_path = config.paths.workspace / "page_segments.parquet"
  url_metrics_path = config.paths.workspace / "url_metrics.parquet"
  segment_metrics_path = config.paths.workspace / "segment_metrics.parquet"
  anchors_path = config.paths.workspace / "anchor_candidates.parquet"
  with recorder.stage("prepare-dashboard-data") as details:
    details.update(
      prepare_dashboard_data(
        input_pages_path=config.paths.pages_parquet,
        input_edges_path=config.paths.edges_parquet,
        input_pri_path=config.paths.pri_scores_parquet,
        output_page_segments_path=page_segments_path,
        output_url_metrics_path=url_metrics_path,
        output_segment_metrics_path=segment_metrics_path,
        input_anchor_candidates_path=anchors_path if anchors_path.exists() else None,
      ),
    )

  payload = recorder.finish(
    extra={
      "status": "ok",
      "run_id": run_id,
      "workspace": str(config.paths.workspace),
      "config": config.to_dict(),
      "page_segments_parquet": str(page_segments_path),
      "url_metrics_parquet": str(url_metrics_path),
      "segment_metrics_parquet": str(segment_metrics_path),
    },
  )

  _write_json(config.paths.run_metrics_json, payload)

  summary_record = {
    "run_id": run_id,
    "timestamp": payload["finished_at"],
    "workspace": str(config.paths.workspace),
    "page_count": _find_stage_value(payload, "prepare-pages", "page_count"),
    "edge_count": _find_stage_value(payload, "build-edges", "edge_count"),
    "pri_sum": _find_stage_value(payload, "compute-pri", "pri_sum"),
    "duration_seconds": payload["duration_seconds"],
    "damping": config.pagerank.damping,
    "included_block_types": ",".join(config.pagerank.included_block_types),
    "max_out_links_per_page": config.rules.max_out_links_per_page,
    "cluster_peer_k": config.rules.cluster_peer_k,
  }
  append_experiment_log(config.paths.experiment_log_parquet, summary_record)
  payload["experiment_log_parquet"] = str(config.paths.experiment_log_parquet)
  payload["run_metrics_json"] = str(config.paths.run_metrics_json)
  return payload


def _handle_run_outlinks_analysis(args: argparse.Namespace) -> dict[str, object]:
  workspace = args.workspace.resolve()
  workspace.mkdir(parents=True, exist_ok=True)

  pages_path = workspace / "pages.parquet"
  edges_path = workspace / "edges.parquet"
  pri_path = workspace / "pri_scores.parquet"
  anchors_path = workspace / "anchor_candidates.parquet"
  scenarios_dir = workspace / "scenarios"
  page_segments_path = workspace / "page_segments.parquet"
  url_metrics_path = workspace / "url_metrics.parquet"
  segment_metrics_path = workspace / "segment_metrics.parquet"
  run_metrics_path = workspace / "run_metrics.json"

  recorder = MetricsRecorder(command="run-outlinks-analysis")
  with recorder.stage("prepare-outlinks") as details:
    details.update(
      prepare_outlinks_dataset(
        input_csv_path=args.input_csv.resolve(),
        output_pages_path=pages_path,
        output_edges_path=edges_path,
        output_anchor_candidates_path=anchors_path,
        source_host=args.source_host,
        max_out_links_per_page=args.max_out_links_per_page,
      ),
    )

  with recorder.stage("compute-pri") as details:
    details.update(
      compute_pri(
        input_pages_path=pages_path,
        input_edges_path=edges_path,
        output_pri_path=pri_path,
        options=ComputePriOptions(
          damping=args.damping,
          include_block_types=None,
          use_weights=True,
        ),
      ),
    )

  with recorder.stage("prepare-dashboard-data") as details:
    details.update(
      prepare_dashboard_data(
        input_pages_path=pages_path,
        input_edges_path=edges_path,
        input_pri_path=pri_path,
        output_page_segments_path=page_segments_path,
        output_url_metrics_path=url_metrics_path,
        output_segment_metrics_path=segment_metrics_path,
        input_anchor_candidates_path=anchors_path if anchors_path.exists() else None,
      ),
    )

  with recorder.stage("model-anchor-scenarios") as details:
    details.update(
      model_anchor_scenarios(
        input_pages_path=pages_path,
        input_edges_path=edges_path,
        output_dir=scenarios_dir,
        input_anchor_candidates_path=anchors_path,
        damping=args.damping,
      ),
    )

  payload = recorder.finish(
    extra={
      "status": "ok",
      "workspace": str(workspace),
      "input_csv": str(args.input_csv.resolve()),
      "source_host": args.source_host,
      "max_out_links_per_page": args.max_out_links_per_page,
      "damping": args.damping,
      "pages_parquet": str(pages_path),
      "edges_parquet": str(edges_path),
      "pri_scores_parquet": str(pri_path),
      "anchor_candidates_parquet": str(anchors_path),
      "scenarios_dir": str(scenarios_dir),
      "page_segments_parquet": str(page_segments_path),
      "url_metrics_parquet": str(url_metrics_path),
      "segment_metrics_parquet": str(segment_metrics_path),
    },
  )
  _write_json(run_metrics_path, payload)
  payload["run_metrics_json"] = str(run_metrics_path)
  return payload


def _handle_benchmark_synthetic(args: argparse.Namespace) -> dict[str, object]:
  config = load_experiment_config(config_path=args.config.resolve())
  workspace = (
    args.workspace.resolve()
    if args.workspace
    else (config.paths.workspace / "benchmark-synthetic").resolve()
  )
  workspace.mkdir(parents=True, exist_ok=True)

  node_count = args.node_count if args.node_count else config.benchmark.node_count
  target_edge_count = args.target_edge_count if args.target_edge_count else config.benchmark.target_edge_count
  cluster_count = args.cluster_count if args.cluster_count else config.benchmark.cluster_count

  pages_path = workspace / "synthetic_pages.parquet"
  edges_path = workspace / "synthetic_edges.parquet"
  pri_path = workspace / "synthetic_pri_scores.parquet"
  metrics_path = workspace / "synthetic_benchmark_metrics.json"

  recorder = MetricsRecorder(command="benchmark-synthetic")
  with recorder.stage("generate-pages") as details:
    details.update(
      generate_synthetic_pages(
        output_pages_path=pages_path,
        node_count=node_count,
        cluster_count=cluster_count,
      ),
    )
  with recorder.stage("generate-edges") as details:
    details.update(
      generate_synthetic_edges(
        output_edges_path=edges_path,
        node_count=node_count,
        target_edge_count=target_edge_count,
      ),
    )
  with recorder.stage("compute-pri") as details:
    details.update(
      compute_pri(
        input_pages_path=pages_path,
        input_edges_path=edges_path,
        output_pri_path=pri_path,
        options=ComputePriOptions(
          damping=args.damping,
          include_block_types=["synthetic"],
          use_weights=True,
        ),
      ),
    )

  payload = recorder.finish(
    extra={
      "status": "ok",
      "workspace": str(workspace),
      "node_count": node_count,
      "target_edge_count": target_edge_count,
      "cluster_count": cluster_count,
    },
  )
  _write_json(metrics_path, payload)
  payload["run_metrics_json"] = str(metrics_path)
  return payload


def _find_stage_value(payload: dict[str, object], stage_name: str, key: str) -> object:
  for stage in payload.get("stages", []):
    if stage.get("name") == stage_name:
      details = stage.get("details", {})
      return details.get(key)
  return None


def _write_metrics_if_needed(payload: dict[str, object], metrics_json: Path | None) -> None:
  if metrics_json is None:
    return
  _write_json(metrics_json.resolve(), payload)


def _write_json(path: Path, payload: dict[str, object]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as file:
    json.dump(payload, file, indent=2, ensure_ascii=False)


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="pri-lab",
    description="Local PRi R&D lab powered by Polars + igraph.",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  default_config = default_config_path()

  prepare_parser = subparsers.add_parser("prepare-pages", help="Flatten ecommerce JSON into pages.parquet.")
  prepare_parser.add_argument(
    "--input-json",
    type=Path,
    default=(default_config.parent / "../../../data/arborescence_ecommerce_auto.json").resolve(),
    help="Path to arborescence JSON input.",
  )
  prepare_parser.add_argument(
    "--output-pages",
    type=Path,
    default=(default_config.parent / "../artifacts/default/pages.parquet").resolve(),
    help="Output pages parquet path.",
  )
  prepare_parser.add_argument("--metrics-json", type=Path, default=None, help="Optional metrics output path.")

  outlinks_prepare_parser = subparsers.add_parser(
    "prepare-outlinks",
    help="Build pages/edges/anchors from a crawled all_outlinks.csv file.",
  )
  outlinks_prepare_parser.add_argument(
    "--input-csv",
    type=Path,
    default=(default_config.parent / "../../../DATA_OUTLIKNS/all_outlinks.csv").resolve(),
    help="Path to the outlinks CSV export.",
  )
  outlinks_prepare_parser.add_argument(
    "--output-pages",
    type=Path,
    default=(default_config.parent / "../artifacts/outlinks/pages.parquet").resolve(),
    help="Output pages parquet path.",
  )
  outlinks_prepare_parser.add_argument(
    "--output-edges",
    type=Path,
    default=(default_config.parent / "../artifacts/outlinks/edges.parquet").resolve(),
    help="Output edges parquet path.",
  )
  outlinks_prepare_parser.add_argument(
    "--output-anchor-candidates",
    type=Path,
    default=(default_config.parent / "../artifacts/outlinks/anchor_candidates.parquet").resolve(),
    help="Output anchor candidates parquet path.",
  )
  outlinks_prepare_parser.add_argument(
    "--source-host",
    type=str,
    default=None,
    help="Optional source host override (e.g. achat-or-et-argent.fr).",
  )
  outlinks_prepare_parser.add_argument(
    "--max-out-links-per-page",
    type=int,
    default=120,
    help="Global cap applied after dedup for each source page.",
  )
  outlinks_prepare_parser.add_argument("--metrics-json", type=Path, default=None, help="Optional metrics output path.")

  build_parser = subparsers.add_parser("build-edges", help="Build edge list parquet from pages parquet.")
  build_parser.add_argument(
    "--input-pages",
    type=Path,
    default=(default_config.parent / "../artifacts/default/pages.parquet").resolve(),
    help="Input pages parquet path.",
  )
  build_parser.add_argument(
    "--output-edges",
    type=Path,
    default=(default_config.parent / "../artifacts/default/edges.parquet").resolve(),
    help="Output edges parquet path.",
  )
  build_parser.add_argument("--disable-hierarchy-up", action="store_true", help="Disable child -> parent links.")
  build_parser.add_argument("--disable-hierarchy-down", action="store_true", help="Disable parent -> child links.")
  build_parser.add_argument("--disable-cluster-peer", action="store_true", help="Disable intra-cluster links.")
  build_parser.add_argument("--cluster-peer-k", type=int, default=8, help="Number of deterministic peers per page.")
  build_parser.add_argument("--max-out-links-per-page", type=int, default=40, help="Global cap after dedup.")
  build_parser.add_argument("--weight-hierarchy-up", type=float, default=1.0, help="Weight for hierarchy_up.")
  build_parser.add_argument("--weight-hierarchy-down", type=float, default=0.7, help="Weight for hierarchy_down.")
  build_parser.add_argument("--weight-cluster-peer", type=float, default=0.4, help="Weight for cluster_peer.")
  build_parser.add_argument("--metrics-json", type=Path, default=None, help="Optional metrics output path.")

  pri_parser = subparsers.add_parser("compute-pri", help="Compute internal PageRank from edge list.")
  pri_parser.add_argument(
    "--input-pages",
    type=Path,
    default=(default_config.parent / "../artifacts/default/pages.parquet").resolve(),
    help="Input pages parquet path.",
  )
  pri_parser.add_argument(
    "--input-edges",
    type=Path,
    default=(default_config.parent / "../artifacts/default/edges.parquet").resolve(),
    help="Input edges parquet path.",
  )
  pri_parser.add_argument(
    "--output-pri",
    type=Path,
    default=(default_config.parent / "../artifacts/default/pri_scores.parquet").resolve(),
    help="Output PRi parquet path.",
  )
  pri_parser.add_argument("--damping", type=float, default=0.85, help="PageRank damping factor.")
  pri_parser.add_argument(
    "--include-block-type",
    action="append",
    default=[],
    help="Limit PRi to one or many block types. Repeat flag to add values.",
  )
  pri_parser.add_argument("--unweighted", action="store_true", help="Ignore rule_weight when computing PRi.")
  pri_parser.add_argument("--metrics-json", type=Path, default=None, help="Optional metrics output path.")

  anchors_parser = subparsers.add_parser(
    "prepare-anchor-dataset",
    help="Generate anchor candidates dataset from pages + edges.",
  )
  anchors_parser.add_argument(
    "--input-pages",
    type=Path,
    default=(default_config.parent / "../artifacts/default/pages.parquet").resolve(),
    help="Input pages parquet path.",
  )
  anchors_parser.add_argument(
    "--input-edges",
    type=Path,
    default=(default_config.parent / "../artifacts/default/edges.parquet").resolve(),
    help="Input edges parquet path.",
  )
  anchors_parser.add_argument(
    "--output-anchor-candidates",
    type=Path,
    default=(default_config.parent / "../artifacts/default/anchor_candidates.parquet").resolve(),
    help="Output anchor candidates parquet path.",
  )
  anchors_parser.add_argument("--metrics-json", type=Path, default=None, help="Optional metrics output path.")

  dashboard_data_parser = subparsers.add_parser(
    "prepare-dashboard-data",
    help="Build dashboard analytics datasets (page_segments, url_metrics, segment_metrics).",
  )
  dashboard_data_parser.add_argument(
    "--workspace",
    type=Path,
    default=(default_config.parent / "../artifacts/default").resolve(),
    help="Workspace containing pages/edges/pri/(optional)anchors artifacts.",
  )
  dashboard_data_parser.add_argument("--metrics-json", type=Path, default=None, help="Optional metrics output path.")

  export_report_parser = subparsers.add_parser(
    "export-report",
    help="Export workspace artifacts to CSV and generate synthesis reports.",
  )
  export_report_parser.add_argument(
    "--workspace",
    type=Path,
    default=(default_config.parent / "../artifacts/default").resolve(),
    help="Workspace containing artifacts to export.",
  )
  export_report_parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Output directory for CSV and synthesis report files.",
  )
  export_report_parser.add_argument(
    "--skip-scenarios",
    action="store_true",
    help="Do not export scenario parquet files from workspace/scenarios.",
  )
  export_report_parser.add_argument("--metrics-json", type=Path, default=None, help="Optional metrics output path.")

  scenarios_parser = subparsers.add_parser(
    "model-anchor-scenarios",
    help="Model and run 3 anchor diversity scenarios with PRi comparison.",
  )
  scenarios_parser.add_argument(
    "--input-pages",
    type=Path,
    default=(default_config.parent / "../artifacts/default/pages.parquet").resolve(),
    help="Input pages parquet path.",
  )
  scenarios_parser.add_argument(
    "--input-edges",
    type=Path,
    default=(default_config.parent / "../artifacts/default/edges.parquet").resolve(),
    help="Input edges parquet path.",
  )
  scenarios_parser.add_argument(
    "--output-dir",
    type=Path,
    default=(default_config.parent / "../artifacts/default/scenarios").resolve(),
    help="Output directory for scenario datasets and results.",
  )
  scenarios_parser.add_argument(
    "--input-anchor-candidates",
    type=Path,
    default=None,
    help="Optional anchor candidates parquet. If omitted, anchors are generated from pages + edges.",
  )
  scenarios_parser.add_argument("--damping", type=float, default=0.85, help="PageRank damping factor.")
  scenarios_parser.add_argument("--metrics-json", type=Path, default=None, help="Optional metrics output path.")

  dashboard_parser = subparsers.add_parser(
    "dashboard",
    help="Launch visual dashboard for PRi artifacts.",
  )
  dashboard_parser.add_argument(
    "--workspace",
    type=Path,
    default=(default_config.parent / "../artifacts/default").resolve(),
    help="Workspace containing pages/edges/pri/scenarios artifacts.",
  )
  dashboard_parser.add_argument("--host", type=str, default="127.0.0.1", help="Dashboard bind host.")
  dashboard_parser.add_argument("--port", type=int, default=8501, help="Dashboard bind port.")

  run_parser = subparsers.add_parser("run-experiment", help="Run the full pipeline from a TOML config.")
  run_parser.add_argument(
    "--config",
    type=Path,
    default=default_config,
    help="Path to TOML experiment config.",
  )
  run_parser.add_argument("--workspace", type=Path, default=None, help="Optional workspace override.")

  run_outlinks_parser = subparsers.add_parser(
    "run-outlinks-analysis",
    help="Run full PRi/CheiRank/anchors/scenarios pipeline from an outlinks CSV.",
  )
  run_outlinks_parser.add_argument(
    "--input-csv",
    type=Path,
    default=(default_config.parent / "../../../DATA_OUTLIKNS/all_outlinks.csv").resolve(),
    help="Path to the outlinks CSV export.",
  )
  run_outlinks_parser.add_argument(
    "--workspace",
    type=Path,
    default=(default_config.parent / "../artifacts/outlinks").resolve(),
    help="Workspace for generated pages/edges/pri/anchors/scenarios artifacts.",
  )
  run_outlinks_parser.add_argument(
    "--source-host",
    type=str,
    default=None,
    help="Optional source host override (e.g. achat-or-et-argent.fr).",
  )
  run_outlinks_parser.add_argument(
    "--max-out-links-per-page",
    type=int,
    default=120,
    help="Global cap applied after dedup for each source page.",
  )
  run_outlinks_parser.add_argument("--damping", type=float, default=0.85, help="PageRank damping factor.")

  benchmark_parser = subparsers.add_parser(
    "benchmark-synthetic",
    help="Generate synthetic graph data and run PRi for a performance smoke check.",
  )
  benchmark_parser.add_argument(
    "--config",
    type=Path,
    default=default_config,
    help="Path to TOML experiment config.",
  )
  benchmark_parser.add_argument("--workspace", type=Path, default=None, help="Optional workspace override.")
  benchmark_parser.add_argument("--node-count", type=int, default=None, help="Number of synthetic nodes.")
  benchmark_parser.add_argument("--target-edge-count", type=int, default=None, help="Target synthetic edges.")
  benchmark_parser.add_argument("--cluster-count", type=int, default=None, help="Synthetic cluster count.")
  benchmark_parser.add_argument("--damping", type=float, default=0.85, help="PageRank damping factor.")

  return parser


if __name__ == "__main__":
  main()
