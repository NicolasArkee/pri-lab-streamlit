from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class PathsConfig:
  input_json: Path
  workspace: Path
  pages_parquet: Path
  edges_parquet: Path
  pri_scores_parquet: Path
  run_metrics_json: Path
  experiment_log_parquet: Path
  maillage_model: Path | None = None


@dataclass(frozen=True)
class BlockConfig:
  """Configuration for a single LBC maillage block."""
  enabled: bool = True
  weight: float = 1.0
  max_links: int = 20
  present_on: list[str] = field(default_factory=lambda: ["all"])


@dataclass(frozen=True)
class PageGenerationConfig:
  """Target page counts per template for LBC graph generation."""
  target_total_pages: int = 2_000_000
  c_category_pages: int = 33_000
  cl_geo_pages: int = 1_300_000
  ck_keyword_pages: int = 550_000
  ckl_hybrid_pages: int = 100_000
  l_location_pages: int = 3_350
  ad_sample_pages: int = 0
  misc_pages: int = 60
  regions: int = 37
  departements: int = 99
  villes_top: int = 500
  categories: int = 80
  super_categories: int = 12


@dataclass(frozen=True)
class RulesConfig:
  enable_hierarchy_up: bool
  enable_hierarchy_down: bool
  enable_cluster_peer: bool
  cluster_peer_k: int
  max_out_links_per_page: int
  weight_hierarchy_up: float
  weight_hierarchy_down: float
  weight_cluster_peer: float


@dataclass(frozen=True)
class PagerankConfig:
  damping: float
  included_block_types: list[str]
  use_weights: bool


@dataclass(frozen=True)
class BenchmarkConfig:
  node_count: int
  target_edge_count: int
  cluster_count: int


@dataclass(frozen=True)
class ExperimentConfig:
  paths: PathsConfig
  rules: RulesConfig
  pagerank: PagerankConfig
  benchmark: BenchmarkConfig
  blocks: dict[str, BlockConfig] = field(default_factory=dict)
  page_generation: PageGenerationConfig = field(default_factory=PageGenerationConfig)

  def to_dict(self) -> dict[str, object]:
    config = asdict(self)
    return _convert_paths_to_str(config)


def default_config_path() -> Path:
  return _repo_root(Path(__file__).resolve()) / "labs" / "pri" / "configs" / "default.toml"


def load_experiment_config(
  config_path: Path | None = None,
  workspace_override: Path | None = None,
) -> ExperimentConfig:
  raw_config_path = config_path or default_config_path()
  resolved_config_path = raw_config_path.expanduser().resolve()
  raw = _read_toml(resolved_config_path)
  config_dir = resolved_config_path.parent

  paths = raw.get("paths", {})
  rules = raw.get("rules", {})
  pagerank = raw.get("pagerank", {})
  benchmark = raw.get("benchmark", {})
  blocks_raw = raw.get("blocks", {})
  page_gen_raw = raw.get("page_generation", {})

  workspace = (
    workspace_override.expanduser().resolve()
    if workspace_override is not None
    else _resolve_path(config_dir, str(paths.get("workspace", "../artifacts/default")))
  )

  maillage_model_raw = paths.get("maillage_model")
  maillage_model_path = (
    _resolve_path(config_dir, str(maillage_model_raw))
    if maillage_model_raw is not None
    else None
  )

  paths_config = PathsConfig(
    input_json=_resolve_path(config_dir, str(paths.get("input_json", "../../../data/arborescence_ecommerce_auto.json"))),
    workspace=workspace,
    pages_parquet=_resolve_artifact_path(workspace, paths.get("pages_parquet", "pages.parquet")),
    edges_parquet=_resolve_artifact_path(workspace, paths.get("edges_parquet", "edges.parquet")),
    pri_scores_parquet=_resolve_artifact_path(workspace, paths.get("pri_scores_parquet", "pri_scores.parquet")),
    run_metrics_json=_resolve_artifact_path(workspace, paths.get("run_metrics_json", "run_metrics.json")),
    experiment_log_parquet=_resolve_artifact_path(workspace, paths.get("experiment_log_parquet", "experiment_runs.parquet")),
    maillage_model=maillage_model_path,
  )

  rules_config = RulesConfig(
    enable_hierarchy_up=bool(rules.get("enable_hierarchy_up", True)),
    enable_hierarchy_down=bool(rules.get("enable_hierarchy_down", True)),
    enable_cluster_peer=bool(rules.get("enable_cluster_peer", True)),
    cluster_peer_k=int(rules.get("cluster_peer_k", 8)),
    max_out_links_per_page=int(rules.get("max_out_links_per_page", 40)),
    weight_hierarchy_up=float(rules.get("weight_hierarchy_up", 1.0)),
    weight_hierarchy_down=float(rules.get("weight_hierarchy_down", 0.7)),
    weight_cluster_peer=float(rules.get("weight_cluster_peer", 0.4)),
  )

  pagerank_config = PagerankConfig(
    damping=float(pagerank.get("damping", 0.85)),
    included_block_types=[str(value) for value in pagerank.get("included_block_types", ["hierarchy_up", "hierarchy_down", "cluster_peer"])],
    use_weights=bool(pagerank.get("use_weights", True)),
  )

  benchmark_config = BenchmarkConfig(
    node_count=int(benchmark.get("node_count", 200_000)),
    target_edge_count=int(benchmark.get("target_edge_count", 4_000_000)),
    cluster_count=int(benchmark.get("cluster_count", 2_000)),
  )

  # Parse LBC block configs
  blocks: dict[str, BlockConfig] = {}
  for block_name, block_data in blocks_raw.items():
    if isinstance(block_data, dict):
      blocks[block_name] = BlockConfig(
        enabled=bool(block_data.get("enabled", True)),
        weight=float(block_data.get("weight", 1.0)),
        max_links=int(block_data.get("max_links", 20)),
        present_on=[str(v) for v in block_data.get("present_on", ["all"])],
      )

  # Parse page generation config
  page_generation = PageGenerationConfig(
    target_total_pages=int(page_gen_raw.get("target_total_pages", 2_000_000)),
    c_category_pages=int(page_gen_raw.get("c_category_pages", 33_000)),
    cl_geo_pages=int(page_gen_raw.get("cl_geo_pages", 1_300_000)),
    ck_keyword_pages=int(page_gen_raw.get("ck_keyword_pages", 550_000)),
    ckl_hybrid_pages=int(page_gen_raw.get("ckl_hybrid_pages", 100_000)),
    l_location_pages=int(page_gen_raw.get("l_location_pages", 3_350)),
    ad_sample_pages=int(page_gen_raw.get("ad_sample_pages", 0)),
    misc_pages=int(page_gen_raw.get("misc_pages", 60)),
    regions=int(page_gen_raw.get("regions", 37)),
    departements=int(page_gen_raw.get("departements", 99)),
    villes_top=int(page_gen_raw.get("villes_top", 500)),
    categories=int(page_gen_raw.get("categories", 80)),
    super_categories=int(page_gen_raw.get("super_categories", 12)),
  )

  return ExperimentConfig(
    paths=paths_config,
    rules=rules_config,
    pagerank=pagerank_config,
    benchmark=benchmark_config,
    blocks=blocks,
    page_generation=page_generation,
  )


def _resolve_artifact_path(workspace: Path, value: object) -> Path:
  raw = str(value)
  artifact_path = Path(raw).expanduser()
  if artifact_path.is_absolute():
    return artifact_path.resolve()
  return (workspace / artifact_path).resolve()


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
  path = Path(raw_path).expanduser()
  if path.is_absolute():
    return path.resolve()
  return (base_dir / path).resolve()


def _read_toml(path: Path) -> dict[str, object]:
  with path.open("rb") as file:
    return tomllib.load(file)


def _repo_root(start: Path) -> Path:
  for candidate in [start, *start.parents]:
    if (candidate / ".git").exists():
      return candidate
  raise RuntimeError("Unable to locate repository root from current file location.")


def _convert_paths_to_str(data: dict[str, object]) -> dict[str, object]:
  for key, value in data.items():
    if isinstance(value, Path):
      data[key] = str(value)
    elif isinstance(value, dict):
      data[key] = _convert_paths_to_str(value)
  return data
