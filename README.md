# PRi Lab (Local R&D)

`labs/pri` is a local, decoupled data-science workspace to model internal-link graphs and compute PRi (internal PageRank) at high volume with Polars + igraph.

## 1. Setup

```bash
cd labs/pri
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Run the full pipeline

```bash
pri-lab run-experiment
```

Default config: `labs/pri/configs/default.toml`  
Default workspace: `labs/pri/artifacts/default/`

## 3. Run commands independently

```bash
pri-lab prepare-pages
pri-lab prepare-outlinks
pri-lab build-edges
pri-lab compute-pri
pri-lab prepare-anchor-dataset
pri-lab prepare-dashboard-data
pri-lab export-report
pri-lab model-anchor-scenarios
pri-lab dashboard
```

You can scope PRi to selected blocks:

```bash
pri-lab compute-pri \
  --include-block-type hierarchy_up \
  --include-block-type cluster_peer
```

## 4. Data contracts

Generated files:

- `pages.parquet`: `page_id:int32`, `path:utf8`, `parent_path:utf8|null`, `depth:int16`, `section:utf8`, `cluster_thematique:utf8`, `is_leaf:bool`
- `edges.parquet`: `source_id:int32`, `target_id:int32`, `block_type:utf8`, `rule_weight:float32`
- `pri_scores.parquet`: `page_id:int32`, `pri_score:float64`, `rank:int32`, `cheirank_score:float64`, `cheirank_rank:int32`, `juice_potential_score:float64`, `out_degree:int32`, `in_degree:int32`, `is_low_power:bool`, `can_give_juice:bool`
- `run_metrics.json`: timing, peak memory, node/edge counts, active config, stage details
- `anchor_candidates.parquet`: `source_id`, `target_id`, `block_type`, `rule_weight`, `anchor_type`, `anchor_text`, `anchor_token_count`
- `page_segments.parquet`: `page_id:int32`, `level:int8`, `segment:utf8`, `segment_path:utf8`, `parent_segment_path:utf8|null`, `depth:int16`, `is_terminal:bool`
- `url_metrics.parquet`: métriques URL enrichies (PRi/CheiRank, degrés, jus, liens entrants/sortants, diversité ancres)
- `segment_metrics.parquet`: agrégats par segment URL (`page_count`, `pri_*`, `cheirank_*`, ratios puissance, degrés moyens, diversité ancres)

## 5. Rules implemented (v1)

- `hierarchy_up`: `child -> parent`
- `hierarchy_down`: `parent -> child`
- `cluster_peer`: deterministic K-neighbor links inside `cluster_thematique`
- Global cap `max_out_links_per_page` applied after dedup, with priority:
  - `hierarchy_up` > `hierarchy_down` > `cluster_peer`

## 6. Synthetic benchmark

```bash
pri-lab benchmark-synthetic --node-count 200000 --target-edge-count 4000000
```

This command writes synthetic pages/edges + PRi results in a dedicated workspace and emits benchmark metrics.

## 7. Anchor diversity scenarios

`model-anchor-scenarios` prepares anchors and executes three deterministic scenarios:

- `scenario_1_exact_focus`
- `scenario_2_balanced_mix`
- `scenario_3_diversity_first`

For each scenario, the lab exports:

- scenario edge dataset with selected anchors and `anchor_diversity_score`
- PRi result parquet
- global comparison file `scenario_pri_comparison.parquet`

You can inject a real anchor dataset:

```bash
pri-lab model-anchor-scenarios \
  --input-pages labs/pri/artifacts/outlinks/pages.parquet \
  --input-edges labs/pri/artifacts/outlinks/edges.parquet \
  --input-anchor-candidates labs/pri/artifacts/outlinks/anchor_candidates.parquet \
  --output-dir labs/pri/artifacts/outlinks/scenarios
```

## 8. Outlinks CSV full analysis

Run the full PRi/CheiRank/anchors/scenarios pipeline directly from an outlinks crawl export:

```bash
pri-lab run-outlinks-analysis \
  --input-csv DATA_OUTLIKNS/all_outlinks.csv \
  --workspace labs/pri/artifacts/outlinks
```

This command now also generates dashboard datasets (`page_segments`, `url_metrics`, `segment_metrics`).

## 9. Exports CSV + rapport de synthèse

Tu peux exporter tous les artefacts parquet du workspace en CSV, avec un rapport de synthèse JSON/Markdown:

```bash
pri-lab export-report \
  --workspace labs/pri/artifacts/outlinks \
  --output-dir labs/pri/artifacts/outlinks/exports/latest
```

Sorties:

- `exports/latest/csv/*.csv` (pages, edges, pri_scores, anchors, segments, scénarios si présents)
- `exports/latest/synthesis_report.json`
- `exports/latest/synthesis_report.md`

Dans le dashboard, la sidebar contient aussi des boutons d'export des vues filtrées (`URL metrics`, `edges`, `anchors`) et un rapport synthétique markdown.

## 10. Tests

```bash
pytest
```

## 11. Visual dashboard

Launch visual dashboard from generated artifacts:

```bash
pri-lab dashboard --workspace labs/pri/artifacts/default --host 127.0.0.1 --port 8501
```

Then open: `http://127.0.0.1:8501`
