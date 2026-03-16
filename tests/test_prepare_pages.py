from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pri_lab.pipeline import flatten_arborescence, infer_cluster_thematique, prepare_pages


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "arbo_small.json"


def test_flatten_arborescence_builds_expected_paths_and_parents() -> None:
  payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
  rows = flatten_arborescence(payload)
  rows_by_path = {row["path"]: row for row in rows}

  assert "catalog/" in rows_by_path
  assert "catalog/brand-a/" in rows_by_path
  assert "catalog/brand-a/_meta" in rows_by_path
  assert "catalog/brand-a/modeles/model-1/" in rows_by_path
  assert "catalog/brand-a/modeles/model-2/" in rows_by_path

  assert rows_by_path["catalog/brand-a/"]["parent_path"] == "catalog/"
  assert rows_by_path["catalog/brand-a/_meta"]["parent_path"] == "catalog/brand-a/"
  assert rows_by_path["catalog/brand-a/modeles/model-1/"]["parent_path"] == "catalog/brand-a/modeles/"

  assert rows_by_path["catalog/"]["depth"] == 1
  assert rows_by_path["catalog/brand-a/modeles/model-1/"]["depth"] == 4
  assert rows_by_path["services/financement/"]["section"] == "services"


def test_cluster_thematique_is_deterministic() -> None:
  path = "catalog/brand-a/modeles/model-1/"
  expected = infer_cluster_thematique(path)

  for _ in range(5):
    assert infer_cluster_thematique(path) == expected


def test_prepare_pages_writes_expected_schema(tmp_path: Path) -> None:
  output_pages = tmp_path / "pages.parquet"
  summary = prepare_pages(
    input_json_path=FIXTURE_PATH,
    output_pages_path=output_pages,
  )

  assert summary["page_count"] > 0
  assert output_pages.exists()

  pages_df = pl.read_parquet(output_pages)
  assert pages_df.columns == [
    "page_id",
    "path",
    "parent_path",
    "depth",
    "section",
    "cluster_thematique",
    "is_leaf",
  ]
  assert pages_df.dtypes == [
    pl.Int32,
    pl.Utf8,
    pl.Utf8,
    pl.Int16,
    pl.Utf8,
    pl.Utf8,
    pl.Boolean,
  ]

  page_ids = pages_df.get_column("page_id").to_list()
  assert page_ids == list(range(1, len(page_ids) + 1))
