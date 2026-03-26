"""Generate a realistic ~2M-page LBC graph from the knowledge_graph taxonomy.

The generator creates pages that mirror leboncoin.fr's real URL structure:
  /                         homepage
  /c/{category}             category listing  (L1/L2/L3 + facets)
  /cl/{category}/{geo}      category x geo
  /ck/{category}/{kw}       Verbolia keywords
  /ckl/{cat}/{kw}/{geo}/{f} keyword + geo + facet (hybrid)
  /l/{geo}                  location-only pages
  /dc/{slug}                legal pages
  /guide/{slug}             editorial guides

Each page carries a `template` column so edge-building knows which
maillage blocs are active on that page.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Taxonomy constants (from knowledge_graph.json + estimation_pages.json)
# ---------------------------------------------------------------------------

SUPER_CATEGORIES: list[str] = [
    "_vehicules_",
    "_immobilier_",
    "_mode_",
    "_maison_jardin_",
    "_multimedia_",
    "_loisirs_",
    "_electronique_",
    "_famille_",
    "_animaux_",
    "_materiel_professionnel_",
    "_services_",
    "_emploi_",
]

# Top-30 categories that generate the bulk of traffic
TOP_CATEGORIES: list[str] = [
    "voitures",
    "locations",
    "ventes_immobilieres",
    "locations_saisonnieres",
    "motos",
    "utilitaires",
    "caravaning",
    "colocations",
    "bureaux_commerces",
    "vetements",
    "chaussures",
    "montres_bijoux",
    "accessoires_bagagerie",
    "ameublement",
    "electromenager",
    "decoration",
    "bricolage",
    "jardin_plantes",
    "ordinateurs",
    "telephones_objets_connectes",
    "consoles",
    "jeux_video",
    "photo_audio_video",
    "sport_plein_air",
    "velos",
    "jeux_jouets",
    "livres",
    "equipement_bebe",
    "cd_musique",
    "instruments_de_musique",
]

# Extra categories to reach ~80 total
EXTRA_CATEGORIES: list[str] = [
    "nautisme",
    "equipement_auto",
    "equipement_moto",
    "equipement_caravaning",
    "equipement_nautisme",
    "equipements_velos",
    "linge_de_maison",
    "arts_de_la_table",
    "papeterie_fournitures_scolaires",
    "accessoires_informatique",
    "tablettes_liseuses",
    "accessoires_telephone_objets_connectes",
    "loisirs_creatifs",
    "modelisme",
    "collection",
    "dvd_films",
    "vins_gastronomie",
    "antiquites",
    "mobilier_enfant",
    "vetements_bebe",
    "animaux",
    "accessoires_animaux",
    "materiel_agricole",
    "materiel_industriel",
    "outillage",
    "fourniture_bureau",
    "fournitures_restaurant",
    "transport_manutention",
    "btp_chantier_gros_oeuvre",
    "equipements_industriels",
    "offres_emploi",
    "services_aux_entreprises",
    "billetterie",
    "cours_particuliers",
    "evenements",
    "services_a_la_personne",
    "covoiturage",
    "camping",
    "chambres_d_hotes",
    "equipement_camping",
    "pieces_detachees_auto",
    "pieces_detachees_moto",
    "materiel_medical",
    "materiel_de_puericulture",
    "produits_bebe",
    "accessoires_poussette",
    "jouets_plein_air",
    "figurines",
    "maquettes",
    "musique_vinyles",
]

ALL_CATEGORIES: list[str] = TOP_CATEGORIES + EXTRA_CATEGORIES

REGIONS: list[str] = [
    "alsace", "aquitaine", "auvergne", "basse-normandie", "bourgogne",
    "bretagne", "centre", "champagne-ardenne", "corse", "franche-comte",
    "guadeloupe", "guyane", "haute-normandie", "ile-de-france",
    "languedoc-roussillon", "limousin", "lorraine", "martinique",
    "midi-pyrenees", "nord-pas-de-calais", "normandie", "pays de la loire",
    "picardie", "poitou-charentes", "provence-alpes-cote d'azur",
    "reunion", "rhone-alpes", "auvergne-rhone-alpes",
    "bourgogne-franche-comte", "centre-val de loire", "grand est",
    "hauts-de-france", "nouvelle-aquitaine", "occitanie",
    "nouvelle-caledonie", "mayotte", "saint-pierre-et-miquelon",
]

# Facet types by vertical (realistic subset)
FACET_TYPES: dict[str, list[str]] = {
    "voitures": ["u_car_brand", "u_car_model", "fuel", "vehicle_type", "gearbox", "critair"],
    "motos": ["u_moto_brand", "u_moto_model", "moto_type", "cubic_capacity"],
    "locations": ["real_estate_type", "rooms", "furnished"],
    "ventes_immobilieres": ["real_estate_type", "rooms", "energy_rate"],
    "telephones_objets_connectes": ["phone_brand", "phone_memory"],
    "vetements": ["clothing_type", "clothing_size", "clothing_color"],
    "chaussures": ["shoe_type", "shoe_size"],
    "ameublement": ["furniture_type"],
    "electromenager": ["home_appliance_product"],
    "velos": ["bicycle_type"],
    "jeux_jouets": ["toy_type"],
}

# top brand values per facet (just enough for combinatorics)
FACET_VALUES: dict[str, list[str]] = {
    "u_car_brand": ["RENAULT", "PEUGEOT", "CITROEN", "BMW", "MERCEDES-BENZ", "VOLKSWAGEN", "AUDI", "TOYOTA", "FORD", "OPEL", "FIAT", "NISSAN", "DACIA", "HYUNDAI", "KIA"],
    "fuel": ["1", "2", "3", "4", "7"],  # essence, diesel, gpl, electrique, hybride
    "vehicle_type": ["berline", "4x4", "break", "monospace", "citadine", "coupe", "cabriolet"],
    "gearbox": ["1", "2"],  # manual, auto
    "u_moto_brand": ["HONDA", "YAMAHA", "KAWASAKI", "BMW", "SUZUKI", "HARLEY-DAVIDSON", "TRIUMPH", "KTM", "DUCATI"],
    "moto_type": ["moto", "scooter", "quad"],
    "real_estate_type": ["1", "2", "3", "4", "5"],  # maison, appart, terrain, parking, autre
    "rooms": ["1", "2", "3", "4", "5"],
    "phone_brand": ["apple", "samsung", "huawei", "xiaomi", "sony", "oneplus", "google"],
    "clothing_type": ["1", "2", "3", "4"],  # femme, maternité, homme, enfant
    "shoe_type": ["1", "2", "3"],
    "bicycle_type": ["course", "vtt", "electrique", "ville", "vtc", "enfant"],
    "furniture_type": ["armoire", "canape", "lit", "table", "fauteuil", "buffet"],
    "toy_type": ["jeuxdesociete", "poupees", "jeuxdeconstruction", "puzzle"],
}

DC_PAGES: list[str] = [
    "accessibilite", "avis_utilisateurs", "cgu", "cgv",
    "charte_de_bonne_conduite", "cookies", "paiement_en_plusieurs_fois",
    "rules", "vos_droits_et_obligations",
]

GUIDE_PAGES: list[str] = [
    "vacances", "biarritz", "gorges-du-verdon", "corse", "bretagne",
    "camargue", "dordogne", "pays-basque", "normandie", "ardeche",
]


def _stable_hash(s: str) -> int:
    """Deterministic hash for reproducible sampling."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Page generation
# ---------------------------------------------------------------------------

def generate_lbc_pages(
    output_path: Path,
    target_total: int = 2_000_000,
    compression: str = "zstd",
) -> dict[str, object]:
    """Generate a realistic LBC page set of ~target_total pages.

    Returns summary dict with counts per template.
    """
    rows: list[dict[str, str | int | bool | None]] = []

    # ── homepage ──
    rows.append(_page("/", None, "homepage", "root"))

    # ── /dc/ legal ──
    for slug in DC_PAGES:
        rows.append(_page(f"/dc/{slug}", "/", "dc", "dc"))

    # ── /guide/ ──
    for slug in GUIDE_PAGES:
        rows.append(_page(f"/guide/{slug}", "/guide/vacances" if slug != "vacances" else "/", "guide", "guide"))

    # ── super-categories /c/_vertical_ ──
    for sc in SUPER_CATEGORIES:
        rows.append(_page(f"/c/{sc}", "/", "c_L0", f"c:{sc}"))

    # ── /c/ categories (L1) ──
    for cat in ALL_CATEGORIES:
        parent_sc = super_category_for(cat)
        rows.append(_page(f"/c/{cat}", f"/c/{parent_sc}", "c_L1", f"c:{cat}"))

    # ── /c/ facets (L2 and L3) — target ~33K ──
    c_budget = 33_000 - len(SUPER_CATEGORIES) - len(ALL_CATEGORIES)
    c_rows = _generate_c_facet_pages(c_budget)
    rows.extend(c_rows)

    # ── /l/ location pages — ~3.3K ──
    l_rows = _generate_l_pages()
    rows.extend(l_rows)

    # ── /cl/ category x geo — ~1.3M (dominant template) ──
    cl_budget = min(1_300_000, int(target_total * 0.65))
    cl_rows = _generate_cl_pages(cl_budget)
    rows.extend(cl_rows)

    # ── /ck/ Verbolia keywords — ~550K ──
    ck_budget = min(550_000, int(target_total * 0.275))
    ck_rows = _generate_ck_pages(ck_budget)
    rows.extend(ck_rows)

    # ── /ckl/ hybrid keyword + geo + facet — fill remainder ──
    current = len(rows)
    ckl_budget = max(0, target_total - current)
    ckl_rows = _generate_ckl_pages(ckl_budget)
    rows.extend(ckl_rows)

    # ── Build DataFrame ──
    pages_df = (
        pl.DataFrame(rows)
        .unique(subset=["path"])
        .sort("path")
        .with_row_index(name="page_id", offset=1)
        .with_columns(
            pl.col("page_id").cast(pl.Int32),
            pl.col("depth").cast(pl.Int16),
            pl.col("is_leaf").cast(pl.Boolean),
        )
        .select(
            "page_id", "path", "parent_path", "depth",
            "section", "cluster_thematique", "template", "is_leaf",
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages_df.write_parquet(output_path, compression=compression)

    # Summary
    template_counts = (
        pages_df.group_by("template").len()
        .sort("len", descending=True)
        .to_dicts()
    )
    return {
        "output_pages_parquet": str(output_path),
        "page_count": pages_df.height,
        "template_counts": template_counts,
    }


def _page(
    path: str,
    parent_path: str | None,
    template: str,
    cluster: str,
) -> dict[str, Any]:
    segments = [s for s in path.strip("/").split("/") if s]
    return {
        "path": path,
        "parent_path": parent_path,
        "depth": len(segments),
        "section": segments[0] if segments else "root",
        "cluster_thematique": cluster,
        "template": template,
        "is_leaf": True,
    }


def _build_category_to_verticale() -> dict[str, str]:
    """Build category → super-category mapping once."""
    mapping: dict[str, str] = {}
    groups: dict[str, set[str]] = {
        "_vehicules_": {"voitures", "motos", "utilitaires", "caravaning", "nautisme",
                        "equipement_auto", "equipement_moto", "equipement_caravaning",
                        "equipement_nautisme", "equipements_velos", "velos",
                        "pieces_detachees_auto", "pieces_detachees_moto"},
        "_immobilier_": {"locations", "ventes_immobilieres", "colocations", "bureaux_commerces",
                         "locations_saisonnieres", "chambres_d_hotes"},
        "_mode_": {"vetements", "chaussures", "montres_bijoux", "accessoires_bagagerie",
                   "vetements_bebe"},
        "_maison_jardin_": {"ameublement", "electromenager", "decoration", "bricolage",
                            "jardin_plantes", "linge_de_maison", "arts_de_la_table",
                            "papeterie_fournitures_scolaires"},
        "_electronique_": {"ordinateurs", "telephones_objets_connectes", "consoles",
                           "jeux_video", "photo_audio_video", "accessoires_informatique",
                           "tablettes_liseuses", "accessoires_telephone_objets_connectes"},
        "_famille_": {"equipement_bebe", "mobilier_enfant", "jeux_jouets",
                      "materiel_de_puericulture", "produits_bebe", "accessoires_poussette",
                      "jouets_plein_air"},
        "_loisirs_": {"sport_plein_air", "livres", "cd_musique", "instruments_de_musique",
                      "loisirs_creatifs", "modelisme", "collection", "dvd_films",
                      "vins_gastronomie", "antiquites", "billetterie", "figurines",
                      "maquettes", "musique_vinyles", "camping", "equipement_camping"},
    }
    for verticale, cats in groups.items():
        for c in cats:
            mapping[c] = verticale
    return mapping


CATEGORY_TO_VERTICALE: dict[str, str] = _build_category_to_verticale()


def super_category_for(cat: str) -> str:
    """Map a category to its super-category."""
    return CATEGORY_TO_VERTICALE.get(cat, "_services_")


# ---------------------------------------------------------------------------
# Template-specific generators
# ---------------------------------------------------------------------------

def _generate_c_facet_pages(budget: int) -> list[dict[str, Any]]:
    """Generate /c/{cat}/{facet}:{value} and combo pages."""
    rows: list[dict[str, Any]] = []
    for cat in ALL_CATEGORIES:
        facets = FACET_TYPES.get(cat, [])
        if not facets:
            # generic: 1 facet with 3 values
            facets = ["generic_facet"]
        for ft in facets:
            values = FACET_VALUES.get(ft, ["val1", "val2", "val3"])
            for v in values:
                rows.append(_page(
                    f"/c/{cat}/{ft}:{v}",
                    f"/c/{cat}",
                    "c_L2",
                    f"c:{cat}",
                ))
                if len(rows) >= budget:
                    return rows
            # combos: take first 2 facet types and cross
            if len(facets) > 1:
                other_ft = [f for f in facets if f != ft][:1]
                for oft in other_ft:
                    ovalues = FACET_VALUES.get(oft, ["val1"])[:3]
                    for v in values[:5]:
                        for ov in ovalues:
                            rows.append(_page(
                                f"/c/{cat}/{ft}:{v}+{oft}:{ov}",
                                f"/c/{cat}/{ft}:{v}",
                                "c_L3",
                                f"c:{cat}",
                            ))
                            if len(rows) >= budget:
                                return rows
    return rows[:budget]


def _generate_l_pages() -> list[dict[str, Any]]:
    """Generate /l/ location-only pages."""
    rows: list[dict[str, Any]] = []
    for region in REGIONS:
        rows.append(_page(f"/l/rp_{region}", "/", "l", f"l:{region}"))
    return rows


def _generate_cl_pages(budget: int) -> list[dict[str, Any]]:
    """Generate /cl/{cat}/{geo} pages — the dominant template.

    Strategy: for each category, generate region + département + top villes.
    """
    rows: list[dict[str, Any]] = []

    for cat in ALL_CATEGORIES:
        # regions
        for region in REGIONS:
            rows.append(_page(
                f"/cl/{cat}/rp_{region}",
                f"/c/{cat}",
                "cl",
                f"cl:{cat}",
            ))
            if len(rows) >= budget:
                return rows

        # départements (use stable hash to pick ~60 per category)
        for dept_num in range(1, 100):
            dept_name = f"dp_{dept_num:02d}"
            rows.append(_page(
                f"/cl/{cat}/{dept_name}",
                f"/c/{cat}",
                "cl",
                f"cl:{cat}",
            ))
            if len(rows) >= budget:
                return rows

        # villes — generate synthetic cp_ pages to fill budget
        # Real LBC has 16K villes; we generate proportionally per cat
        villes_per_cat = max(1, (budget - len(rows)) // max(1, len(ALL_CATEGORIES) - ALL_CATEGORIES.index(cat)))
        villes_per_cat = min(villes_per_cat, 16_164)
        for v in range(villes_per_cat):
            cp = f"{10000 + (_stable_hash(f'{cat}_{v}') % 90000):05d}"
            ville = f"ville{v}"
            rows.append(_page(
                f"/cl/{cat}/cp_{ville}_{cp}",
                f"/c/{cat}",
                "cl",
                f"cl:{cat}",
            ))
            if len(rows) >= budget:
                return rows

        # geo + facet pages for top categories
        if cat in FACET_TYPES:
            for region in REGIONS[:10]:
                for ft in FACET_TYPES[cat][:2]:
                    for fv in FACET_VALUES.get(ft, [])[:5]:
                        rows.append(_page(
                            f"/cl/{cat}/rp_{region}/{ft}:{fv}",
                            f"/cl/{cat}/rp_{region}",
                            "cl",
                            f"cl:{cat}",
                        ))
                        if len(rows) >= budget:
                            return rows

    return rows[:budget]


def _generate_ck_pages(budget: int) -> list[dict[str, Any]]:
    """Generate /ck/{cat}/{keyword-slug} Verbolia keyword pages."""
    rows: list[dict[str, Any]] = []
    kw_per_cat = max(1, budget // len(ALL_CATEGORIES))

    for cat in ALL_CATEGORIES:
        for i in range(kw_per_cat):
            slug = f"kw-{_stable_hash(f'{cat}_{i}') % 999999:06d}"
            rows.append(_page(
                f"/ck/{cat}/{slug}",
                f"/c/{cat}",
                "ck",
                f"ck:{cat}",
            ))
            if len(rows) >= budget:
                return rows
    return rows[:budget]


def _generate_ckl_pages(budget: int) -> list[dict[str, Any]]:
    """Generate /ckl/{cat}/{kw}/{geo}/{facet}:{value} hybrid pages."""
    rows: list[dict[str, Any]] = []
    if budget <= 0:
        return rows

    cats_with_facets = [c for c in ALL_CATEGORIES if c in FACET_TYPES]
    if not cats_with_facets:
        cats_with_facets = ALL_CATEGORIES[:10]

    per_cat = max(1, budget // len(cats_with_facets))

    for cat in cats_with_facets:
        facets = FACET_TYPES.get(cat, [])[:2]
        count = 0
        for ft in facets:
            values = FACET_VALUES.get(ft, ["val1"])[:5]
            for region in REGIONS[:15]:
                for fv in values:
                    for kw_i in range(max(1, per_cat // (len(facets) * 15 * len(values)))):
                        slug = f"kw-{_stable_hash(f'{cat}_{ft}_{fv}_{region}_{kw_i}') % 999999:06d}"
                        cp = f"{10000 + (_stable_hash(f'{region}_{kw_i}') % 90000):05d}"
                        rows.append(_page(
                            f"/ckl/{cat}/{slug}/cp_{region}_{cp}/{ft}:{fv}",
                            f"/ck/{cat}/{slug}",
                            "ckl",
                            f"ckl:{cat}",
                        ))
                        count += 1
                        if count >= per_cat or len(rows) >= budget:
                            break
                    if count >= per_cat or len(rows) >= budget:
                        break
                if count >= per_cat or len(rows) >= budget:
                    break
            if count >= per_cat or len(rows) >= budget:
                break
        if len(rows) >= budget:
            return rows[:budget]

    return rows[:budget]
