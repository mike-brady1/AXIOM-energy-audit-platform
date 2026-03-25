"""
AXIOM — Step 1: ADEME Multi-Dataset Loader
==========================================
Loads, validates and cleans all three ADEME/OPERAT datasets:

  1. COMMUNE   — consommation-tertiaire-commune.csv
                 Geographic aggregation: commune × climate zone × year
                 → Decret Tertiaire compliance maps

  2. ACTIVITE  — consommation-tertiaire-activite.csv
                 Sector benchmarks: meta/categorie/sous_categorie × year
                 → Reference EUI targets per activity type

  3. RATIO     — operat03-ratio-conso-ajustee.csv
                 Building-level (EFA): climate-adjusted EUI + energy mix %
                 → PRIMARY ML training dataset (RF, LinReg, K-Means)

Usage:
    from src.ingestion.ademe_loader import load_commune, load_activite, load_ratio
    df_commune  = load_commune("consommation-tertiaire-commune.csv")
    df_activite = load_activite("consommation-tertiaire-activite.csv")
    df_ratio    = load_ratio("operat03-ratio-conso-ajustee.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

ZONE_ENCODING = {
    "H1a": 0, "H1b": 1, "H1c": 2,
    "H2a": 3, "H2b": 4, "H2c": 5, "H2d": 6,
    "H3":  7,
    "GUA": 8, "GUY": 9, "MAR": 10, "REU": 11, "MAY": 12,
}

EUI_MIN =   10.0   # kWh/m²/yr — below = data error or unoccupied
EUI_MAX = 2000.0   # kWh/m²/yr — above = data error

# French grid emission factor (kgCO₂e/kWh) — ADEME 2024
EF_ELECTRICITY  = 0.0571
EF_GAS_NETWORK  = 0.2272
EF_GAS_LPG      = 0.2740
EF_FUEL_OIL     = 0.3240
EF_WOOD         = 0.0300
EF_HEAT_NETWORK = 0.1100
EF_COAL         = 0.3860


def _read_csv(filepath: Path, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """Safe CSV reader with existence check."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath, sep=sep, encoding=encoding, low_memory=False)
    logger.info("✓ Loaded %s — raw shape %s", filepath.name, df.shape)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 1 — COMMUNE
# ═══════════════════════════════════════════════════════════════════════════

_COMMUNE_MAP = {
    "annee_consommation":    "year",
    "zone_climatique":       "climate_zone",
    "code_region":           "region_code",
    "nom_region":            "region",
    "code_departement":      "dept_code",
    "nom_departement":       "dept",
    "nom_commune":           "commune",
    "nombre_declaration":    "n_buildings",
    "surface_declaree":      "surface_m2",
    "consommation_declaree": "consumption_kwh",
}


def load_commune(
    filepath: str | Path,
    *,
    reference_year_only: bool = False,
) -> pd.DataFrame:
    """
    Load COMMUNE dataset — geographic aggregation by commune × climate zone.

    Output columns:
        year, climate_zone, climate_zone_code, region_code, region,
        dept_code, dept, commune, n_buildings, surface_m2,
        consumption_kwh, eui_kwh_m2
    """
    df = _read_csv(Path(filepath))
    df = df.rename(columns=_COMMUNE_MAP)

    if reference_year_only:
        df = df[df["year"].str.startswith("0 -")].copy()

    for col in ("surface_m2", "consumption_kwh", "n_buildings"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df[(df["surface_m2"] > 0) & (df["consumption_kwh"] > 0)].dropna(
        subset=["surface_m2", "consumption_kwh"]
    )
    logger.info("COMMUNE: dropped %d zero/null rows", before - len(df))

    df["eui_kwh_m2"] = df["consumption_kwh"] / df["surface_m2"]

    before = len(df)
    df = df[(df["eui_kwh_m2"] >= EUI_MIN) & (df["eui_kwh_m2"] <= EUI_MAX)]
    logger.info("COMMUNE: dropped %d EUI outliers", before - len(df))

    df["climate_zone_code"] = df["climate_zone"].map(ZONE_ENCODING).fillna(-1).astype(int)

    for col in ("commune", "region", "dept"):
        df[col] = df[col].str.strip().str.upper()

    df = df[[
        "year", "climate_zone", "climate_zone_code",
        "region_code", "region", "dept_code", "dept", "commune",
        "n_buildings", "surface_m2", "consumption_kwh", "eui_kwh_m2",
    ]].reset_index(drop=True)

    logger.info("COMMUNE clean shape: %s", df.shape)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 2 — ACTIVITE
# ═══════════════════════════════════════════════════════════════════════════

_ACTIVITE_MAP = {
    "annee_consommation":       "year",
    "meta_categorie_activite":  "meta_category",
    "categorie_activite":       "category",
    "sous_categorie_activite":  "subcategory",
    "nombre_declaration":       "n_buildings",
    "surface_declaree":         "surface_m2",
    "consommation_declaree":    "consumption_kwh",
}


def load_activite(
    filepath: str | Path,
    *,
    reference_year_only: bool = False,
) -> pd.DataFrame:
    """
    Load ACTIVITE dataset — sector-level benchmarks.

    Output columns:
        year, meta_category, category, subcategory,
        n_buildings, surface_m2, consumption_kwh, eui_kwh_m2
    """
    df = _read_csv(Path(filepath))
    df = df.rename(columns=_ACTIVITE_MAP)

    if reference_year_only:
        df = df[df["year"].str.startswith("0 -")].copy()

    for col in ("surface_m2", "consumption_kwh", "n_buildings"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df[(df["surface_m2"] > 0) & (df["consumption_kwh"] > 0)].dropna(
        subset=["surface_m2", "consumption_kwh"]
    )
    logger.info("ACTIVITE: dropped %d zero/null rows", before - len(df))

    df["eui_kwh_m2"] = df["consumption_kwh"] / df["surface_m2"]

    for col in ("meta_category", "category", "subcategory"):
        df[col] = df[col].str.strip()

    df = df[[
        "year", "meta_category", "category", "subcategory",
        "n_buildings", "surface_m2", "consumption_kwh", "eui_kwh_m2",
    ]].reset_index(drop=True)

    logger.info("ACTIVITE clean shape: %s", df.shape)
    return df


def activite_benchmarks(df_activite: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EUI benchmark statistics per subcategory.
    Used as reference targets for RF classifier and compliance engine.

    Output columns:
        subcategory, category, meta_category,
        eui_mean, eui_median, eui_p25, eui_p75,
        tertiaire_target_40pct, tertiaire_target_50pct, tertiaire_target_60pct
    """
    grp = df_activite.groupby(["subcategory", "category", "meta_category"])
    bench = grp["eui_kwh_m2"].agg(
        eui_mean="mean",
        eui_median="median",
        eui_p25=lambda x: x.quantile(0.25),
        eui_p75=lambda x: x.quantile(0.75),
    ).reset_index()
    bench["tertiaire_target_40pct"] = bench["eui_median"] * 0.60
    bench["tertiaire_target_50pct"] = bench["eui_median"] * 0.50
    bench["tertiaire_target_60pct"] = bench["eui_median"] * 0.40
    return bench.round(1)


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 3 — RATIO (PRIMARY ML DATASET)
# ═══════════════════════════════════════════════════════════════════════════

_RATIO_MAP = {
    "id_efa":                                          "building_id",
    "annee_de_consommation":                           "year",
    "cas_assujettissement_efa":                        "compliance_case",
    "categorie_activite_majoritaire_efa":              "category",
    "sous_categorie_activite_majoritaire_efa":         "subcategory",
    "nombre_de_categories_activite_distinctes":        "n_categories",
    "nombre_de_sous_categories_activite_distinctes":   "n_subcategories",
    "ratio_de_consommation_ajustee_du_climat_kwh_par_m2": "eui_adjusted",
    "ratio_de_consommation_brut_kwh_par_m2":           "eui_raw",
    "consommation_individuelle_%":                     "pct_individual",
    "consommation_espaces_communs_%":                  "pct_common",
    "consommation_repartie_%":                         "pct_shared",
    "electricite_%":                                   "pct_electricity",
    "gaz_naturel_reseau_%":                            "pct_gas_network",
    "gaz_naturel_liquefie_%":                          "pct_gas_lng",
    "gaz_propane_%":                                   "pct_gas_propane",
    "gaz_butane_%":                                    "pct_gas_butane",
    "fioul_domestique_%":                              "pct_fuel_oil",
    "charbon_%":                                       "pct_coal",
    "houille_%":                                       "pct_anthracite",
    "bois_%":                                          "pct_wood",
    "reseau_de_chaleur_%":                             "pct_heat_network",
    "reseau_de_froid_%":                               "pct_cold_network",
    "gazole_non_routier_%":                            "pct_diesel",
}

_ENERGY_PCT_COLS = [
    "pct_electricity", "pct_gas_network", "pct_gas_lng", "pct_gas_propane",
    "pct_gas_butane", "pct_fuel_oil", "pct_coal", "pct_anthracite",
    "pct_wood", "pct_heat_network", "pct_cold_network", "pct_diesel",
]

_EF_MAP = {
    "pct_electricity":  EF_ELECTRICITY,
    "pct_gas_network":  EF_GAS_NETWORK,
    "pct_gas_lng":      EF_GAS_LPG,
    "pct_gas_propane":  EF_GAS_LPG,
    "pct_gas_butane":   EF_GAS_LPG,
    "pct_fuel_oil":     EF_FUEL_OIL,
    "pct_coal":         EF_COAL,
    "pct_anthracite":   EF_COAL,
    "pct_wood":         EF_WOOD,
    "pct_heat_network": EF_HEAT_NETWORK,
    "pct_cold_network": 0.0,
    "pct_diesel":       EF_FUEL_OIL,
}


def load_ratio(
    filepath: str | Path,
    *,
    eui_min: float = EUI_MIN,
    eui_max: float = EUI_MAX,
) -> pd.DataFrame:
    """
    Load RATIO dataset — building-level climate-adjusted EUI + energy mix.
    PRIMARY dataset for RF, LinearRegression and K-Means.

    Output columns:
        building_id, year, compliance_case, category, subcategory,
        n_categories, n_subcategories,
        eui_adjusted, eui_raw,
        pct_individual, pct_common, pct_shared,
        pct_electricity, pct_gas_network, ... (all energy mix %)
        carbon_intensity_kgco2e_kwh   <- computed from energy mix
        scope1_pct, scope2_pct        <- Scope 1/2 split
    """
    df = _read_csv(Path(filepath))
    df = df.rename(columns=_RATIO_MAP)

    # Cast numerics
    numeric_cols = ["eui_adjusted", "eui_raw", "n_categories", "n_subcategories",
                    "pct_individual", "pct_common", "pct_shared"] + _ENERGY_PCT_COLS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop null EUI
    before = len(df)
    df = df[df["eui_adjusted"].notna() & (df["eui_adjusted"] > 0)]
    logger.info("RATIO: dropped %d null/zero EUI rows", before - len(df))

    # Filter EUI outliers
    before = len(df)
    df = df[(df["eui_adjusted"] >= eui_min) & (df["eui_adjusted"] <= eui_max)]
    logger.info("RATIO: dropped %d EUI outliers (outside %s–%s)", before - len(df), eui_min, eui_max)

    # Fill energy mix NaN with 0
    df[_ENERGY_PCT_COLS] = df[_ENERGY_PCT_COLS].fillna(0.0)

    # Compute blended carbon intensity (kgCO2e per kWh)
    df["carbon_intensity_kgco2e_kwh"] = sum(
        df[col] / 100.0 * ef
        for col, ef in _EF_MAP.items()
        if col in df.columns
    ).round(4)

    # Scope 1 (direct combustion) vs Scope 2 (electricity + heat network)
    df["scope2_pct"] = (df["pct_electricity"] + df["pct_heat_network"]).clip(0, 100)
    df["scope1_pct"] = (100 - df["scope2_pct"]).clip(0, 100)

    # Clean string columns
    for col in ("category", "subcategory", "compliance_case"):
        if col in df.columns:
            df[col] = df[col].str.strip()

    df = df.reset_index(drop=True)
    logger.info("RATIO clean shape: %s | EUI range: %.1f – %.1f kWh/m²",
                df.shape, df["eui_adjusted"].min(), df["eui_adjusted"].max())
    return df


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def summary_commune(df: pd.DataFrame) -> dict:
    return {
        "rows":                  len(df),
        "communes":              df["commune"].nunique(),
        "climate_zones":         sorted(df["climate_zone"].unique().tolist()),
        "eui_mean_kwh_m2":       round(df["eui_kwh_m2"].mean(), 1),
        "eui_median_kwh_m2":     round(df["eui_kwh_m2"].median(), 1),
        "eui_p95_kwh_m2":        round(df["eui_kwh_m2"].quantile(0.95), 1),
        "surface_total_m2":      int(df["surface_m2"].sum()),
        "consumption_total_gwh": round(df["consumption_kwh"].sum() / 1e9, 1),
    }


def summary_activite(df: pd.DataFrame) -> dict:
    return {
        "rows":           len(df),
        "meta_categories": df["meta_category"].nunique(),
        "categories":      df["category"].nunique(),
        "subcategories":   df["subcategory"].nunique(),
        "eui_mean_kwh_m2": round(df["eui_kwh_m2"].mean(), 1),
        "eui_min_kwh_m2":  round(df["eui_kwh_m2"].min(), 1),
        "eui_max_kwh_m2":  round(df["eui_kwh_m2"].max(), 1),
    }


def summary_ratio(df: pd.DataFrame) -> dict:
    return {
        "rows":                    len(df),
        "buildings":               df["building_id"].nunique(),
        "years":                   sorted(df["year"].unique().tolist()),
        "categories":              df["category"].nunique(),
        "subcategories":           df["subcategory"].nunique(),
        "eui_adj_mean_kwh_m2":     round(df["eui_adjusted"].mean(), 1),
        "eui_adj_median_kwh_m2":   round(df["eui_adjusted"].median(), 1),
        "eui_adj_p95_kwh_m2":      round(df["eui_adjusted"].quantile(0.95), 1),
        "avg_carbon_kgco2e_kwh":   round(df["carbon_intensity_kgco2e_kwh"].mean(), 4),
        "avg_scope2_pct":          round(df["scope2_pct"].mean(), 1),
    }
