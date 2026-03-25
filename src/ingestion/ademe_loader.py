"""
AXIOM — Step 1: ADEME Tertiaire Data Loader
============================================
Loads, validates, cleans and normalises the ADEME
'consommation-tertiaire-commune' CSV into a guaranteed-schema
DataFrame consumed by all downstream AXIOM modules.

Usage:
    from src.ingestion.ademe_loader import load_ademe
    df = load_ademe("consommation-tertiaire-commune.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Column mapping: raw French → clean English snake_case ────────────────────
COLUMN_MAP = {
    "annee_consommation":   "year",
    "zone_climatique":      "climate_zone",
    "code_region":          "region_code",
    "nom_region":           "region",
    "code_departement":     "dept_code",
    "nom_departement":      "dept",
    "nom_commune":          "commune",
    "nombre_declaration":   "n_buildings",
    "surface_declaree":     "surface_m2",
    "consommation_declaree": "consumption_kwh",
}

# ── Climate zone encoding (H1a…H3, overseas) ────────────────────────────────
ZONE_ENCODING = {
    "H1a": 0, "H1b": 1, "H1c": 2,
    "H2a": 3, "H2b": 4, "H2c": 5, "H2d": 6,
    "H3":  7,
    "GUA": 8, "GUY": 9, "MAR": 10, "REU": 11, "MAY": 12,
}

# ── EUI sanity bounds (kWh/m²/yr) ────────────────────────────────────────────
EUI_MIN =   10.0   # below this → data entry error or unoccupied
EUI_MAX = 2000.0   # above this → data entry error (e.g. missing m²)


def load_ademe(
    filepath: str | Path,
    *,
    separator: str = ",",
    encoding: str = "utf-8",
    reference_year_only: bool = False,
) -> pd.DataFrame:
    """
    Load and validate the ADEME tertiaire consumption CSV.

    Parameters
    ----------
    filepath          : path to the CSV file
    separator         : column delimiter (default ',')
    encoding          : file encoding (default 'utf-8')
    reference_year_only : if True, keep only '0 - année de référence' rows

    Returns
    -------
    pd.DataFrame with guaranteed columns:
        year, climate_zone, climate_zone_code, region_code, region,
        dept_code, dept, commune, n_buildings, surface_m2,
        consumption_kwh, eui_kwh_m2
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ADEME CSV not found: {filepath}")

    logger.info("Loading ADEME CSV: %s", filepath)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(filepath, sep=separator, encoding=encoding, low_memory=False)
    logger.info("Raw shape: %s", df.shape)

    # ── 2. Rename columns ─────────────────────────────────────────────────────
    df = df.rename(columns=COLUMN_MAP)
    missing = [c for c in COLUMN_MAP.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after rename: {missing}")

    # ── 3. Optional year filter ───────────────────────────────────────────────
    if reference_year_only:
        df = df[df["year"].str.startswith("0 -")].copy()
        logger.info("After year filter: %s rows", len(df))

    # ── 4. Cast numeric types ─────────────────────────────────────────────────
    for col in ("surface_m2", "consumption_kwh", "n_buildings"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 5. Drop zero / null surface (division-by-zero guard) ─────────────────
    before = len(df)
    df = df[df["surface_m2"].notna() & (df["surface_m2"] > 0)]
    df = df[df["consumption_kwh"].notna() & (df["consumption_kwh"] > 0)]
    logger.info("Dropped %d rows with null/zero surface or consumption", before - len(df))

    # ── 6. Compute EUI (kWh/m²/yr) ───────────────────────────────────────────
    df["eui_kwh_m2"] = df["consumption_kwh"] / df["surface_m2"]

    # ── 7. Filter EUI outliers ────────────────────────────────────────────────
    before = len(df)
    df = df[(df["eui_kwh_m2"] >= EUI_MIN) & (df["eui_kwh_m2"] <= EUI_MAX)]
    logger.info("Dropped %d EUI outliers (outside %s–%s kWh/m²)", before - len(df), EUI_MIN, EUI_MAX)

    # ── 8. Encode climate zone ────────────────────────────────────────────────
    df["climate_zone_code"] = df["climate_zone"].map(ZONE_ENCODING).fillna(-1).astype(int)
    unknown_zones = df[df["climate_zone_code"] == -1]["climate_zone"].unique()
    if len(unknown_zones) > 0:
        logger.warning("Unknown climate zones (encoded -1): %s", unknown_zones)

    # ── 9. Clean string columns ───────────────────────────────────────────────
    for col in ("commune", "region", "dept"):
        df[col] = df[col].str.strip().str.upper()

    # ── 10. Final column order ────────────────────────────────────────────────
    df = df[[
        "year", "climate_zone", "climate_zone_code",
        "region_code", "region", "dept_code", "dept", "commune",
        "n_buildings", "surface_m2", "consumption_kwh", "eui_kwh_m2",
    ]].reset_index(drop=True)

    logger.info("Clean shape: %s | EUI range: %.1f – %.1f kWh/m²",
                df.shape, df["eui_kwh_m2"].min(), df["eui_kwh_m2"].max())

    return df


def summary(df: pd.DataFrame) -> dict:
    """
    Return a quick diagnostic summary dict — useful for Colab display.

    Example
    -------
    >>> s = summary(df)
    >>> print(s)
    {
      'rows': 12453,
      'communes': 8721,
      'climate_zones': ['GUA', 'GUY', 'H1a', ...],
      'eui_mean': 243.1,
      'eui_median': 198.4,
      'eui_p95': 612.3,
      'surface_total_m2': 1_482_000_000,
      'consumption_total_gwh': 312_000
    }
    """
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
