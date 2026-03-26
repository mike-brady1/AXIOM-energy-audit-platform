"""
AXIOM — Step 2: Feature Matrix Preprocessor
============================================
Transforms cleaned ADEME DataFrames into ML-ready feature matrices.

Key design decisions:

1. TARGET ENCODING (leakage-safe):
   Built from training fold only. Replaces LabelEncoder integers with
   mean EUI per sector.

2. USAGE PATTERN FEATURES:
   pct_individual / pct_common / pct_shared explain within-sector variance.

3. TEMPORAL FEATURE:
   year column parsed robustly (handles '2021', '2010-2019' decade ranges).

4. DPE TERTIAIRE JOIN FEATURES (from df_dpe_tertiaire via sector_stats join):
   log_surface_median   — log(median floor area) per sector category
   median_period_ord    — median construction period ordinal (1=pre-1948 → 10=post-2021)
   pct_pre1975          — % of sector buildings built before 1975 (proxy for renovation need)
   These must be pre-joined into df_ratio before calling build_feature_matrix.

Pipeline
--------
  1. Sector-aware EUI outlier filter
  2. Parse year column (handles string ranges)
  3. Fill + compute energy mix columns
  4. Derive energy features
  5. Assign DPE label (A→G) and EUI tier
  6. Décret Tertiaire gap vs sector benchmark
  7. Target-encode category + subcategory (training fold only)
  8. Add usage pattern + temporal + climate correction features
  9. One-hot-encode compliance_case
  10. StandardScaler
  11. Return X, y (raw EUI), meta, artifacts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
DPE_THRESHOLDS = [70, 110, 180, 250, 330, 420]
DPE_LABELS     = ["A", "B", "C", "D", "E", "F", "G"]

HIGH_EUI_SECTORS = {
    "Salles serveurs et centres d'exploitation informatique",
    'Blanchisserie dite "industrielle"',
    "Blanchisserie",
    "Serveurs & IT",
}
HIGH_EUI_MAX = 6000.0
STD_EUI_MAX  = 2000.0
EUI_MIN      =   10.0

TERTIAIRE_TARGETS = {2030: 0.40, 2040: 0.50, 2050: 0.60}

_EF = {
    "pct_electricity":  0.0571,
    "pct_gas_network":  0.2272,
    "pct_gas_lng":      0.2740,
    "pct_gas_propane":  0.2740,
    "pct_gas_butane":   0.2740,
    "pct_fuel_oil":     0.3240,
    "pct_coal":         0.3860,
    "pct_anthracite":   0.3860,
    "pct_wood":         0.0300,
    "pct_heat_network": 0.1100,
    "pct_cold_network": 0.0000,
    "pct_diesel":       0.3240,
}
_ENERGY_COLS = list(_EF.keys())
_USAGE_COLS  = ["pct_individual", "pct_common", "pct_shared"]

# DPE Tertiaire join features (pre-joined into df_ratio via sector_stats)
_DPE_JOIN_COLS = ["log_surface_median", "median_period_ord", "pct_pre1975"]


# ───────────────────────────────────────────────────────────────────────────
@dataclass
class PreprocessArtifacts:
    scaler:                 StandardScaler
    feature_names:          list[str]
    category_target_map:    dict  = field(default_factory=dict)
    subcategory_target_map: dict  = field(default_factory=dict)
    global_mean_eui:        float = 155.0
    year_min:               float = 2016.0
    year_max:               float = 2023.0
    # DPE join defaults for inference on unenriched buildings
    dpe_join_defaults:      dict  = field(default_factory=dict)
    label_enc_category:     Optional[LabelEncoder] = None
    label_enc_subcategory:  Optional[LabelEncoder] = None
    sector_benchmarks:      Optional[pd.DataFrame] = None


# ───────────────────────────────────────────────────────────────────────────
def assign_dpe_label(eui: Union[float, pd.Series]) -> Union[str, pd.Series]:
    """DPE 2021: A≤70 | B≤110 | C≤180 | D≤250 | E≤330 | F≤420 | G>420"""
    def _label(v):
        for t, l in zip(DPE_THRESHOLDS, DPE_LABELS):
            if float(v) <= t: return l
        return "G"
    return eui.apply(_label) if isinstance(eui, pd.Series) else _label(eui)


def assign_eui_tier(eui: Union[float, pd.Series], p33: float, p66: float) -> Union[str, pd.Series]:
    def _tier(v):
        return "Low" if v <= p33 else ("Medium" if v <= p66 else "High")
    return eui.apply(_tier) if isinstance(eui, pd.Series) else _tier(float(eui))


def compute_tertiaire_gap(
    eui: Union[float, pd.Series],
    reference_eui: Union[float, pd.Series],
    horizon: int = 2030,
) -> Union[float, pd.Series]:
    """Gap vs Décret Tertiaire target. >0 = non-compliant. <0 = compliant."""
    rate   = TERTIAIRE_TARGETS.get(horizon, 0.40)
    target = reference_eui * (1.0 - rate)
    result = (eui - target) / target * 100
    return result.round(1) if isinstance(result, pd.Series) else round(float(result), 1)


def _parse_year_col(series: pd.Series) -> pd.Series:
    """
    Robustly parse year column:
      '2021'       → 2021.0
      '2010-2019'  → 2014.5  (midpoint of decade range)
      NaN/unknown  → 2019.0
    """
    def _parse(v) -> float:
        try:
            s = str(v).strip()
            if "-" in s:
                parts = s.split("-")
                nums  = [float(p) for p in parts if p.strip().isdigit()]
                if len(nums) == 2:
                    return (nums[0] + nums[1]) / 2.0
            return float(s)
        except Exception:
            return 2019.0
    return series.apply(_parse)


def _sector_aware_eui_filter(df: pd.DataFrame) -> pd.DataFrame:
    is_high = df["category"].isin(HIGH_EUI_SECTORS)
    mask = (
        (df["eui_adjusted"] >= EUI_MIN) &
        (( is_high & (df["eui_adjusted"] <= HIGH_EUI_MAX)) |
         (~is_high & (df["eui_adjusted"] <= STD_EUI_MAX)))
    )
    logger.info("EUI filter: dropped %d rows", (~mask).sum())
    return df[mask].copy()


def _ensure_energy_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in _ENERGY_COLS + _USAGE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[_ENERGY_COLS + _USAGE_COLS] = df[_ENERGY_COLS + _USAGE_COLS].fillna(0.0)
    return df


def _compute_carbon_scope(df: pd.DataFrame) -> pd.DataFrame:
    df["carbon_intensity_kgco2e_kwh"] = sum(
        df[col] / 100.0 * ef for col, ef in _EF.items() if col in df.columns
    ).round(4)
    df["scope2_pct"] = (df.get("pct_electricity", 0) + df.get("pct_heat_network", 0)).clip(0, 100)
    df["scope1_pct"] = (100 - df["scope2_pct"]).clip(0, 100)
    return df


def _derive_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    available          = [c for c in _ENERGY_COLS if c in df.columns]
    df["n_fuel_types"]  = (df[available] > 1.0).sum(axis=1).astype(int)
    fossil_cols         = [c for c in [
        "pct_gas_network", "pct_gas_lng", "pct_gas_propane", "pct_gas_butane",
        "pct_fuel_oil", "pct_coal", "pct_anthracite", "pct_diesel",
    ] if c in df.columns]
    df["pct_fossil"]      = df[fossil_cols].sum(axis=1).clip(0, 100)
    renew_cols            = [c for c in ["pct_wood", "pct_heat_network"] if c in df.columns]
    df["pct_renewable"]   = df[renew_cols].sum(axis=1).clip(0, 100)
    df["is_mixed_energy"] = (df[available] > 5.0).sum(axis=1).gt(1).astype(int)
    if "eui_raw" in df.columns:
        df["eui_correction_ratio"] = (
            df["eui_adjusted"] / df["eui_raw"].replace(0, np.nan)
        ).fillna(1.0).clip(0.5, 2.0)
    return df


def _build_target_maps(df: pd.DataFrame) -> tuple[dict, dict, float]:
    """Build mean-EUI target maps. Call on training fold only."""
    global_mean = float(df["eui_adjusted"].mean())
    cat_map     = df.groupby("category")["eui_adjusted"].mean().to_dict()
    sub_map     = df.groupby("subcategory")["eui_adjusted"].mean().to_dict()
    logger.info("Target encoding: %d cats %d subcats global_mean=%.1f",
                len(cat_map), len(sub_map), global_mean)
    return cat_map, sub_map, global_mean


# ───────────────────────────────────────────────────────────────────────────
def build_feature_matrix(
    df_ratio: pd.DataFrame,
    df_activite: Optional[pd.DataFrame] = None,
    *,
    scale: bool = True,
    tertiaire_horizon: int = 2030,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, PreprocessArtifacts]:
    """
    Build ML-ready feature matrix with leakage-safe target encoding.
    Pass df_ratio_enriched (with DPE join columns) for best R².
    Returns X, y (raw EUI kWh/m²), meta, artifacts.
    """
    from sklearn.model_selection import train_test_split as _tts

    df = df_ratio.copy()
    logger.info("Preprocessor start: %d rows", len(df))

    if "year" in df.columns:
        df["year"] = _parse_year_col(df["year"])

    df = _sector_aware_eui_filter(df)
    df = _ensure_energy_cols(df)
    df = _compute_carbon_scope(df)
    df = _derive_energy_features(df)

    df["dpe_label"] = assign_dpe_label(df["eui_adjusted"])
    p33 = df["eui_adjusted"].quantile(0.33)
    p66 = df["eui_adjusted"].quantile(0.66)
    df["eui_tier"] = assign_eui_tier(df["eui_adjusted"], p33, p66)

    if df_activite is not None:
        from src.ingestion.ademe_loader import activite_benchmarks
        bench     = activite_benchmarks(df_activite)
        bench_map = bench.set_index("subcategory")["eui_median"].to_dict()
        df["sector_ref_eui"] = df["subcategory"].map(bench_map)
    else:
        sub_median = df.groupby("subcategory")["eui_adjusted"].median()
        df["sector_ref_eui"] = df["subcategory"].map(sub_median)
    df["sector_ref_eui"]    = df["sector_ref_eui"].fillna(df["eui_adjusted"].median())
    df["tertiaire_gap_pct"] = compute_tertiaire_gap(
        df["eui_adjusted"], df["sector_ref_eui"], horizon=tertiaire_horizon
    )

    # Leakage-safe target encoding (training fold only)
    train_idx, _ = _tts(df.index, test_size=test_size, random_state=random_state)
    cat_map, sub_map, global_mean = _build_target_maps(df.loc[train_idx])
    df["category_target_eui"]    = df["category"].map(cat_map).fillna(global_mean)
    df["subcategory_target_eui"] = df["subcategory"].map(sub_map).fillna(global_mean)

    # Year feature
    year_min   = float(df["year"].min()) if "year" in df.columns else 2016.0
    year_max   = float(df["year"].max()) if "year" in df.columns else 2023.0
    year_range = max(year_max - year_min, 1.0)
    df["year_norm"] = ((df["year"] - year_min) / year_range).clip(0, 1) if "year" in df.columns else 0.5

    # Fill DPE join columns with global medians if not pre-joined
    dpe_join_defaults = {}
    for col in _DPE_JOIN_COLS:
        if col in df.columns:
            default = float(df[col].median())
        else:
            default = 0.0
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        dpe_join_defaults[col] = default
    joined = sum(df[c].ne(0).any() for c in _DPE_JOIN_COLS)
    logger.info("DPE join columns active: %d/%d", joined, len(_DPE_JOIN_COLS))

    compliance_dummies = pd.get_dummies(
        df["compliance_case"].fillna("Unknown"), prefix="case", dtype=int
    )
    df        = pd.concat([df, compliance_dummies], axis=1)
    case_cols = compliance_dummies.columns.tolist()

    numeric_features = [
        "category_target_eui", "subcategory_target_eui",
        "n_categories", "n_subcategories",
        # Usage pattern
        "pct_individual", "pct_common", "pct_shared",
        # Temporal
        "year_norm",
        # Climate correction
        "eui_correction_ratio",
        # DPE Tertiaire join features ← NEW
        "log_surface_median",    # log(median floor area per sector)
        "median_period_ord",     # construction era ordinal (1=pre-1948, 10=post-2021)
        "pct_pre1975",           # % buildings pre-1975 (renovation pressure proxy)
        # Energy mix
        "pct_electricity", "pct_gas_network", "pct_fuel_oil",
        "pct_heat_network", "pct_wood",
        "pct_fossil", "pct_renewable",
        "n_fuel_types", "is_mixed_energy",
        "carbon_intensity_kgco2e_kwh",
        "scope1_pct", "scope2_pct",
    ] + case_cols

    feature_cols = [c for c in numeric_features if c in df.columns]
    X_df         = df[feature_cols].fillna(0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values) if scale else X_df.values
    if not scale:
        scaler.fit(X_df.values)

    y = df["eui_adjusted"].values

    meta = df[[
        "building_id", "year", "category", "subcategory",
        "eui_adjusted", "eui_raw", "dpe_label", "eui_tier",
        "sector_ref_eui", "tertiaire_gap_pct",
        "carbon_intensity_kgco2e_kwh", "scope1_pct", "scope2_pct",
    ]].copy().reset_index(drop=True)

    artifacts = PreprocessArtifacts(
        scaler=scaler,
        feature_names=feature_cols,
        category_target_map=cat_map,
        subcategory_target_map=sub_map,
        global_mean_eui=global_mean,
        year_min=year_min,
        year_max=year_max,
        dpe_join_defaults=dpe_join_defaults,
    )

    logger.info("Done: X=%s features=%s", X.shape, feature_cols)
    return X, y, meta, artifacts


def transform_single_building(
    building: dict,
    artifacts: PreprocessArtifacts,
) -> np.ndarray:
    """Transform a single building dict into a scaled feature vector."""
    row = pd.DataFrame([building])
    row = _ensure_energy_cols(row)
    row = _compute_carbon_scope(row)
    row = _derive_energy_features(row)

    cat_val = str(row["category"].iloc[0])    if "category"    in row.columns else "Unknown"
    sub_val = str(row["subcategory"].iloc[0]) if "subcategory" in row.columns else "Unknown"
    row["category_target_eui"]    = artifacts.category_target_map.get(cat_val,    artifacts.global_mean_eui)
    row["subcategory_target_eui"] = artifacts.subcategory_target_map.get(sub_val, artifacts.global_mean_eui)

    year_range  = max(artifacts.year_max - artifacts.year_min, 1.0)
    raw_year    = _parse_year_col(row["year"]).iloc[0] if "year" in row.columns else artifacts.year_max
    row["year_norm"] = (raw_year - artifacts.year_min) / year_range

    # DPE join features: use stored defaults if not provided
    for col, default in artifacts.dpe_join_defaults.items():
        if col not in row.columns:
            row[col] = default

    for fname in artifacts.feature_names:
        if fname not in row.columns:
            row[fname] = 0

    vec = row[artifacts.feature_names].fillna(0.0).values
    return artifacts.scaler.transform(vec)
