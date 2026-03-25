"""
AXIOM — Step 2: Feature Matrix Preprocessor
============================================
Transforms cleaned ADEME DataFrames into ML-ready feature matrices.

Pipeline
--------
  1. Sector-aware EUI outlier filter
  2. Fill + compute energy mix columns
  3. Derive energy features
  4. Assign DPE label (A→G)
  5. EUI tier (Low/Medium/High)
  6. Décret Tertiaire gap vs sector benchmark
  7. Label-encode category/subcategory
  8. One-hot-encode compliance_case
  9. StandardScaler
  10. Return X, y, meta, artifacts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
DPE_THRESHOLDS  = [70, 110, 180, 250, 330, 420]
DPE_LABELS      = ["A", "B", "C", "D", "E", "F", "G"]

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


# ───────────────────────────────────────────────────────────────────────────
@dataclass
class PreprocessArtifacts:
    label_enc_category:    LabelEncoder
    label_enc_subcategory: LabelEncoder
    scaler:                StandardScaler
    feature_names:         list[str]
    sector_benchmarks:     Optional[pd.DataFrame] = None


# ───────────────────────────────────────────────────────────────────────────
def assign_dpe_label(eui: Union[float, pd.Series]) -> Union[str, pd.Series]:
    """
    DPE 2021: A≤70 | B≤110 | C≤180 | D≤250 | E≤330 | F≤420 | G>420
    Accepts float, int, or pd.Series.
    """
    def _label(v: float) -> str:
        for t, l in zip(DPE_THRESHOLDS, DPE_LABELS):
            if float(v) <= t:
                return l
        return "G"
    if isinstance(eui, pd.Series):
        return eui.apply(_label)
    return _label(eui)


def assign_eui_tier(eui: Union[float, pd.Series], p33: float, p66: float) -> Union[str, pd.Series]:
    def _tier(v):
        return "Low" if v <= p33 else ("Medium" if v <= p66 else "High")
    return eui.apply(_tier) if isinstance(eui, pd.Series) else _tier(float(eui))


def compute_tertiaire_gap(
    eui: Union[float, pd.Series],
    reference_eui: Union[float, pd.Series],
    horizon: int = 2030,
) -> Union[float, pd.Series]:
    """
    Gap vs Décret Tertiaire target.
    Gap > 0 → non-compliant.  Gap < 0 → compliant.
    Works with both float scalars and pd.Series.
    """
    rate   = TERTIAIRE_TARGETS.get(horizon, 0.40)
    target = reference_eui * (1.0 - rate)
    result = (eui - target) / target * 100
    if isinstance(result, pd.Series):
        return result.round(1)
    return round(float(result), 1)


def _sector_aware_eui_filter(df: pd.DataFrame) -> pd.DataFrame:
    is_high = df["category"].isin(HIGH_EUI_SECTORS)
    mask = (
        (df["eui_adjusted"] >= EUI_MIN) &
        (
            ( is_high & (df["eui_adjusted"] <= HIGH_EUI_MAX)) |
            (~is_high & (df["eui_adjusted"] <= STD_EUI_MAX))
        )
    )
    logger.info("Sector-aware EUI filter: dropped %d rows", (~mask).sum())
    return df[mask].copy()


def _ensure_energy_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in _ENERGY_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[_ENERGY_COLS] = df[_ENERGY_COLS].fillna(0.0)
    return df


def _compute_carbon_scope(df: pd.DataFrame) -> pd.DataFrame:
    df["carbon_intensity_kgco2e_kwh"] = sum(
        df[col] / 100.0 * ef for col, ef in _EF.items() if col in df.columns
    ).round(4)
    df["scope2_pct"] = (df.get("pct_electricity", 0) + df.get("pct_heat_network", 0)).clip(0, 100)
    df["scope1_pct"] = (100 - df["scope2_pct"]).clip(0, 100)
    return df


def _derive_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    available         = [c for c in _ENERGY_COLS if c in df.columns]
    df["n_fuel_types"] = (df[available] > 1.0).sum(axis=1).astype(int)
    fossil_cols       = [c for c in [
        "pct_gas_network", "pct_gas_lng", "pct_gas_propane", "pct_gas_butane",
        "pct_fuel_oil", "pct_coal", "pct_anthracite", "pct_diesel",
    ] if c in df.columns]
    df["pct_fossil"]     = df[fossil_cols].sum(axis=1).clip(0, 100)
    renew_cols            = [c for c in ["pct_wood", "pct_heat_network"] if c in df.columns]
    df["pct_renewable"]  = df[renew_cols].sum(axis=1).clip(0, 100)
    df["is_mixed_energy"] = (df[available] > 5.0).sum(axis=1).gt(1).astype(int)
    return df


# ───────────────────────────────────────────────────────────────────────────
def build_feature_matrix(
    df_ratio: pd.DataFrame,
    df_activite: Optional[pd.DataFrame] = None,
    *,
    scale: bool = True,
    tertiaire_horizon: int = 2030,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, PreprocessArtifacts]:
    """
    Build ML-ready feature matrix. Returns X, y (log-EUI), meta, artifacts.
    y is log1p(eui_adjusted) — use np.expm1(y_pred) to recover kWh/m².
    """
    df = df_ratio.copy()
    logger.info("Preprocessor start: %d rows", len(df))

    df = _sector_aware_eui_filter(df)
    df = _ensure_energy_cols(df)
    df = _compute_carbon_scope(df)
    df = _derive_energy_features(df)

    df["dpe_label"] = assign_dpe_label(df["eui_adjusted"])

    p33 = df["eui_adjusted"].quantile(0.33)
    p66 = df["eui_adjusted"].quantile(0.66)
    df["eui_tier"] = assign_eui_tier(df["eui_adjusted"], p33, p66)
    logger.info("EUI tiers: p33=%.1f  p66=%.1f kWh/m²", p33, p66)

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

    le_cat = LabelEncoder()
    le_sub = LabelEncoder()
    df["category_code"]    = le_cat.fit_transform(df["category"].fillna("Unknown"))
    df["subcategory_code"] = le_sub.fit_transform(df["subcategory"].fillna("Unknown"))

    compliance_dummies = pd.get_dummies(
        df["compliance_case"].fillna("Unknown"), prefix="case", dtype=int
    )
    df        = pd.concat([df, compliance_dummies], axis=1)
    case_cols = compliance_dummies.columns.tolist()

    numeric_features = [
        "category_code", "subcategory_code",
        "n_categories", "n_subcategories",
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

    # ✔ log1p transform on y — compresses right tail from data centres
    y = np.log1p(df["eui_adjusted"].values)

    meta = df[[
        "building_id", "year", "category", "subcategory",
        "eui_adjusted", "eui_raw", "dpe_label", "eui_tier",
        "sector_ref_eui", "tertiaire_gap_pct",
        "carbon_intensity_kgco2e_kwh", "scope1_pct", "scope2_pct",
    ]].copy().reset_index(drop=True)

    artifacts = PreprocessArtifacts(
        label_enc_category=le_cat,
        label_enc_subcategory=le_sub,
        scaler=scaler,
        feature_names=feature_cols,
    )

    logger.info("Preprocessor done: X=%s | y=log1p(EUI) | DPE dist: %s",
                X.shape, meta["dpe_label"].value_counts().to_dict())
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

    for col, le in [("category",    artifacts.label_enc_category),
                    ("subcategory", artifacts.label_enc_subcategory)]:
        val = str(row[col].fillna("Unknown").iloc[0]) if col in row.columns else "Unknown"
        row[f"{col}_code"] = le.transform([val])[0] if val in le.classes_ else -1

    for fname in artifacts.feature_names:
        if fname not in row.columns:
            row[fname] = 0

    vec = row[artifacts.feature_names].fillna(0.0).values
    return artifacts.scaler.transform(vec)
