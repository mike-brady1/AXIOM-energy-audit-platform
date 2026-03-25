"""
AXIOM — Step 2: Feature Matrix Preprocessor
============================================
Transforms cleaned ADEME DataFrames into ML-ready feature matrices.

Inputs  : df_ratio (from load_ratio), df_activite benchmarks (optional)
Outputs : X (feature matrix), y (EUI target), metadata DataFrame

Pipeline
--------
  1. Sector-aware EUI outlier filter   (Data Centres get higher ceiling)
  2. Label-encode category/subcategory
  3. One-hot-encode compliance_case
  4. Derive energy mix features
  5. Assign DPE 2021 label (A→G)
  6. Compute Décret Tertiaire gap vs sector benchmark
  7. StandardScaler on numeric features
  8. Return X, y, feature names, scalers

Usage:
    from src.processing.preprocessor import build_feature_matrix
    X, y, meta, artifacts = build_feature_matrix(df_ratio, df_activite)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# DPE 2021 THRESHOLDS (kWh EP/m²/yr)
# Source: Décret n°2021-872, arrêté du 31 mars 2021
# ───────────────────────────────────────────────────────────────────────────
DPE_THRESHOLDS = [70, 110, 180, 250, 330, 420]   # A B C D E F | >420 = G
DPE_LABELS     = ["A", "B", "C", "D", "E", "F", "G"]

# Sectors whose EUI legitimately exceeds standard 2000 kWh/m² ceiling
HIGH_EUI_SECTORS = {
    "Salles serveurs et centres d'exploitation informatique",
    "Blanchisserie dite \"industrielle\"",
    "Blanchisserie",
    "Serveurs & IT",
}
HIGH_EUI_MAX = 6000.0   # kWh/m²/yr for high-intensity sectors
STD_EUI_MAX  = 2000.0
EUI_MIN      =   10.0

# Décret Tertiaire targets (relative reduction from reference year)
TERTIAIRE_TARGETS = {2030: 0.40, 2040: 0.50, 2050: 0.60}

# Energy mix columns present in RATIO dataset
_ENERGY_COLS = [
    "pct_electricity", "pct_gas_network", "pct_gas_lng", "pct_gas_propane",
    "pct_gas_butane",  "pct_fuel_oil",    "pct_coal",    "pct_anthracite",
    "pct_wood",        "pct_heat_network", "pct_cold_network", "pct_diesel",
]


# ───────────────────────────────────────────────────────────────────────────
# OUTPUT CONTAINER
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessArtifacts:
    """Serialisable objects needed to transform new/unseen buildings."""
    label_enc_category:    LabelEncoder
    label_enc_subcategory: LabelEncoder
    scaler:                StandardScaler
    feature_names:         list[str]
    sector_benchmarks:     Optional[pd.DataFrame] = None  # from activite_benchmarks()


# ───────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────

def assign_dpe_label(eui: float | pd.Series) -> str | pd.Series:
    """
    Assign French DPE 2021 energy label based on EUI (kWh EP/m²/yr).
    Thresholds: A≤70 | B≤110 | C≤180 | D≤250 | E≤330 | F≤420 | G>420
    """
    def _label(v: float) -> str:
        for threshold, label in zip(DPE_THRESHOLDS, DPE_LABELS):
            if v <= threshold:
                return label
        return "G"

    if isinstance(eui, pd.Series):
        return eui.apply(_label)
    return _label(float(eui))


def assign_eui_tier(eui: float | pd.Series, p33: float, p66: float) -> str | pd.Series:
    """
    Assign EUI performance tier: Low / Medium / High (energy use).
    Thresholds are computed per-sector from the training data distribution.
    """
    def _tier(v: float) -> str:
        if v <= p33:
            return "Low"
        if v <= p66:
            return "Medium"
        return "High"

    if isinstance(eui, pd.Series):
        return eui.apply(_tier)
    return _tier(float(eui))


def compute_tertiaire_gap(
    eui: float | pd.Series,
    reference_eui: float | pd.Series,
    horizon: int = 2030,
) -> float | pd.Series:
    """
    Compute percentage gap vs Décret Tertiaire target.

    Gap > 0  → building exceeds target (needs reduction)
    Gap < 0  → building already meets target (compliant)

    Formula: gap = (eui - target) / target * 100
    where target = reference_eui * (1 - reduction_rate)
    """
    rate   = TERTIAIRE_TARGETS.get(horizon, 0.40)
    target = reference_eui * (1.0 - rate)
    return ((eui - target) / target * 100).round(1)


def _sector_aware_eui_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply EUI bounds with higher ceiling for high-intensity sectors.
    """
    is_high = df["category"].isin(HIGH_EUI_SECTORS)
    mask = (
        (df["eui_adjusted"] >= EUI_MIN) &
        (
            (is_high  & (df["eui_adjusted"] <= HIGH_EUI_MAX)) |
            (~is_high & (df["eui_adjusted"] <= STD_EUI_MAX))
        )
    )
    dropped = (~mask).sum()
    logger.info("Sector-aware EUI filter: dropped %d rows", dropped)
    return df[mask].copy()


def _derive_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive higher-level energy mix features from raw % columns.
    """
    available = [c for c in _ENERGY_COLS if c in df.columns]

    # Number of distinct fuel types used (> 1% share)
    df["n_fuel_types"] = (df[available] > 1.0).sum(axis=1).astype(int)

    # Fossil fuel share (gas + oil + coal + diesel)
    fossil_cols = [c for c in [
        "pct_gas_network", "pct_gas_lng", "pct_gas_propane", "pct_gas_butane",
        "pct_fuel_oil", "pct_coal", "pct_anthracite", "pct_diesel",
    ] if c in df.columns]
    df["pct_fossil"] = df[fossil_cols].sum(axis=1).clip(0, 100)

    # Renewable share (wood + heat network proxy)
    renew_cols = [c for c in ["pct_wood", "pct_heat_network"] if c in df.columns]
    df["pct_renewable"] = df[renew_cols].sum(axis=1).clip(0, 100)

    # Flag: mixed energy (uses more than one fuel > 5%)
    df["is_mixed_energy"] = (df[available] > 5.0).sum(axis=1).gt(1).astype(int)

    return df


# ───────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ───────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df_ratio: pd.DataFrame,
    df_activite: Optional[pd.DataFrame] = None,
    *,
    scale: bool = True,
    tertiaire_horizon: int = 2030,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, PreprocessArtifacts]:
    """
    Build ML-ready feature matrix from RATIO dataset.

    Parameters
    ----------
    df_ratio           : output of load_ratio()
    df_activite        : output of load_activite() — used for sector benchmarks
    scale              : apply StandardScaler to numeric features (default True)
    tertiaire_horizon  : compliance target year (2030 / 2040 / 2050)

    Returns
    -------
    X         : np.ndarray  shape (n_samples, n_features)
    y         : np.ndarray  shape (n_samples,)  — eui_adjusted
    meta      : pd.DataFrame with building_id, dpe_label, eui_tier,
                tertiaire_gap, category, subcategory
    artifacts : PreprocessArtifacts (encoders + scaler for inference)
    """
    df = df_ratio.copy()
    logger.info("Preprocessor start: %d rows", len(df))

    # ── 1. Sector-aware EUI filter ──────────────────────────────────────────
    df = _sector_aware_eui_filter(df)

    # ── 2. Derive energy mix features ────────────────────────────────────────
    df = _derive_energy_features(df)

    # ── 3. DPE label ─────────────────────────────────────────────────────────
    df["dpe_label"] = assign_dpe_label(df["eui_adjusted"])

    # ── 4. EUI tier (Low / Medium / High) ───────────────────────────────────
    p33 = df["eui_adjusted"].quantile(0.33)
    p66 = df["eui_adjusted"].quantile(0.66)
    df["eui_tier"] = assign_eui_tier(df["eui_adjusted"], p33, p66)
    logger.info("EUI tiers: p33=%.1f p66=%.1f kWh/m²", p33, p66)

    # ── 5. Décret Tertiaire gap vs sector median benchmark ─────────────────
    if df_activite is not None:
        from src.ingestion.ademe_loader import activite_benchmarks
        bench = activite_benchmarks(df_activite)
        bench_map = bench.set_index("subcategory")["eui_median"].to_dict()
        df["sector_ref_eui"] = df["subcategory"].map(bench_map)
    else:
        # Fall back to per-subcategory median from RATIO itself
        sub_median = df.groupby("subcategory")["eui_adjusted"].median()
        df["sector_ref_eui"] = df["subcategory"].map(sub_median)

    df["sector_ref_eui"] = df["sector_ref_eui"].fillna(df["eui_adjusted"].median())
    df["tertiaire_gap_pct"] = compute_tertiaire_gap(
        df["eui_adjusted"], df["sector_ref_eui"], horizon=tertiaire_horizon
    )

    # ── 6. Label-encode category + subcategory ────────────────────────────
    le_cat  = LabelEncoder()
    le_sub  = LabelEncoder()
    df["category_code"]    = le_cat.fit_transform(df["category"].fillna("Unknown"))
    df["subcategory_code"] = le_sub.fit_transform(df["subcategory"].fillna("Unknown"))

    # ── 7. One-hot-encode compliance_case ─────────────────────────────────
    compliance_dummies = pd.get_dummies(
        df["compliance_case"].fillna("Unknown"),
        prefix="case", dtype=int
    )
    df = pd.concat([df, compliance_dummies], axis=1)
    case_cols = compliance_dummies.columns.tolist()

    # ── 8. Assemble feature matrix ───────────────────────────────────────
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

    # Keep only columns that actually exist in df
    feature_cols = [c for c in numeric_features if c in df.columns]
    X_df = df[feature_cols].fillna(0.0)

    # ── 9. Scale ──────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    if scale:
        X = scaler.fit_transform(X_df.values)
    else:
        X = X_df.values
        scaler.fit(X_df.values)   # fit anyway so artifacts are complete

    # ── 10. Target vector ──────────────────────────────────────────────────────
    y = df["eui_adjusted"].values

    # ── 11. Metadata (for audit reporting + K-Means labelling) ─────────────
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

    logger.info(
        "Preprocessor done: X=%s | features=%d | DPE dist: %s",
        X.shape, len(feature_cols),
        meta["dpe_label"].value_counts().to_dict()
    )
    return X, y, meta, artifacts


def transform_single_building(
    building: dict,
    artifacts: PreprocessArtifacts,
) -> np.ndarray:
    """
    Transform a single new building dict into a feature vector for inference.
    Keys must match RATIO column names (English snake_case after load_ratio).

    Example
    -------
    >>> vec = transform_single_building({
    ...     "category":        "Bureaux – Services Publics - Banque",
    ...     "subcategory":     "Bureaux - Bureaux standards",
    ...     "compliance_case": "1A",
    ...     "pct_electricity": 100.0,
    ...     "n_categories":    1,
    ... }, artifacts)
    >>> model.predict([vec])
    """
    row = pd.DataFrame([building])

    # Encode category / subcategory — handle unseen labels gracefully
    for col, le in [("category", artifacts.label_enc_category),
                    ("subcategory", artifacts.label_enc_subcategory)]:
        val = row[col].fillna("Unknown").iloc[0]
        if val in le.classes_:
            row[f"{col}_code"] = le.transform([val])[0]
        else:
            row[f"{col}_code"] = -1   # unseen sector

    row = _derive_energy_features(row)

    # Compliance case dummies
    for fname in artifacts.feature_names:
        if fname.startswith("case_") and fname not in row.columns:
            row[fname] = 0

    vec = row[artifacts.feature_names].fillna(0.0).values
    return artifacts.scaler.transform(vec)
