"""
AXIOM — Step 3: Random Forest EUI Regressor
============================================
Predicts climate-adjusted EUI (kWh/m²/yr) for any tertiary building,
then derives DPE label (A→G) and Decret Tertiaire compliance from
the predicted value.

Why regression, not classification:
  DPE label = deterministic threshold applied to EUI.
  Predicting EUI directly (regression) then applying thresholds
  yields far higher DPE accuracy than classifying label directly.

Model
-----
  RandomForestRegressor
  - n_estimators : 200
  - max_features : 'sqrt'
  - min_samples_leaf : 5
  - n_jobs : -1  (all cores)

Evaluation
----------
  - R², MAE, RMSE on held-out 20% test set
  - DPE label accuracy derived from predicted EUI
  - Feature importances

Usage:
    from src.ai_engine.classifier import train_regressor, predict_building
    result = train_regressor(X, y, meta, artifacts)
    pred   = predict_building({...}, result, artifacts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.processing.preprocessor import (
    PreprocessArtifacts,
    transform_single_building,
    assign_dpe_label,
    compute_tertiaire_gap,
    TERTIAIRE_TARGETS,
)

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# OUTPUT CONTAINER
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class RegressorResult:
    model:                  RandomForestRegressor
    r2:                     float
    mae:                    float          # kWh/m²/yr
    rmse:                   float          # kWh/m²/yr
    dpe_accuracy:           float          # % correct DPE label from predicted EUI
    dpe_adjacent_accuracy:  float          # % within ±1 DPE band
    feature_importances:    pd.DataFrame
    y_test:                 np.ndarray     # actual EUI
    y_pred:                 np.ndarray     # predicted EUI
    dpe_actual:             np.ndarray     # actual DPE labels
    dpe_predicted:          np.ndarray     # predicted DPE labels


# ───────────────────────────────────────────────────────────────────────────
# TRAINING
# ───────────────────────────────────────────────────────────────────────────

def train_regressor(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    *,
    n_estimators: int = 200,
    test_size: float = 0.20,
    random_state: int = 42,
    sample_size: Optional[int] = None,
) -> RegressorResult:
    """
    Train Random Forest Regressor to predict EUI (kWh/m²/yr).

    Parameters
    ----------
    X             : feature matrix from build_feature_matrix()
    y             : EUI target vector from build_feature_matrix()
    meta          : metadata DataFrame (contains dpe_label for evaluation)
    artifacts     : PreprocessArtifacts
    n_estimators  : number of trees (default 200)
    test_size     : held-out test fraction (default 0.20)
    random_state  : reproducibility seed
    sample_size   : optional row subsample for faster iteration

    Returns
    -------
    RegressorResult with trained model + full evaluation metrics
    """
    y_dpe = meta["dpe_label"].values

    # ── Optional subsample ───────────────────────────────────────────────
    if sample_size and sample_size < len(X):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X, y, y_dpe = X[idx], y[idx], y_dpe[idx]
        logger.info("Subsampled to %d rows", sample_size)

    # ── Train / test split ───────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te, dpe_tr, dpe_te = train_test_split(
        X, y, y_dpe,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info("Train: %d  Test: %d", len(X_tr), len(X_te))

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("Training RF regressor (%d trees)...", n_estimators)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_tr, y_tr)

    # ── Evaluate EUI regression ─────────────────────────────────────────
    y_pred = model.predict(X_te)
    r2     = r2_score(y_te, y_pred)
    mae    = mean_absolute_error(y_te, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    logger.info("EUI regression — R²=%.3f  MAE=%.1f  RMSE=%.1f kWh/m²", r2, mae, rmse)

    # ── Derive DPE labels from predicted EUI ──────────────────────────
    dpe_pred_series  = assign_dpe_label(pd.Series(y_pred))
    dpe_pred         = dpe_pred_series.values
    dpe_accuracy     = float((dpe_pred == dpe_te).mean())

    # Adjacent accuracy: within ±1 DPE band (A↔B, B↔C, etc.)
    labels_order = ["A", "B", "C", "D", "E", "F", "G"]
    label_idx    = {l: i for i, l in enumerate(labels_order)}
    adj = np.abs(
        np.array([label_idx.get(p, 0) for p in dpe_pred]) -
        np.array([label_idx.get(a, 0) for a in dpe_te])
    ) <= 1
    dpe_adj_accuracy = float(adj.mean())
    logger.info("DPE exact=%.3f  adjacent=%.3f", dpe_accuracy, dpe_adj_accuracy)

    # ── Feature importances ─────────────────────────────────────────────
    fi = pd.DataFrame({
        "feature":    artifacts.feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return RegressorResult(
        model=model,
        r2=round(r2, 4),
        mae=round(mae, 1),
        rmse=round(rmse, 1),
        dpe_accuracy=round(dpe_accuracy, 4),
        dpe_adjacent_accuracy=round(dpe_adj_accuracy, 4),
        feature_importances=fi,
        y_test=y_te,
        y_pred=y_pred,
        dpe_actual=dpe_te,
        dpe_predicted=dpe_pred,
    )


# ───────────────────────────────────────────────────────────────────────────
# INFERENCE
# ───────────────────────────────────────────────────────────────────────────

def predict_building(
    building: dict,
    result: RegressorResult,
    artifacts: PreprocessArtifacts,
    *,
    sector_ref_eui: Optional[float] = None,
    tertiaire_horizon: int = 2030,
) -> dict:
    """
    Predict EUI, DPE label and Decret Tertiaire compliance for a single building.

    Parameters
    ----------
    building          : dict with building attributes
    result            : RegressorResult from train_regressor()
    artifacts         : PreprocessArtifacts from build_feature_matrix()
    sector_ref_eui    : reference EUI for Tertiaire gap (kWh/m²/yr)
                        If None, predicted EUI is used as its own reference
    tertiaire_horizon : compliance target year (2030/2040/2050)

    Returns
    -------
    dict with:
        predicted_eui          : float  kWh/m²/yr
        dpe_label              : str    A–G
        tertiaire_gap_pct      : float  % above/below 2030 target
        tertiaire_compliant    : bool
        carbon_intensity       : float  kgCO₂e/kWh (from input energy mix)
        savings_potential_pct  : float  % reduction needed to reach target

    Example
    -------
    >>> predict_building({
    ...     "category":        "Enseignement",
    ...     "subcategory":     "Enseignement primaire - École élémentaire",
    ...     "compliance_case": "1A",
    ...     "pct_electricity": 75.0,
    ...     "pct_gas_network": 25.0,
    ...     "n_categories":    1,
    ...     "n_subcategories": 1,
    ... }, result, artifacts, sector_ref_eui=120.0)
    """
    vec           = transform_single_building(building, artifacts)
    predicted_eui = float(result.model.predict(vec)[0])
    dpe_label     = assign_dpe_label(predicted_eui)

    ref_eui   = sector_ref_eui if sector_ref_eui is not None else predicted_eui
    gap       = float(compute_tertiaire_gap(predicted_eui, ref_eui, horizon=tertiaire_horizon))
    compliant = gap <= 0.0

    rate             = TERTIAIRE_TARGETS.get(tertiaire_horizon, 0.40)
    target_eui       = ref_eui * (1.0 - rate)
    savings_pct      = max(0.0, round((predicted_eui - target_eui) / predicted_eui * 100, 1))

    # Carbon intensity from input energy mix (already computed in transform)
    carbon = building.get("carbon_intensity_kgco2e_kwh",
                          0.0571 * building.get("pct_electricity", 100) / 100)

    return {
        "predicted_eui":       round(predicted_eui, 1),
        "dpe_label":           dpe_label,
        "tertiaire_gap_pct":   gap,
        "tertiaire_compliant": compliant,
        "target_eui":          round(target_eui, 1),
        "savings_potential_pct": savings_pct,
        "carbon_intensity_kgco2e_kwh": round(carbon, 4),
    }


def print_summary(result: RegressorResult) -> None:
    """Print a clean evaluation summary."""
    print("\n" + "="*60)
    print("AXIOM EUI REGRESSOR EVALUATION")
    print("="*60)
    print(f"\n📊 EUI Regression (kWh/m²/yr)")
    print(f"   R²       : {result.r2:.3f}")
    print(f"   MAE      : {result.mae:.1f} kWh/m²")
    print(f"   RMSE     : {result.rmse:.1f} kWh/m²")
    print(f"\n🏠 DPE Label (derived from predicted EUI)")
    print(f"   Exact    : {result.dpe_accuracy:.1%}")
    print(f"   ±1 band  : {result.dpe_adjacent_accuracy:.1%}")
    print(f"\n🔍 Top 10 Feature Importances:")
    print(result.feature_importances.head(10).to_string(index=False))
    print(f"\n📉 EUI Distribution (predicted vs actual):")
    print(f"   Actual  — mean: {result.y_test.mean():.1f}  median: {np.median(result.y_test):.1f}")
    print(f"   Predict — mean: {result.y_pred.mean():.1f}  median: {np.median(result.y_pred):.1f}")
