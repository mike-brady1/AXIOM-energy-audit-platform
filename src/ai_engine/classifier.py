"""
AXIOM — Step 3: Random Forest EUI Regressor
============================================
Predicts log1p(EUI), converts back to kWh/m²/yr via expm1,
then derives DPE label and Décret Tertiaire compliance.

log1p transform on y compresses the right tail caused by
high-EUI sectors (data centres, laundries), improving R²
and reducing MAE for the typical building range.

Usage:
    from src.ai_engine.classifier import train_regressor, predict_building
    result = train_regressor(X, y, meta, artifacts)
    pred   = predict_building({...}, result, artifacts, sector_ref_eui=120.0)
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


@dataclass
class RegressorResult:
    model:                  RandomForestRegressor
    r2:                     float
    mae:                    float
    rmse:                   float
    dpe_accuracy:           float
    dpe_adjacent_accuracy:  float
    feature_importances:    pd.DataFrame
    y_test:                 np.ndarray    # actual EUI (kWh/m², not log)
    y_pred:                 np.ndarray    # predicted EUI (kWh/m², not log)
    dpe_actual:             np.ndarray
    dpe_predicted:          np.ndarray


def train_regressor(
    X: np.ndarray,
    y: np.ndarray,                        # log1p(EUI) from build_feature_matrix
    meta: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    *,
    n_estimators: int = 200,
    test_size: float = 0.20,
    random_state: int = 42,
    sample_size: Optional[int] = None,
) -> RegressorResult:
    """
    Train RF Regressor on log1p(EUI). Evaluates on back-transformed kWh/m².
    """
    y_dpe = meta["dpe_label"].values

    if sample_size and sample_size < len(X):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X, y, y_dpe = X[idx], y[idx], y_dpe[idx]
        logger.info("Subsampled to %d rows", sample_size)

    X_tr, X_te, y_tr, y_te, dpe_tr, dpe_te = train_test_split(
        X, y, y_dpe, test_size=test_size, random_state=random_state
    )
    logger.info("Train: %d  Test: %d", len(X_tr), len(X_te))

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    logger.info("Training RF regressor on log1p(EUI)...")
    model.fit(X_tr, y_tr)

    # Back-transform predictions to kWh/m²
    y_pred_log  = model.predict(X_te)
    y_pred      = np.expm1(y_pred_log)
    y_test      = np.expm1(y_te)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info("R²=%.3f  MAE=%.1f  RMSE=%.1f kWh/m²", r2, mae, rmse)

    dpe_pred         = assign_dpe_label(pd.Series(y_pred)).values
    dpe_accuracy     = float((dpe_pred == dpe_te).mean())

    labels_order = ["A", "B", "C", "D", "E", "F", "G"]
    label_idx    = {l: i for i, l in enumerate(labels_order)}
    adj = np.abs(
        np.array([label_idx.get(p, 0) for p in dpe_pred]) -
        np.array([label_idx.get(a, 0) for a in dpe_te])
    ) <= 1
    dpe_adj_accuracy = float(adj.mean())
    logger.info("DPE exact=%.3f  adj=%.3f", dpe_accuracy, dpe_adj_accuracy)

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
        y_test=y_test,
        y_pred=y_pred,
        dpe_actual=dpe_te,
        dpe_predicted=dpe_pred,
    )


def predict_building(
    building: dict,
    result: RegressorResult,
    artifacts: PreprocessArtifacts,
    *,
    sector_ref_eui: Optional[float] = None,
    tertiaire_horizon: int = 2030,
) -> dict:
    """
    Predict EUI, DPE label and Décret Tertiaire compliance for one building.
    """
    vec           = transform_single_building(building, artifacts)
    predicted_eui = float(np.expm1(result.model.predict(vec)[0]))
    dpe_label     = assign_dpe_label(predicted_eui)

    ref_eui   = sector_ref_eui if sector_ref_eui is not None else predicted_eui
    gap       = compute_tertiaire_gap(predicted_eui, ref_eui, horizon=tertiaire_horizon)
    compliant = gap <= 0.0

    rate        = TERTIAIRE_TARGETS.get(tertiaire_horizon, 0.40)
    target_eui  = ref_eui * (1.0 - rate)
    savings_pct = max(0.0, round((predicted_eui - target_eui) / predicted_eui * 100, 1))

    carbon = building.get(
        "carbon_intensity_kgco2e_kwh",
        round(0.0571 * building.get("pct_electricity", 100) / 100, 4)
    )

    return {
        "predicted_eui":              round(predicted_eui, 1),
        "dpe_label":                  dpe_label,
        "tertiaire_gap_pct":          gap,
        "tertiaire_compliant":        compliant,
        "target_eui":                 round(target_eui, 1),
        "savings_potential_pct":      savings_pct,
        "carbon_intensity_kgco2e_kwh": round(float(carbon), 4),
    }


def print_summary(result: RegressorResult) -> None:
    print("\n" + "="*60)
    print("AXIOM EUI REGRESSOR EVALUATION")
    print("="*60)
    print(f"\n📊 EUI Regression (kWh/m²/yr)  [y = log1p → expm1]")
    print(f"   R²       : {result.r2:.3f}")
    print(f"   MAE      : {result.mae:.1f} kWh/m²")
    print(f"   RMSE     : {result.rmse:.1f} kWh/m²")
    print(f"\n🏠 DPE Label (derived from predicted EUI)")
    print(f"   Exact    : {result.dpe_accuracy:.1%}")
    print(f"   ±1 band  : {result.dpe_adjacent_accuracy:.1%}")
    print(f"\n🔍 Top 10 Feature Importances:")
    print(result.feature_importances.head(10).to_string(index=False))
    print(f"\n📉 EUI (predicted vs actual):")
    print(f"   Actual  — mean: {result.y_test.mean():.1f}  median: {np.median(result.y_test):.1f}")
    print(f"   Predict — mean: {result.y_pred.mean():.1f}  median: {np.median(result.y_pred):.1f}")
