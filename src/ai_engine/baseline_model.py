"""
AXIOM — Linear Regression Baseline Model
=========================================
Predicts expected EUI (kWh/m²/yr) from building characteristics.
Residual = actual EUI - predicted EUI → fed into K-Means anomaly triage.

Usage:
    from src.ai_engine.baseline_model import train_baseline, predict_savings
    result = train_baseline(df)
    savings = predict_savings(actual_eui, predicted_eui, floor_area, carrier)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ENERGY_PRICE = {
    "Electricite": 0.20, "Gaz": 0.08, "Fioul": 0.12,
    "Reseau de chaleur": 0.09, "Reseau de froid": 0.06, "Bois": 0.04,
}
CARBON_FACTORS = {
    "Electricite": 0.052, "Gaz": 0.227, "Fioul": 0.324,
    "Reseau de chaleur": 0.110, "Reseau de froid": 0.025, "Bois": 0.030,
}
EU_ETS_PRICE = 65.0  # EUR/tCO2e


@dataclass
class BaselineResult:
    model:        LinearRegression
    scaler:       StandardScaler
    r2:           float
    mae:          float
    feature_names: list[str]
    y_test:       np.ndarray
    y_pred:       np.ndarray
    residuals:    np.ndarray   # actual - predicted EUI


def train_baseline(
    df: pd.DataFrame,
    *,
    target_col: str = "eui_kwh_m2",
    test_size: float = 0.20,
    random_state: int = 42,
) -> BaselineResult:
    """
    Train a Linear Regression to predict EUI from:
      - floor_area_m2 (log-transformed)
      - activity_enc  (label-encoded sector)
      - hdd           (heating degree days — climate zone)
      - pct_electricity, pct_gas, pct_heat_network
    Returns residuals for K-Means anomaly triage.
    """
    feature_cols = [
        c for c in [
            "log_floor_area", "activity_enc", "hdd",
            "pct_electricity", "pct_gas", "pct_heat_network",
        ] if c in df.columns
    ]
    if not feature_cols:
        # Fallback: use whatever numeric columns exist
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c != target_col]

    df = df.dropna(subset=feature_cols + [target_col]).copy()

    # Log-transform floor area if raw column present
    if "floor_area_m2" in df.columns and "log_floor_area" not in df.columns:
        df["log_floor_area"] = np.log1p(df["floor_area_m2"])
        if "log_floor_area" not in feature_cols:
            feature_cols.append("log_floor_area")

    X = df[feature_cols].values
    y = df[target_col].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LinearRegression()
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)
    r2  = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    residuals = y_te - y_pred

    logger.info("Baseline Linear Regression — R²=%.3f  MAE=%.1f kWh/m²", r2, mae)

    return BaselineResult(
        model=model,
        scaler=scaler,
        r2=round(r2, 4),
        mae=round(mae, 1),
        feature_names=feature_cols,
        y_test=y_te,
        y_pred=y_pred,
        residuals=residuals,
    )


def predict_savings(
    actual_eui: float,
    predicted_eui: float,
    floor_area: float,
    carrier: str = "Electricite",
) -> dict:
    """
    Compute savings potential from gap between actual and predicted EUI.
    actual_eui    — measured EUI (kWh/m²/yr)
    predicted_eui — model baseline EUI (kWh/m²/yr)
    Returns kWh/yr, EUR/yr, kgCO2e/yr, carbon value EUR/yr, adjusted payback.
    """
    gap_eui   = max(0.0, actual_eui - predicted_eui)   # only positive gaps
    sav_kwh   = round(gap_eui * floor_area)
    price     = ENERGY_PRICE.get(carrier, 0.15)
    cf        = CARBON_FACTORS.get(carrier, 0.1)
    sav_eur   = round(sav_kwh * price)
    sav_co2   = round(sav_kwh * cf)
    carbon_val = round((sav_co2 / 1000) * EU_ETS_PRICE)
    total_benefit = sav_eur + carbon_val

    return {
        "gap_eui_kwh_m2":         round(gap_eui, 1),
        "savings_kwh_yr":         sav_kwh,
        "savings_eur_yr":         sav_eur,
        "savings_kgco2e_yr":      sav_co2,
        "carbon_value_eur_yr":    carbon_val,
        "total_benefit_eur_yr":   total_benefit,
        "actual_eui":             round(actual_eui, 1),
        "predicted_eui":          round(predicted_eui, 1),
        "floor_area_m2":          floor_area,
        "carrier":                carrier,
    }


def print_summary(result: BaselineResult) -> None:
    print("\n" + "="*55)
    print("AXIOM LINEAR REGRESSION BASELINE")
    print("="*55)
    print(f"  R²   : {result.r2:.3f}")
    print(f"  MAE  : {result.mae:.1f} kWh/m²/yr")
    print(f"  Features: {result.feature_names}")
    print(f"  Residuals — mean: {result.residuals.mean():.1f}  std: {result.residuals.std():.1f}")
    coefs = dict(zip(result.feature_names, result.model.coef_))
    print("  Coefficients:")
    for k, v in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {k:30s}: {v:+.3f}")
