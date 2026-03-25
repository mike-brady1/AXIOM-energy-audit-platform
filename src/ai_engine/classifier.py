"""
AXIOM — Step 3: Random Forest Classifier
=========================================
Predicts DPE label (A→G) and EUI tier (Low/Medium/High) for any
tertiary building using sector, energy mix and compliance features.

Model
-----
  RandomForestClassifier
  - n_estimators : 200
  - class_weight : 'balanced'  (handles A/G imbalance)
  - max_features : 'sqrt'      (standard for classification)
  - n_jobs       : -1          (all cores)

Inputs  : X, y from build_feature_matrix() + meta DataFrame
Outputs : trained model, evaluation report, feature importances

Usage:
    from src.ai_engine.classifier import train_classifier, predict_building
    result = train_classifier(X, y_dpe, meta, artifacts)
    label  = predict_building({...}, result.model, artifacts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from src.processing.preprocessor import PreprocessArtifacts, transform_single_building

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# OUTPUT CONTAINER
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class ClassifierResult:
    model_dpe:              RandomForestClassifier
    model_tier:             RandomForestClassifier
    accuracy_dpe:           float
    accuracy_tier:          float
    f1_macro_dpe:           float
    f1_macro_tier:          float
    cv_scores_dpe:          np.ndarray          # 5-fold CV F1 macro
    report_dpe:             str                 # sklearn classification_report
    report_tier:            str
    confusion_dpe:          np.ndarray
    feature_importances:    pd.DataFrame        # feature → importance score
    dpe_classes:            list[str]
    tier_classes:           list[str]


# ───────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ───────────────────────────────────────────────────────────────────────────

def train_classifier(
    X: np.ndarray,
    meta: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    *,
    n_estimators: int = 200,
    test_size: float = 0.20,
    random_state: int = 42,
    run_cv: bool = True,
    sample_size: Optional[int] = None,
) -> ClassifierResult:
    """
    Train Random Forest classifiers for DPE label and EUI tier.

    Parameters
    ----------
    X             : feature matrix from build_feature_matrix()
    meta          : metadata DataFrame from build_feature_matrix()
    artifacts     : PreprocessArtifacts (encoders + scaler)
    n_estimators  : number of trees (default 200)
    test_size     : train/test split ratio (default 0.20)
    random_state  : reproducibility seed
    run_cv        : run 5-fold cross-validation (slower but gives confidence interval)
    sample_size   : optional subsample for faster iteration (e.g. 200_000)
                    Set None to train on full dataset

    Returns
    -------
    ClassifierResult with both trained models + evaluation metrics
    """
    y_dpe  = meta["dpe_label"].values
    y_tier = meta["eui_tier"].values

    # ── Optional subsample ────────────────────────────────────────────────
    if sample_size and sample_size < len(X):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_s, y_dpe_s, y_tier_s = X[idx], y_dpe[idx], y_tier[idx]
        logger.info("Subsampled to %d rows for training", sample_size)
    else:
        X_s, y_dpe_s, y_tier_s = X, y_dpe, y_tier

    # ── Train / test split (stratified on DPE label) ────────────────────
    X_tr, X_te, y_dpe_tr, y_dpe_te = train_test_split(
        X_s, y_dpe_s,
        test_size=test_size,
        stratify=y_dpe_s,
        random_state=random_state,
    )
    _, _, y_tier_tr, y_tier_te = train_test_split(
        X_s, y_tier_s,
        test_size=test_size,
        stratify=y_tier_s,
        random_state=random_state,
    )

    logger.info("Train: %d  Test: %d", len(X_tr), len(X_te))

    # ── Model definitions ─────────────────────────────────────────────────
    rf_params = dict(
        n_estimators=n_estimators,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    model_dpe  = RandomForestClassifier(**rf_params)
    model_tier = RandomForestClassifier(**rf_params)

    # ── Train DPE model ─────────────────────────────────────────────────
    logger.info("Training DPE classifier...")
    model_dpe.fit(X_tr, y_dpe_tr)
    y_dpe_pred   = model_dpe.predict(X_te)
    acc_dpe      = accuracy_score(y_dpe_te, y_dpe_pred)
    f1_dpe       = f1_score(y_dpe_te, y_dpe_pred, average="macro")
    report_dpe   = classification_report(y_dpe_te, y_dpe_pred)
    conf_dpe     = confusion_matrix(y_dpe_te, y_dpe_pred, labels=model_dpe.classes_)
    logger.info("DPE  — accuracy=%.3f  F1 macro=%.3f", acc_dpe, f1_dpe)

    # ── Train EUI tier model ─────────────────────────────────────────────
    logger.info("Training EUI tier classifier...")
    model_tier.fit(X_tr, y_tier_tr)
    y_tier_pred  = model_tier.predict(X_te)
    acc_tier     = accuracy_score(y_tier_te, y_tier_pred)
    f1_tier      = f1_score(y_tier_te, y_tier_pred, average="macro")
    report_tier  = classification_report(y_tier_te, y_tier_pred)
    logger.info("Tier — accuracy=%.3f  F1 macro=%.3f", acc_tier, f1_tier)

    # ── 5-fold cross-validation on DPE (optional) ───────────────────────
    cv_scores = np.array([])
    if run_cv:
        logger.info("Running 5-fold CV on DPE model...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(
            RandomForestClassifier(**rf_params),
            X_s, y_dpe_s,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        logger.info("CV F1 macro: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    # ── Feature importances ───────────────────────────────────────────────
    fi = pd.DataFrame({
        "feature":    artifacts.feature_names,
        "importance": model_dpe.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return ClassifierResult(
        model_dpe=model_dpe,
        model_tier=model_tier,
        accuracy_dpe=round(acc_dpe, 4),
        accuracy_tier=round(acc_tier, 4),
        f1_macro_dpe=round(f1_dpe, 4),
        f1_macro_tier=round(f1_tier, 4),
        cv_scores_dpe=cv_scores,
        report_dpe=report_dpe,
        report_tier=report_tier,
        confusion_dpe=conf_dpe,
        feature_importances=fi,
        dpe_classes=model_dpe.classes_.tolist(),
        tier_classes=model_tier.classes_.tolist(),
    )


# ───────────────────────────────────────────────────────────────────────────
# INFERENCE
# ───────────────────────────────────────────────────────────────────────────

def predict_building(
    building: dict,
    result: ClassifierResult,
    artifacts: PreprocessArtifacts,
) -> dict:
    """
    Predict DPE label, EUI tier and class probabilities for a single building.

    Parameters
    ----------
    building  : dict with building attributes (same keys as RATIO columns)
    result    : ClassifierResult from train_classifier()
    artifacts : PreprocessArtifacts from build_feature_matrix()

    Returns
    -------
    dict with keys:
        dpe_label       : str  (A–G)
        dpe_probability : float (confidence 0–1)
        eui_tier        : str  (Low/Medium/High)
        tier_probability: float
        all_dpe_probs   : dict {label: probability}

    Example
    -------
    >>> predict_building({
    ...     "category":        "Enseignement",
    ...     "subcategory":     "Enseignement primaire - École élémentaire",
    ...     "compliance_case": "1A",
    ...     "pct_electricity": 80.0,
    ...     "pct_gas_network": 20.0,
    ...     "n_categories":    1,
    ... }, result, artifacts)
    """
    vec = transform_single_building(building, artifacts)

    dpe_pred   = result.model_dpe.predict(vec)[0]
    dpe_proba  = result.model_dpe.predict_proba(vec)[0]
    tier_pred  = result.model_tier.predict(vec)[0]
    tier_proba = result.model_tier.predict_proba(vec)[0]

    dpe_conf  = float(dpe_proba.max())
    tier_conf = float(tier_proba.max())

    all_probs = {
        label: round(float(p), 3)
        for label, p in zip(result.dpe_classes, dpe_proba)
    }

    return {
        "dpe_label":        dpe_pred,
        "dpe_probability":  round(dpe_conf, 3),
        "eui_tier":         tier_pred,
        "tier_probability": round(tier_conf, 3),
        "all_dpe_probs":    all_probs,
    }


def print_summary(result: ClassifierResult) -> None:
    """Print a clean evaluation summary to stdout."""
    print("\n" + "="*60)
    print("AXIOM CLASSIFIER EVALUATION")
    print("="*60)
    print(f"\n🏠 DPE Label (A→G)")
    print(f"   Accuracy  : {result.accuracy_dpe:.3f}")
    print(f"   F1 macro  : {result.f1_macro_dpe:.3f}")
    if len(result.cv_scores_dpe) > 0:
        print(f"   CV F1     : {result.cv_scores_dpe.mean():.3f} ± {result.cv_scores_dpe.std():.3f}")
    print(f"\n{result.report_dpe}")
    print(f"\n📊 EUI Tier (Low/Med/High)")
    print(f"   Accuracy  : {result.accuracy_tier:.3f}")
    print(f"   F1 macro  : {result.f1_macro_tier:.3f}")
    print(f"\n🔍 Top 10 Feature Importances (DPE model):")
    print(result.feature_importances.head(10).to_string(index=False))
