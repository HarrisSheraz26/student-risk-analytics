"""
Benchmarking module - compares multiple ML models using 5-fold cross-validation.

Outputs:
- reports/tables/model_comparison.csv
- reports/figures/roc_comparison.png

This module is part of the Student Risk Analytics pipeline.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.common.utils import load_yaml, project_path


def infer_target_column(df: pd.DataFrame) -> str:
    """
    Infer target column name from common conventions used in this project.
    """
    candidates = [
        "__target__",
        "pass_fail",
        "target",
        "label",
        "at_risk",
        "risk",
        "y",
        "G3",  # common in student performance datasets
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not infer target column. Expected one of: "
        "__target__/pass_fail/target/label/at_risk/risk/y or G3."
    )


def build_models(random_state: int = 42) -> Dict[str, object]:
    """
    Build baseline models for comparison.
    """
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, n_jobs=None, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        # probability=True keeps things consistent for ROC/PR utilities
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
    }

    # XGBoost is optional but expected in your thesis plan
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )
    except Exception:
        warnings.warn(
            "xgboost not available. Skipping XGBoost in benchmark.",
            RuntimeWarning,
        )

    return models


def load_train_dataframe(cfg: dict) -> pd.DataFrame:
    """
    Load processed train.csv using config paths,
    with a small fallback to interim/cleaned.csv if needed.
    """
    processed_dir = project_path(cfg["paths"]["processed"])
    train_path = processed_dir / "train.csv"

    if train_path.exists():
        return pd.read_csv(train_path)

    # fallback (should rarely be needed)
    interim_dir = project_path(cfg["paths"].get("interim", "data/interim"))
    cleaned_path = interim_dir / "cleaned.csv"
    if cleaned_path.exists():
        return pd.read_csv(cleaned_path)

    raise FileNotFoundError(
        "Could not find processed train.csv or interim cleaned.csv. "
        "Run: python -m src.data.validate and python -m src.data.preprocess"
    )


def save_roc_comparison(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    reports_dir: Path,
    cv: StratifiedKFold,
) -> None:
    """
    Create ROC comparison plot using cross-validated predictions.
    """
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ROC_PLOT = fig_dir / "roc_comparison.png"

    plt.figure()

    for name, model in models.items():
        # Prefer predict_proba for consistency
        method = "predict_proba"
        try:
            y_scores = cross_val_predict(
                model, X, y, cv=cv, method=method, n_jobs=-1
            )[:, 1]
        except Exception:
            # fallback to decision_function if needed
            method = "decision_function"
            y_scores = cross_val_predict(
                model, X, y, cv=cv, method=method, n_jobs=-1
            )
            # decision scores may be unbounded - roc_curve handles this fine

        auc = roc_auc_score(y, y_scores)
        fpr, tpr, _ = roc_curve(y, y_scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison (5-fold CV)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PLOT, dpi=300)
    plt.close()


def main():
    cfg = load_yaml("config/experiment_base.yaml")

    reports_dir = project_path(cfg["paths"]["reports"])
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_train_dataframe(cfg)

    target_col = infer_target_column(df)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    random_state = int(cfg.get("random_state", 42))
    models = build_models(random_state=random_state)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    scoring = {
        "acc": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    print("Running 5-fold CV benchmark for all models...\n")

    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        row = {
            "model": name,
            "accuracy": float(np.mean(scores["test_acc"])),
            "f1": float(np.mean(scores["test_f1"])),
            "roc_auc": float(np.mean(scores["test_roc_auc"])),
            "pr_auc": float(np.mean(scores["test_pr_auc"])),
        }
        rows.append(row)

        print(
            f"[{name}] "
            f"acc={row['accuracy']:.3f} "
            f"f1={row['f1']:.3f} "
            f"roc_auc={row['roc_auc']:.3f} "
            f"pr_auc={row['pr_auc']:.3f}"
        )

    results = pd.DataFrame(rows).sort_values(
        by=["pr_auc", "roc_auc"], ascending=False
    )

    out_csv = tables_dir / "model_comparison.csv"
    results.to_csv(out_csv, index=False)

    save_roc_comparison(models, X, y, reports_dir, cv)

    print("\n✅ Benchmark complete.")
    print(f"✅ Model comparison table: {out_csv}")
    print(f"✅ ROC comparison plot:    {reports_dir / 'figures' / 'roc_comparison.png'}")


if __name__ == "__main__":
    main()
