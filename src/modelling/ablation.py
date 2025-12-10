"""Ablation study module - evaluates model performance with and without specific features (G1/G2)
to quantify their contribution to predictions.
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Optional XGBoost - gracefully handle if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.common.utils import load_yaml, project_path


def build_models():
    """Initialize all candidate models with baseline hyperparameters.
    
    Returns:
        Dictionary mapping model names to initialized model instances
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )
    return models


def run_cv(df, scenario_label, cv):
    if "__target__" not in df.columns:
        raise ValueError("Expected '__target__' in train.csv. Run preprocess first.")

    X = df.drop(columns="__target__").values
    y = df["__target__"].values

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    rows = []
    for name, model in build_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

        rows.append({
            "Scenario": scenario_label,
            "Model": name,
            "accuracy": float(np.mean(scores["test_accuracy"])),
            "f1": float(np.mean(scores["test_f1"])),
            "roc_auc": float(np.mean(scores["test_roc_auc"])),
            "pr_auc": float(np.mean(scores["test_pr_auc"])),
        })

    return pd.DataFrame(rows)


def main():
    """Ablation study comparing model performance with vs without G1/G2 features:
    1. Load preprocessed training data
    2. Run 5-fold CV benchmark WITH G1/G2 features
    3. Run 5-fold CV benchmark WITHOUT G1/G2 features
    4. Compare performance deltas to quantify G1/G2 importance
    5. Save combined results table
    """
    cfg = load_yaml("config/experiment_base.yaml")
    processed = project_path(cfg["paths"]["processed"])
    reports = project_path(cfg["paths"]["reports"])

    train_path = processed / "train.csv"
    df = pd.read_csv(train_path)

    # Setup 5-fold stratified cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=cfg.get("random_seed", 42)
    )

    # Scenario A: Include G1 and G2 (first and second period grades)
    df_with = df.copy()

    # Scenario B: Exclude G1 and G2 to test their impact
    drop_cols = [c for c in ["G1", "G2"] if c in df.columns]
    df_without = df.drop(columns=drop_cols)

    # Run cross-validation for both feature sets
    res_with = run_cv(df_with, "With G1/G2", cv)
    res_without = run_cv(df_without, "Without G1/G2", cv)

    # Combine results from both scenarios for comparison
    final = pd.concat([res_with, res_without], ignore_index=True)

    # Save ablation study results to reports
    tables_dir = reports / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    out_path = tables_dir / "ablation_g1_g2.csv"
    final.to_csv(out_path, index=False)

    print("✅ Ablation complete.")
    print("✅ Saved:", out_path)


if __name__ == "__main__":
    main()