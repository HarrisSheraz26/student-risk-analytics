"""Fairness analysis module - evaluates model for demographic parity and bias
across protected attributes (e.g., gender) to ensure equitable predictions."""
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from src.common.utils import load_yaml, project_path


def infer_group_from_onehot(df: pd.DataFrame, prefixes):
    """
    Try to infer a categorical group label from one-hot columns.
    Example: prefix='sex' -> columns like sex_F, sex_M
    Returns (group_name, series) or (None, None) if not found.
    """
    for prefix in prefixes:
        cols = [c for c in df.columns if c.startswith(prefix + "_")]
        if cols:
            arr = df[cols].to_numpy()
            row_sum = arr.sum(axis=1)

            # For rows with at least one indicator set, pick the max col
            idx = np.argmax(arr, axis=1)
            labels = [cols[i].split(prefix + "_", 1)[1] for i in idx]

            labels = np.where(row_sum == 0, "unknown", labels)
            return prefix, pd.Series(labels, name=prefix)
    return None, None


def confusion_rates(y_true, y_pred):
    """
    Returns TPR, FPR with safe zero-division handling.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    return tpr, fpr


def safe_auc(func, y_true, y_prob):
    """
    ROC-AUC / PR-AUC can fail if a group has only one class.
    """
    try:
        return float(func(y_true, y_prob))
    except Exception:
        return np.nan


def main():
    cfg = load_yaml("config/experiment_base.yaml")
    processed = project_path(cfg["paths"]["processed"])
    reports = project_path(cfg["paths"]["reports"])
    models_dir = project_path("models")

    threshold = cfg.get("decision_threshold", 0.5)

    test_path = processed / "test.csv"
    df = pd.read_csv(test_path)

    if "__target__" not in df.columns:
        raise ValueError("Expected '__target__' in processed/test.csv. Run preprocess first.")

    # Infer group column from one-hot features in processed data
    group_name, group_series = infer_group_from_onehot(
        df,
        prefixes=["sex", "gender"]   # flexible naming
    )

    if group_name is None:
        print("⚠️ Could not find one-hot gender/sex columns in processed data.")
        print("   This fairness script currently expects columns like 'sex_F', 'sex_M'.")
        print("   You can extend prefixes or add a mapping export in preprocess if needed.")
        return

    X = df.drop(columns="__target__")
    y_true = df["__target__"].to_numpy()

    model_path = models_dir / "best_model.joblib"
    model = joblib.load(model_path)

    # Predict
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        # fallback for rare models without predict_proba
        y_prob = model.decision_function(X)
        # squash to (0,1) roughly
        y_prob = 1 / (1 + np.exp(-y_prob))

    y_pred = (y_prob >= threshold).astype(int)

    # Overall metrics
    overall = {
        "GroupAttr": group_name,
        "GroupValue": "ALL",
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": safe_auc(roc_auc_score, y_true, y_prob),
        "pr_auc": safe_auc(average_precision_score, y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "positive_prediction_rate": float(np.mean(y_pred)),
    }
    overall["tpr"], overall["fpr"] = confusion_rates(y_true, y_pred)

    rows = [overall]

    # Group metrics
    df_groups = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        group_name: group_series
    })

    for gval, sub in df_groups.groupby(group_name):
        yt = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        ypb = sub["y_prob"].to_numpy()

        row = {
            "GroupAttr": group_name,
            "GroupValue": str(gval),
            "n": int(len(sub)),
            "accuracy": float(accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan,
            "roc_auc": safe_auc(roc_auc_score, yt, ypb),
            "pr_auc": safe_auc(average_precision_score, yt, ypb),
            "brier": float(brier_score_loss(yt, ypb)),
            "positive_prediction_rate": float(np.mean(yp)),
        }
        row["tpr"], row["fpr"] = confusion_rates(yt, yp)
        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Parity-style summaries (simple, no extra libs)
    subgroup = out_df[out_df["GroupValue"] != "ALL"].copy()

    def span(col):
        vals = subgroup[col].dropna().to_numpy()
        if len(vals) == 0:
            return np.nan
        return float(np.max(vals) - np.min(vals))

    fairness_summary = {
        "decision_threshold": float(threshold),
        "demographic_parity_diff(ppr_span)": span("positive_prediction_rate"),
        "equal_opportunity_diff(tpr_span)": span("tpr"),
        "fpr_diff(fpr_span)": span("fpr"),
    }

    tables_dir = reports / "tables"
    metrics_dir = reports / "metrics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    out_path = tables_dir / "fairness_by_sex.csv"
    out_df.to_csv(out_path, index=False)

    summary_path = metrics_dir / "fairness_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(fairness_summary, f, indent=2)

    print("✅ Fairness audit complete.")
    print("✅ Group attribute:", group_name)
    print("✅ Saved subgroup metrics:", out_path)
    print("✅ Saved fairness summary:", summary_path)
    print("\nFairness spans (higher = more disparity):")
    for k, v in fairness_summary.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
