"""Diagnostics module - generates detailed model evaluation metrics, confusion matrices,
threshold analysis, and feature importance rankings on test set."""
﻿import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.common import utils


def main():
    cfg = utils.load_yaml("config/experiment_base.yaml")
    processed = utils.project_path(cfg["paths"]["processed"])
    reports = utils.project_path(cfg["paths"]["reports"])
    (reports / "tables").mkdir(parents=True, exist_ok=True)
    (reports / "figures").mkdir(parents=True, exist_ok=True)

    # Load model + data
    model_path = utils.project_path("models/best_model.joblib")
    model = joblib.load(model_path)

    test = pd.read_csv(processed / "test.csv")
    X = test.drop(columns="__target__")
    y = test["__target__"].values

    # Probabilities and default predictions (0.50)
    proba = model.predict_proba(X.values)[:, 1]
    yhat_default = (proba >= 0.5).astype(int)

    # 1) Classification report (at 0.50)
    report_dict = classification_report(y, yhat_default, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report_dict).transpose()
    rep_df.to_csv(reports / "tables" / "classification_report.csv")

    # 2) Confusion matrix (CSV + PNG, at 0.50)
    cm = confusion_matrix(y, yhat_default)
    pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
        reports / "tables" / "confusion_matrix.csv"
    )
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (threshold=0.50)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.colorbar()
    plt.savefig(reports / "figures" / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close()

    # 3) Threshold sweep and selection
    target_recall = 0.80
    rows = []
    for t in np.linspace(0.0, 1.0, 101):
        yhat_t = (proba >= t).astype(int)
        r = recall_score(y, yhat_t, zero_division=0)
        p = precision_score(y, yhat_t, zero_division=0)
        f = f1_score(y, yhat_t, zero_division=0)
        rows.append({"threshold": t, "recall": r, "precision": p, "f1": f})
    thr_df = pd.DataFrame(rows)
    thr_df.to_csv(reports / "tables" / "threshold_curve.csv", index=False)

    # Pick the best threshold:
    # 1) among thresholds with recall >= target, choose the one with MAX F1
    # 2) if none reach the target recall, choose the global MAX F1
    subset = thr_df[thr_df["recall"] >= target_recall]
    if not subset.empty:
        best_row = subset.loc[subset["f1"].idxmax()]
        t_star = float(best_row["threshold"])
    else:
        best_row = thr_df.loc[thr_df["f1"].idxmax()]
        t_star = float(best_row["threshold"])

    Path(utils.project_path("models/threshold.json")).write_text(
        json.dumps({"mode": "recall_at", "value": target_recall, "threshold": t_star}, indent=2),
        encoding="utf-8",
    )

    # Plot precision/recall vs threshold
    plt.figure()
    plt.plot(thr_df["threshold"], thr_df["recall"], label="Recall")
    plt.plot(thr_df["threshold"], thr_df["precision"], label="Precision")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs Threshold")
    plt.legend()
    plt.savefig(reports / "figures" / "precision_recall_vs_threshold.png", dpi=160, bbox_inches="tight")
    plt.close()

    # 4) Top features
    try:
        meta = json.loads((processed / "meta.json").read_text(encoding="utf-8"))
        feat_names = meta.get("feature_names", list(range(X.shape[1])))
    except Exception:
        feat_names = list(range(X.shape[1]))  # fallback index names

    top_path = reports / "tables" / "top_features.csv"
    if hasattr(model, "coef_"):
        coefs = pd.Series(model.coef_.ravel(), index=feat_names)
        top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(20)
        top.to_csv(top_path, header=["coefficient"])
        plt.figure()
        top.sort_values().plot(kind="barh")
        plt.title("Top features (|coef|) - Logistic Regression")
        plt.tight_layout()
        plt.savefig(reports / "figures" / "top_features.png", dpi=160, bbox_inches="tight")
        plt.close()
    elif hasattr(model, "feature_importances_"):
        imps = pd.Series(model.feature_importances_, index=feat_names)
        top = imps.sort_values(ascending=False).head(20)
        top.to_csv(top_path, header=["importance"])
        plt.figure()
        top.sort_values().plot(kind="barh")
        plt.title("Top features - Tree-based model")
        plt.tight_layout()
        plt.savefig(reports / "figures" / "top_features.png", dpi=160, bbox_inches="tight")
        plt.close()

    # Console summary for quick sanity check
    yhat_star = (proba >= t_star).astype(int)
    r_star = recall_score(y, yhat_star, zero_division=0)
    p_star = precision_score(y, yhat_star, zero_division=0)
    f_star = f1_score(y, yhat_star, zero_division=0)
    print(f"[diagnostics] threshold={t_star:.3f}  recall={r_star:.3f}  precision={p_star:.3f}  f1={f_star:.3f}")
    print("Wrote: classification_report.csv, confusion_matrix.*, threshold_curve.csv, models/threshold.json, top_features.*")


if __name__ == "__main__":
    main()
