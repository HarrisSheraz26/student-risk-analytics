"""SHAP analysis module - generates SHAP (SHapley Additive exPlanations) values
to provide model-agnostic explanations of individual predictions and global feature importance."""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from src.common.utils import load_yaml, project_path


def _to_2d_shap_values(shap_out, X):
    """
    Converts SHAP output into a 2D array shaped (n_samples, n_features)
    for binary classification.

    Handles:
    - list of arrays (TreeExplainer style)
    - shap.Explanation objects
    - raw numpy arrays
    """

    # 1) TreeExplainer sometimes returns list per class
    if isinstance(shap_out, list):
        # Use positive class if available
        sv = shap_out[1] if len(shap_out) > 1 else shap_out[0]
        return sv

    # 2) New SHAP API often returns Explanation
    if hasattr(shap_out, "values"):
        sv = shap_out.values
    else:
        sv = shap_out

    # 3) If sv is 3D: (samples, features, classes)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        # Pick positive class if it exists
        if sv.shape[2] >= 2:
            sv = sv[:, :, 1]
        else:
            sv = sv[:, :, 0]

    # 4) Final safety check
    if not (isinstance(sv, np.ndarray) and sv.ndim == 2 and sv.shape[1] == X.shape[1]):
        raise ValueError(
            f"Unexpected SHAP shape: {getattr(sv, 'shape', None)} "
            f"for X shape: {X.shape}"
        )

    return sv


def main():
    cfg = load_yaml("config/experiment_base.yaml")

    processed = project_path(cfg["paths"]["processed"])
    reports = project_path(cfg["paths"]["reports"])
    models_dir = project_path("models")

    train_path = processed / "train.csv"
    df = pd.read_csv(train_path)

    if "__target__" not in df.columns:
        raise ValueError("Expected '__target__' in processed/train.csv. Run preprocess first.")

    feature_names = [c for c in df.columns if c != "__target__"]
    X = df.drop(columns="__target__")

    # Optional: sample for speed & cleaner plots
    max_rows = 800
    if len(X) > max_rows:
        X_sample = X.sample(max_rows, random_state=42)
    else:
        X_sample = X

    model_path = models_dir / "best_model.joblib"
    model = joblib.load(model_path)

    figures_dir = reports / "figures"
    tables_dir = reports / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Prefer TreeExplainer for tree models
    try:
        explainer = shap.TreeExplainer(model)
        shap_out = explainer.shap_values(X_sample)
    except Exception:
        explainer = shap.Explainer(model, X_sample)
        shap_out = explainer(X_sample)

    sv = _to_2d_shap_values(shap_out, X_sample)

    # --- Global importance table ---
    mean_abs = np.abs(sv).mean(axis=0)

    shap_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    shap_table_path = tables_dir / "shap_importance.csv"
    shap_imp.to_csv(shap_table_path, index=False)

    # --- Summary plot (beeswarm) ---
    plt.figure()
    shap.summary_plot(sv, X_sample, show=False)
    summary_path = figures_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Bar plot ---
    plt.figure()
    shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
    bar_path = figures_dir / "shap_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ SHAP analysis complete.")
    print("✅ Saved table:", shap_table_path)
    print("✅ Saved summary plot:", summary_path)
    print("✅ Saved bar plot:", bar_path)


if __name__ == "__main__":
    main()
