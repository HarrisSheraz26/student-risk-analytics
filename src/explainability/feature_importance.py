import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.common.utils import load_yaml, project_path


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

    model_path = models_dir / "best_model.joblib"
    model = joblib.load(model_path)

    # Ensure the model supports feature importances
    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            f"Loaded model type {type(model).__name__} does not support feature_importances_."
        )

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(
            "Feature importance length mismatch. "
            "This can happen if train columns changed after tuning. "
            "Re-run tune.py if needed."
        )

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # Save table
    tables_dir = reports / "tables"
    figures_dir = reports / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fi_path = tables_dir / "feature_importance.csv"
    fi.to_csv(fi_path, index=False)

    # Plot top 20
    top = fi.head(20).iloc[::-1]  # reverse for horizontal bar chart

    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top 20 Feature Importances (Best Model)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plot_path = figures_dir / "feature_importance_top20.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("✅ Feature importance complete.")
    print("✅ Saved table:", fi_path)
    print("✅ Saved plot:", plot_path)


if __name__ == "__main__":
    main()
