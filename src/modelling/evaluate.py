"""Model evaluation module - evaluates trained model on test set and generates metrics and plots."""
import pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, brier_score_loss
from src.common.utils import load_yaml, project_path

def main():
    """Evaluate model on test set:
    1. Load trained model and test data
    2. Generate probability predictions
    3. Calculate ROC-AUC, PR-AUC, and Brier score metrics
    4. Generate PR and ROC curve plots
    5. Save metrics table and visualizations
    """
    # Load configuration and create output directories
    cfg = load_yaml("config/experiment_base.yaml")
    processed = project_path(cfg["paths"]["processed"])
    reports   = project_path(cfg["paths"]["reports"])
    (reports/"figures").mkdir(parents=True, exist_ok=True)
    (reports/"tables").mkdir(parents=True, exist_ok=True)

    # Load trained model and test dataset
    model = joblib.load(project_path("models/best_model.joblib"))
    test = pd.read_csv(processed / "test.csv")
    X = test.drop(columns="__target__").values  # Test features
    y = test["__target__"].values  # Test target

    # Generate probability predictions and compute evaluation metrics
    p = model.predict_proba(X)[:,1]  # Probability of positive class
    roc = roc_auc_score(y, p); pr = average_precision_score(y, p); brier = brier_score_loss(y, p)
    # Save test metrics to CSV file
    pd.DataFrame([{"roc_auc": roc, "pr_auc": pr, "brier": brier}]).to_csv(reports/"tables/test_metrics.csv", index=False)

    # Calculate precision-recall and ROC curves
    prec, rec, _ = precision_recall_curve(y, p)  # PR curve data
    fpr, tpr, _  = roc_curve(y, p)  # ROC curve data
    
    # Plot and save Precision-Recall curve
    plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (Test)")
    plt.savefig(reports/"figures/pr_curve.png", dpi=160, bbox_inches="tight"); plt.close()
    
    # Plot and save ROC curve
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Test)")
    plt.savefig(reports/"figures/roc_curve.png", dpi=160, bbox_inches="tight"); plt.close()
    
    # Print summary metrics to console
    print(f"[evaluate] ROC={roc:.3f} PR={pr:.3f} Brier={brier:.3f}")

if __name__ == "__main__":
    main()
