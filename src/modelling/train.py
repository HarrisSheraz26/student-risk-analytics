"""Model training module - trains baseline logistic regression model on preprocessed data."""
import joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from src.common.utils import load_yaml, project_path

def main():
    """Train baseline logistic regression model:
    1. Load preprocessed training data
    2. Initialize logistic regression with balanced class weights
    3. Fit model on training features and target
    4. Save trained model to models directory
    """
    # Load configuration and preprocessed data path
    cfg = load_yaml("config/experiment_base.yaml")
    processed = project_path(cfg["paths"]["processed"])
    models = project_path("models")
    models.mkdir(parents=True, exist_ok=True)

    # Load preprocessed training dataset
    train = pd.read_csv(processed / "train.csv")
    X = train.drop(columns="__target__").values  # Features (excluding target)
    y = train["__target__"].values  # Target variable

    # Train logistic regression with balanced class weighting to handle imbalance
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, y)
    
    # Save trained model as joblib file for later use in evaluation and inference
    joblib.dump(model, models / "best_model.joblib")
    print("[train] Saved models/best_model.joblib")

if __name__ == "__main__":
    main()
