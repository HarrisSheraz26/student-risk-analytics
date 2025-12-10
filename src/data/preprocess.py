"""Data preprocessing module - handles feature engineering, encoding, scaling,
and train-test split creation with proper column transformation.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.common.utils import load_yaml, project_path

def make_target(df, target_name: str):
    """Create target variable from configuration.
    
    Args:
        df: Input DataFrame containing grade columns
        target_name: Either 'pass_fail' (binary: G3 >= 10) or 'G3' (regression)
    
    Returns:
        numpy array with target values
    """
    if target_name == "pass_fail":
        # Binary classification: 1 if final grade >= 10, else 0
        return (df["G3"] >= 10).astype(int)
    elif target_name == "G3":
        # Regression target: raw final grade values
        return df["G3"]
    raise ValueError("Unknown target")

def main():
    """Main preprocessing pipeline:
    1. Load configuration and cleaned data
    2. Create target variable (pass/fail or G3 regression)
    3. Identify numeric and categorical feature columns from config
    4. Apply OneHotEncoder to categorical and StandardScaler to numeric
    5. Split into train/test sets with stratification
    6. Save transformed datasets and feature metadata
    """
    # Load experiment configuration and feature specifications
    cfg = load_yaml("config/experiment_base.yaml")
    fcfg = load_yaml("config/features.yaml")
    interim = project_path(cfg["paths"]["interim"])
    processed = project_path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    # Load validated/cleaned data and create target variable
    df = pd.read_csv(interim / "cleaned.csv", sep=";")
    y = make_target(df, cfg["target"])
    X = df.copy()
    
    # Remove target column from features if doing binary classification
    if cfg["target"] == "pass_fail" and "G3" in X.columns:
        X = X.drop(columns=["G3"])

    # Extract numeric and categorical columns that exist in data
    num_cols = [c for c in fcfg.get("numeric", []) if c in X.columns]
    cat_cols = [c for c in fcfg.get("categorical", []) if c in X.columns]

    # Create preprocessing pipeline: OneHotEncode categorical, StandardScale numeric
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop"  # Ignore any columns not in cat_cols or num_cols
    )

    # Split data into train and test sets with stratification for balanced classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("test_size", 0.2), random_state=cfg.get("random_seed", 42),
        stratify=(y if len(np.unique(y)) <= 20 else None)  # Stratify if < 20 unique values
    )

    # Apply transformations: fit on train, transform both train and test
    Xt_train = ct.fit_transform(X_train)
    Xt_test  = ct.transform(X_test)

    # Build feature names list: one-hot encoded names + numeric column names
    feat_names = []
    if cat_cols:
        ohe = ct.named_transformers_["cat"]
        feat_names.extend(list(ohe.get_feature_names_out(cat_cols)))
    if num_cols:
        feat_names.extend(num_cols)

    # Create DataFrames with processed features and target variable
    train_df = pd.DataFrame(Xt_train, columns=feat_names); train_df["__target__"] = np.array(y_train)
    test_df  = pd.DataFrame(Xt_test,  columns=feat_names); test_df["__target__"]  = np.array(y_test)

    # Export processed datasets and metadata to processed directory
    train_df.to_csv(processed/"train.csv", index=False)
    test_df.to_csv(processed/"test.csv", index=False)
    # Save feature names metadata for later reference in modeling/explainability
    (processed/"meta.json").write_text(json.dumps({"feature_names": feat_names}, indent=2), encoding="utf-8")
    print("[preprocess] train.csv, test.csv, meta.json written")

if __name__ == "__main__":
    main()
