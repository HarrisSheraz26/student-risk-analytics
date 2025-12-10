\"\"\"Hyperparameter tuning module - uses RandomizedSearchCV to optimize all models
aiming to maximize PR-AUC (average precision) on the validation set.
\"\"\"\nimport os\nimport json\nimport joblib\nimport numpy as np\nimport pandas as pd\n\nfrom sklearn.model_selection import StratifiedKFold, RandomizedSearchCV\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.svm import SVC\n\n# Optional XGBoost\ntry:\n    from xgboost import XGBClassifier\n    HAS_XGB = True\nexcept ImportError:\n    HAS_XGB = False\n\nfrom src.common.utils import load_yaml, project_path\n\n\ndef main():\n    \"\"\"Hyperparameter tuning pipeline using RandomizedSearchCV:\n    1. Load preprocessed training data\n    2. Define parameter search spaces for each model\n    3. Run RandomizedSearchCV (30 iterations) with PR-AUC scoring\n    4. Save best model for each type and best parameters JSON\n    5. Select overall best model and save as best_model.joblib\n    6. Export results table and metrics\n    \"\"\"\n    # Load configuration and data paths\n    cfg = load_yaml("config/experiment_base.yaml")
    processed = project_path(cfg["paths"]["processed"])
    reports = project_path(cfg["paths"]["reports"])
    models_dir = project_path("models")

    train_path = processed / "train.csv"
    df = pd.read_csv(train_path)

    if "__target__" not in df.columns:
        raise ValueError(
            "Expected a '__target__' column in processed/train.csv. "
            "Make sure you have run src/data/preprocess.py first."
        )

    # Extract features and target from training data
    X = df.drop(columns="__target__").values
    y = df["__target__"].values

    # Setup 5-fold stratified cross-validation for hyperparameter evaluation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=cfg.get("random_seed", 42)
    )

    # Initialize models with baseline hyperparameters
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

    # Define hyperparameter search spaces for each model
    param_distributions = {
        "Logistic Regression": {
            "C": np.logspace(-3, 3, 50),  # Regularization strength
            "solver": ["lbfgs", "liblinear"],  # Optimization algorithm
        },
        "Decision Tree": {
            "max_depth": [None] + list(range(2, 31)),  # Tree depth limit
            "min_samples_split": list(range(2, 21)),  # Min samples to split node
            "min_samples_leaf": list(range(1, 21)),  # Min samples in leaf
            "criterion": ["gini", "entropy", "log_loss"],  # Split criterion
        },
        "Random Forest": {
            "n_estimators": list(range(100, 801, 50)),  # Number of trees
            "max_depth": [None] + list(range(3, 31)),  # Tree depth
            "min_samples_split": list(range(2, 21)),  # Min samples to split
            "min_samples_leaf": list(range(1, 11)),  # Min samples in leaf
            "max_features": ["sqrt", "log2", None],  # Features to consider
        },
        "SVM": {
            "C": np.logspace(-3, 3, 50),  # Regularization parameter
            "kernel": ["rbf", "linear"],  # Kernel type
            "gamma": ["scale", "auto"],  # Kernel coefficient
        },
        "XGBoost": {
            "n_estimators": list(range(100, 801, 50)),  # Number of boosting rounds
            "learning_rate": np.linspace(0.01, 0.3, 30),  # Shrinkage rate
            "max_depth": list(range(2, 11)),  # Max tree depth
            "subsample": np.linspace(0.6, 1.0, 9),  # Row subsampling ratio
            "colsample_bytree": np.linspace(0.6, 1.0, 9),  # Column subsampling ratio
            "reg_lambda": np.logspace(-2, 2, 20),  # L2 regularization
        }
    }

    # Create directories for tuned models and results
    tuned_dir = models_dir / "tuned_models"
    tuned_dir.mkdir(parents=True, exist_ok=True)

    reports_tables = reports / "tables"
    reports_tables.mkdir(parents=True, exist_ok=True)
    reports_metrics = reports / "metrics"
    reports_metrics.mkdir(parents=True, exist_ok=True)

    tuning_results = []
    best_params_all = {}

    best_model = None
    best_score = -1.0
    best_name = None

    print("Running RandomizedSearchCV for all models...\n")

    # Tune each model with RandomizedSearchCV
    for name, model in models.items():
        search_space = param_distributions.get(name, {})
        if not search_space:
            print(f"[{name}] WARNING: empty search space, skipping.")
            continue

        # Run randomized search with 30 iterations, optimizing for PR-AUC
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=search_space,
            n_iter=30,  # Number of random parameter combinations to try
            scoring="average_precision",  # Optimize for PR-AUC (best for imbalanced data)
            cv=cv,  # Use 5-fold cross-validation
            random_state=cfg.get("random_seed", 42),
            n_jobs=-1,  # Parallel processing using all CPU cores
            verbose=1
        )

        # Fit the search on training data
        search.fit(X, y)

        best_est = search.best_estimator_
        best_cv_score = float(search.best_score_)
        best_params_all[name] = search.best_params_

        # Save the best model for this model type as joblib file
        model_filename = f"{name.lower().replace(' ', '_')}_tuned.joblib"
        model_path = tuned_dir / model_filename
        joblib.dump(best_est, model_path)

        tuning_results.append({
            "Model": name,
            "Best_CV_PR_AUC": best_cv_score,
            "Best_Model_Path": str(model_path)
        })

        print(f"[{name}] best CV PR-AUC = {best_cv_score:.4f}")

        # Track the overall best model across all types
        if best_cv_score > best_score:
            best_score = best_cv_score
            best_model = best_est
            best_name = name

    # Save detailed tuning results table sorted by PR-AUC
    results_df = (
        pd.DataFrame(tuning_results)
        .sort_values("Best_CV_PR_AUC", ascending=False)
    )
    results_df.to_csv(reports_tables / "tuning_results.csv", index=False)

    # Save best hyperparameters for all models as JSON
    with open(reports_metrics / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_all, f, indent=2)

    # Save overall best model (best PR-AUC across all model types)
    if best_model is not None:
        best_model_path = models_dir / "best_model.joblib"
        joblib.dump(best_model, best_model_path)
        print(f"\n✅ Best overall model: {best_name} (CV PR-AUC = {best_score:.4f})")
        print("✅ Saved as:", best_model_path)
    else:
        print("No best model was selected – check your tuning configuration.")

    print("\n✅ Tuning complete.")
    print("✅ Full tuning results:", reports_tables / "tuning_results.csv")
    print("✅ Best parameters JSON:", reports_metrics / "best_params.json")


if __name__ == "__main__":
    main()
