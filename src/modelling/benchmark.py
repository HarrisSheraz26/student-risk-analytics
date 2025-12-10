\"\"\"Benchmarking module - compares multiple ML models using 5-fold cross-validation
on the training data to identify best-performing baseline models.\n\"\"\"\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom sklearn.model_selection import StratifiedKFold, cross_validate\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.svm import SVC\n\nfrom sklearn.metrics import (\n    make_scorer,\n    accuracy_score,\n    f1_score,\n    roc_auc_score,\n    average_precision_score,\n    roc_curve,\n)\n\n# Optional XGBoost - gracefully handle if not installed\ntry:\n    from xgboost import XGBClassifier\n    HAS_XGB = True\nexcept ImportError:\n    HAS_XGB = False\n\nfrom src.common.utils import load_yaml, project_path\n\n\ndef main():\n    \"\"\"Benchmark all models using 5-fold stratified cross-validation:\n    1. Load preprocessed training data\n    2. Initialize multiple model types (LR, DT, RF, SVM, XGB)\n    3. Run cross-validation with multiple scoring metrics\n    4. Generate model comparison table and ROC curve visualization\n    5. Save results to reports directory\n    \"\"\""
    # Load configuration and data paths
    cfg = load_yaml("config/experiment_base.yaml")
    processed = project_path(cfg["paths"]["processed"])
    reports = project_path(cfg["paths"]["reports"])

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

    # Initialize all candidate models with baseline hyperparameters
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42)

    # Define scoring metrics: accuracy, F1, ROC-AUC, and Precision-Recall AUC
    scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    }

    # Setup 5-fold stratified cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=cfg.get("random_seed", 42)
    )

    results = []

    print("Running 5-fold CV benchmark for all models...\n")

    # Train and evaluate each model using cross-validation
    for name, model in models.items():
        # Perform cross-validation and collect scores for all metrics
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

        # Calculate mean scores across folds
        row = {
    "Model": name,
    "accuracy": float(np.mean(scores["test_accuracy"])),
    "f1": float(np.mean(scores["test_f1"])),
    "roc_auc": float(np.mean(scores["test_roc_auc"])),
    "pr_auc": float(np.mean(scores["test_pr_auc"])),
        }
        results.append(row)

        print(
            f"[{name}] "
            f"acc={row['accuracy']:.3f} "
            f"f1={row['f1']:.3f} "
            f"roc_auc={row['roc_auc']:.3f} "
            f"pr_auc={row['pr_auc']:.3f}"
        )

    # Create output directories for tables and figures
    tables_dir = reports / "tables"
    figures_dir = reports / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Sort results by PR-AUC (primary metric) and save comparison table
    df_results = (
        pd.DataFrame(results)
        .set_index("Model")
        .sort_values("pr_auc", ascending=False)
    )
    comparison_path = tables_dir / "model_comparison.csv"
    df_results.to_csv(comparison_path)

    # Generate ROC curves for visual comparison (fit on full training data)
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        # Train each model on full training data for visualization
        model.fit(X, y)
        y_prob = model.predict_proba(X)[:, 1]
        # Calculate ROC curve coordinates
        fpr, tpr, _ = roc_curve(y, y_prob)
        plt.plot(fpr, tpr, label=name)

    # Plot random classifier baseline
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curves - Model Comparison (Train)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    # Save ROC comparison plot
    roc_plot_path = figures_dir / "roc_comparison.png"
    plt.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✅ Benchmark complete.")
    print("✅ Model comparison table:", comparison_path)
    print("✅ ROC comparison plot:   ", roc_plot_path)


if __name__ == "__main__":
    main()
