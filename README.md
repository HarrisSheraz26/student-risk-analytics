# Student Risk Analytics

An early-warning analytics pipeline to predict student academic risk using machine learning.

## Project Structure

- `src/data/` - Data validation and preprocessing modules
- `src/modelling/` - Model training, evaluation, benchmarking, tuning, ablation, and fairness analysis
- `src/explainability/` - Feature importance and SHAP analysis for model interpretability
- `src/dashboard/` - Streamlit interactive demo application
- `data/` - Raw and processed datasets
  - `raw/` - Original student datasets
  - `interim/` - Cleaned intermediate data
  - `processed/` - Final train/test splits and metadata
- `reports/` - Generated analysis outputs
  - `tables/` - CSV exports of results (model comparisons, metrics, fairness)
  - `figures/` - Visualizations (SHAP plots, etc.)
  - `metrics/` - JSON outputs (best parameters, fairness summaries)
- `models/` - Trained and tuned model files (joblib format)

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Full Pipeline Run

### Data Validation & Preprocessing

```bash
python -m src.data.validate
python -m src.data.preprocess
```

### Model Training & Evaluation

```bash
python -m src.modelling.train
python -m src.modelling.evaluate
```

### Benchmarking & Hyperparameter Tuning

```bash
python -m src.modelling.benchmark
python -m src.modelling.tune
python -m src.modelling.evaluate
```

### Ablation & Fairness Analysis

```bash
python -m src.modelling.ablation
python -m src.modelling.fairness
```

### Explainability Analysis

```bash
python -m src.explainability.feature_importance
python -m src.explainability.shap_analysis
```

## Dashboard

Launch the interactive Streamlit dashboard:

```bash
python -m pip install streamlit
streamlit run src/dashboard/app.py
```

The dashboard provides interactive visualizations and model predictions for exploring student risk assessment.

## Outputs

### Model Comparisons
- **Model comparison metrics:** `reports/tables/model_comparison.csv`
- **Tuning results:** `reports/tables/tuning_results.csv`

### Analysis Results
- **Ablation study:** `reports/tables/ablation_g1_g2.csv`
- **Fairness metrics:** `reports/tables/fairness_by_sex.csv`
- **Feature importance:** `reports/tables/feature_importance.csv`
- **SHAP analysis:** `reports/tables/shap_importance.csv` and `reports/figures/shap_*.png`

### Model Files
- **Best model:** `models/best_model.joblib`
- **Tuned models:** `models/tuned_models/*.joblib`
- **Decision thresholds:** `models/threshold.json`

## Key Features

- **Early-Warning System:** Identifies at-risk students early for intervention
- **Multiple Models:** Compares decision trees, logistic regression, random forests, SVM, and XGBoost
- **Fairness Analysis:** Evaluates demographic parity and bias across protected attributes
- **Model Interpretability:** Provides SHAP values and feature importance rankings
- **Hyperparameter Optimization:** Uses Bayesian and grid-based tuning strategies
- **Ablation Studies:** Analyzes contribution of feature groups to model performance

## Requirements

See `requirements.txt` for all dependencies. Key packages include:
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning models and evaluation
- xgboost - Gradient boosting
- shap - Model explainability
- streamlit - Interactive dashboard

## Notes

This project is designed for academic risk prediction and includes fairness-aware machine learning practices to ensure equitable predictions across different student demographics.
