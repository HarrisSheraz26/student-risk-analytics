"""Streamlit dashboard application - interactive web interface for student risk predictions
and batch scoring on uploaded data."""
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Ensure imports work when running streamlit
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import load_yaml, project_path


st.set_page_config(
    page_title="Student Risk Analytics",
    page_icon="🎓",
    layout="wide"
)


def load_artifacts():
    cfg = load_yaml("config/experiment_base.yaml")
    processed = project_path(cfg["paths"]["processed"])
    models_dir = project_path("models")

    model = joblib.load(models_dir / "best_model.joblib")

    train_path = processed / "train.csv"
    test_path = processed / "test.csv"

    df_train = pd.read_csv(train_path) if train_path.exists() else None
    df_test = pd.read_csv(test_path) if test_path.exists() else None

    threshold = cfg.get("decision_threshold", 0.5)

    return model, df_train, df_test, float(threshold)


def predict_proba_safe(model, X_df):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_df)[:, 1]
    scores = model.decision_function(X_df)
    return 1 / (1 + np.exp(-scores))


def main():
    st.title("🎓 Student Risk Analytics Dashboard")
    st.caption("Demo dashboard powered by the tuned best model.")

    model, df_train, df_test, default_threshold = load_artifacts()

    st.sidebar.header("Settings")
    threshold = st.sidebar.slider(
        "Decision Threshold",
        0.10, 0.90, default_threshold, 0.01
    )

    tab1, tab2 = st.tabs(["🔍 Predict from Test Set", "📤 Upload Processed CSV"])

    # -------------------
    # Tab 1: test row
    # -------------------
    with tab1:
        st.subheader("Predict using processed test records")

        if df_test is None:
            st.warning("processed/test.csv not found. Run preprocess first.")
        else:
            feature_cols = [c for c in df_test.columns if c != "__target__"]
            st.write(f"Detected **{len(feature_cols)}** feature columns.")

            idx = st.number_input(
                "Select row index",
                min_value=0,
                max_value=max(0, len(df_test) - 1),
                value=0,
                step=1
            )

            row = df_test.iloc[[int(idx)]]
            X_row = row[feature_cols]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.dataframe(X_row, use_container_width=True)

            with col2:
                if st.button("Run Early-Warning Prediction", type="primary"):
                    prob = float(predict_proba_safe(model, X_row)[0])
                    label = 1 if prob >= threshold else 0

                    st.metric("Risk Probability", f"{prob:.3f}")
                    st.success("⚠️ At Risk" if label == 1 else "✅ Not At Risk")

                    if "__target__" in row.columns:
                        st.caption(f"Actual label: {int(row['__target__'].iloc[0])}")

    # -------------------
    # Tab 2: upload CSV
    # -------------------
    with tab2:
        st.subheader("Batch scoring from uploaded processed CSV")

        st.info(
            "Upload a CSV with the same processed feature columns. "
            "If '__target__' exists, it will be ignored."
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            df_up = pd.read_csv(uploaded)

            if "__target__" in df_up.columns:
                df_up = df_up.drop(columns="__target__")

            # Use train columns for exact ordering
            if df_train is not None and "__target__" in df_train.columns:
                expected = [c for c in df_train.columns if c != "__target__"]

                missing = [c for c in expected if c not in df_up.columns]
                extra = [c for c in df_up.columns if c not in expected]

                if missing:
                    st.error(f"Missing expected columns: {missing}")
                    return

                if extra:
                    st.warning(f"Extra columns will be ignored: {extra}")

                df_up = df_up[expected]

            probs = predict_proba_safe(model, df_up)
            labels = (probs >= threshold).astype(int)

            out = df_up.copy()
            out["risk_probability"] = probs
            out["risk_label"] = labels

            st.dataframe(out.head(50), use_container_width=True)
            st.caption("Showing first 50 rows.")

            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="student_risk_predictions.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
