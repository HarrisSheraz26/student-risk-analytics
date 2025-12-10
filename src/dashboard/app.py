from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from src.common.utils import load_yaml, project_path


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "__target__",
        "pass_fail",
        "target",
        "label",
        "at_risk",
        "risk",
        "y",
        "G3",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_threshold(threshold_path: Path, default: float = 0.5) -> float:
    if threshold_path.exists():
        try:
            with open(threshold_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            val = float(obj.get("threshold", default))
            return max(0.0, min(1.0, val))
        except Exception:
            return default
    return default


def get_feature_columns(df: pd.DataFrame, target_col: Optional[str]) -> List[str]:
    cols = df.columns.tolist()
    if target_col and target_col in cols:
        cols.remove(target_col)
    # ignore any internal columns if you ever add them later
    cols = [c for c in cols if not c.startswith("__")]
    return cols


def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    # fallback to decision_function if needed
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores).ravel()
        # convert to pseudo-prob with sigmoid
        return 1 / (1 + np.exp(-scores))

    # last resort
    preds = model.predict(X)
    return np.asarray(preds).astype(float).ravel()


def main():
    st.set_page_config(
        page_title="Student Risk Analytics Dashboard",
        page_icon="🎓",
        layout="wide",
    )

    cfg = load_yaml("config/experiment_base.yaml")

    processed_dir = project_path(cfg["paths"]["processed"])
    reports_dir = project_path(cfg["paths"]["reports"])
    models_dir = project_path("models")

    test_path = processed_dir / "test.csv"

    # best model saved by tuning script
    best_model_path = models_dir / "best_model.joblib"

    # fallback if needed
    fallback_rf = models_dir / "tuned_models" / "random_forest_tuned.joblib"

    threshold_path = models_dir / "threshold.json"
    default_threshold = load_threshold(threshold_path, default=0.5)

    # Sidebar
    st.sidebar.title("Settings")
    decision_threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.01,
    )

    # Header
    st.markdown(
        "## 🎓 Student Risk Analytics Dashboard\n"
        "Demo dashboard powered by the tuned best model."
    )

    # Load model
    model = None
    if best_model_path.exists():
        model = joblib.load(best_model_path)
    elif fallback_rf.exists():
        model = joblib.load(fallback_rf)

    if model is None:
        st.error(
            "Model not found. Please run:\n"
            "- python -m src.modelling.train\n"
            "- python -m src.modelling.tune"
        )
        return

    tab1, tab2 = st.tabs(["🔍 Predict from Test Set", "📤 Upload Processed CSV"])

    # -------------------------
    # Tab 1: Predict from test
    # -------------------------
    with tab1:
        st.markdown("### Predict using processed test records")

        if not test_path.exists():
            st.error(
                f"Processed test set not found at: {test_path}\n\n"
                "Run:\n"
                "- python -m src.data.validate\n"
                "- python -m src.data.preprocess"
            )
            return

        df_test = pd.read_csv(test_path)
        target_col = infer_target_column(df_test)
        feature_cols = get_feature_columns(df_test, target_col)

        st.write(f"Detected **{len(feature_cols)}** feature columns.")

        if len(df_test) == 0:
            st.warning("Test dataset is empty.")
            return

        idx = st.number_input(
            "Select row index",
            min_value=0,
            max_value=max(0, len(df_test) - 1),
            value=0,
            step=1,
        )

        row_df = df_test.loc[[int(idx)], feature_cols]

        st.dataframe(row_df, width="stretch")

        colA, colB = st.columns([1, 2])

        with colA:
            run = st.button("Run Early-Warning Prediction")

        with colB:
            if run:
                probs = predict_proba_safe(model, row_df)
                risk_prob = float(probs[0])
                pred_label = int(risk_prob >= decision_threshold)

                st.markdown("#### Risk Probability")
                st.markdown(
                    f"<div style='font-size:36px; font-weight:700;'>{risk_prob:.3f}</div>",
                    unsafe_allow_html=True,
                )

                if pred_label == 1:
                    st.warning("⚠️ At Risk")
                else:
                    st.success("✅ Not At Risk")

                if target_col:
                    actual = int(df_test.loc[int(idx), target_col])
                    st.caption(f"Actual label: {actual}")

    # ---------------------------------
    # Tab 2: Upload processed CSV
    # ---------------------------------
    with tab2:
        st.markdown("### Upload a processed CSV for batch prediction")
        st.info(
            "Upload a CSV with the same processed feature columns used by the model. "
            "If a target column exists, it will be ignored for prediction."
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            df_up = pd.read_csv(uploaded)

            up_target = infer_target_column(df_up)
            up_feature_cols = get_feature_columns(df_up, up_target)

            # We should align uploaded columns to expected columns from test set
            # If test exists, use it as reference for feature schema
            expected_cols = None
            if test_path.exists():
                df_ref = pd.read_csv(test_path)
                ref_target = infer_target_column(df_ref)
                expected_cols = get_feature_columns(df_ref, ref_target)

            if expected_cols is not None:
                missing = [c for c in expected_cols if c not in df_up.columns]
                if missing:
                    st.error(
                        "Your uploaded file is missing required processed columns:\n\n"
                        + ", ".join(missing)
                    )
                    st.stop()

                X_up = df_up[expected_cols].copy()
            else:
                X_up = df_up[up_feature_cols].copy()

            probs = predict_proba_safe(model, X_up)
            preds = (probs >= decision_threshold).astype(int)

            out = df_up.copy()
            out["risk_probability"] = probs
            out["predicted_label"] = preds

            st.success(f"Predicted {len(out)} rows successfully.")

            st.dataframe(out.head(50), width="stretch")

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=csv_bytes,
                file_name="student_risk_predictions.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
