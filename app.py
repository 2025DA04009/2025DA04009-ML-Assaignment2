import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# App Settings
# ============================================================

st.set_page_config(
    page_title="Credit Default Classifier Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Default Prediction Dashboard")
st.caption("A Streamlit dashboard for evaluating saved classification models on credit default test data.")


# ============================================================
# Constants
# ============================================================

MODEL_FOLDER = Path("model")

TARGET_NAME = "default.payment.next.month"

MODEL_REGISTRY = {
    "Logistic Regression": {
        "file": "logistic_regression.pkl",
        "needs_scaling": True
    },
    "Decision Tree": {
        "file": "decision_tree.pkl",
        "needs_scaling": False
    },
    "K-Nearest Neighbors": {
        "file": "knn.pkl",
        "needs_scaling": True
    },
    "Naive Bayes": {
        "file": "naive_bayes.pkl",
        "needs_scaling": True
    },
    "Random Forest": {
        "file": "random_forest.pkl",
        "needs_scaling": False
    }
}

SCALER_FILE = "scaler.pkl"


# ============================================================
# Cached Loaders
# ============================================================

@st.cache_resource
def get_model(model_filename: str):
    """Load a trained model from disk."""
    model_path = MODEL_FOLDER / model_filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not available: {model_path}")

    return joblib.load(model_path)


@st.cache_resource
def get_scaler():
    """Load the fitted scaler from disk."""
    scaler_path = MODEL_FOLDER / SCALER_FILE

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not available: {scaler_path}")

    return joblib.load(scaler_path)


# ============================================================
# Data Utilities
# ============================================================

def read_input_file(uploaded_file):
    """Read CSV or Excel data uploaded through Streamlit."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format. Please upload CSV, XLSX, or XLS file.")


def clean_credit_data(raw_df: pd.DataFrame):
    """
    Prepare uploaded credit card data for prediction.

    Expected:
    - Dataset must include target column.
    - ID column is removed if present.
    - Feature values are converted to numeric where possible.
    """
    df = raw_df.copy()

    df.columns = [col.strip() for col in df.columns]

    if TARGET_NAME not in df.columns:
        raise ValueError(f"Target column '{TARGET_NAME}' was not found in uploaded data.")

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    y = df[TARGET_NAME]
    X = df.drop(columns=[TARGET_NAME])

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    missing_feature_rows = X.isnull().sum().sum()
    missing_target_rows = y.isnull().sum()

    if missing_feature_rows > 0:
        X = X.fillna(X.median(numeric_only=True))

    if missing_target_rows > 0:
        valid_rows = y.notnull()
        X = X.loc[valid_rows]
        y = y.loc[valid_rows]

    return X, y.astype(int)


def apply_preprocessing(X: pd.DataFrame, model_display_name: str):
    """Scale features only for models that require scaling."""
    model_info = MODEL_REGISTRY[model_display_name]

    if model_info["needs_scaling"]:
        scaler = get_scaler()
        transformed = scaler.transform(X)
        return pd.DataFrame(transformed, columns=X.columns, index=X.index)

    return X


# ============================================================
# Evaluation Utilities
# ============================================================

def get_prediction_scores(model, X):
    """
    Return probability scores if supported.

    Some models may not expose predict_proba. In that case, return None.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    return None


def calculate_scores(y_true, y_pred, y_prob=None):
    """Calculate common classification metrics."""
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        results["ROC AUC"] = roc_auc_score(y_true, y_prob)
    else:
        results["ROC AUC"] = np.nan

    return results


def evaluate_saved_model(model_name: str, X_raw: pd.DataFrame, y_true: pd.Series):
    """Load, preprocess, predict, and evaluate one selected model."""
    model_details = MODEL_REGISTRY[model_name]
    model = get_model(model_details["file"])

    X_ready = apply_preprocessing(X_raw, model_name)

    predictions = model.predict(X_ready)
    probabilities = get_prediction_scores(model, X_ready)

    metric_values = calculate_scores(y_true, predictions, probabilities)

    return {
        "model": model,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metric_values
    }


def plot_confusion_matrix(y_true, y_pred, title):
    """Create a matplotlib confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Default", "Default"])
    ax.set_yticklabels(["No Default", "Default"])

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, cm[row, col], ha="center", va="center")

    return fig


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Navigation")

uploaded_dataset = st.sidebar.file_uploader(
    "Upload test dataset",
    type=["csv", "xlsx", "xls"]
)

st.sidebar.info(
    "The uploaded file should contain the target column: "
    f"`{TARGET_NAME}`"
)


# ============================================================
# Main Layout
# ============================================================

overview_tab, single_model_tab, comparison_tab, data_tab = st.tabs(
    [
        "📌 Overview",
        "🔍 Single Model Evaluation",
        "📊 Model Comparison",
        "🧾 Dataset Preview"
    ]
)


# ============================================================
# Overview Tab
# ============================================================

with overview_tab:
    st.subheader("Project Overview")

    st.write(
        """
        This application evaluates trained classification models for predicting whether
        a credit card customer will default in the next month.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Available Models", len(MODEL_REGISTRY))

    with col2:
        st.metric("Target Column", TARGET_NAME)

    with col3:
        st.metric("Model Directory", str(MODEL_FOLDER))

    st.markdown("### Models Included")

    model_summary = pd.DataFrame(
        [
            {
                "Model": name,
                "File Name": details["file"],
                "Uses Scaler": "Yes" if details["needs_scaling"] else "No"
            }
            for name, details in MODEL_REGISTRY.items()
        ]
    )

    st.dataframe(model_summary, use_container_width=True)

    st.markdown("### How to Use")
    st.markdown(
        """
        1. Upload the test dataset from the sidebar.
        2. Open the **Single Model Evaluation** tab to review one model.
        3. Open the **Model Comparison** tab to benchmark all saved models.
        4. Check the **Dataset Preview** tab to inspect uploaded data.
        """
    )


# ============================================================
# Load Uploaded Dataset
# ============================================================

data_is_ready = False
X_test = None
y_test = None
uploaded_df = None

if uploaded_dataset is not None:
    try:
        uploaded_df = read_input_file(uploaded_dataset)
        X_test, y_test = clean_credit_data(uploaded_df)
        data_is_ready = True
    except Exception as error:
        st.error(f"Unable to process uploaded file: {error}")


# ============================================================
# Single Model Evaluation Tab
# ============================================================

with single_model_tab:
    st.subheader("Evaluate One Classification Model")

    if not data_is_ready:
        st.warning("Please upload a valid test dataset from the sidebar.")
    else:
        selected_model = st.selectbox(
            "Choose a model to evaluate",
            list(MODEL_REGISTRY.keys())
        )

        if st.button("Run Selected Model", type="primary"):
            try:
                output = evaluate_saved_model(selected_model, X_test, y_test)

                st.success(f"Evaluation completed for {selected_model}")

                metric_cols = st.columns(6)
                metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "MCC"]

                for col, metric_name in zip(metric_cols, metric_names):
                    value = output["metrics"].get(metric_name, np.nan)
                    display_value = "N/A" if pd.isna(value) else f"{value:.4f}"
                    col.metric(metric_name, display_value)

                st.markdown("### Confusion Matrix")
                fig = plot_confusion_matrix(
                    y_test,
                    output["predictions"],
                    f"{selected_model} Confusion Matrix"
                )
                st.pyplot(fig)

                st.markdown("### Classification Report")
                report = classification_report(
                    y_test,
                    output["predictions"],
                    zero_division=0,
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)

                st.markdown("### Prediction Sample")
                prediction_view = X_test.copy()
                prediction_view["Actual"] = y_test.values
                prediction_view["Predicted"] = output["predictions"]

                if output["probabilities"] is not None:
                    prediction_view["Default Probability"] = output["probabilities"]

                st.dataframe(prediction_view.head(20), use_container_width=True)

            except Exception as error:
                st.error(f"Model evaluation failed: {error}")


# ============================================================
# Model Comparison Tab
# ============================================================

with comparison_tab:
    st.subheader("Compare All Available Models")

    if not data_is_ready:
        st.warning("Please upload a valid test dataset from the sidebar.")
    else:
        if st.button("Evaluate All Models", type="primary"):
            comparison_rows = []

            for model_name in MODEL_REGISTRY:
                try:
                    result = evaluate_saved_model(model_name, X_test, y_test)

                    row = {"Model": model_name}
                    row.update(result["metrics"])
                    comparison_rows.append(row)

                except Exception as error:
                    comparison_rows.append(
                        {
                            "Model": model_name,
                            "Accuracy": np.nan,
                            "Precision": np.nan,
                            "Recall": np.nan,
                            "F1 Score": np.nan,
                            "ROC AUC": np.nan,
                            "MCC": np.nan,
                            "Error": str(error)
                        }
                    )

            comparison_df = pd.DataFrame(comparison_rows)

            st.markdown("### Performance Table")
            st.dataframe(comparison_df, use_container_width=True)

            valid_results = comparison_df.dropna(subset=["F1 Score"])

            if not valid_results.empty:
                best_row = valid_results.sort_values(
                    by="F1 Score",
                    ascending=False
                ).iloc[0]

                st.success(
                    f"Best model based on F1 Score: "
                    f"{best_row['Model']} with F1 Score = {best_row['F1 Score']:.4f}"
                )

                chart_df = valid_results.set_index("Model")[
                    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
                ]

                st.markdown("### Metric Comparison Chart")
                st.bar_chart(chart_df)

            csv_output = comparison_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Comparison Results",
                data=csv_output,
                file_name="model_comparison_results.csv",
                mime="text/csv"
            )


# ============================================================
# Dataset Preview Tab
# ============================================================

with data_tab:
    st.subheader("Uploaded Dataset Preview")

    if not data_is_ready:
        st.info("Upload a dataset to preview its structure.")
    else:
        st.markdown("### Raw Uploaded Data")
        st.dataframe(uploaded_df.head(25), use_container_width=True)

        st.markdown("### Dataset Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Rows", uploaded_df.shape[0])

        with c2:
            st.metric("Columns", uploaded_df.shape[1])

        with c3:
            default_rate = y_test.mean() * 100
            st.metric("Default Rate", f"{default_rate:.2f}%")

        st.markdown("### Feature Columns Used for Prediction")
        st.write(list(X_test.columns))

        st.markdown("### Missing Values After Cleaning")
        missing_summary = X_test.isnull().sum().reset_index()
        missing_summary.columns = ["Feature", "Missing Count"]
        st.dataframe(missing_summary, use_container_width=True)