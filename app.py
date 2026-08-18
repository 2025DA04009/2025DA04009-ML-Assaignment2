import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
    classification_report,
)


# ============================================================
# Application Settings
# ============================================================

APP_TITLE = "Telco Customer Churn Prediction"
MODEL_DIRECTORY = Path("model")

MODEL_REGISTRY: Dict[str, str] = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

MODELS_REQUIRING_SCALING: List[str] = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Naive Bayes",
]

TARGET_COLUMN = "Churn"


# ============================================================
# Page Setup
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Machine Learning Assignment 2")
st.subheader("Telco Customer Churn Prediction Using Classification Models")

st.markdown(
    """
This application evaluates trained classification models on uploaded Telco customer data.
It supports model-level evaluation, confusion matrix visualization, classification reports,
and comparison of multiple models.
"""
)


# ============================================================
# Cached Loaders
# ============================================================

@st.cache_resource
def load_saved_model(model_label: str):
    """
    Load a trained model from the model directory.
    """
    model_file = MODEL_REGISTRY[model_label]
    model_path = MODEL_DIRECTORY / model_file

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


@st.cache_resource
def load_saved_scaler():
    """
    Load the fitted scaler used during training.
    """
    scaler_path = MODEL_DIRECTORY / "scaler.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    return joblib.load(scaler_path)


# ============================================================
# Data Processing Helpers
# ============================================================

def clean_total_charges(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TotalCharges to numeric and handle invalid values.
    """
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    return data


def remove_unused_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are identifiers or not useful for model prediction.
    """
    columns_to_remove = ["customerID"]

    existing_columns = [col for col in columns_to_remove if col in data.columns]
    if existing_columns:
        data = data.drop(columns=existing_columns)

    return data


def encode_target_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Churn values from Yes/No to 1/0 if needed.
    """
    if TARGET_COLUMN in data.columns:
        if data[TARGET_COLUMN].dtype == "object":
            data[TARGET_COLUMN] = data[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    return data


def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical columns into numeric values using one-hot encoding.
    """
    categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

    if categorical_columns:
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    return data


def align_features_with_training(
    features: pd.DataFrame,
    model
) -> pd.DataFrame:
    """
    Align uploaded feature columns with the feature names used during training.

    This works if the model was trained using scikit-learn with feature names.
    If feature names are unavailable, the function returns the uploaded feature set.
    """
    if hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)

        for column in expected_columns:
            if column not in features.columns:
                features[column] = 0

        features = features[expected_columns]

    return features


def prepare_uploaded_dataset(
    uploaded_data: pd.DataFrame,
    model
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Prepare raw uploaded Telco churn data for prediction and evaluation.

    Returns:
        X: Processed feature matrix
        y: Target values if Churn column is available, otherwise None
    """
    data = uploaded_data.copy()

    data = remove_unused_columns(data)
    data = clean_total_charges(data)
    data = encode_target_column(data)

    if TARGET_COLUMN in data.columns:
        y = data[TARGET_COLUMN]
        X = data.drop(columns=[TARGET_COLUMN])
    else:
        y = None
        X = data

    X = encode_categorical_features(X)
    X = align_features_with_training(X, model)

    return X, y


def apply_scaling_if_required(
    model_name: str,
    features: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply the saved scaler only for models that require scaled input.
    """
    if model_name not in MODELS_REQUIRING_SCALING:
        return features

    scaler = load_saved_scaler()
    scaled_values = scaler.transform(features)

    return pd.DataFrame(
        scaled_values,
        columns=features.columns,
        index=features.index,
    )


# ============================================================
# Evaluation Helpers
# ============================================================

def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    predicted_probability: Optional[np.ndarray]
) -> Dict[str, float]:
    """
    Calculate standard classification metrics.
    """
    metrics = {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1 Score": f1_score(actual, predicted, zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }

    if predicted_probability is not None:
        try:
            metrics["ROC AUC"] = roc_auc_score(actual, predicted_probability)
        except ValueError:
            metrics["ROC AUC"] = np.nan
    else:
        metrics["ROC AUC"] = np.nan

    return metrics


def get_prediction_probability(model, features: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Return positive-class probabilities if the model supports predict_proba.
    """
    if hasattr(model, "predict_proba"):
        probability_values = model.predict_proba(features)
        return probability_values[:, 1]

    return None


def evaluate_classifier(
    model_name: str,
    features: pd.DataFrame,
    actual: pd.Series
) -> Tuple[Dict[str, float], np.ndarray, str]:
    """
    Load model, generate predictions, and calculate metrics.
    """
    model = load_saved_model(model_name)

    processed_features = align_features_with_training(features.copy(), model)
    processed_features = apply_scaling_if_required(model_name, processed_features)

    predictions = model.predict(processed_features)
    probabilities = get_prediction_probability(model, processed_features)

    metrics = calculate_metrics(actual, predictions, probabilities)
    report = classification_report(actual, predictions, zero_division=0)

    return metrics, predictions, report


def show_metric_cards(metrics: Dict[str, float]) -> None:
    """
    Display model metrics in Streamlit metric cards.
    """
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    metric_items = list(metrics.items())
    columns = [col1, col2, col3, col4, col5, col6]

    for index, (metric_name, value) in enumerate(metric_items):
        display_value = "N/A" if pd.isna(value) else f"{value:.4f}"
        columns[index].metric(metric_name, display_value)


def plot_confusion_matrix(actual: pd.Series, predicted: np.ndarray) -> None:
    """
    Display a confusion matrix using matplotlib.
    """
    matrix = confusion_matrix(actual, predicted)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                matrix[row, col],
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "black",
                fontsize=12,
            )

    fig.colorbar(image, ax=ax)
    st.pyplot(fig)


def build_comparison_table(
    features: pd.DataFrame,
    actual: pd.Series
) -> pd.DataFrame:
    """
    Evaluate all registered models and return metrics as a DataFrame.
    """
    comparison_rows = []

    for model_name in MODEL_REGISTRY:
        try:
            metrics, _, _ = evaluate_classifier(model_name, features, actual)
            row = {"Model": model_name}
            row.update(metrics)
            comparison_rows.append(row)

        except Exception as error:
            comparison_rows.append(
                {
                    "Model": model_name,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1 Score": np.nan,
                    "MCC": np.nan,
                    "ROC AUC": np.nan,
                    "Error": str(error),
                }
            )

    return pd.DataFrame(comparison_rows)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Navigation")

selected_section = st.sidebar.radio(
    "Choose a section",
    [
        "Upload Test Data",
        "Evaluate Single Model",
        "Compare All Models",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
Upload a Telco Customer Churn test dataset in CSV format.
If the dataset includes a `Churn` column, the app will calculate evaluation metrics.
"""
)


# ============================================================
# Session State
# ============================================================

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if "prepared_features" not in st.session_state:
    st.session_state.prepared_features = None

if "target_values" not in st.session_state:
    st.session_state.target_values = None


# ============================================================
# Section 1: Upload Test Data
# ============================================================

if selected_section == "Upload Test Data":
    st.header("Upload Test Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_df = uploaded_df

            st.success("File uploaded successfully.")

            st.subheader("Dataset Preview")
            st.dataframe(uploaded_df.head())

            st.subheader("Dataset Summary")
            col1, col2, col3 = st.columns(3)

            col1.metric("Rows", uploaded_df.shape[0])
            col2.metric("Columns", uploaded_df.shape[1])
            col3.metric(
                "Target Available",
                "Yes" if TARGET_COLUMN in uploaded_df.columns else "No",
            )

            if TARGET_COLUMN in uploaded_df.columns:
                st.subheader("Target Column Distribution")
                st.write(uploaded_df[TARGET_COLUMN].value_counts())

            st.info(
                "Go to the evaluation or comparison section after uploading the dataset."
            )

        except Exception as error:
            st.error(f"Unable to read the uploaded file: {error}")

    else:
        st.warning("Please upload a CSV file to continue.")


# ============================================================
# Section 2: Evaluate Single Model
# ============================================================

elif selected_section == "Evaluate Single Model":
    st.header("Evaluate a Selected Model")

    if st.session_state.uploaded_df is None:
        st.warning("Please upload a dataset first from the Upload Test Data section.")

    else:
        chosen_model_name = st.selectbox(
            "Select a trained model",
            list(MODEL_REGISTRY.keys()),
        )

        if st.button("Run Evaluation"):
            try:
                selected_model = load_saved_model(chosen_model_name)

                X, y = prepare_uploaded_dataset(
                    st.session_state.uploaded_df,
                    selected_model,
                )

                if y is None:
                    st.error(
                        "The uploaded dataset does not contain a Churn column, "
                        "so evaluation metrics cannot be calculated."
                    )
                else:
                    X_for_prediction = apply_scaling_if_required(chosen_model_name, X)

                    predictions = selected_model.predict(X_for_prediction)
                    probabilities = get_prediction_probability(
                        selected_model,
                        X_for_prediction,
                    )

                    metrics = calculate_metrics(y, predictions, probabilities)
                    report = classification_report(y, predictions, zero_division=0)

                    st.subheader(f"Evaluation Results: {chosen_model_name}")
                    show_metric_cards(metrics)

                    st.subheader("Confusion Matrix")
                    plot_confusion_matrix(y, predictions)

                    st.subheader("Classification Report")
                    st.text(report)

                    result_df = pd.DataFrame(
                        {
                            "Actual": y,
                            "Predicted": predictions,
                        }
                    )

                    if probabilities is not None:
                        result_df["Churn Probability"] = probabilities

                    st.subheader("Prediction Results")
                    st.dataframe(result_df.head(20))

            except Exception as error:
                st.error(f"Evaluation failed: {error}")


# ============================================================
# Section 3: Compare All Models
# ============================================================

elif selected_section == "Compare All Models":
    st.header("Compare All Classification Models")

    if st.session_state.uploaded_df is None:
        st.warning("Please upload a dataset first from the Upload Test Data section.")

    else:
        if st.button("Compare Models"):
            try:
                example_model_name = list(MODEL_REGISTRY.keys())[0]
                example_model = load_saved_model(example_model_name)

                X, y = prepare_uploaded_dataset(
                    st.session_state.uploaded_df,
                    example_model,
                )

                if y is None:
                    st.error(
                        "The uploaded dataset does not contain a Churn column, "
                        "so model comparison cannot be performed."
                    )
                else:
                    comparison_df = build_comparison_table(X, y)

                    st.subheader("Model Comparison Table")
                    st.dataframe(comparison_df)

                    numeric_columns = [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score",
                        "ROC AUC",
                        "MCC",
                    ]

                    available_numeric_columns = [
                        col for col in numeric_columns if col in comparison_df.columns
                    ]

                    st.subheader("Metric Comparison Chart")

                    chart_df = comparison_df.set_index("Model")[available_numeric_columns]
                    st.bar_chart(chart_df)

                    best_model_row = comparison_df.sort_values(
                        by="F1 Score",
                        ascending=False,
                    ).head(1)

                    if not best_model_row.empty:
                        best_model_name = best_model_row.iloc[0]["Model"]
                        best_f1_score = best_model_row.iloc[0]["F1 Score"]

                        st.success(
                            f"Best model based on F1 Score: "
                            f"{best_model_name} with score {best_f1_score:.4f}"
                        )

            except Exception as error:
                st.error(f"Model comparison failed: {error}")