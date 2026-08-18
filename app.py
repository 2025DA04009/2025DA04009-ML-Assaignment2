import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    layout="wide"
)

st.title("Machine Learning Assignment 2")
st.subheader("Telco Customer Churn Prediction using Classification Models")

st.write(
    """
    This Streamlit application evaluates multiple machine learning classification
    models on uploaded Telco Customer Churn test data. The app displays evaluation
    metrics, confusion matrix, classification report, and model comparison results.
    """
)


# ------------------------------------------------------------
# Model Folder and Model Files
# ------------------------------------------------------------

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

SCALED_MODELS = [
    "Logistic Regression",
    "KNN",
    "Naive Bayes"
]


# ------------------------------------------------------------
# Load Saved Models and Scaler
# ------------------------------------------------------------

@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    return joblib.load(model_path)


@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    return joblib.load(scaler_path)


# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------

st.sidebar.header("Navigation")

section = st.sidebar.radio(
    "Choose a section",
    [
        "Upload Test Data",
        "Evaluate Single Model",
        "Compare All Models"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    Upload a Telco Customer Churn test dataset in CSV format.
    If the dataset includes a Churn column, the app will calculate
    evaluation metrics.
    """
)


# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None


# ------------------------------------------------------------
# Encoding Helper
# ------------------------------------------------------------

def encode_telco_columns(df):
    """
    Encodes Telco categorical columns into numeric values.

    Important:
    This version avoids pd.get_dummies(), because your trained models expect
    original feature names such as Contract, Dependents, InternetService, etc.
    """

    encoded_df = df.copy()

    mapping_rules = {
        "gender": {
            "Female": 0,
            "Male": 1
        },
        "Partner": {
            "No": 0,
            "Yes": 1
        },
        "Dependents": {
            "No": 0,
            "Yes": 1
        },
        "PhoneService": {
            "No": 0,
            "Yes": 1
        },
        "MultipleLines": {
            "No": 0,
            "No phone service": 1,
            "Yes": 2
        },
        "InternetService": {
            "DSL": 0,
            "Fiber optic": 1,
            "No": 2
        },
        "OnlineSecurity": {
            "No": 0,
            "No internet service": 1,
            "Yes": 2
        },
        "OnlineBackup": {
            "No": 0,
            "No internet service": 1,
            "Yes": 2
        },
        "DeviceProtection": {
            "No": 0,
            "No internet service": 1,
            "Yes": 2
        },
        "TechSupport": {
            "No": 0,
            "No internet service": 1,
            "Yes": 2
        },
        "StreamingTV": {
            "No": 0,
            "No internet service": 1,
            "Yes": 2
        },
        "StreamingMovies": {
            "No": 0,
            "No internet service": 1,
            "Yes": 2
        },
        "Contract": {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        },
        "PaperlessBilling": {
            "No": 0,
            "Yes": 1
        },
        "PaymentMethod": {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3
        },
        "Churn": {
            "No": 0,
            "Yes": 1
        }
    }

    for column, mapping in mapping_rules.items():
        if column in encoded_df.columns:
            encoded_df[column] = encoded_df[column].map(mapping)

    return encoded_df


# ------------------------------------------------------------
# Prepare Uploaded Dataset
# ------------------------------------------------------------

def prepare_uploaded_data(uploaded_df, model=None):
    """
    Prepares uploaded test data for prediction.

    This function:
    1. Removes customerID.
    2. Separates Churn as target if available.
    3. Converts TotalCharges to numeric.
    4. Encodes categorical columns without creating dummy columns.
    5. Aligns feature columns with the selected model's training columns.
    """

    df = uploaded_df.copy()

    # Remove customerID because it is not useful for prediction
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Separate target column if available
    y_test = None

    if "Churn" in df.columns:
        y_test = df["Churn"].copy()
        df.drop(columns=["Churn"], inplace=True)

        if y_test.dtype == "object":
            y_test = y_test.map({
                "No": 0,
                "Yes": 1
            })

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing numeric values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    # Fill missing categorical values
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    # Encode categorical columns
    df = encode_telco_columns(df)

    # If any unmapped categorical values remain, convert safely
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = pd.factorize(df[column])[0]

    # Align columns with model training features
    if model is not None and hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)

        missing_columns = [col for col in expected_columns if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in expected_columns]

        if missing_columns:
            st.warning(
                "Some expected model features were missing from the uploaded file. "
                "They were added with value 0."
            )
            st.write("Missing columns:", missing_columns)

            for column in missing_columns:
                df[column] = 0

        if extra_columns:
            st.info(
                "Some uploaded columns were not used by the model and were removed."
            )
            st.write("Removed columns:", extra_columns)

        df = df[expected_columns]

    return df, y_test


# ------------------------------------------------------------
# Evaluation Function
# ------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a classification model and returns predictions and metrics.
    """

    y_pred = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(X_test)[:, 1]
        results["ROC AUC"] = roc_auc_score(y_test, y_probability)
    else:
        results["ROC AUC"] = np.nan

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return y_pred, results, cm, report


# ------------------------------------------------------------
# Display Confusion Matrix
# ------------------------------------------------------------

def display_confusion_matrix(cm):
    fig, ax = plt.subplots()

    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)


# ------------------------------------------------------------
# Section 1: Upload Test Data
# ------------------------------------------------------------

if section == "Upload Test Data":

    st.header("Upload Test Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = uploaded_df

            st.success("File uploaded successfully.")

            st.subheader("Dataset Preview")
            st.dataframe(uploaded_df.head())

            st.subheader("Dataset Shape")
            st.write(f"Rows: {uploaded_df.shape[0]}")
            st.write(f"Columns: {uploaded_df.shape[1]}")

            st.subheader("Column Names")
            st.write(uploaded_df.columns.tolist())

        except Exception as error:
            st.error(f"Failed to read uploaded file: {error}")

    else:
        st.info("Please upload a CSV file to continue.")


# ------------------------------------------------------------
# Section 2: Evaluate Single Model
# ------------------------------------------------------------

elif section == "Evaluate Single Model":

    st.header("Evaluate Single Model")

    if st.session_state.uploaded_data is None:
        st.warning("Please upload a test dataset first from the Upload Test Data section.")

    else:
        selected_model_name = st.selectbox(
            "Select a trained model",
            list(MODEL_FILES.keys())
        )

        if st.button("Run Evaluation"):
            try:
                model = load_model(selected_model_name)

                X_test, y_test = prepare_uploaded_data(
                    st.session_state.uploaded_data,
                    model=model
                )

                if selected_model_name in SCALED_MODELS:
                    scaler = load_scaler()
                    X_test = pd.DataFrame(
                        scaler.transform(X_test),
                        columns=X_test.columns
                    )

                st.subheader("Prepared Feature Data")
                st.dataframe(X_test.head())

                if y_test is None:
                    st.warning(
                        "The uploaded dataset does not contain a Churn column. "
                        "Only predictions can be generated."
                    )

                    predictions = model.predict(X_test)

                    prediction_df = pd.DataFrame({
                        "Predicted Churn": predictions
                    })

                    prediction_df["Predicted Churn"] = prediction_df["Predicted Churn"].map({
                        0: "No",
                        1: "Yes"
                    })

                    st.subheader("Predictions")
                    st.dataframe(prediction_df)

                else:
                    y_pred, metrics, cm, report = evaluate_model(
                        model,
                        X_test,
                        y_test
                    )

                    st.subheader("Evaluation Metrics")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                        st.metric("Precision", f"{metrics['Precision']:.4f}")

                    with col2:
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")

                    with col3:
                        st.metric("MCC", f"{metrics['MCC']:.4f}")

                        if not np.isnan(metrics["ROC AUC"]):
                            st.metric("ROC AUC", f"{metrics['ROC AUC']:.4f}")
                        else:
                            st.metric("ROC AUC", "N/A")

                    st.subheader("Confusion Matrix")
                    display_confusion_matrix(cm)

                    st.subheader("Classification Report")
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)

            except Exception as error:
                st.error(f"Evaluation failed: {error}")


# ------------------------------------------------------------
# Section 3: Compare All Models
# ------------------------------------------------------------

elif section == "Compare All Models":

    st.header("Compare All Models")

    if st.session_state.uploaded_data is None:
        st.warning("Please upload a test dataset first from the Upload Test Data section.")

    else:
        if st.button("Compare Models"):
            comparison_results = []

            for model_name in MODEL_FILES.keys():
                try:
                    model = load_model(model_name)

                    X_test, y_test = prepare_uploaded_data(
                        st.session_state.uploaded_data,
                        model=model
                    )

                    if y_test is None:
                        st.error(
                            "The uploaded dataset does not contain a Churn column. "
                            "Model comparison requires actual target values."
                        )
                        comparison_results = []
                        break

                    if model_name in SCALED_MODELS:
                        scaler = load_scaler()
                        X_test = pd.DataFrame(
                            scaler.transform(X_test),
                            columns=X_test.columns
                        )

                    _, metrics, _, _ = evaluate_model(
                        model,
                        X_test,
                        y_test
                    )

                    comparison_results.append({
                        "Model": model_name,
                        "Accuracy": metrics["Accuracy"],
                        "Precision": metrics["Precision"],
                        "Recall": metrics["Recall"],
                        "F1 Score": metrics["F1 Score"],
                        "ROC AUC": metrics["ROC AUC"],
                        "MCC": metrics["MCC"]
                    })

                except Exception as error:
                    st.error(f"{model_name} failed: {error}")

            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)

                st.subheader("Model Comparison Table")
                st.dataframe(comparison_df)

                st.subheader("Model Comparison Chart")

                chart_df = comparison_df.set_index("Model")[
                    [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score",
                        "ROC AUC",
                        "MCC"
                    ]
                ]

                st.bar_chart(chart_df)

                best_model = comparison_df.sort_values(
                    by="F1 Score",
                    ascending=False
                ).iloc[0]

                st.success(
                    f"Best model based on F1 Score: {best_model['Model']} "
                    f"with F1 Score = {best_model['F1 Score']:.4f}"
                )