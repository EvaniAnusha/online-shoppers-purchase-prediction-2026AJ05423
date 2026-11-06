
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Online Shoppers Purchase Prediction",
    page_icon="🛍️",
    layout="wide"
)

# --------------------------------------------------
# APPLICATION TITLE
# --------------------------------------------------

st.title("Online Shoppers Purchase Prediction")
st.write(
    "Compare five machine learning classifiers "
    "to predict whether an online shopping session "
    "will result in a purchase."
)

# --------------------------------------------------
# LOAD TRAINED MODELS AND PREPROCESSOR
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    models = {
        "Logistic Regression": joblib.load(
            "saved_models/logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            "saved_models/decision_tree.pkl"
        ),
        "K-Nearest Neighbors": joblib.load(
            "saved_models/k-nearest_neighbors.pkl"
        ),
        "Gaussian Naive Bayes": joblib.load(
            "saved_models/gaussian_naive_bayes.pkl"
        ),
        "Support Vector Machine": joblib.load(
            "saved_models/support_vector_machine.pkl"
        )
    }

    preprocessor = joblib.load(
        "saved_models/preprocessor.pkl"
    )

    return models, preprocessor


models, preprocessor = load_artifacts()

st.success("All five trained models and the preprocessing pipeline loaded successfully!")

# --------------------------------------------------
# DATASET FEATURE DEFINITIONS
# --------------------------------------------------

TARGET_COLUMN = "Revenue"

REQUIRED_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

REQUIRED_COLUMNS = REQUIRED_FEATURES + [TARGET_COLUMN]

# --------------------------------------------------
# TEST DATA UPLOAD
# --------------------------------------------------

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV containing the 17 input features and Revenue target."
)

if uploaded_file is not None:

    try:
        uploaded_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset Preview")
        st.dataframe(uploaded_data.head())

        missing_columns = [
            column for column in REQUIRED_COLUMNS
            if column not in uploaded_data.columns
        ]

        if missing_columns:
            st.error(
                "The uploaded CSV is missing these required columns: "
                + ", ".join(missing_columns)
            )

        else:
            uploaded_data = uploaded_data[REQUIRED_COLUMNS]

            st.success("All required columns are present!")

            st.write("Dataset shape:", uploaded_data.shape)


            # Validate missing values
            if uploaded_data[REQUIRED_COLUMNS].isnull().any().any():

                st.error(
                    "The uploaded dataset contains missing values. "
                    "Please upload a dataset without missing values."
                )

            else:

                # Separate input features and actual target values
                X_uploaded = uploaded_data[REQUIRED_FEATURES].copy()

                # Convert target values into boolean labels
                y_values = (
                    uploaded_data[TARGET_COLUMN]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False
                    })
                )

                if y_values.isnull().any():

                    st.error(
                        "Revenue must contain only True/False "
                        "or 1/0 values."
                    )

                else:

                    y_actual = y_values.astype(bool)

                    # Apply the previously fitted preprocessing pipeline
                    X_processed = preprocessor.transform(X_uploaded)

                    st.success(
                        "Uploaded data preprocessed successfully!"
                    )

                    # Select a trained classification model
                    st.subheader("Select Classification Model")

                    selected_model_name = st.selectbox(
                        "Choose a model for evaluation:",
                        list(models.keys())
                    )

                    selected_model = models[selected_model_name]

                    st.info(
                        f"Selected Model: {selected_model_name}"
                    )
                    # Handle Gaussian Naive Bayes input format
                    if selected_model_name == "Gaussian Naive Bayes":
                        X_for_prediction = (
                            X_processed.toarray()
                            if hasattr(X_processed, "toarray")
                            else X_processed
                        )
                    else:
                        X_for_prediction = X_processed

                    # Generate predictions
                    y_pred = selected_model.predict(X_for_prediction)

                    # Generate prediction probabilities for AUC
                    y_probability = selected_model.predict_proba(
                        X_for_prediction
                    )

                    positive_class_index = list(
                        selected_model.classes_
                    ).index(True)

                    y_scores = y_probability[:, positive_class_index]

                    # Calculate evaluation metrics
                    accuracy = accuracy_score(y_actual, y_pred)

                    auc = (
                        roc_auc_score(y_actual, y_scores)
                        if y_actual.nunique() == 2
                        else None
                    )

                    precision = precision_score(
                        y_actual, y_pred, zero_division=0
                    )

                    recall = recall_score(
                        y_actual, y_pred, zero_division=0
                    )

                    f1 = f1_score(
                        y_actual, y_pred, zero_division=0
                    )

                    mcc = matthews_corrcoef(y_actual, y_pred)
                    # Display evaluation metrics
                    st.header("Model Evaluation Results")

                    st.subheader(
                        f"Performance of {selected_model_name}"
                    )

                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    metric_col1.metric(
                        "Accuracy", f"{accuracy:.4f}"
                    )

                    metric_col2.metric(
                        "AUC",
                        f"{auc:.4f}" if auc is not None else "N/A"
                    )

                    metric_col3.metric(
                        "Precision", f"{precision:.4f}"
                    )

                    metric_col4, metric_col5, metric_col6 = st.columns(3)

                    metric_col4.metric(
                        "Recall", f"{recall:.4f}"
                    )

                    metric_col5.metric(
                        "F1 Score", f"{f1:.4f}"
                    )

                    metric_col6.metric(
                        "MCC", f"{mcc:.4f}"
                    )

                    # Display confusion matrix
                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(
                        y_actual,
                        y_pred,
                        labels=[False, True]
                    )

                    fig, ax = plt.subplots(figsize=(6, 4))

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["No Purchase", "Purchase"],
                        yticklabels=["No Purchase", "Purchase"],
                        ax=ax
                    )

                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("Actual Label")
                    ax.set_title(
                        f"Confusion Matrix - {selected_model_name}"
                    )

                    st.pyplot(fig)
                    plt.close(fig)

                    # Display classification report
                    st.subheader("Classification Report")

                    report = classification_report(
                        y_actual,
                        y_pred,
                        labels=[False, True],
                        target_names=["No Purchase", "Purchase"],
                        zero_division=0,
                        output_dict=True
                    )

                    report_df = pd.DataFrame(report).transpose()

                    st.dataframe(
                        report_df.round(4),
                        use_container_width=True
                    )



    except Exception as error:
        st.error(f"Unable to read the uploaded CSV: {error}")
