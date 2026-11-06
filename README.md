# Online Shoppers Purchase Prediction

## Project Overview

This project compares five machine learning classification algorithms to predict whether an online shopping session will result in a purchase.

The models are trained and evaluated using the **Online Shoppers Purchasing Intention** dataset from the UCI Machine Learning Repository.

## Dataset Information

- **Dataset:** Online Shoppers Purchasing Intention
- **Source:** UCI Machine Learning Repository
- **Total instances after cleaning:** 12,205
- **Input features:** 17
- **Target variable:** Revenue
- **Classification type:** Binary Classification

## Machine Learning Models

The following five classifiers are implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Evaluation Metrics

Each model is evaluated using the following metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Confusion matrices and classification reports are also generated for model evaluation.

## Project Structure

```text
2026AJ05423_Project/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── saved_models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── k-nearest_neighbors.pkl
    ├── gaussian_naive_bayes.pkl
    ├── support_vector_machine.pkl
    └── preprocessor.pkl
```

## Installation and Setup

### Prerequisites

- Python 3.12
- Git
- Visual Studio Code (or any Python IDE)

### 1. Clone the Repository

Once the GitHub repository is created, clone it using:

```bash
git clone https://github.com/EvaniAnusha/online-shoppers-purchase-prediction-2026AJ05423.git
```

### 2. Navigate to the Project Directory

```bash
cd 2026AJ05423_Project
```

### 3. Create a Virtual Environment

```bash
python3.12 -m venv .venv
```

### 4. Activate the Virtual Environment

For macOS/Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Streamlit Application

Start the application using:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

## Application Features

- Upload a CSV test dataset.
- Validate the presence of all 17 required input features and the target column (`Revenue`).
- Preview the uploaded dataset.
- Select one of the five trained classification models.
- Evaluate the uploaded dataset using the selected saved model.
- Display all six evaluation metrics.
- Visualize the confusion matrix.
- Display the classification report.

The application uses previously trained and saved machine learning models. It does not retrain models when a CSV file is uploaded.

## Model Performance

The five classifiers were evaluated using the same test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8869 | 0.8996 | 0.7524 | 0.4136 | 0.5338 | 0.5032 |
| Decision Tree | 0.8558 | 0.7376 | 0.5373 | 0.5654 | 0.5510 | 0.4654 |
| K-Nearest Neighbors | 0.8734 | 0.7958 | 0.6622 | 0.3901 | 0.4909 | 0.4435 |
| Gaussian Naive Bayes | 0.2773 | 0.7502 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| Support Vector Machine | 0.8955 | 0.8733 | 0.7361 | 0.5183 | 0.6083 | 0.5614 |

## Deployment

The application is intended to be deployed using Streamlit Community Cloud.

Deployment steps:

1. Upload the project repository to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the project repository.
5. Choose `app.py` as the main application file.
6. Deploy the application.
7. Verify that the deployed application loads and evaluates the test dataset successfully.

## Conclusion

Among the five evaluated classifiers, Support Vector Machine achieved the highest Accuracy (0.8955), F1 Score (0.6083), and MCC (0.5614). Logistic Regression achieved the highest AUC (0.8996).

Gaussian Naive Bayes achieved the highest Recall (0.9791), but its Precision and Accuracy were considerably lower.

The comparison demonstrates the performance differences between multiple machine learning classification algorithms on the Online Shoppers Purchasing Intention dataset.