# Online Shoppers Purchase Prediction

## Problem Statement

This project compares five machine-learning classification algorithms to predict whether an online shopping session will result in a purchase.

The models are trained and evaluated using the **Online Shoppers Purchasing Intention** dataset from the UCI Machine Learning Repository. The objective is to compare the performance of different classification algorithms using standard evaluation metrics and demonstrate the trained models through an interactive Streamlit web application.

## Dataset Information

- **Dataset:** Online Shoppers Purchasing Intention
- **Source:** UCI Machine Learning Repository
- **Dataset ID:** 468
- **Original instances:** 12,330
- **Total instances after cleaning:** 12,205
- **Input features:** 17
- **Numerical features:** 10
- **Categorical features:** 7
- **Target variable:** Revenue
- **Classification type:** Binary Classification

The original dataset contained 12,330 rows and 18 columns. After removing 125 duplicate rows, 12,205 instances remained. No missing values were found.

A stratified 80:20 train-test split was used, resulting in 9,764 training instances and 2,441 testing instances.

Numerical features were standardized using `StandardScaler`, while categorical features were transformed using `OneHotEncoder`. The preprocessing pipeline was fitted only on the training data to avoid data leakage.

## GitHub Repository Link

**GitHub Repository:**

https://github.com/EvaniAnusha/online-shoppers-purchase-prediction-2026AJ05423

The repository contains the application source code, trained classification models, preprocessing pipeline, test dataset, dependency information, and project documentation.

## Machine Learning Models

The following five classification algorithms were implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Evaluation Metrics

Each model was evaluated using the following six metrics:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures how many predicted positive instances were actually positive.
- **Recall:** Measures how many actual positive instances were correctly identified.
- **F1 Score:** Harmonic mean of Precision and Recall.
- **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classifications, considering all four confusion-matrix categories.

## Model Performance Comparison

All five models were evaluated on the same held-out test dataset containing 2,441 instances. The results are rounded to four decimal places.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8869 | 0.8996 | 0.7524 | 0.4136 | 0.5338 | 0.5032 |
| Decision Tree | 0.8558 | 0.7376 | 0.5373 | 0.5654 | 0.5510 | 0.4654 |
| K-Nearest Neighbors | 0.8734 | 0.7958 | 0.6622 | 0.3901 | 0.4909 | 0.4435 |
| Gaussian Naive Bayes | 0.2773 | 0.7502 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| Support Vector Machine | 0.8955 | 0.8733 | 0.7361 | 0.5183 | 0.6083 | 0.5614 |

## Observations on Model Performance

### Logistic Regression

Logistic Regression achieved an accuracy of 0.8869 and the highest AUC score of 0.8996. It also achieved good precision of 0.7524. However, its recall of 0.4136 indicates that some actual purchasing sessions were not identified correctly.

### Decision Tree

The Decision Tree classifier achieved an accuracy of 0.8558. It demonstrated a moderate balance between precision (0.5373) and recall (0.5654). Its F1 score was 0.5510, although its AUC score was comparatively lower at 0.7376.

### K-Nearest Neighbors (KNN)

KNN achieved an accuracy of 0.8734 and precision of 0.6622. However, its recall of 0.3901 was relatively low, indicating that it missed a considerable number of actual purchasing sessions.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved the highest recall of 0.9791 among all five models. However, its precision (0.1756) and accuracy (0.2773) were considerably lower due to a large number of false-positive predictions.

### Support Vector Machine (SVM)

Support Vector Machine achieved the highest accuracy (0.8955), F1 score (0.6083), and MCC (0.5614). It demonstrated a stronger overall balance between precision and recall compared with the other classifiers.

### Overall Winner

**Support Vector Machine (SVM)** is the overall winner for this dataset based on its highest accuracy, F1 score, and MCC.

However, Logistic Regression achieved the highest AUC, while Gaussian Naive Bayes achieved the highest recall. The results demonstrate that the best-performing classifier can vary depending on the evaluation metric considered.

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
- Visual Studio Code or another Python IDE

### 1. Clone the Repository

```bash
git clone https://github.com/EvaniAnusha/online-shoppers-purchase-prediction-2026AJ05423.git
```

### 2. Navigate to the Project Directory

```bash
cd online-shoppers-purchase-prediction-2026AJ05423
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

```text
http://localhost:8501
```

## Application Features

The Streamlit application provides the following features:

- Upload a CSV test dataset.
- Validate all 17 required input features and the target column (`Revenue`).
- Preview the uploaded dataset.
- Select one of the five trained classification models through a dropdown.
- Evaluate the uploaded dataset using the selected saved model.
- Display all six evaluation metrics: Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Visualize the confusion matrix.
- Display the classification report.

The application loads previously trained models and the saved preprocessing pipeline. It evaluates uploaded test data without retraining the models.

## Deployment

The application is deployed using **Streamlit Community Cloud**.

The GitHub repository is connected to Streamlit Community Cloud, with `app.py` selected as the main application file.

**Live Streamlit Application:**

https://online-shoppers-purchase-prediction-2026aj05423-oevtditwfy8tjg.streamlit.app/

The deployed application was verified in both a normal browser session and an Incognito browser session. It successfully displayed the CSV upload interface and confirmed that all five trained models and the preprocessing pipeline loaded successfully.

## Conclusion

This project demonstrates the implementation and comparative evaluation of five machine-learning classification algorithms for predicting online shopping purchase intention.

Among the evaluated models, Support Vector Machine achieved the best overall performance based on Accuracy (0.8955), F1 Score (0.6083), and MCC (0.5614). Logistic Regression achieved the highest AUC (0.8996), while Gaussian Naive Bayes achieved the highest Recall (0.9791), but with considerably lower Precision and Accuracy.

The project also demonstrates an end-to-end machine-learning workflow, including data preprocessing, model training, evaluation, artifact storage, and deployment of an interactive Streamlit application.