# 💳 Credit Card Fraud Detection Using Machine Learning

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a very small percentage of all transactions.

The objective of this project is to build and compare multiple machine learning models for detecting fraudulent credit card transactions while focusing on metrics such as **Precision, Recall, F1 Score, ROC-AUC, and PR-AUC** rather than relying only on accuracy.

The models evaluated in this project are:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

After comparing the models, XGBoost was further optimized using hyperparameter tuning and validation-based threshold selection.

---

## 🎯 Problem Statement

Credit card fraud datasets are highly imbalanced, meaning legitimate transactions greatly outnumber fraudulent transactions.

Because of this imbalance, accuracy alone can be misleading.

For example, a model that predicts every transaction as legitimate could achieve very high accuracy while failing to detect fraudulent transactions.

Therefore, this project focuses primarily on:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

---

## 📊 Dataset

The project uses the Credit Card Fraud Detection dataset containing European cardholder transactions.

The dataset contains:

- `Time` – Seconds elapsed since the first transaction
- `V1` to `V28` – Anonymized PCA-transformed numerical features
- `Amount` – Transaction amount
- `Class` – Target variable

Target classes:

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

The dataset is extremely imbalanced, with fraudulent transactions representing only a very small percentage of the total transactions.

The dataset itself is not committed to this repository and should be placed inside:

```text
data/creditcard.csv
```

---

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure and characteristics of the dataset.

The analysis included:

- Dataset shape and column inspection
- Missing-value analysis
- Duplicate analysis
- Class distribution
- Transaction amount distribution
- Amount skewness
- Transaction-time analysis
- Fraud rate by derived hour
- Feature correlation analysis
- Correlation heatmap

The `Amount` feature showed strong positive skewness.

The target variable also showed severe class imbalance, making metrics such as Recall, F1 Score, and PR-AUC particularly important.

---

## 🧹 Data Cleaning

The following data-quality checks were performed:

- Checked for missing values
- Checked for duplicate transactions
- Verified feature data types
- Verified target-class values

The dataset contains numerical features, so categorical encoding was not required.

---

## 🛠️ Feature Engineering

An additional `Hour` feature was derived from the `Time` column:

```python
df["Hour"] = ((df["Time"] / 3600) % 24).astype(int)
```

This converts elapsed transaction time into a repeating 24-hour representation that can be used to investigate temporal transaction patterns.

---

## ⚙️ Data Preprocessing

The preprocessing workflow included:

1. Separating independent features and the target variable.
2. Performing stratified data splitting to preserve the fraud/legitimate class distribution.
3. Using algorithm-appropriate preprocessing.
4. Accounting for severe class imbalance during model development.

The data was separated into training, validation, and testing components.

The validation data was used for model and threshold decisions, while the test data was used for final evaluation.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

Logistic Regression was used as the baseline classification model.

It achieved high recall but extremely low precision, resulting in a large number of legitimate transactions being incorrectly classified as fraud.

### 2. Decision Tree

A Decision Tree classifier was trained to capture nonlinear relationships.

It improved precision compared with Logistic Regression while maintaining relatively high recall.

### 3. Random Forest

Random Forest significantly improved precision and F1 score.

It produced very few false-positive fraud alerts but had lower recall than Logistic Regression and Decision Tree.

### 4. XGBoost

XGBoost provided the strongest overall balance among the evaluated models.

It was further optimized using hyperparameter tuning and threshold analysis.

---

## 🔧 Hyperparameter Tuning

XGBoost hyperparameters were optimized using `RandomizedSearchCV`.

The search included parameters such as:

- Number of estimators
- Maximum tree depth
- Learning rate
- Subsample ratio
- Column sampling ratio
- Minimum child weight
- Gamma

Average Precision was used as the scoring metric because the dataset is highly imbalanced.

Cross-validation was used during hyperparameter search to obtain a more reliable estimate of model performance.

---

## 🎚️ Classification Threshold Selection

Instead of automatically relying on the default classification threshold, different probability thresholds were evaluated using validation data.

Threshold selection is important in fraud detection because changing the threshold affects the trade-off between:

- False positives
- False negatives
- Precision
- Recall

The threshold was selected using validation performance rather than directly optimizing it on the final evaluation data.

---

## 📈 Model Performance

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0555 | 0.8737 | 0.1043 | 0.9685 | 0.6683 |
| Decision Tree | 0.1382 | 0.8000 | 0.2357 | 0.9074 | 0.4963 |
| Random Forest | 0.9583 | 0.7263 | 0.8263 | 0.9340 | 0.8089 |
| **XGBoost** | **0.9481** | **0.7684** | **0.8488** | **0.9659** | **0.8340** |

---

## 🏆 Final Model

XGBoost was selected as the final model because it provided a strong overall balance between fraud-detection capability and false-positive control.

Final XGBoost performance:

```text
Precision : 94.81%
Recall    : 76.84%
F1 Score  : 84.88%
ROC-AUC   : 96.59%
PR-AUC    : 83.40%
```

This means the model achieves very high precision while detecting a substantial proportion of fraudulent transactions.

The results also demonstrate an important fraud-detection trade-off:

- Logistic Regression produced higher recall but extremely poor precision.
- Random Forest produced excellent precision but lower recall.
- XGBoost provided the strongest overall F1 score and PR-AUC among the evaluated models.

---

## 🔎 Feature Importance

XGBoost feature importance was analyzed to determine which features contributed most strongly to model predictions.

Because `V1` through `V28` are anonymized PCA-transformed variables, their original business meanings are unavailable.

Therefore, feature importance indicates which transformed variables were most useful to the model but should not be interpreted as evidence that those features cause fraud.

Feature-importance results are stored in:

```text
results/feature_importance.csv
```

---

## 💾 Model Saving

The trained model and selected classification threshold were saved using `joblib`.

```python
model_artifact = {
    "model": best_xgb_model,
    "threshold": best_threshold
}

joblib.dump(
    model_artifact,
    "../models/fraud_detection_model.pkl"
)
```

Saving both the model and threshold ensures that the same classification decision rule can be reused for future predictions.

---

## 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_detection_model.pkl
│   └── xgboost_fraud_model.pkl
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── results/
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   └── threshold_analysis.csv
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook
- Git
- GitHub Codespaces

---

## 📊 Evaluation Metrics

### Precision

Precision measures how many transactions predicted as fraudulent were actually fraudulent.

High precision is important because excessive false-positive fraud alerts can inconvenience legitimate customers.

### Recall

Recall measures how many actual fraudulent transactions were successfully detected.

High recall is important because false negatives represent fraudulent transactions that the system failed to detect.

### F1 Score

F1 Score provides a balance between Precision and Recall.

It is particularly useful when working with imbalanced classification problems.

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between the two classes across classification thresholds.

### PR-AUC

PR-AUC summarizes the Precision-Recall relationship across thresholds.

Because fraudulent transactions are extremely rare in this dataset, PR-AUC is especially useful when comparing the fraud-detection models.

---

## 💡 Key Findings

The experiments demonstrated that model selection for fraud detection cannot be based on accuracy alone.

Logistic Regression achieved high fraud recall but generated many false-positive alerts.

Random Forest achieved extremely high precision but missed more fraudulent transactions.

XGBoost provided the strongest overall balance, achieving approximately **94.8% precision, 76.8% recall, 84.9% F1 score, and 83.4% PR-AUC**.

The project also demonstrates that classification-threshold selection should depend on the business cost associated with false positives and false negatives.

---

## ⚠️ Limitations

- The dataset contains anonymized PCA-transformed features, limiting business-level interpretation.
- Fraud patterns can change over time due to concept drift.
- The dataset represents historical transactions and may not represent current fraud behavior.
- Different businesses may require different precision-recall trade-offs.
- Earlier exploratory experiments inspected test-set performance, so a production-grade study should reserve a completely untouched holdout dataset or use a stricter nested evaluation strategy.

---

## 🚀 Future Improvements

Possible future improvements include:

- Cost-sensitive threshold optimization
- SHAP-based model explainability
- Advanced imbalance-handling techniques
- Real-time fraud prediction API
- Model monitoring and drift detection
- Automated retraining pipeline
- Deployment using FastAPI or a cloud platform

---

## ▶️ How to Run the Project

Clone the repository:

```bash
git clone <repository-url>
cd credit-card-fraud-detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
data/creditcard.csv
```

Then open:

```text
notebooks/fraud_detection.ipynb
```

and run the notebook cells in order.

---

## 📌 Conclusion

This project demonstrates an end-to-end machine learning workflow for highly imbalanced credit card fraud detection.

Multiple classification algorithms were evaluated, including Logistic Regression, Decision Tree, Random Forest, and XGBoost.

The project demonstrates the importance of selecting appropriate evaluation metrics for imbalanced classification and understanding the trade-off between precision and recall.

Among the evaluated models, XGBoost achieved the strongest overall balance and was selected as the final fraud-detection model.