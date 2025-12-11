# 🏦 End-to-End Credit Scoring System with MLOps & XAI

## 📋 Project Overview
The goal of this project is to build an automated **Credit Scoring System** based on **Home Credit** data. The system predicts the probability of a client's default, supporting the decision-making process in financial institutions.

The main business challenge was to **maximize the detection of risky clients** (high ROC AUC) while maintaining model interpretability (Explainable AI), which is a key regulatory requirement in the banking sector.

## 🛠 Tech Stack
The project utilizes a modern Data Science tech stack:

* **Modeling:** `LightGBM`, `XGBoost` (Gradient Boosting Machines).
* **Orchestration & MLOps:** `MLflow` (Experiment tracking, model versioning, artifact registry).
* **Optimization:** `Optuna` (Bayesian hyperparameter optimization - TPE).
* **Pipeline:** `Scikit-Learn Pipeline` & `ColumnTransformer` (Data leakage prevention).
* **XAI (Interpretability):** `SHAP` (SHapley Additive exPlanations).

## ⚙️ Methodology

### 1. Data Engineering & Preprocessing
The entire data processing workflow is encapsulated within a `Pipeline` object to ensure full reproducibility in production (preventing Training-Serving Skew).
* **Missing Values:** Median imputation strategy (robust to outliers in financial data).
* **Encoding:** `OneHotEncoder` handling unknown categories (`handle_unknown='ignore'`).
* **Scaling:** Standardization of numerical variables.

### 2. Training Strategy & Class Imbalance
The dataset is characterized by a strong Class Imbalance.
* **Stratification:** Utilized stratified splits (`stratify`) for validation sets.
* **Metric:** Optimized for **ROC AUC**, which better reflects the model's ability to rank risk compared to raw Accuracy.
* **Regularization:** Tuned regularization parameters in tree-based models to prevent overfitting on the majority class.

### 3. Hyperparameter Tuning
Instead of classic GridSearch, **Optuna** was used. This allowed for searching a broader parameter space in less time by leveraging Bayesian estimators that "learn" from previous trials.

## 📊 Results & Interpretation (XAI)

### Model Performance
Comparative experiments were conducted between XGBoost and LightGBM.
* **Selected Model:** [ENTER MODEL HERE, E.G., XGBoost]
* **ROC AUC Score:** [ENTER SCORE HERE, E.G., 0.7XX]

### Key Risk Factors (Feature Importance)
SHAP analysis revealed that the most significant factors influencing the credit decision are:
1.  **EXT_SOURCE (1, 2, 3):** External credit scores – strong negative correlation (higher score = lower default risk).
2.  **[ENTER FEATURE HERE]:** [Short description].

By using **SHAP Summary Plots**, the model is not a "black box" – every decision can be justified to stakeholders or regulators.

## 🚀 How to Run (Reproducibility)

Required libraries are listed in `requirements.txt`.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook home_credit_default_risk.ipynb
