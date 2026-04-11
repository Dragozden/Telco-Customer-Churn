# Telco Customer Churn Prediction 📡

This project implements a complete Machine Learning pipeline to predict customer churn for a telecommunications company. It addresses real-world challenges such as **class imbalance**, **feature engineering**, and **model interpretability**.

## Key Features
- **End-to-End Pipeline**: Automated data preprocessing, scaling, and classification using `scikit-learn` and `imblearn`.
- **Handling Imbalance**: Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) within cross-validation to prevent data leakage and improve recall for churning customers.
- **Model Tuning**: Systematic hyperparameter optimization using **GridSearchCV** across multiple algorithms:
  - Logistic Regression (Winner)
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors
- **Business Insights**: Detailed EDA and model coefficient analysis to identify top churn drivers.

## Results
The final **Logistic Regression** model achieved:
- **ROC AUC**: 0.8424
- **Recall (Churn Class)**: 0.80 (Successfully identifying 80% of at-risk customers)
- **Stability**: Consistent performance between Cross-Validation and unseen test data.

## Project Structure
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Raw dataset.
- `01_Business_Insights_and_Model_Evaluation.ipynb`: Interactive EDA and visual model evaluation.
- `train.py`: Production-grade script for model training and serialization.
- `requirements.txt`: Project dependencies.
- `churn_logisticregression_pipeline.pkl`: Serialized production-ready pipeline.

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourNickname/telco-customer-churn-prediction.git](https://github.com/YourNickname/telco-customer-churn-prediction.git)
2.Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3.Run the training pipeline:
    ```
    python train.py
    ```
