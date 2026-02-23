# Telco Customer Churn Prediction

## Project Overview
Customer churn (the loss of clients or customers) is one of the most significant metrics for businesses, particularly in the telecommunications sector. This project focuses on building a machine learning pipeline to predict whether a customer will cancel their service. 

The goal is to identify at-risk customers early, allowing the business to take proactive retention actions. The solution covers the full data science lifecycle: from data cleaning and preprocessing to model training and evaluation using a baseline classification algorithm.

## Dataset
The project utilizes the **Telco Customer Churn** dataset (IBM).
* **Source:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
* **Target Variable:** `Churn` (Binary: Yes/No)
* **Features:** The dataset includes customer demographics (gender, senior citizen status, partner, dependents), account information (tenure, contract type, payment method), and subscribed services (phone, internet, streaming, etc.).

## Technologies Used
* **Python 3.x**
* **Pandas:** Data manipulation and cleaning.
* **Scikit-Learn:** Data preprocessing (scaling, encoding) and model building.
* **NumPy:** Numerical operations.

## Methodology

### 1. Data Preprocessing
The raw data required cleaning to ensure model stability:
* **Data Type Conversion:** The `TotalCharges` column was converted from object type to numeric, with empty strings coerced to NaN.
* **Handling Missing Values:** Rows with missing `TotalCharges` values were removed to maintain data integrity.
* **Feature Selection:** The `customerID` column was dropped as it holds no predictive value.

### 2. Feature Engineering
To prepare the data for the Logistic Regression model:
* **Categorical Encoding:** `LabelEncoder` was applied to transform categorical text variables into a machine-readable numeric format.
* **Feature Scaling:** `StandardScaler` was used to normalize numerical features. This is a critical step for distance-based and linear models to ensure convergence and prevent features with larger magnitudes from dominating the objective function.

### 3. Model Training
* **Algorithm:** Logistic Regression was selected as the baseline model due to its interpretability and efficiency in binary classification tasks.
* **Data Split:** The dataset was split into training (80%) and testing (20%) sets to validate performance on unseen data.

## Model Performance
The model is evaluated using standard classification metrics:
* **Accuracy:** Overall correctness of the model.
* **Confusion Matrix:** Breakdown of True Positives, True Negatives, False Positives, and False Negatives.
* **Classification Report:** Precision, Recall, and F1-Score for both churn and non-churn classes.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [Your-Repository-Link-Here]
    ```

2.  **Install dependencies:**
    Ensure you have the required libraries installed. You can install them via pip:
    ```bash
    pip install pandas scikit-learn numpy
    ```

3.  **Execute the script:**
    Run the main pipeline script:
    ```bash
    python main.py
    ```

## Future Improvements
To further improve predictive performance, the following steps are planned:
* Testing tree-based models (Random Forest, XGBoost) to capture non-linear relationships.
* Implementing SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance, as churn is typically a minority class.
* Using One-Hot Encoding for nominal categorical variables to improve linear model performance.
* Hyperparameter tuning using GridSearchCV.
