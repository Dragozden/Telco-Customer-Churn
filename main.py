import pandas as pd
import numpy as np
import logging
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score

# IMPORTANT: Using imblearn Pipeline to prevent Data Leakage during Cross-Validation
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Configure logging for pipeline tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw data and performs basic cleaning operations.
    """
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Coerce TotalCharges to numeric, drop resulting NaNs to ensure data integrity
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    initial_shape = df.shape
    df.dropna(subset=['TotalCharges'], inplace=True)
    logging.info(f"Dropped {initial_shape[0] - df.shape[0]} rows due to NaN values in TotalCharges.")
    
    # customerID carries no predictive signal
    df.drop(columns=["customerID"], inplace=True)
    
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives new predictive features from existing columns.
    """
    logging.info("Engineering new features...")
    df_eng = df.copy()
    
    # Vectorized count of additional subscribed services
    service_columns = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df_eng['AdditionalServices'] = (df_eng[service_columns] == 'Yes').sum(axis=1)
    
    # Financial ratio to capture spending intensity over time (+1 prevents ZeroDivisionError)
    df_eng['ChargePer_TenureLength'] = df_eng['MonthlyCharges'] / (df_eng['tenure'] + 1)
    
    # Binarize target variable
    df_eng['Churn'] = df_eng['Churn'].map({'Yes': 1, 'No': 0})
    
    return df_eng

def main():
    # 1. Data Preparation
    df = load_and_clean_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = engineer_features(df)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Check class distribution
    class_balance = y.value_counts(normalize=True) * 100
    logging.info(f"Class distribution - Retained (0): {class_balance[0]:.1f}%, Churn (1): {class_balance[1]:.1f}%")
    logging.info("Target variable is imbalanced. SMOTE will be applied inside the CV pipeline.")
    
    # Stratified split to maintain class ratios in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 2. Preprocessor Setup
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AdditionalServices', 'ChargePer_TenureLength']
    categorical_features = [col for col in X_train.columns if col not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    # 3. Model Dictionary & Hyperparameter Grids Definition
    # Defined explicitly to allow systematic GridSearchCV iterations
    models_config = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__class_weight': [None, 'balanced']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [5, 10, None],
                'classifier__min_samples_split': [2, 5]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__max_depth': [3, 5]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'classifier__n_neighbors': [5, 9, 15],
                'classifier__weights': ['uniform', 'distance']
            }
        }
    }
    
    # 4. Model Training & Tuning Loop
    best_overall_model = None
    best_overall_score = -1
    best_model_name = ""
    
    # Stratified K-Fold to ensure robust validation on imbalanced target
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, config in models_config.items():
        logging.info(f"--- Training & Tuning: {name} ---")
        
        # Build Imblearn Pipeline: Preprocessing -> SMOTE -> Classifier
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', config['model'])
        ])
        
        # GridSearch maximizing ROC AUC (standard metric for imbalanced classification)
        grid_search = GridSearchCV(
            pipeline, 
            param_grid=config['params'], 
            cv=cv_strategy, 
            scoring='roc_auc', 
            n_jobs=-1, # Utilize all available CPU cores
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_cv_score = grid_search.best_score_
        
        logging.info(f"Best CV ROC AUC for {name}: {best_cv_score:.4f}")
        logging.info(f"Best params: {grid_search.best_params_}")
        
        # Track the absolute best model
        if best_cv_score > best_overall_score:
            best_overall_score = best_cv_score
            best_overall_model = best_model
            best_model_name = name

    # 5. Final Evaluation on Hold-out Test Set
    logging.info(f"\n==============================================")
    logging.info(f"BEST MODEL FOUND: {best_model_name}")
    logging.info(f"Evaluating best model on unseen test data...")
    logging.info(f"==============================================")
    
    y_pred = best_overall_model.predict(X_test)
    y_pred_proba = best_overall_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Final Test Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # 6. Model Serialization
    model_filename = f"churn_{best_model_name.lower()}_pipeline.pkl"
    joblib.dump(best_overall_model, model_filename)
    logging.info(f"Production-ready pipeline saved to: {model_filename}")

if __name__ == "__main__":
    main()