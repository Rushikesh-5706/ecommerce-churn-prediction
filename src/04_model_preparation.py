"""
Model Preparation Script (Phase 6.1)

This script performs the following steps:
1. Loads engineered features (customer_features.csv)
2. Separates features (X) and target (y)
3. Performs stratified train/validation/test split
4. Scales numerical features
5. Saves processed datasets and scaler

Rubric Requirements:
- Remove CustomerID from features
- Encode categorical variables
- Stratified split maintaining churn ratio
- Scale only numerical features
- Save all artifacts
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

def prepare_model_data():
    print("Loading customer features...")
    df = pd.read_csv('data/processed/customer_features.csv')
    
    # 1. Separate ID and Churn
    X = df.drop(['CustomerID', 'Churn'], axis=1)
    y = df['Churn']
    
    # 2. Categorical Encoding & 4. Scaling Combined
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    
    # Identify numerical vs categorical features
    numerical_features = [col for col in X.columns if col not in ['CustomerSegment', 'PreferredDay', 'PreferredHour']]
    categorical_features = ['CustomerSegment'] if 'CustomerSegment' in X.columns else []
    
    # One-hot encode if needed
    if categorical_features:
        X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    else:
        X_encoded = X
        
    feature_names = list(X_encoded.columns)
    
    # 3. Stratified Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_encoded, y, test_size=0.30, stratify=y, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    # Scale ONLY numerical features
    numerical_cols_encoded = [col for col in feature_names 
                              if not any(cat in col for cat in (categorical_features if categorical_features else []))]
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols_encoded] = scaler.fit_transform(X_train[numerical_cols_encoded])
    X_val_scaled[numerical_cols_encoded] = scaler.transform(X_val[numerical_cols_encoded])
    X_test_scaled[numerical_cols_encoded] = scaler.transform(X_test[numerical_cols_encoded])
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # 5. Save Artifacts
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("Saving processed datasets...")
    # Features
    X_train_df.to_csv('data/processed/X_train.csv', index=False)
    X_val_df.to_csv('data/processed/X_val.csv', index=False)
    X_test_df.to_csv('data/processed/X_test.csv', index=False)
    
    # Targets
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # Scaler and Feature Names
    joblib.dump(scaler, 'models/scaler.pkl')
    
    with open('data/processed/feature_names.json', 'w') as f:
        json.dump({'feature_names': feature_names}, f, indent=4)
        
    # Summary
    print("\nData Preparation Summary:")
    print(f"- Original features: {X.shape[1]}")
    print(f"- Features after encoding: {len(feature_names)}")
    print(f"- Training samples: {len(X_train)} ({y_train.mean()*100:.1f}% churn)")
    print(f"- Validation samples: {len(X_val)} ({y_val.mean()*100:.1f}% churn)")
    print(f"- Test samples: {len(X_test)} ({y_test.mean()*100:.1f}% churn)")
    print("\nAll files saved successfully.")

if __name__ == "__main__":
    prepare_model_data()
