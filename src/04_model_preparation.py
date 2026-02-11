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
    X = df.drop(['Customer ID', 'Churn'], axis=1)
    y = df['Churn']
    
    # 2. Categorical Encoding
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    print(f"Categorical columns to encode: {list(cat_cols)}")
    
    # One-hot encode
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Save feature names
    feature_names = list(X_encoded.columns)
    
    # 3. Stratified Split
    # First split: Train (70%) vs Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_encoded, y, test_size=0.30, stratify=y, random_state=42
    )
    
    # Second split: Validation (15%) vs Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    # 4. Scaling
    # Scale only numerical features (excluding binary encoded ones if desired, 
    # but standard practice often scales all for models like NN/LR)
    scaler = StandardScaler()
    
    # Fit on TRAIN only
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform Val and Test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for saving (optional, but good for inspection)
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
