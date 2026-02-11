"""
Model Training Script with SMOTE for Class Imbalance

This script trains all 5 models with SMOTE applied to training data
to improve performance on imbalanced churn dataset (41.92% churn rate).

Target: ROC-AUC ≥ 0.75
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE

print("="*60)
print("MODEL TRAINING WITH SMOTE (Class Imbalance Handling)")
print("="*60)

# Load data
print("\nLoading prepared data...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()

print(f"Original training set: {X_train.shape[0]} samples")
print(f"  Churn: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"  Active: {(1-y_train).sum()} ({(1-y_train.mean())*100:.2f}%)")

# Apply SMOTE to balance classes in training set
print("\n" + "="*60)
print("APPLYING SMOTE TO TRAINING DATA")
print("="*60)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Balanced training set: {X_train_balanced.shape[0]} samples")
print(f"  Churn: {y_train_balanced.sum()} ({y_train_balanced.mean()*100:.2f}%)")
print(f"  Active: {(1-y_train_balanced).sum()} ({(1-y_train_balanced.mean())*100:.2f}%)")
print(f"✓ Classes balanced successfully!")

# Model configurations with optimized hyperparameters
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        random_state=42,
        C=1.0,
        solver='lbfgs'
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,  # Reduced from 15
        min_samples_split=20, # Increased from 10
        min_samples_leaf=10, # Increased from 5
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,   # Reduced from 5
        learning_rate=0.05,
        subsample=0.7, # Reduced from 0.8
        min_samples_split=20,
        random_state=42
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(64, 32), # Simplified
        activation='relu',
        solver='adam',
        alpha=0.01, # Increased regularization
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
}

# Train all models
results = []
trained_models = {}

for model_name, model in models.items():
    print("\n" + "="*60)
    print(f"TRAINING {model_name.upper()}")
    print("="*60)
    
    # Train on balanced data
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predict on validation set (original imbalanced distribution)
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    # Store results
    result = {
        'model': model_name,
        'roc_auc': round(roc_auc, 4),
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4)
    }
    results.append(result)
    trained_models[model_name] = model
    
    # Print metrics
    print(f"✓ {model_name} trained successfully")
    print(f"\nValidation Metrics:")
    print(f"  ROC-AUC:   {roc_auc:.4f} {'✅' if roc_auc >= 0.75 else '❌'}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Save model
    model_path = f'models/{model_name.lower().replace(" ", "_")}_smote.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Saved to: {model_path}")

# Print comparison
print("\n" + "="*60)
print("MODEL COMPARISON (WITH SMOTE)")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('roc_auc', ascending=False)

print(results_df.to_string(index=False))

# Save comparison
results_df.to_csv('models/model_comparison_smote.csv', index=False)
print(f"\n✓ Comparison saved to: models/model_comparison_smote.csv")

# Identify best model
best_model_name = results_df.iloc[0]['model']
best_roc_auc = results_df.iloc[0]['roc_auc']

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)
print(f"🏆 {best_model_name}")
print(f"   ROC-AUC: {best_roc_auc:.4f}")

if best_roc_auc >= 0.75:
    print(f"   ✅ MEETS MINIMUM REQUIREMENT (≥ 0.75)")
else:
    print(f"   ❌ BELOW MINIMUM ({0.75 - best_roc_auc:.4f} short)")

# Save best model separately
best_model = trained_models[best_model_name]
joblib.dump(best_model, 'models/best_model.pkl')
print(f"\n✓ Best model saved to: models/best_model.pkl")

# Save metadata
metadata = {
    'best_model': best_model_name,
    'best_roc_auc': float(best_roc_auc),
    'training_method': 'SMOTE (Synthetic Minority Over-sampling)',
    'training_samples': int(X_train_balanced.shape[0]),
    'validation_samples': int(X_val.shape[0]),
    'feature_count': int(X_train.shape[1]),
    'meets_requirement': bool(best_roc_auc >= 0.75)
}

with open('models/training_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"✓ Metadata saved to: models/training_metadata.json")
print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)
