"""
Hyperparameter Tuning Script

This script performs hyperparameter tuning to achieve ROC-AUC >= 0.75
Uses Grid Search for each model type
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load prepared data"""
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    return X_train, X_val, y_train, y_val

def tune_random_forest(X_train, y_train):
    """Tune Random Forest with expanded parameter grid"""
    print("\n" + "="*60)
    print("TUNING RANDOM FOREST")
    print("="*60)
    
    param_grid = {
        'n_estimators': [150, 200],
        'max_depth': [None, 20, 25],
        'min_samples_split': [10, 15],
        'min_samples_leaf': [5, 8],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_xgboost(X_train, y_train):
    """Tune XGBoost (replaces Gradient Boosting)"""
    print("\n" + "="*60)
    print("TUNING XGBOOST")
    print("="*60)
    
    from xgboost import XGBClassifier
    
    # Estimate scale_pos_weight
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"Estimated scale_pos_weight: {ratio:.2f}")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 0.8],
        'scale_pos_weight': [1, ratio],
        'min_child_weight': [1, 3]
    }
    
    xgb = XGBClassifier(
        random_state=42, 
        eval_metric='logloss',
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_neural_network(X_train, y_train):
    """Tune Neural Network"""
    print("\n" + "="*60)
    print("TUNING NEURAL NETWORK")
    print("=" *60)
    
    param_grid = {
        'hidden_layer_sizes': [(100, 50), (128, 64, 32)],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    }
    
    mlp = MLPClassifier(
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    grid_search = GridSearchCV(
        mlp,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate model on validation set"""
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\n{model_name} Validation ROC-AUC: {roc_auc:.4f}")
    
    if roc_auc >= 0.78:
        print(f"   ✅ EXCEEDS TARGET (≥ 0.78)!")
    elif roc_auc >= 0.75:
        print(f"   ✅ MEETS MINIMUM (≥ 0.75)")
    else:
        print(f"   ❌ BELOW MINIMUM")
    
    return roc_auc

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Load data
    X_train, X_val, y_train, y_val = load_data()
    
    results = {}
    
    # Tune Random Forest
    rf_tuned = tune_random_forest(X_train, y_train)
    rf_score = evaluate_model(rf_tuned, X_val, y_val, "Random Forest (Tuned)")
    results['Random Forest (Tuned)'] = rf_score
    joblib.dump(rf_tuned, 'models/random_forest_tuned.pkl')
    
    # Tune XGBoost
    xgb_tuned = tune_xgboost(X_train, y_train)
    xgb_score = evaluate_model(xgb_tuned, X_val, y_val, "XGBoost (Tuned)")
    results['XGBoost (Tuned)'] = xgb_score
    joblib.dump(xgb_tuned, 'models/xgboost_tuned.pkl')
    
    # Tune Neural Network
    nn_tuned = tune_neural_network(X_train, y_train)
    nn_score = evaluate_model(nn_tuned, X_val, y_val, "Neural Network (Tuned)")
    results['Neural Network (Tuned)'] = nn_score
    joblib.dump(nn_tuned, 'models/neural_network_tuned.pkl')
    
    # Select best model
    print("\n" + "="*60)
    print("TUNING SUMMARY")
    print("="*60)
    
    best_model_name = max(results, key=results.get)
    best_score = results[best_model_name]
    
    print("\nModel Performance After Tuning:")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:.4f}")
    
    print(f"\n🏆 BEST TUNED MODEL: {best_model_name}")
    print(f"   Validation ROC-AUC: {best_score:.4f}")
    
    # Load and save as best model
    best_model_file = best_model_name.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.pkl'
    best_model = joblib.load(f'models/{best_model_file}')
    joblib.dump(best_model, 'models/best_model.pkl')
    
    print(f"\n✓ Saved best model to models/best_model.pkl")
    
    # Save results
    with open('models/tuning_results.json', 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=4)
    
    print(f"✓ Saved tuning results to models/tuning_results.json")
    
    if best_score >= 0.75:
        print("\n✅ SUCCESS: Met minimum ROC-AUC requirement (≥ 0.75)")
    else:
        print(f"\n⚠️  WARNING: Best score {best_score:.4f} still below 0.75")
        print("   Consider additional feature engineering or ensemble methods")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
