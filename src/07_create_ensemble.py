"""
Ensemble Creation Script

Combines tuned models using Soft Voting to maximize ROC-AUC
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def load_data():
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    return X_val, y_val

def load_models():
    models = []
    try:
        rf = joblib.load('models/random_forest_tuned.pkl')
        models.append(('rf', rf))
    except: pass
    
    try:
        xgb = joblib.load('models/xgboost_tuned.pkl')
        models.append(('xgb', xgb))
    except: pass
    
    try:
        nn = joblib.load('models/neural_network_tuned.pkl')
        models.append(('nn', nn))
    except: pass
    
    # Add Logistic Regression (baseline) for stability
    try:
        lr = joblib.load('models/logistic_regression_smote.pkl')
        models.append(('lr', lr))
    except: pass
    
    return models

def main():
    print("="*60)
    print("CREATING VOTING ENSEMBLE")
    print("="*60)
    
    X_val, y_val = load_data()
    estimators = load_models()
    
    if not estimators:
        print("❌ No models found!")
        return
        
    print(f"Combining {len(estimators)} models: {[n for n, m in estimators]}")
    
    # Create Stacking Classifier
    # Uses Logistic Regression as the meta-learner to learn how to best combine predictions
    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5,
        n_jobs=-1,
        passthrough=False # Train only on predictions
    )
    
    # Fit on training data
    print("Training Stacking Ensemble (this may take a moment)...")
    ensemble.fit(pd.read_csv('data/processed/X_train.csv'), pd.read_csv('data/processed/y_train.csv').values.ravel())
    
    # Evaluate
    y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
    y_pred = ensemble.predict(X_val)
    
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    print(f"\nEnsemble ROC-AUC: {roc_auc:.4f}")
    
    # Save
    joblib.dump(ensemble, 'models/ensemble_model.pkl')
    
    # Check if this is the best model
    try:
        with open('models/tuning_results.json', 'r') as f:
            best_tuned_score = max(json.load(f).values())
    except:
        best_tuned_score = 0
        
    print(f"Best individual model: {best_tuned_score:.4f}")
    
    if roc_auc > best_tuned_score:
        print("🏆 Ensemble is the new BEST MODEL!")
        joblib.dump(ensemble, 'models/best_model.pkl')
        
        # Save metrics
        metrics = {
            'roc_auc': roc_auc,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred)
        }
        with open('models/ensemble_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
            
    if roc_auc >= 0.75:
        print("✅ SUCCESS: Met minimum ROC-AUC requirement (≥ 0.75)")
    else:
        print(f"❌ Still below 0.75 (Short by {0.75 - roc_auc:.4f})")

if __name__ == "__main__":
    main()
