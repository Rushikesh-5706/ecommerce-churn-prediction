"""
Model Training and Evaluation Script

This script trains all 5 models and compares their performance:
1. Logistic Regression (baseline)
2. Decision Tree
3. Random Forest
4. XGBoost
5. Neural Network (MLP)

Target: ROC-AUC >= 0.75 (minimum), >= 0.78 (target)
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier

def load_prepared_data():
    """Load prepared train/val/test sets"""
    print("\n" + "="*60)
    print("LOADING PREPARED DATA")
    print("="*60)
    
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_val = pd.read_csv('data/processed/X_val.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    print(f"✓ Loaded datasets")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X, y, dataset_name=""):
    """
    Evaluate model and return metrics
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    # Use custom threshold to boost precision (0.53)
    y_pred = (y_pred_proba >= 0.53).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }
    
    if dataset_name:
        print(f"\n{dataset_name} Metrics:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
    
    return metrics

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression (baseline)"""
    print("\n" + "="*60)
    print("1. LOGISTIC REGRESSION (BASELINE)")
    print("="*60)
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    
    return model, train_metrics, val_metrics

def train_decision_tree(X_train, y_train, X_val, y_val):
    """Train Decision Tree"""
    print("\n" + "="*60)
    print("2. DECISION TREE")
    print("="*60)
    
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    
    return model, train_metrics, val_metrics

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest"""
    print("\n" + "="*60)
    print("3. RANDOM FOREST")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=15,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    
    return model, train_metrics, val_metrics

def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting (sklearn)"""
    print("\n" + "="*60)
    print("4. GRADIENT BOOSTING")
    print("="*60)
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    
    return model, train_metrics, val_metrics

def train_neural_network(X_train, y_train, X_val, y_val):
    """Train Neural Network (MLP)"""
    print("\n" + "="*60)
    print("5. NEURAL NETWORK (MLP)")
    print("="*60)
    
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    model.fit(X_train, y_train)
    
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    
    return model, train_metrics, val_metrics

def compare_models(results):
    """
    Compare all models and select the best
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison = []
    for name, metrics in results.items():
        comparison.append({
            'Model': name,
            'Val_ROC_AUC': metrics['val']['roc_auc'],
            'Val_Precision': metrics['val']['precision'],
            'Val_Recall': metrics['val']['recall'],
            'Val_F1': metrics['val']['f1'],
            'Train_ROC_AUC': metrics['train']['roc_auc'],
            'Overfitting': metrics['train']['roc_auc'] - metrics['val']['roc_auc']
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('Val_ROC_AUC', ascending=False)
    
    print("\nPerformance Ranking:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    print(f"\n✓ Saved comparison to models/model_comparison.csv")
    
    # Select best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_roc_auc = comparison_df.iloc[0]['Val_ROC_AUC']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Validation ROC-AUC: {best_roc_auc:.4f}")
    
    # Check if meets minimum requirement
    if best_roc_auc >= 0.78:
        print(f"   ✅ EXCEEDS TARGET (≥ 0.78)")
    elif best_roc_auc >= 0.75:
        print(f"   ✅ MEETS MINIMUM (≥ 0.75)")
    else:
        print(f"   ❌ BELOW MINIMUM (<0.75) - Need improvement!")
    
    return best_model_name, comparison_df

def save_best_model(models, best_model_name):
    """Save the best model"""
    best_model = models[best_model_name]
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"\n✓ Saved best model ({best_model_name}) to models/best_model.pkl")
    
    # Also save all models
    for name, model in models.items():
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(model, f'models/{safe_name}.pkl')
    
    print(f"✓ Saved all 5 models to models/ directory")

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("TRAINING ALL MODELS")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_prepared_data()
    
    # Train all models
    models = {}
    results = {}
    
    # 1. Logistic Regression
    lr_model, lr_train, lr_val = train_logistic_regression(X_train, y_train, X_val, y_val)
    models['Logistic Regression'] = lr_model
    results['Logistic Regression'] = {'train': lr_train, 'val': lr_val}
    
    # 2. Decision Tree
    dt_model, dt_train, dt_val = train_decision_tree(X_train, y_train, X_val, y_val)
    models['Decision Tree'] = dt_model
    results['Decision Tree'] = {'train': dt_train, 'val': dt_val}
    
    # 3. Random Forest
    rf_model, rf_train, rf_val = train_random_forest(X_train, y_train, X_val, y_val)
    models['Random Forest'] = rf_model
    results['Random Forest'] = {'train': rf_train, 'val': rf_val}
    
    # 4. Gradient Boosting
    gb_model, gb_train, gb_val = train_gradient_boosting(X_train, y_train, X_val, y_val)
    models['Gradient Boosting'] = gb_model
    results['Gradient Boosting'] = {'train': gb_train, 'val': gb_val}
    
    # 5. Neural Network
    nn_model, nn_train, nn_val = train_neural_network(X_train, y_train, X_val, y_val)
    models['Neural Network'] = nn_model
    results['Neural Network'] = {'train': nn_train, 'val': nn_val}
    
    # Compare models
    best_model_name, comparison_df = compare_models(results)
    
    # Save best model
    save_best_model(models, best_model_name)
    
    # Save results
    with open('models/training_results.json', 'w') as f:
        # Convert numpy types to native Python types
        results_serializable = {}
        for model_name, metrics_dict in results.items():
            results_serializable[model_name] = {
                'train': {k: float(v) for k, v in metrics_dict['train'].items()},
                'val': {k: float(v) for k, v in metrics_dict['val'].items()}
            }
        json.dump(results_serializable, f, indent=4)
    
    print(f"\n✓ Saved training results to models/training_results.json")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED")
    print("="*60)
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Val ROC-AUC: {comparison_df.iloc[0]['Val_ROC_AUC']:.4f}")
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    best_model = models[best_model_name]
    test_metrics = evaluate_model(best_model, X_test, y_test, f"Test Set ({best_model_name})")
    
    # Generate submission.json
    submission = {
        "student_id": "DS2026001",
        "project_name": "Customer Churn Prediction System",
        "repository_url": "https://github.com/Rushikesh-5706/ecommerce-churn-prediction",
        "streamlit_app_url": "https://ecommerce-churn-prediction-rushi5706.streamlit.app/",
        "model_metrics": {
            "roc_auc": round(test_metrics['roc_auc'], 4),
            "precision": round(test_metrics['precision'], 4),
            "recall": round(test_metrics['recall'], 4),
            "f1_score": round(test_metrics['f1'], 4)
        },
        "deployment_status": "active"
    }
    
    with open('submission.json', 'w') as f:
        json.dump(submission, f, indent=4)
    
    print(f"\n✓ Generated submission.json with metrics: AUC={submission['model_metrics']['roc_auc']}, P={submission['model_metrics']['precision']}, R={submission['model_metrics']['recall']}")

if __name__ == "__main__":
    main()

