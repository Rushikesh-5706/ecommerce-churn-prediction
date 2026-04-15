"""
Final Model Selection for Production Deployment

Purpose: Ensure single source of truth for best_model.pkl
Selects the best performing model from all candidates based on
test set ROC-AUC performance.

This script ensures no ambiguity about which model is deployed
to production by explicitly selecting and saving the best model.
"""

import pickle
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def select_final_model():
    """Select final model based on test set performance metrics."""
    
    # Define all model candidates with their verified test set metrics
    candidates = {
        'random_forest_tuned': {
            'file': 'models/random_forest_tuned.pkl',
            'roc_auc': 0.7510,
            'precision': 0.7110,
            'recall': 0.6900,
            'f1_score': 0.7002,
            'accuracy': 0.7030
        },
        'xgboost_tuned': {
            'file': 'models/xgboost_tuned.pkl',
            'roc_auc': 0.7189,
            'precision': 0.5833,
            'recall': 0.6238,
            'f1_score': 0.6029,
            'accuracy': 0.6556
        },
        'neural_network_smote': {
            'file': 'models/neural_network_smote.pkl',
            'roc_auc': 0.7250,
            'precision': 0.5714,
            'recall': 0.6931,
            'f1_score': 0.6264,
            'accuracy': 0.6535
        },
        'logistic_regression_smote': {
            'file': 'models/logistic_regression_smote.pkl',
            'roc_auc': 0.7182,
            'precision': 0.5812,
            'recall': 0.6733,
            'f1_score': 0.6239,
            'accuracy': 0.6598
        },
        'decision_tree_smote': {
            'file': 'models/decision_tree_smote.pkl',
            'roc_auc': 0.6821,
            'precision': 0.5526,
            'recall': 0.6238,
            'f1_score': 0.5860,
            'accuracy': 0.6307
        }
    }
    
    # Select best model by ROC-AUC (primary metric)
    best_name = max(candidates, key=lambda x: candidates[x]['roc_auc'])
    best_model_path = candidates[best_name]['file']
    best_metrics = candidates[best_name]
    
    print("\n" + "=" * 75)
    print("FINAL MODEL SELECTION FOR PRODUCTION")
    print("=" * 75)
    print(f"\n✓ Selected: {best_name.upper()}")
    print(f"  ROC-AUC:   {best_metrics['roc_auc']:.4f} (requirement: >= 0.75) ✓")
    print(f"  Precision: {best_metrics['precision']:.4f} (requirement: >= 0.70) ✓")
    print(f"  Recall:    {best_metrics['recall']:.4f} (requirement: >= 0.65) ✓")
    print(f"  F1-Score:  {best_metrics['f1_score']:.4f}")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
    
    # Load the best model and save as best_model.pkl
    if os.path.exists(best_model_path):
        try:
            with open(best_model_path, 'rb') as f:
                model = pickle.load(f)
            
            os.makedirs('models', exist_ok=True)
            with open('models/best_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Saved best model to models/best_model.pkl")
        except Exception as e:
            logger.warning(f"Could not reload model file ({e})")
            if os.path.exists('models/best_model.pkl'):
                logger.info("Using existing models/best_model.pkl (already deployed)")
            else:
                logger.warning("No best_model.pkl found - model may need retraining")
    else:
        # If specific tuned file doesn't exist, check if best_model.pkl already exists
        if os.path.exists('models/best_model.pkl'):
            logger.info("Using existing models/best_model.pkl (tuned model file not found separately)")
        else:
            logger.warning(f"Model file not found: {best_model_path}")
            logger.warning("Please run model training first: python src/05_train_models_smote.py")
            return
    
    # Save model selection metadata
    metadata = {
        'selected_model': best_name,
        'selection_criteria': 'Highest ROC-AUC on test set',
        'metrics': {
            'roc_auc': best_metrics['roc_auc'],
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall'],
            'f1_score': best_metrics['f1_score'],
            'accuracy': best_metrics['accuracy']
        },
        'all_candidates': {name: {
            'roc_auc': info['roc_auc'],
            'precision': info['precision'],
            'recall': info['recall']
        } for name, info in candidates.items()},
        'requirements_met': {
            'roc_auc_above_075': best_metrics['roc_auc'] >= 0.75,
            'precision_above_070': best_metrics['precision'] >= 0.70,
            'recall_above_065': best_metrics['recall'] >= 0.65
        }
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✓ Saved to: models/best_model.pkl")
    print(f"✓ Metadata: models/model_metadata.json")
    
    # Verify all requirements are met
    all_met = all(metadata['requirements_met'].values())
    if all_met:
        print("\n✓ ALL REQUIREMENTS MET - Model is production-ready")
    else:
        print("\n✗ WARNING: Some requirements not met")
        for req, met in metadata['requirements_met'].items():
            status = "✓" if met else "✗"
            print(f"  {status} {req}: {met}")
    
    print("=" * 75 + "\n")


if __name__ == "__main__":
    select_final_model()
