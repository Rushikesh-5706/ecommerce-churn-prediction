# Advanced Models Notebook - See src/05_train_models_smote.py for implementation

All 5 models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Neural Network) are trained with SMOTE in the Python script.

## Model Comparison Results

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Random Forest | 0.7307 | 0.6701 | 0.5939 | 0.6733 | 0.6311 |
| Neural Network | 0.7250 | 0.6535 | 0.5714 | 0.6931 | 0.6264 |
| Gradient Boosting | 0.7189 | 0.6556 | 0.5833 | 0.6238 | 0.6029 |
| Logistic Regression | 0.7182 | 0.6598 | 0.5812 | 0.6733 | 0.6239 |
| Decision Tree | 0.6821 | 0.6307 | 0.5526 | 0.6238 | 0.5860 |

## Selected Model: Random Forest
- Best ROC-AUC: 0.7307
- Balanced precision-recall trade-off
- Interpretable feature importance
- Production-ready

Run:
```bash
python src/05_train_models_smote.py
```

See full details in docs/11_model_selection.md
