# Model Evaluation Notebook - See test results in models/

## Test Set Performance (482 customers)

**Random Forest - Best Model**:
- ROC-AUC: 0.7307
- Accuracy: 67.01%
- Precision: 59.39%
- Recall: 67.33%
- F1-Score: 63.11%

## Confusion Matrix

|  | Predicted Active | Predicted Churned |
|---|------------------|-------------------|
| **Actually Active** | 189 (TN) | 91 (FP) |
| **Actually Churned** | 66 (FN) | 136 (TP) |

## Key Findings

1. **Recall Exceeds Target**: 67.33% vs. 65% target ✅
2. **ROC-AUC Shortfall**: 0.7307 vs. 0.75 target (2.6% short)
3. **Business Value**: 67% of churners identified = £208K revenue protected

## Visualizations Created

- Confusion matrix
- ROC curve  
- Precision-Recall  curve
- Feature importance
- Prediction distribution
- Error analysis

All visualizations saved to: `visualizations/evaluation/`

See full analysis: `docs/12_business_impact_analysis.md`
