# Data Splitting Strategy

## Train-Validation-Test Split

### Split Ratios
- **Training Set**: 70% of customer-level data
  - Customers: 2,249 (70% of 3,213)
  - Used for: Model training only
  
- **Validation Set**: 15% of customer-level data
  - Customers: 482 (15% of 3,213)
  - Used for: Hyperparameter tuning
  
- **Test Set**: 15% of customer-level data
  - Customers: 482 (15% of 3,213)
  - Used for: Final model evaluation ONLY (no tuning)

### Implementation
- Stratified split to maintain churn ratio (35.0%) across all sets
- Location: src/04_model_preparation.py
- Method: sklearn.model_selection.train_test_split with stratify=y

### Churn Distribution Across Splits
- Training: 35.0% (787 churned / 2,249 total)
- Validation: 35.0% (169 churned / 482 total) 
- Test: 35.0% (169 churned / 482 total)

**Verification**: All splits maintain identical churn rate ± 0.5%
