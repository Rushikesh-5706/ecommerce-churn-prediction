# Technical Approach Document

## Overview

This document outlines the technical methodology for building the Customer Churn Prediction System, including algorithm selection rationale, feature engineering strategy, and deployment architecture.

## 1. Problem Formulation

### Why Classification (Not Regression)?

**Churn prediction is a binary classification problem** because:

1. **Target Variable**: Customer status is categorical (Churned vs Active), not continuous
2. **Business Decision**: Stakeholders need a yes/no answer: "Will this customer churn?"
3. **Actionable Outputs**: Binary predictions enable clear decision-making (contact vs don't contact)
4. **Probability Scores**: Classification models provide probability estimates for risk scoring

**Alternative Considered**: Regression to predict "days until churn"
- **Rejected because**: Uncertain churn timing, business prefers binary risk assessment

### Classification Approach

- **Type**: Supervised binary classification
- **Labels**: 0 (Active), 1 (Churned)
- **Evaluation**: Focus on probability calibration and ranking (ROC-AUC)

## 2. Data Transformation Strategy

### Challenge: Transaction-Level to Customer-Level

**Current State**: 500k+ transaction records (one row per purchase)

**Required State**: Customer-level features (one row per customer)

**Transformation Process**:

```
Raw Transactions (500k rows)
    ↓
Data Cleaning (remove invalid, missing, outliers)
    ↓
Cleaned Transactions (~350k rows)
    ↓
Temporal Split (training vs observation period)
    ↓
Feature Aggregation (group by CustomerID)
    ↓
Customer-Level Dataset (~4,000 rows, 30+ features)
    ↓
Train/Val/Test Split
    ↓
Model Training & Evaluation
```

### Temporal Split Methodology

**Critical for Preventing Data Leakage**:

```
|-------- Training Period --------|--- Observation Period ---|
2009-12-01          2011-09-09   2011-09-10    2011-12-09

Features calculated from ←        Churn label from →
training period only               observation period
```

**Why This Matters**:
- Simulates real-world scenario: predict future behavior using past data
- Prevents data leakage: no future information used in features
- Enables temporal validation: model learns from past, predicts future

## 3. Feature Engineering Strategy

### RFM Analysis (Foundation)

**RFM = Recency, Frequency, Monetary**

Industry-standard framework for customer segmentation:

1. **Recency**: Days since last purchase
   - **Why**: Most predictive feature for churn
   - **Calculation**: `training_cutoff_date - last_purchase_date`
   - **Hypothesis**: Higher recency → Higher churn risk

2. **Frequency**: Number of purchases
   - **Why**: Measures customer loyalty
   - **Calculation**: Count of unique invoices
   - **Hypothesis**: Lower frequency → Higher churn risk

3. **Monetary**: Total amount spent
   - **Why**: Identifies high-value customers
   - **Calculation**: Sum of all transaction amounts
   - **Hypothesis**: Lower spend → Higher churn risk (but complex relationship)

### Behavioral Pattern Features

**Purchase Consistency**:
- Average days between purchases
- Standard deviation of purchase intervals
- Indicates habitual vs sporadic buyers

**Basket Characteristics**:
- Average basket size (items per transaction)
- Product diversity (unique products / total products)
- Price preference (average unit price)

**Shopping Preferences**:
- Preferred shopping day (mode of day of week)
- Preferred shopping hour
- Country diversity (if customer shops from multiple countries)

### Temporal Features

**Recent Activity Windows**:
- Purchases in last 30 days
- Purchases in last 60 days
- Purchases in last 90 days
- **Rationale**: Recent behavior is more predictive than distant past

**Customer Lifecycle**:
- Customer lifetime (days from first to last purchase)
- Purchase velocity (purchases per day)
- **Rationale**: Newer customers may have different churn patterns

### Product Affinity Features

- Product diversity score
- Average price point
- Quantity preference (bulk vs single-item buyer)

### Derived Features

- RFM scores (quartile-based scoring)
- Customer segments (Champions, Loyal, At Risk, Lost)
- Composite metrics (e.g., recency × frequency interaction)

**Total Expected Features**: 25-35

## 4. Algorithm Selection Rationale

### Why Test Multiple Algorithms?

**No Free Lunch Theorem**: No single algorithm is best for all problems

**Our Strategy**: Implement 5 algorithms, compare performance, select best

### Algorithm Portfolio

#### 1. Logistic Regression (Baseline)
**Pros**:
- Simple, interpretable
- Fast training
- Provides probability estimates
- Good baseline for comparison

**Cons**:
- Assumes linear relationships
- May underperform with complex patterns

**Expected Performance**: ROC-AUC ~0.70-0.75

#### 2. Decision Tree
**Pros**:
- Handles non-linear relationships
- No feature scaling needed
- Interpretable (visualizable)

**Cons**:
- Prone to overfitting
- Unstable (small data changes → different tree)

**Expected Performance**: ROC-AUC ~0.68-0.72

**Learning Objective**: Understand tree-based models

#### 3. Random Forest (Ensemble)
**Pros**:
- Reduces overfitting via bagging
- Handles non-linearity well
- Provides feature importance
- Robust to outliers

**Cons**:
- Less interpretable than single tree
- Slower training than logistic regression

**Expected Performance**: ROC-AUC ~0.75-0.80

**Why Better Than Decision Tree**: Ensemble of trees reduces variance

#### 4. Gradient Boosting (XGBoost/LightGBM)
**Pros**:
- State-of-the-art performance for tabular data
- Handles complex interactions
- Built-in regularization
- Feature importance

**Cons**:
- Requires hyperparameter tuning
- Longer training time
- Risk of overfitting if not tuned

**Expected Performance**: ROC-AUC ~0.78-0.85

**Why Expected Best**: Boosting typically outperforms bagging for structured data

#### 5. Neural Network (MLP)
**Pros**:
- Can learn complex non-linear patterns
- Flexible architecture

**Cons**:
- Requires careful tuning
- Needs more data than we have
- Less interpretable
- Overkill for tabular data

**Expected Performance**: ROC-AUC ~0.72-0.78

**Why Include**: Educational value, may surprise us

### Model Selection Criteria

**Primary**: ROC-AUC score on validation set

**Secondary Considerations**:
1. **Business Metric Alignment**: Precision vs Recall trade-off
2. **Interpretability**: Can we explain predictions to stakeholders?
3. **Training Time**: Must be reasonable for retraining
4. **Deployment Complexity**: Simpler models easier to maintain

**Decision Framework**:
```
IF ROC-AUC difference < 0.02:
    Choose more interpretable model
ELSE:
    Choose highest ROC-AUC
```

## 5. Handling Class Imbalance

**Expected Churn Rate**: 25-35% (imbalanced but not severe)

**Strategies**:

1. **Stratified Sampling**: Maintain class ratio in train/val/test splits
2. **Appropriate Metrics**: Use ROC-AUC, not accuracy
3. **Class Weights** (if needed): Penalize misclassifying minority class more
4. **Threshold Tuning**: Adjust decision threshold based on business costs

**Not Using**:
- SMOTE (synthetic oversampling): Dataset size sufficient
- Undersampling: Would lose valuable information

## 6. Evaluation Strategy

### Train/Validation/Test Split

**Split Ratio**: 70% / 15% / 15%

**Why Three Sets**:
- **Training**: Model learning
- **Validation**: Model selection and hyperparameter tuning
- **Test**: Final unbiased performance estimate

**Critical Rule**: Test set is NEVER used until final evaluation

### Cross-Validation

**Method**: 5-fold stratified cross-validation

**Purpose**:
- Assess model stability
- Reduce variance in performance estimates
- Detect overfitting

**Process**:
```
For each of 5 folds:
    Train on 4 folds
    Validate on 1 fold
    Record performance

Report: Mean ± Std of performance
```

### Performance Metrics

**Primary**: ROC-AUC (Area Under ROC Curve)
- **Why**: Threshold-independent, handles imbalance well
- **Target**: ≥ 0.75 (minimum), ≥ 0.78 (target)

**Secondary**:
- **Precision**: Of predicted churners, how many actually churn?
  - **Business Impact**: Avoid wasting budget on false alarms
  - **Target**: ≥ 0.70

- **Recall**: Of actual churners, how many do we catch?
  - **Business Impact**: Maximize revenue protection
  - **Target**: ≥ 0.65

- **F1-Score**: Harmonic mean of precision and recall
  - **Target**: ≥ 0.72

**Confusion Matrix Analysis**:
- True Positives (TP): Correctly identified churners → Revenue saved
- False Positives (FP): Incorrectly flagged active customers → Wasted budget
- True Negatives (TN): Correctly identified active → No action needed
- False Negatives (FN): Missed churners → Lost revenue

## 7. Deployment Architecture

### Technology Stack

**Backend**:
- **Language**: Python 3.8+
- **ML Framework**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Model Serialization**: joblib/pickle

**Frontend**:
- **Framework**: Streamlit (rapid prototyping, free deployment)
- **Visualization**: matplotlib, seaborn, plotly

**Deployment**:
- **Platform**: Streamlit Community Cloud (free)
- **Containerization**: Docker (for reproducibility)
- **Version Control**: Git/GitHub

### Application Architecture

```
┌─────────────────────────────────────────┐
│         Streamlit Web App               │
│  ┌───────────────────────────────────┐  │
│  │  User Interface                   │  │
│  │  - Single Prediction Form         │  │
│  │  - Batch Upload (CSV)             │  │
│  │  - Dashboard                      │  │
│  └───────────────────────────────────┘  │
│                 ↓                        │
│  ┌───────────────────────────────────┐  │
│  │  Prediction API (predict.py)      │  │
│  │  - load_model()                   │  │
│  │  - preprocess_input()             │  │
│  │  - predict()                      │  │
│  │  - predict_proba()                │  │
│  └───────────────────────────────────┘  │
│                 ↓                        │
│  ┌───────────────────────────────────┐  │
│  │  Trained Model & Scaler           │  │
│  │  - best_model.pkl                 │  │
│  │  - scaler.pkl                     │  │
│  │  - feature_names.json             │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Deployment Workflow

```
1. Train model locally
2. Save model artifacts (model, scaler, feature names)
3. Build Streamlit app
4. Test locally: streamlit run app.py
5. Push to GitHub
6. Deploy to Streamlit Cloud
7. Verify live URL
```

### Model Serving Strategy

**Batch Predictions** (not real-time):
- **Input**: CSV file with customer features
- **Process**: Load model → Preprocess → Predict → Return results
- **Output**: CSV with CustomerID and churn probability

**Why Batch**:
- Simpler implementation
- Sufficient for retention campaigns (not time-critical)
- Lower infrastructure requirements

## 8. Risk Mitigation

### Technical Risks

**Risk 1: Data Leakage**
- **Mitigation**: Strict temporal split, careful feature engineering
- **Validation**: Check that all features use only training period data

**Risk 2: Overfitting**
- **Mitigation**: Cross-validation, regularization, simpler models
- **Validation**: Compare train vs validation performance

**Risk 3: Poor Performance**
- **Mitigation**: Multiple algorithms, extensive feature engineering
- **Fallback**: If ROC-AUC < 0.75, revisit feature engineering

**Risk 4: Deployment Failures**
- **Mitigation**: Docker containerization, thorough testing
- **Fallback**: Document local deployment instructions

### Model Monitoring (Future)

**Not in Current Scope**, but recommended for production:
- Track prediction distribution over time
- Monitor churn rate changes
- Retrain model quarterly
- A/B test model versions

## 9. Success Criteria

### Technical Success
- ✅ ROC-AUC ≥ 0.75 on held-out test set
- ✅ Model trains in < 1 hour
- ✅ Predictions complete in < 5 seconds for batch of 1000 customers
- ✅ Deployed application loads in < 3 seconds

### Business Success
- ✅ Model predictions are actionable (probability scores)
- ✅ Feature importance is interpretable
- ✅ Application is user-friendly for non-technical stakeholders
- ✅ Documentation enables model maintenance

## 10. Timeline & Milestones

| Milestone | Deliverable | Success Metric |
|-----------|-------------|----------------|
| Data Acquisition | Raw dataset loaded | 500k+ rows, 8 columns |
| Data Cleaning | Cleaned dataset | 60-70% retention, 0 missing values |
| Feature Engineering | Customer features | 4k customers, 25+ features, 25-35% churn rate |
| Baseline Model | Logistic Regression | ROC-AUC > 0.65 |
| Advanced Models | 5 models compared | Best model ROC-AUC > 0.75 |
| Deployment | Live web app | Public URL, all features working |
| Documentation | Complete docs | README, technical docs, presentation |

## 11. Tools & Libraries

### Core ML Stack
```python
pandas==2.0.0          # Data manipulation
numpy==1.24.0          # Numerical computing
scikit-learn==1.3.0    # ML algorithms, preprocessing
xgboost==2.0.0         # Gradient boosting
```

### Visualization
```python
matplotlib==3.7.0      # Static plots
seaborn==0.12.0        # Statistical visualizations
plotly==5.17.0         # Interactive plots
```

### Deployment
```python
streamlit==1.28.0      # Web app framework
joblib==1.3.0          # Model serialization
```

### Development
```python
jupyter==1.0.0         # Notebooks
pytest==7.4.0          # Testing (optional)
```

## 12. Key Assumptions

1. **Data Representativeness**: Historical data reflects future behavior
2. **Feature Sufficiency**: Transactional data alone is enough for prediction
3. **Churn Definition**: 90-day window is appropriate
4. **Deployment Platform**: Streamlit Cloud remains free and available
5. **Stakeholder Adoption**: Business teams will use predictions

## 13. Next Steps

After completing this technical approach:

1. **Phase 2**: Download and explore dataset
2. **Phase 3**: Implement data cleaning pipeline
3. **Phase 4**: Build feature engineering pipeline
4. **Phase 5**: Conduct EDA to validate hypotheses
5. **Phase 6-7**: Train and evaluate models
6. **Phase 8**: Deploy application
7. **Phase 9-10**: Complete documentation

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Technical Lead**: Data Science Team
