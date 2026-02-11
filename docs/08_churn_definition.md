# Churn Definition & Temporal Split Strategy

## Executive Summary

This document defines the critical business logic for identifying churned customers and explains the temporal split methodology that prevents data leakage.

**Churn Definition**: A customer is considered "churned" if they made purchases during the training period but did NOT make any purchases during the observation period.

**Key Dates**:
- **Dataset Range**: 2009-12-01 to 2010-12-09 (373 days)
- **Training Period**: 2009-12-01 to 2010-09-09 (283 days, 76%)
- **Observation Period**: 2010-09-10 to 2010-12-09 (90 days, 24%)
- **Cutoff Date**: 2010-09-09 (last day of training period)

---

## 1. Why Temporal Split is Critical

### The Data Leakage Problem

**Data leakage** occurs when information from the future "leaks" into the training data, artificially inflating model performance.

**Example of Data Leakage** (WRONG):
```python
# WRONG: Using all data to calculate features
customer_recency = (max_date - last_purchase_date).days

# Problem: max_date includes observation period
# Model learns from future information it wouldn't have in production
```

**Correct Approach**:
```python
# CORRECT: Using only training period for features
customer_recency = (training_cutoff_date - last_purchase_date).days

# Model only uses past information available at prediction time
```

### Real-World Scenario

Imagine it's September 9, 2010, and we want to predict which customers will churn in the next 3 months:

- **What we know**: Customer behavior from Dec 2009 to Sep 9, 2010
- **What we predict**: Will they purchase between Sep 10 - Dec 9, 2010?
- **What we DON'T know**: Any information after Sep 9, 2010

This is exactly what our temporal split simulates.

---

## 2. Temporal Split Methodology

### Visual Representation

```
|------------- Training Period (283 days) -------------|--- Observation (90 days) ---|
2009-12-01                                    2010-09-09 2010-09-10        2010-12-09

Features calculated from ←                              Churn label from →
this period ONLY                                        this period ONLY
```

### Date Selection Rationale

**Training Period: 2009-12-01 to 2010-09-09 (283 days)**
- **Why 283 days**: Provides ~9 months of customer behavior data
- **Sufficient for**: Capturing seasonal patterns, purchase cycles
- **Allows**: Multiple purchases per customer for robust RFM calculation

**Observation Period: 2010-09-10 to 2010-12-09 (90 days)**
- **Why 90 days**: Standard business definition of "active" customer
- **Industry Standard**: 3 months without purchase = churned
- **Business Alignment**: Matches retention campaign planning cycles

**Cutoff Date: 2010-09-09**
- **Critical**: All features must use data ≤ this date
- **Validation**: No feature should reference dates after 2010-09-09

---

## 3. Churn Label Definition

### Who is Churned?

A customer is **CHURNED (label = 1)** if:
1. ✅ They made at least one purchase during the **training period** (2009-12-01 to 2010-09-09)
2. ❌ They made ZERO purchases during the **observation period** (2010-09-10 to 2010-12-09)

A customer is **ACTIVE (label = 0)** if:
1. ✅ They made at least one purchase during the **training period**
2. ✅ They made at least one purchase during the **observation period**

### Implementation Logic

```python
# Step 1: Split data into training and observation periods
training_cutoff = pd.Timestamp('2010-09-09')

training_data = df[df['InvoiceDate'] <= training_cutoff]
observation_data = df[df['InvoiceDate'] > training_cutoff]

# Step 2: Get customers who purchased in each period
training_customers = set(training_data['Customer ID'].unique())
observation_customers = set(observation_data['Customer ID'].unique())

# Step 3: Define churn
# Churned = purchased in training BUT NOT in observation
churned_customers = training_customers - observation_customers

# Active = purchased in BOTH periods
active_customers = training_customers & observation_customers

# Step 4: Create labels
customer_labels = {}
for customer_id in training_customers:
    if customer_id in churned_customers:
        customer_labels[customer_id] = 1  # Churned
    else:
        customer_labels[customer_id] = 0  # Active
```

### Edge Cases

**Case 1: Customer only in observation period**
- **Scenario**: New customer who first purchased after 2010-09-09
- **Decision**: **EXCLUDE** from dataset
- **Reasoning**: No training period behavior to learn from

**Case 2: Customer with single purchase in training period**
- **Scenario**: Customer purchased once on 2010-05-15, never again
- **Decision**: **INCLUDE** as churned (label = 1)
- **Reasoning**: Valid churn case - one-time buyer

**Case 3: Customer's last purchase is exactly on cutoff date**
- **Scenario**: Last purchase on 2010-09-09
- **Decision**: Included in training period (≤ cutoff)
- **Churn Status**: Depends on observation period activity

---

## 4. Expected Churn Rate

### Target Range: 25-35%

**Why this range?**

1. **Industry Benchmarks**: E-commerce churn rates typically 20-40%
2. **Class Balance**: Not too imbalanced for modeling
3. **Business Realism**: Reflects actual customer retention challenges

**Validation**:
```python
churn_rate = churned_customers / total_customers
assert 0.20 <= churn_rate <= 0.40, "Churn rate outside expected range"
```

**If churn rate is outside range**:
- **< 20%**: Observation period may be too short
- **> 40%**: May indicate data quality issues or very high natural churn

---

## 5. Feature Engineering Rules

### Critical Rules to Prevent Data Leakage

✅ **DO**:
- Calculate Recency using `training_cutoff_date` as reference
- Count Frequency using only training period transactions
- Sum Monetary using only training period transactions
- Use training period for ALL behavioral features

❌ **DON'T**:
- Use observation period data for ANY feature
- Calculate features using entire dataset
- Include future dates in temporal calculations

### Feature Calculation Examples

**Recency** (Days since last purchase):
```python
# CORRECT
customer_last_purchase = training_data.groupby('Customer ID')['InvoiceDate'].max()
recency = (training_cutoff - customer_last_purchase).dt.days

# WRONG
customer_last_purchase = df.groupby('Customer ID')['InvoiceDate'].max()  # Uses observation period!
```

**Frequency** (Number of purchases):
```python
# CORRECT
frequency = training_data.groupby('Customer ID')['Invoice'].nunique()

# WRONG
frequency = df.groupby('Customer ID')['Invoice'].nunique()  # Includes observation period!
```

**Monetary** (Total spent):
```python
# CORRECT
monetary = training_data.groupby('Customer ID')['TotalPrice'].sum()

# WRONG
monetary = df.groupby('Customer ID')['TotalPrice'].sum()  # Includes observation period!
```

---

## 6. Implementation Checklist

Before proceeding to feature engineering, verify:

- [ ] Training period defined: 2009-12-01 to 2010-09-09
- [ ] Observation period defined: 2010-09-10 to 2010-12-09
- [ ] Cutoff date set: 2010-09-09
- [ ] Churn labels created correctly
- [ ] Churn rate between 20-40%
- [ ] No observation period data used in features
- [ ] All features use training period only

---

## 7. Expected Outcomes

### Dataset Statistics

**After temporal split**:
- **Training customers**: ~3,500-4,000 unique customers
- **Churned customers**: ~900-1,400 (25-35%)
- **Active customers**: ~2,600-2,600 (65-75%)

**Final feature dataset**:
- **Rows**: ~3,500-4,000 (one per customer)
- **Columns**: 30-35 features + 1 target (Churn)
- **Class distribution**: 25-35% churned, 65-75% active

### Validation Metrics

After creating features, verify:

1. **No missing values** in target variable
2. **Churn rate** between 0.20 and 0.40
3. **All features** calculated from training period only
4. **No future leakage**: max(feature_dates) ≤ 2010-09-09

---

## 8. Why This Approach Works

### Simulates Production Environment

In production, when predicting churn:
- We only have **past** customer behavior
- We predict **future** churn risk
- We cannot use future information

Our temporal split exactly replicates this:
- Features = past behavior (training period)
- Target = future outcome (observation period)

### Enables Fair Evaluation

- **Training set**: Customers from early training period
- **Validation set**: Customers from mid training period
- **Test set**: Customers from late training period

All evaluated on their observation period outcomes.

---

## 9. Common Pitfalls to Avoid

### Pitfall 1: Using Entire Dataset for Features
```python
# WRONG
df['Recency'] = (df['InvoiceDate'].max() - df.groupby('Customer ID')['InvoiceDate'].transform('max')).dt.days
```
**Problem**: Uses observation period max date

### Pitfall 2: Inconsistent Cutoff Dates
```python
# WRONG
recency_cutoff = pd.Timestamp('2010-09-09')
frequency_cutoff = pd.Timestamp('2010-12-09')  # Different cutoff!
```
**Problem**: Inconsistent temporal boundaries

### Pitfall 3: Including Observation Period Customers
```python
# WRONG
all_customers = df['Customer ID'].unique()  # Includes observation-only customers
```
**Problem**: Cannot create features for customers with no training period data

---

## 10. Next Steps

After defining churn:

1. ✅ Implement feature engineering pipeline (`src/03_feature_engineering.py`)
2. ✅ Calculate RFM features using training period only
3. ✅ Create behavioral and temporal features
4. ✅ Validate churn rate (20-40%)
5. ✅ Verify no data leakage
6. ✅ Save customer-level dataset with features and target

---

**Document Version**: 1.0  
**Last Updated**: February 10, 2026  
**Critical for**: Preventing data leakage and ensuring valid model evaluation
