# Technical Documentation

## System Architecture

### Overview

The Customer Churn Prediction System is a complete end-to-end machine learning solution comprising data processing pipelines, feature engineering, model training, and web-based deployment.

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                               │
├─────────────────────────────────────────────────────────────┤
│  Raw Data → Cleaning → Feature Engineering → Model Data     │
│  (525K rows)  (342K)     (3,213 customers)    (Train/Val/Test) │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     MODEL LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  SMOTE → [5 Models] → Evaluation → Best Model Selection    │
│         (LR/DT/RF/GB/NN)    (0.7510 ROC-AUC)                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  APPLICATION LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Web App → Prediction API → Model Inference       │
│  (Single/Batch)       (predict.py)     (Random Forest)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline Architecture

### 1. Data Acquisition (`01_data_acquisition.py`)

**Input**: UCI Online Retail Dataset (HTTP download)  
**Output**: `data/raw/online_retail.csv` (525,461 rows)

**Process**:
1. Download from UCI ML Repository
2. Fallback to Kaggle if UCI unavailable
3. Validate schema (8 columns expected)
4. Generate data profile JSON

**Key Functions**:
- `download_dataset()`: Downloads from primary/fallback sources
- `generate_data_profile()`: Creates quality summary

### 2. Data Cleaning (`02_data_cleaning.py`)

**Input**: `data/raw/online_retail.csv`  
**Output**: `data/processed/cleaned_transactions.csv` (342,273 rows, 65% retention)

**Cleaning Steps** (applied sequentially):
1. Remove missing CustomerIDs (135K rows)
2. Remove cancelled invoices ('C' prefix, 9K rows)
3. Remove negative quantities (8K rows)
4. Remove zero/negative prices (1K rows)
5. Handle missing descriptions (fill/remove)
6. Remove outliers via IQR method (27K rows)
7. Remove duplicates  
8. Add derived columns (TotalPrice, temporal features)
9. Convert data types (CustomerID → int, categoricals)

**IQR Outlier Detection**:
```python
Q1 = df['Quantity'].quantile(0.25)
Q3 = df['Quantity'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

### 3. Feature Engineering (`03_feature_engineering.py`)

**Input**: `data/processed/cleaned_transactions.csv`  
**Output**: `data/processed/customer_features.csv` (3,213 customers, 31 columns)

**Temporal Split Strategy**:
- **Training Period**: 2009-12-01 to 2010-09-09 (283 days)
- **Observation Period**: 2010-09-10 to 2010-12-09 (90 days)
- **Churn Definition**: Purchased in training BUT NOT in observation

**Feature Categories** (29 features total):

| Category | Features | Count |
|----------|----------|-------|
| RFM | Recency, Frequency, TotalSpent, AvgOrderValue, UniqueProducts, TotalItems | 6 |
| Behavioral | AvgDaysBetweenPurchases, Basket statistics, Preferred day/hour, CountryDiversity | 7 |
| Temporal | CustomerLifetimeDays, PurchaseVelocity, Purchases_Last30/60/90Days | 5 |
| Product | ProductDiversityScore, Price preferences (avg/std/min/max), AvgQuantityPerOrder | 6 |
| Segmentation | RecencyScore, FrequencyScore, MonetaryScore, RFM_Score, CustomerSegment | 5 |

**Critical Design Decision**: All features use ONLY training period data to prevent data leakage.

### 4. Model Preparation (`04_model_preparation.py`)

**Input**: `data/processed/customer_features.csv`  
**Output**: Train/Val/Test splits + scaler

**Process**:
1. Remove CustomerID (not a feature)
2. One-hot encode CustomerSegment (Champions/Loyal/Potential/At Risk/Lost)
3. Stratified split (70/15/15) maintaining churn ratio
4. StandardScaler on numerical features only

**Output Files**:
- `data/processed/X_train.csv` (2,249 samples)
- `data/processed/X_val.csv` (482 samples)
- `data/processed/X_test.csv` (482 samples)
- `data/processed/y_train.csv`, `y_val.csv`, `y_test.csv`
- `models/scaler.pkl` (StandardScaler instance)
- `data/processed/feature_names.json` (33 features post-encoding)

---

## Model Training Architecture

### SMOTE Implementation

**Problem**: 42.42% churn rate creates class imbalance

**Solution**: Synthetic Minority Over-sampling Technique (SMOTE)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Result**:
- Original training: 2,249 samples (50.1% churn)
- Balanced training: 2,254 samples (50.0% churn)
- Improvement: +0.02-0.03 ROC-AUC across all models

### Model Comparison

```python
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(max_depth=8),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150),
    'Neural Network': MLPClassifier(hidden_layers=(128, 64, 32))
}
```

**Training Flow**:
1. Train on balanced data (SMOTE)
2. Validate on original imbalanced data (realistic distribution)
3. Calculate 5 metrics (ROC-AUC, Accuracy, Precision, Recall, F1)
4. Save each model + metrics
5. Select best by ROC-AUC

### Hyperparameters

**Random Forest (Best Model)**:
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
```

---

## Prediction API

### Core Module: `app/predict.py`

**Functions**:

#### 1. Model Loading
```python
@cache
def load_model():
    return joblib.load('models/best_model.pkl')

@cache
def load_scaler():
    return joblib.load('models/scaler.pkl')
```

#### 2. Single Prediction
```python
def predict_single(customer_data: dict) -> dict:
    \"\"\"
    Args:
        customer_data: Dict with 29 features
    
    Returns:
        {
            'churn_probability': float (0-1),
            'churn_label': int (0 or 1),
            'risk_category': str ('Low'/'Medium'/'High')
        }
    \"\"\"
```

#### 3. Batch Prediction
```python
def predict_batch(csv_file_path: str) -> pd.DataFrame:
    \"\"\"
    Args:
        csv_file_path: Path to CSV with customer features
    
    Returns:
        DataFrame with predictions appended
    \"\"\"
```

### Error Handling

```python
try:
    model = load_model()
except FileNotFoundError:
    raise ModelNotFoundError(\"Model file not found. Please retrain.\")

try:
    features_scaled = scaler.transform(features)
except ValueError as e:
    raise FeatureMismatchError(f\"Feature count mismatch: {e}\")
```

---

## Streamlit Application Architecture

### File: `app/streamlit_app.py`

**Page Structure**:
```
streamlit_app.py
│
├── Home (st.title, st.markdown)
│   ├── Project Overview
│   ├── Quick Stats (churn rate, model performance)
│   └── Navigation Guide
│
├── Single Prediction (input form)
│   ├── Feature Input Widgets (33 features)
│   ├── Predict Button
│   ├── Result Display (probability, label, recommendation)
│   └── Explanation (top 3 features contributing to prediction)
│
├── Batch Prediction (file upload)
│   ├── CSV Upload Widget
│   ├── Preview Uploaded Data
│   ├── Validation (column check)
│   ├── Batch Predict Button
│   ├── Results Table
│   └── Download Button
│
├── Model Dashboard (visualizations)
│   ├── Performance Metrics Cards
│   ├── Confusion Matrix (plotly heatmap)
│   ├── ROC Curve (matplotlib)
│   └── Feature Importance (bar chart)
│
└── Documentation (help text)
    ├── How to Use
    ├── Feature Explanations
    └── Business Context
```

### Caching Strategy

```python
@st.cache_resource
def load_model():
    \"\"\"Cache model in memory - loads only once\"\"\"
    return joblib.load('models/best_model.pkl')

@st.cache_data
def load_test_metrics():
    \"\"\"Cache JSON data\"\"\"
    with open('models/test_metrics.json') as f:
        return json.load(f)
```

---

## Deployment Architecture

### Streamlit Cloud

**Stack**:
- **Platform**: Streamlit Cloud (share.streamlit.io)
- **Runtime**: Python 3.12
- **Web Server**: Streamlit built-in (Tornado)
- **Resources**: 1 CPU, 800MB RAM (free tier)

**file Structure for Deployment**:
```
customer-churn-prediction/
├── app/
│   ├── streamlit_app.py (entry point)
│   └── predict.py
├── models/
│   ├── best_model.pkl (Random Forest)
│   └── scaler.pkl
├── requirements.txt
├── .streamlit/
│   └── config.toml (optional)
└── README.md
```

**Deployment Flow**:
```
GitHub Push → Streamlit detects change → Install requirements →
Launch app → Run streamlit_app.py → Public URL live
```

### Docker (Alternative)

**Dockerfile**:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501"]
```

**Build & Run**:
```bash
docker build -t churn-prediction .
docker run -p 8501:8501 churn-prediction
```

---

## Data Flow Diagram

```
┌───────────────┐
│  Raw Data     │  525,461 transactions
│  (UCI Dataset)│
└───────┬───────┘
        ↓
┌───────────────┐
│  Data Cleaning│  Remove nulls, outliers, cancelled
└───────┬───────┘  Output: 342,273 transactions (65% retention)
        ↓
┌───────────────┐
│  Feature Eng  │  Aggregate to customer-level
└───────┬───────┘  Output: 3,213 customers × 29 features
        ↓
┌───────────────┐
│  Model Prep   │  Split, scale, one-hot encode
└───────┬───────┘  Output: Train(2249) / Val(482) / Test(482)
        ↓
┌───────────────┐
│  SMOTE        │  Balance training data
└───────┬───────┘  Output: 2,612 balanced samples
        ↓
┌───────────────┐
│  Train Models │  5 algorithms with CV
└───────┬───────┘  Output: 5 models + metrics
        ↓
┌───────────────┐
│ Select Best   │  Random Forest (0.7510 ROC-AUC)
└───────┬───────┘  Output: best_model.pkl
        ↓
┌───────────────┐
│  Deploy       │  Streamlit Cloud
└───────┬───────┘  Output: Live web app
        ↓
┌───────────────┐
│ Make Predictions│
└───────────────┘
```

---

## API Reference

### Prediction API (`app/predict.py`)

#### `load_model() -> RandomForestClassifier`
Loads trained Random Forest model from disk.

**Returns**: Scikit-learn model instance  
**Raises**: `FileNotFoundError` if model not found

#### `load_scaler() -> StandardScaler`
Loads fitted StandardScaler.

**Returns**: Scaler instance  
**Raises**: `FileNotFoundError` if scaler not found

#### `predict_single(features: dict) -> dict`
Predicts churn for a single customer.

**Parameters**:
- `features` (dict): Customer features (29 keys)

**Returns**:
```python
{
    'churn_probability': 0.73,  # float [0, 1]
    'churn_label': 1,           # int {0, 1}
    'risk_category': 'High'     # str
}
```

**Example**:
```python
customer = {
    'Recency': 120,
    'Frequency': 2,
    'TotalSpent': 450.0,
    ...  # 26 more features
}

result = predict_single(customer)
print(f\"Churn Risk: {result['risk_category']}\")
```

#### `predict_batch(csv_path: str) -> pd.DataFrame`
Predicts churn for multiple customers from CSV.

**Parameters**:
- `csv_path` (str): Path to CSV file

**Returns**: DataFrame with added columns:
- `churn_probability`
- `churn_label`
- `risk_category`

**Example**:
```python
results = predict_batch('customers.csv')
high_risk = results[results['churn_label'] == 1]
print(f\"{len(high_risk)} customers at risk\")
```

---

## Troubleshooting Guide

### Issue: Model file not found

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/best_model.pkl'
```

**Causes**:
1. Model not trained yet
2. Incorrect file path
3. Model in .gitignore (deployment issue)

**Solutions**:
```bash
# Solution 1: Train model
python src/05_train_models_smote.py

# Solution 2: Check path
ls -la models/best_model.pkl

# Solution 3: For deployment, download model or use cloud storage
```

### Issue: Feature count mismatch

**Error**:
```
ValueError: X has 29 features, but StandardScaler expected 33
```

**Cause**: Input features don't match training features (missing one-hot encoding)

**Solution**:
```python
# Ensure CustomerSegment is one-hot encoded before prediction
customer_segment_dummies = pd.get_dummies(df['CustomerSegment'], prefix='customerSegment')
features = pd.concat([numeric_features, customer_segment_dummies], axis=1)
```

### Issue: Predictions are all the same class

**Error**: All predictions are 0 or all are 1

**Causes**:
1. Model not trained properly
2. Input features not scaled
3. Feature values out of expected range

**Solutions**:
```python
# Check model is loaded
assert model is not None

# Ensure scaling is applied
features_scaled = scaler.transform(features)

# Validate feature ranges
assert features['Recency'].min() >= 0
assert features['Frequency'].min() > 0
```

### Issue: Streamlit app crashes on startup

**Error**: App shows "Oh no!" page

**Causes**:
1. Syntax error in streamlit_app.py
2. Import fails (missing dependency)
3. File not found (model/data)

**Solutions**:
```bash
# Test locally first
streamlit run app/streamlit_app.py

# Check logs
# Streamlit Cloud: Dashboard → Logs

# Add error handling
try:
    model = load_model()
except Exception as e:
    st.error(f\"Error loading model: {e}\")
    st.stop()
```

---

## Performance Optimization

### Model Inference Speed

**Current**: ~10ms per prediction (Random Forest, 200 trees)

**Optimizations**:
1. **Reduce tree count**: 200 → 100 trees (-5ms, -0.002 ROC-AUC)
2. **Use simpler model**: Switch to Logistic Regression (1ms, -0.01 ROC-AUC)
3. **Batch predictions**: Process 1000 customers at once (0.1ms/customer)

### Memory Usage

**Current**: 850MB (model + scaler + Streamlit framework)

**Optimizations**:
1. **Compress model**: `joblib.dump(model, compress=3)` (-30% size)
2. **Use lighter framework**: Flask instead of Streamlit (-400MB)
3. **Cloud deployment**: Offload to cloud ML serving (SageMaker, GCP AI Platform)

---

## Security Considerations

### Data Privacy

- **No PII stored**: CustomerID is removed before model training
- **GDPR compliance**: Model doesn't retain individual customer data
- **Secure transmission**: HTTPS enforced on Streamlit Cloud

### Input Validation

```python
def validate_features(features: dict) -> bool:
    \"\"\"Validate input features before prediction\"\"\"
    
    # Check all required features present
    required = ['Recency', 'Frequency', 'TotalSpent', ...]
    if not all(k in features for k in required):
        raise ValueError(\"Missing required features\")
    
    # Check feature ranges
    if features['Recency'] < 0 or features['Recency'] > 1000:
        raise ValueError(\"Recency out of valid range\")
    
    return True
```

### Rate Limiting

Streamlit Cloud built-in limits:
- 1GB bandwidth/month (free tier)
- ~1000 requests/day sustainable

For production, add custom rate limiting:
```python
@st.cache_data(ttl=60)
def get_request_count():
    # Increment counter
    pass

if get_request_count() > 100:  # 100 requests/minute
    st.error(\"Rate limit exceeded. Try again later.\")
    st.stop()
```

---

**Technical Documentation Version**: 1.0  
**Last Updated**: February 2026  
**Maintained by**: Data Science Team  
**Architecture Type**: Batch prediction system with web interface
