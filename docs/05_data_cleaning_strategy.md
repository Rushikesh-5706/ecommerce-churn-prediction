# Data Cleaning Strategy

## Executive Summary

This document outlines the comprehensive data cleaning strategy for the UCI Online Retail dataset. Based on initial data quality assessment, we have identified several critical issues that must be addressed before feature engineering.

**Key Statistics**:
- Original dataset: 525,461 rows × 8 columns
- Date range: 2009-12-01 to 2010-12-09
- Missing CustomerIDs: 0 (0.00%) - Good!
- Negative quantities: 12,326 (2.3%) - Returns/cancellations
- Duplicate rows: 6,865 (1.3%)
- Expected retention after cleaning: 60-70%

---

## 1. Missing Values Strategy

### 1.1 CustomerID (Missing: 0%)
**Status**: ✅ **No action needed**

**Finding**: The dataset has 0% missing CustomerIDs, which is excellent news!

**Decision**: No imputation or removal needed for this field.

**Impact**: All 525,461 rows can be used for customer-level feature engineering.

### 1.2 Description (Missing: TBD - will check in exploration)
**Decision**: 
- **Option A**: Remove rows with missing descriptions (if < 5% missing)
- **Option B**: Fill with "Unknown Product" (if 5-15% missing)

**Reasoning**: Product descriptions may be useful for product affinity analysis, but not critical for churn prediction.

**Impact**: Will assess during data cleaning phase.

---

## 2. Handling Cancellations

### Issue: Cancelled/Returned Orders
**Detection Method**: 
- Invoices starting with 'C' (cancellations)
- Negative quantities (returns)

**Current Status**:
- Cancelled invoices (InvoiceNo starting with 'C'): 0 detected
- Negative quantities: 12,326 rows (2.3%)

**Strategy**: **REMOVE all negative quantities**

**Reasoning**:
1. **Churn Definition Clarity**: Returns complicate the definition of "active" vs "churned"
   - If a customer returns items, are they still "active"?
   - Returns don't represent genuine purchasing behavior
   
2. **Feature Engineering Simplicity**: 
   - Negative quantities would skew RFM calculations
   - TotalSpent could become negative
   - Frequency counts would be ambiguous

3. **Business Focus**: 
   - We're predicting future purchases, not returns
   - Retention campaigns target customers to buy more, not return less

**Alternative Considered**: Create "return_rate" feature
- **Rejected because**: Adds complexity without clear predictive value for churn

**Impact**: Remove ~12,326 rows (2.3% of dataset)

---

## 3. Negative Quantities

**Issue**: 12,326 rows have negative quantities

**Root Cause**: Product returns or order cancellations

**Strategy**: **REMOVE all rows with Quantity ≤ 0**

**Reasoning**:
- Same as cancellation handling above
- Negative quantities represent reverse transactions
- Not representative of normal purchasing behavior

**Implementation**:
```python
df = df[df['Quantity'] > 0]
```

**Impact**: Remove 12,326 rows

---

## 4. Outliers

### 4.1 Quantity Outliers

**Detection Method**: **IQR (Interquartile Range) Method**

**Threshold**: 
- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 - Q1
- Lower bound = Q1 - 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR

**Reasoning**:
- IQR is robust to extreme values
- 1.5 × IQR is standard statistical practice
- Captures ~99.3% of data in normal distribution

**Action**: 
- Remove quantities outside [Lower bound, Upper bound]
- If too aggressive (>40% removal), adjust to 2.0 × IQR or 3.0 × IQR

**Business Consideration**:
- Some bulk purchases may be legitimate (B2B customers)
- Will review distribution before final decision

### 4.2 Price Outliers

**Detection Method**: Same IQR method as Quantity

**Threshold**: Q1 - 1.5 × IQR to Q3 + 1.5 × IQR

**Special Cases**:
- Zero prices: **REMOVE** (likely data errors or samples)
- Negative prices: **REMOVE** (data errors)
- Extremely high prices: Review manually before removing

**Action**:
```python
# Remove zero/negative prices first
df = df[df['UnitPrice'] > 0]

# Then apply IQR method
Q1 = df['UnitPrice'].quantile(0.25)
Q3 = df['UnitPrice'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['UnitPrice'] >= Q1 - 1.5*IQR) & (df['UnitPrice'] <= Q3 + 1.5*IQR)]
```

---

## 5. Data Type Conversions

### 5.1 InvoiceDate
**Current Type**: String or Object  
**Target Type**: datetime64[ns]  
**Method**: `pd.to_datetime(df['InvoiceDate'])`  
**Importance**: Critical for temporal feature engineering

### 5.2 CustomerID
**Current Type**: Float (due to missing values in original data)  
**Target Type**: int64  
**Method**: `df['CustomerID'].astype(int)` (after removing any missing)  
**Importance**: Required for grouping by customer

### 5.3 Categorical Columns
**Columns**: StockCode, Country  
**Target Type**: category (for memory efficiency)  
**Method**: `df['StockCode'] = df['StockCode'].astype('category')`  
**Benefit**: Reduces memory usage by ~50% for these columns

### 5.4 Numerical Columns
**Columns**: Quantity, UnitPrice  
**Current Type**: Already numeric  
**Action**: Verify and ensure no strings

---

## 6. Duplicate Handling

**Issue**: 6,865 duplicate rows detected (1.3%)

**Definition of Duplicate**:
- Same InvoiceNo, StockCode, CustomerID, Quantity, UnitPrice, InvoiceDate

**Strategy**: **REMOVE exact duplicates**

**Reasoning**:
- Exact duplicates are likely data entry errors
- Same invoice shouldn't appear twice with identical details

**Implementation**:
```python
df = df.drop_duplicates()
```

**Validation**: 
- Check if duplicate removal is reasonable
- Verify no legitimate transactions are removed

**Impact**: Remove 6,865 rows (1.3%)

---

## 7. Derived Columns (To Add During Cleaning)

### 7.1 TotalPrice
**Formula**: `Quantity × UnitPrice`  
**Purpose**: Calculate transaction value  
**Importance**: Critical for Monetary (M) in RFM analysis

### 7.2 Date Components
**Columns to create**:
- `Year`: Extract year from InvoiceDate
- `Month`: Extract month (1-12)
- `DayOfWeek`: Extract day of week (0=Monday, 6=Sunday)
- `Hour`: Extract hour (0-23)

**Purpose**: 
- Identify temporal patterns
- Create behavioral features (preferred shopping time)

**Implementation**:
```python
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Hour'] = df['InvoiceDate'].dt.hour
```

---

## 8. Data Cleaning Pipeline Order

**Critical**: Steps must be executed in this order to avoid errors

1. **Load Data** with proper encoding (latin1 if UTF-8 fails)
2. **Convert InvoiceDate** to datetime
3. **Remove Missing CustomerIDs** (if any found during exploration)
4. **Remove Cancelled Invoices** (InvoiceNo starting with 'C')
5. **Remove Negative Quantities** (Quantity ≤ 0)
6. **Remove Zero/Negative Prices** (UnitPrice ≤ 0)
7. **Handle Missing Descriptions** (remove or fill)
8. **Remove Outliers** (IQR method for Quantity and UnitPrice)
9. **Remove Duplicates**
10. **Add Derived Columns** (TotalPrice, date components)
11. **Convert Data Types** (CustomerID to int, categorical columns)

---

## 9. Expected Outcomes

### 9.1 Data Retention Target
**Target**: 60-70% of original data

**Calculation**:
```
Original rows: 525,461
Expected removals:
- Negative quantities: 12,326 (2.3%)
- Duplicates: 6,865 (1.3%)
- Outliers (estimated): 80,000-100,000 (15-19%)
- Other issues: 20,000-30,000 (4-6%)

Total removed: ~120,000-150,000 (23-29%)
Retained: ~375,000-405,000 (71-77%)
```

**Validation**: If retention < 50%, review outlier thresholds

### 9.2 Data Quality Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Missing Values | TBD | 0 | 100% |
| Duplicates | 6,865 | 0 | 100% |
| Invalid Prices | TBD | 0 | 100% |
| Negative Quantities | 12,326 | 0 | 100% |

### 9.3 Final Dataset Characteristics
**Expected**:
- Rows: 375,000-405,000
- Columns: 13 (8 original + 5 derived)
- Memory: ~150-200 MB
- Date range: 2009-12-01 to 2010-12-09
- Unique customers: ~4,000-4,500

---

## 10. Validation Checks

After cleaning, we will verify:

✅ **No missing values** in any column  
✅ **All quantities > 0**  
✅ **All prices > 0**  
✅ **CustomerID is integer type**  
✅ **InvoiceDate is datetime type**  
✅ **No duplicates**  
✅ **Retention rate between 50-80%**  
✅ **Date range is reasonable**  

---

## 11. Risk Mitigation

### Risk 1: Over-aggressive Outlier Removal
**Mitigation**: 
- Start with 1.5 × IQR
- If retention < 50%, adjust to 2.0 × IQR
- Review distribution plots before final decision

### Risk 2: Removing Legitimate Transactions
**Mitigation**:
- Document all removal criteria
- Save removed rows to separate file for review
- Validate business logic with domain experts (if available)

### Risk 3: Data Type Conversion Errors
**Mitigation**:
- Use try-except blocks
- Validate conversions with assertions
- Print summary statistics before/after

---

## 12. Next Steps

After completing data cleaning:

1. ✅ Run validation notebook (`notebooks/02_data_validation.ipynb`)
2. ✅ Generate cleaning statistics report
3. ✅ Save cleaned data to `data/processed/cleaned_transactions.csv`
4. ✅ Document cleaning process in `docs/07_data_cleaning_report.md`
5. ✅ Proceed to Phase 4: Feature Engineering

---

**Document Version**: 1.0  
**Last Updated**: February 10, 2026  
**Status**: Ready for Implementation
