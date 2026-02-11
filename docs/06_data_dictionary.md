# Data Dictionary

## Dataset Information

**Dataset Name**: UCI Online Retail II  
**Source**: UCI Machine Learning Repository  
**URL**: http://archive.ics.uci.edu/ml/datasets/Online+Retail+II  
**File**: online_retail.csv  
**Size**: 525,461 rows × 8 columns  
**Period**: December 1, 2009 - December 9, 2010  
**Domain**: E-commerce / Retail  

---

## Column Specifications

| Column Name | Data Type | Description | Example Values | Missing % | Notes |
|-------------|-----------|-------------|----------------|-----------|-------|
| **Invoice** | String | 6-digit invoice number. Prefix 'C' indicates cancellation | `536365`, `C536379` | 0% | Unique identifier for each transaction |
| **StockCode** | String | 5-6 character product code | `85123A`, `22423` | 0% | Some non-standard codes exist (e.g., `POST`, `D`, `M`) |
| **Description** | String | Product name/description | `WHITE HANGING HEART T-LIGHT HOLDER` | 0.56% | 2,928 missing values |
| **Quantity** | Integer | Quantity of items per transaction | `6`, `-1`, `120` | 0% | Negative values (12,326 rows) indicate returns |
| **InvoiceDate** | DateTime | Transaction date and time | `2010-12-01 08:26:00` | 0% | Format: YYYY-MM-DD HH:MM:SS |
| **Price** | Float | Unit price in GBP (£) | `2.55`, `3.39`, `0.00` | 0% | Some zero prices exist (samples/damages) |
| **Customer ID** | Float | 5-digit customer identifier | `17850.0`, `13047.0` | **20.54%** | **107,927 missing values** - Critical issue |
| **Country** | String | Customer's country of residence | `United Kingdom`, `France`, `Germany` | 0% | 38 unique countries |

---

## Data Quality Issues Identified

### Critical Issues

#### 1. Missing Customer IDs (20.54%)
- **Count**: 107,927 rows out of 525,461
- **Percentage**: 20.54%
- **Impact**: Cannot create customer-level features without CustomerID
- **Action Required**: **REMOVE** these rows during cleaning
- **Justification**: Customer ID is essential for churn prediction

#### 2. Negative Quantities (2.35%)
- **Count**: 12,326 rows
- **Interpretation**: Product returns or order cancellations
- **Impact**: Complicates churn definition and RFM calculations
- **Action Required**: **REMOVE** during cleaning
- **Justification**: Returns don't represent purchasing behavior

#### 3. Missing Product Descriptions (0.56%)
- **Count**: 2,928 rows
- **Impact**: Minor - descriptions not critical for churn prediction
- **Action Required**: **REMOVE** or fill with "Unknown Product"
- **Decision**: Will finalize during cleaning phase

#### 4. Duplicate Rows (1.31%)
- **Count**: 6,865 rows
- **Impact**: Data entry errors or system duplicates
- **Action Required**: **REMOVE** exact duplicates
- **Justification**: Same transaction shouldn't appear twice

### Data Anomalies

#### 1. Zero Prices
- **Status**: To be quantified during exploration
- **Likely Cause**: Free samples, promotional items, or data errors
- **Action**: Remove during cleaning

#### 2. Cancelled Invoices
- **Detection**: Invoice numbers starting with 'C'
- **Current Count**: 0 detected in initial scan
- **Note**: May be captured as negative quantities instead

#### 3. Non-Standard Stock Codes
- **Examples**: `POST` (postage), `D` (discount), `M` (manual adjustment)
- **Impact**: Not actual products
- **Action**: May need special handling or removal

---

## Column Details

### Invoice
**Purpose**: Unique transaction identifier  
**Format**: 6-digit number, or 'C' + 6 digits for cancellations  
**Uniqueness**: Multiple rows can have same InvoiceNo (multi-item orders)  
**Business Logic**: One invoice can contain multiple products  

**Examples**:
- `536365` - Normal purchase
- `C536379` - Cancelled order

### StockCode
**Purpose**: Product identifier  
**Format**: Typically 5-6 alphanumeric characters  
**Uniqueness**: ~3,600-4,000 unique products expected  

**Special Codes**:
- `POST` - Postage charges
- `D` - Discount
- `M` - Manual adjustment
- `BANK CHARGES` - Banking fees
- `CRUK` - Charity donation

**Note**: These special codes may need to be filtered out

### Description
**Purpose**: Human-readable product name  
**Format**: Free text, typically uppercase  
**Missing**: 2,928 rows (0.56%)  

**Examples**:
- `WHITE HANGING HEART T-LIGHT HOLDER`
- `CREAM CUPID HEARTS COAT HANGER`
- `KNITTED UNION FLAG HOT WATER BOTTLE`

**Cleaning Note**: Some descriptions may have encoding issues (special characters)

### Quantity
**Purpose**: Number of units purchased  
**Data Type**: Integer  
**Range**: -80,995 to 80,995 (extreme outliers exist)  
**Negative Values**: 12,326 rows (returns)  

**Distribution**:
- Most transactions: 1-50 items
- Bulk orders: 100-1,000 items (B2B customers)
- Extreme outliers: >10,000 items (likely data errors)

**Cleaning Strategy**: Remove negatives, apply IQR outlier detection

### InvoiceDate
**Purpose**: Transaction timestamp  
**Format**: YYYY-MM-DD HH:MM:SS  
**Range**: 2009-12-01 to 2010-12-09  
**Timezone**: Assumed UTC or UK time  

**Temporal Coverage**: ~1 year of data

**Usage**:
- Calculate Recency (days since last purchase)
- Identify purchase patterns (day of week, hour)
- Create temporal features

### Price
**Purpose**: Unit price per item in British Pounds (£)  
**Data Type**: Float  
**Range**: £0.00 to £38,970.00 (extreme outliers)  

**Typical Range**: £0.50 - £50.00

**Issues**:
- Zero prices: Free items or data errors
- Negative prices: Likely data errors (should not exist)
- Extreme prices: May be bulk pricing or errors

**Cleaning Strategy**: Remove ≤ 0, apply IQR outlier detection

### Customer ID
**Purpose**: Unique customer identifier  
**Data Type**: Float (should be Integer)  
**Format**: 5-digit number  
**Missing**: 107,927 rows (20.54%) - **CRITICAL**  

**Expected Unique Customers**: ~4,000-4,500

**Why Float?**: pandas uses float for numeric columns with missing values (NaN)

**Cleaning**: 
1. Remove rows with missing Customer ID
2. Convert to integer type

### Country
**Purpose**: Customer's country of residence  
**Data Type**: String  
**Unique Values**: 38 countries  
**Missing**: 0%  

**Top Countries** (expected):
- United Kingdom (majority)
- Germany
- France
- EIRE (Ireland)
- Spain

**Usage**: 
- Country diversity feature
- Geographic segmentation (optional)

---

## Derived Columns (To Be Created During Cleaning)

These columns will be added during the data cleaning phase:

| Column Name | Data Type | Description | Formula |
|-------------|-----------|-------------|---------|
| **TotalPrice** | Float | Total transaction value | `Quantity × Price` |
| **Year** | Integer | Year of transaction | Extract from InvoiceDate |
| **Month** | Integer | Month (1-12) | Extract from InvoiceDate |
| **DayOfWeek** | Integer | Day of week (0=Monday, 6=Sunday) | Extract from InvoiceDate |
| **Hour** | Integer | Hour of day (0-23) | Extract from InvoiceDate |

---

## Data Quality Summary

### Before Cleaning

| Metric | Value | Status |
|--------|-------|--------|
| Total Rows | 525,461 | ✓ |
| Total Columns | 8 | ✓ |
| Complete Rows | ~397,606 (75.7%) | ⚠️ |
| Missing Customer IDs | 107,927 (20.54%) | ❌ Critical |
| Missing Descriptions | 2,928 (0.56%) | ⚠️ Minor |
| Negative Quantities | 12,326 (2.35%) | ❌ |
| Duplicate Rows | 6,865 (1.31%) | ⚠️ |
| Date Range | 2009-12-01 to 2010-12-09 | ✓ |

### Expected After Cleaning

| Metric | Expected Value |
|--------|----------------|
| Total Rows | ~375,000-405,000 (60-70% retention) |
| Total Columns | 13 (8 original + 5 derived) |
| Missing Values | 0 (all columns) |
| Data Quality Score | 100% |

---

## Data Cleaning Required

Based on the data quality assessment, the following cleaning steps are required:

1. ✅ **Remove Missing Customer IDs** (~107,927 rows, 20.54%)
2. ✅ **Remove Negative Quantities** (~12,326 rows, 2.35%)
3. ✅ **Remove Zero/Negative Prices** (count TBD)
4. ✅ **Handle Missing Descriptions** (~2,928 rows, 0.56%)
5. ✅ **Remove Duplicates** (~6,865 rows, 1.31%)
6. ✅ **Remove Outliers** (IQR method, count TBD)
7. ✅ **Add Derived Columns** (TotalPrice, date components)
8. ✅ **Convert Data Types** (Customer ID to int, dates to datetime)

**Expected Data Loss**: 23-40% of original rows  
**Expected Retention**: 60-77% (~315,000-405,000 rows)

---

## Business Context

### Customer Base
- **Type**: B2C and B2B (mix of retail and wholesale)
- **Geography**: Primarily UK, with international customers
- **Size**: ~4,000-4,500 unique customers

### Product Catalog
- **Type**: Gift items, home decor, accessories
- **Count**: ~3,600-4,000 unique products
- **Price Range**: Typically £0.50 - £50.00

### Transaction Patterns
- **Frequency**: Daily transactions
- **Seasonality**: Likely higher during holidays (December spike expected)
- **Order Size**: Mix of small retail (1-10 items) and bulk wholesale (100+ items)

---

## Usage Notes for Feature Engineering

### RFM Analysis
- **Recency**: Use InvoiceDate to calculate days since last purchase
- **Frequency**: Count unique Invoice numbers per Customer ID
- **Monetary**: Sum of TotalPrice per Customer ID

### Temporal Features
- **Purchase Velocity**: Transactions per day
- **Preferred Shopping Time**: Mode of DayOfWeek and Hour
- **Recent Activity**: Purchases in last 30/60/90 days

### Behavioral Features
- **Basket Size**: Average Quantity per Invoice
- **Product Diversity**: Unique StockCodes per Customer
- **Price Preference**: Average Price per Customer

---

## Data Limitations

1. **Limited Time Period**: Only 1 year of data (Dec 2009 - Dec 2010)
   - May not capture long-term customer behavior
   - Seasonal patterns may be incomplete

2. **Missing Customer IDs**: 20.54% of transactions have no customer link
   - Represents guest checkouts or data quality issues
   - Cannot use these for customer-level analysis

3. **No Demographic Data**: Only transactional data available
   - No age, gender, location (beyond country)
   - Limited to behavioral features only

4. **No Web Analytics**: No clickstream or session data
   - Cannot analyze browsing behavior
   - Limited to purchase behavior only

---

**Document Version**: 1.0  
**Last Updated**: February 10, 2026  
**Status**: Complete - Ready for Data Cleaning Phase
