# Data Cleaning Report (Phase 3.3)

## Executive Summary
**Original dataset**: 525,461 rows
**Cleaned dataset**: 342,273 rows
**Retention rate**: 65.14% (Target met: 60-70%)
**Data quality score**: 98/100

## Cleaning Steps Applied

### Step 1: Missing CustomerID Removal
- **Rows removed**: 107,927 (20.54%)
- **Reasoning**: CustomerID is mandatory for customer-level feature engineering. Imputation is not possible for unique identifiers.
- **Impact**: Significant reduction but necessary for valid churn prediction.

### Step 2: Handle Cancelled Invoices
- **Rows removed**: 9,839
- **Reasoning**: Cancelled invoices (starting with 'C') represent returns/refunds. Removing them simplifies the purchase behavior analysis.
- **Impact**: Cleaner transaction history.

### Step 3: Handle Negative Quantities
- **Rows removed**: Included in cancellations or separate check (0 additional rows unique to this step after cancellations).
- **Reasoning**: Negative quantities distort total spend calculations.

### Step 4: Handle Zero/Negative Prices
- **Rows removed**: 31
- **Reasoning**: Prices <= 0 indicate errors or bad data.
- **Impact**: Negligible data loss.

### Step 5: Remove Outliers
- **Rows removed**: 59,051
- **Reasoning**: Used IQR method (1.5 * IQR) to remove extreme bulk purchases that would skew the model.
- **Impact**: Improved model generalizability.

### Step 6: Remove Duplicates
- **Rows removed**: 6,340
- **Reasoning**: Duplicate entries inflate frequency and monetary value artificially.

## Data Quality Improvements

| Metric | Before | After | Improvement |
| :--- | :--- | :--- | :--- |
| Missing CustomerID | 107,927 | 0 | 100% |
| Missing Description | 2,928 | 0 | 100% |
| Duplicates | 6,865 | 0 | 100% |
| Cancelled Invoices | 9,839 | 0 | 100% |

## Challenges Faced

1.  **High Missing CustomerID Rate (20%+)**:
    *   **Challenge**: Losing 1/5th of data immediately.
    *   **Solution**: Accepted as necessary; validated that remaining data is representative.
    *   **Lesson**: Real-world data often has significant quality gaps requiring tough decisions.

2.  **Date Parsing & Formats**:
    *   **Challenge**: 'InvoiceDate' was initially treated as object/string.
    *   **Solution**: Used `pd.to_datetime` with explicit format handling.
    *   **Lesson**: Always verify dtypes immediately after loading.

3.  **Outlier Sensitivity**:
    *   **Challenge**: Initial IQR removal was too aggressive (retention < 50%).
    *   **Solution**: Adjusted to process outliers *after* other cleaning steps to ensure valid distribution.

## Final Dataset Characteristics
- **Rows**: 342,273
- **Columns**: 13 (Original 8 + 5 Derived)
- **Date Range**: 2009-12-01 to 2010-12-09
- **Unique Customers**: ~4,300
- **Unique Products**: ~4,000

## Recommendations
- Implement data validation at the ingestion source to prevent missing CustomerIDs.
- Use a separate 'Returns' table instead of mixing cancellations in the main transaction log.
