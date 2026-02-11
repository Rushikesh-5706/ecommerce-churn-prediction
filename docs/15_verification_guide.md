# Application Verification Guide

This guide provides test cases to verify the deployed Streamlit application.

## 1. Single Customer Prediction
Use the "Predict Single Customer" tab to test these scenarios.

### 🔴 Test Case A: High Risk Customer (Likely Churn)
Simulate a customer who bought once a long time ago.

*   **Recency**: 320 (days since last purchase)
*   **Frequency**: 1 (purchase)
*   **Total Spend**: 50.00
*   **Customer Lifetime**: 0
*   **Num Unique Products**: 1
*   **Avg Days Between Purchases**: 0
*   **Spending Trend**: 0

**Expected Result**:
*   **Prediction**: **Churn Risk** (Red)
*   **Probability**: > 80%

---

### 🟢 Test Case B: Loyal Customer (Likely Active)
Simulate a frequent shopper who bought recently.

*   **Recency**: 5 (days since last purchase)
*   **Frequency**: 25 (purchases)
*   **Total Spend**: 5000.00
*   **Customer Lifetime**: 300
*   **Num Unique Products**: 50
*   **Avg Days Between Purchases**: 12
*   **Spending Trend**: 1.2 (Spending more recently)

**Expected Result**:
*   **Prediction**: **Active / Low Risk** (Green)
*   **Probability**: < 20%

---

## 2. Batch Prediction
Use the "Batch Prediction" tab.

1.  **Download Sample**: I have created a sample file for you at `data/sample_batch_test.csv` in the repository.
2.  **Upload Info**:
    *   This file contains 10 mixed customers.
    *   Upload it to the "Upload CSV" widget.
3.  **Expected Result**:
    *   The app should display a table with 10 rows.
    *   It should show a mix of "Churn" and "Active" predictions.
    *   You should see a "Download Predictions" button.
