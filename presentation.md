# Customer Churn Prediction System Project Presentation

---

## Slide 1: Title Slide

**Project**: Customer Churn Prediction System  
**Presenter**: Rushikesh  
**Date**: February 10, 2026  
**Role**: Data Scientist  

*Predicting customer retention in e-commerce using machine learning.*

---

## Slide 2: Business Problem

**Context**: High customer acquisition costs (£50 vs £10 retention) make churn prevention critical.  
**Problem**: E-commerce platform losing ~42% of customers annually.  
**Goal**: Identify at-risk customers 3 months in advance.  
**Impact**:  
- **Revenue at Risk**: £1.55M annually  
- **Success Metric**: ROC-AUC > 0.75 (Ability to rank risk)  

---

## Slide 3: Dataset Overview

**Source**: UCI Online Retail II Dataset  
**Scale**:  
- **Raw**: 525,461 transactions  
- **Processed**: 342,273 valid transactions  
- **Customers**: 3,213 unique entities  
- **Period**: Dec 2009 - Dec 2010 (1 year)  

**Key Challenges**:  
- High Churn Rate (41.92%)  
- Missing CustomerIDs (20% of data)  
- No explicit "churn" label (Must be inferred)  

---

## Slide 4: Data Cleaning Pipeline

**Objective**: Ensure high-quality input for modeling.  

**Steps Taken**:  
1. **Removed Missing IDs**: 107k rows dropped (Mandatory for customer-level analysis).  
2. **Handled Cancellations**: Excluded 9k returns to simplify purchase behavior.  
3. **Outlier Removal**: Removed extreme bulk buyers (top 1%) to prevent skewing.  
4. **Validation**: Confirmed 0 nulls, 0 negative prices.  

**Result**: 65% Data Retention (Met 60-70% target).  

---

## Slide 5: Feature Engineering

**Strategy**: "rfm" + "behavioral" + "temporal"  

| Category | Key Features | Business Insight |
| :--- | :--- | :--- |
| **RFM** | Recency, Frequency, Monetary | Core predictors of value. |
| **Temporal** | PurchaseVelocity, DaysBetween | Captures change in habit. |
| **Product** | DiversityScore, AvgPrice | Differentifies bulk vs. casual. |

**Total Features**: 29 Customer-level attributes.  
**Target Definition**: No purchase in next 65 days.  

---

## Slide 6: Models Evaluated

**Approach**: Tested 5 algorithms with SMOTE (handling imbalance).  

| Model | ROC-AUC | Precision | Recall | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.718 | 0.58 | 0.67 | Good baseline. |
| **Decision Tree** | 0.682 | 0.55 | 0.66 | Prone to overfitting. |
| **Gradient Boosting** | 0.719 | 0.57 | 0.49 | High precision, low recall. |
| **Neural Network** | 0.725 | 0.60 | 0.58 | Strong but complex. |
| **Random Forest** | **0.752** | 0.69 | **0.75** | **Selected Champion.** |

---

## Slide 7: Final Model Performance

**Champion Model**: Random Forest (Ensemble)  

**Key Metrics**:  
- **ROC-AUC**: **0.7517** (Discriminative power)  
- **Recall**: **74.79%** (Catches 3 out of 4 churners)  
- **Accuracy**: 70.33%  

**Why this model?**  
- Best balance of Recall vs Precision.  
- High interpretability (Feature Importance).  
- Robust to outliers.  

---

## Slide 8: Business Impact Prediction

**Scenario Analysis**:  
Targeting top 30% riskiest customers.  

- **Campaign Cost**: £10/customer  
- **Customer Value (LTV)**: £1,150  
- **Success Rate**: Assumed 15% retention from campaign.  

**ROI Calculation**:  
- **Cost**: ~£15,000 (Discovery + Offer)  
- **Revenue Saved**: ~£182,000  
- **Net Benefit**: **£167,000 / year**  
- **ROI**: **121.6%**  

---

## Slide 9: Deployment

**Architecture**: Streamlit Web App + Docker  

**Features**:  
1. **Single Prediction**: For customer support agents.  
2. **Batch Prediction**: For marketing team (CSV upload).  
3. **Dashboard**: Real-time performance tracking.  

**Status**:  
- ✅ Local: Tested & working.  
- ✅ Cloud: Ready for Streamlit Share.  
- ✅ Docker: Container build successful.  

---

## Slide 10: Key Learnings & Challenges

**Challenges**:  
1. **Churn Definition**: 42% natural churn rate is very high. Hard to separate signal from noise.  
   - *Fix*: Adjusted observation window to 65 days.  
2. **Imbalance**:  
   - *Fix*: Implementation of SMOTE improved ROC-AUC by ~0.02.  

**Learnings**:  
- **Recency** is the single strongest predictor.  
- **Recall > Precision** for this business case (Missed churn is expensive).  

---

## Slide 11: Future Improvements

1.  **Data Quality**: Capture customer demographics (Age, Location) to improve model.  
2.  **Advanced Modeling**: Test LSTM/GRU on sequence of baskets.  
3.  **Real-Time**: Connect API to checkout system for "at-risk" alerts during session.  

---

## Slide 12: Thank You

**Repo**: `github.com/rushikeshkunisetty/Customer-Churn-Prediction-System`  
**Live App**: `customer-churn-prediction.streamlit.app`  

**Questions?**  
