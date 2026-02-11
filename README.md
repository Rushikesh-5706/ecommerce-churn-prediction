# E-Commerce Customer Churn Prediction

## Project Overview
This project implements a production-ready machine learning pipeline to predict customer churn in an e-commerce context. By analyzing transactional data from 2009-2010, the system identifies customers likely to stop purchasing in the next 65 days. The solution is built with Python, Scikit-Learn, and Streamlit, featuring a comprehensive data processing pipeline and an interactive dashboard.

## Business Problem
**Context**: E-commerce businesses face high customer acquisition costs. Retaining existing customers is 5-25x cheaper.
**Problem**: The platform is losing ~50% of its customers annually but lacks visibility into who is at risk.
**Goal**: Predict churn (no purchase in 65 days) with >75% ROC-AUC to enable proactive retention campaigns.
**Impact**: Targeted retention could save estimated £167,000 annually.

## Dataset
- **Source**: UCI Machine Learning Repository (Online Retail II)
- **Size**: 525,461 initial rows, 342,273 cleaned rows
- **Period**: 2009-12-01 to 2010-12-09
- **Customers**: 3,213 unique customers

## Methodology

### 1. Data Cleaning
- **Removed**: Missing CustomerIDs (25%), Cancelled Invoices, Negative Quantities.
- **Outliers**: Handled using IQR method (1.5x threshold).
- **Retention**: 65% of original data retained for analysis.

### 2. Feature Engineering
- **RFM**: Recency, Frequency, Monetary scores.
- **Behavioral**: Average days between purchases, basket size stats.
- **Temporal**: Purchase velocity, recent activity (30/60/90 days), trends.
- **Total Features**: 39 engineered features.

### 3. Models Evaluated
| Model | ROC-AUC | Precision | Recall |
| :--- | :--- | :--- | :--- |
| **Random Forest (SMOTE)** | **0.7517** | **0.69** | **0.75** |
| Neural Network | 0.7250 | 0.57 | 0.69 |
| Gradient Boosting | 0.7189 | 0.58 | 0.62 |
| Logistic Regression | 0.7182 | 0.58 | 0.67 |

### 4. Final Model
- **Selected**: Random Forest Classifier (with SMOTE balancing)
- **Performance**: ROC-AUC 0.7517, Recall 75%, Precision 69%
- **Justification**: Best balance of discriminatory power and recall (business priority).

## Installation & Usage

### Local Setup

#### Clone repository
```bash
git clone https://github.com/Rushikesh-5706/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction
```

#### Install dependencies
```bash
pip install -r requirements.txt
```

#### Run data pipeline
```bash
python src/01_data_acquisition.py
python src/02_data_cleaning.py
python src/03_feature_engineering.py
```

#### Run model training
```bash
python src/05_train_models.py
```

#### Launch web app
```bash
streamlit run app/streamlit_app.py
```

### Live Application
**URL**: https://ecommerce-churn-prediction-rushi5706.streamlit.app/

## Project Structure
```
project-root/
├── data/               # Raw and processed datasets
├── src/                # Source code for pipeline
├── notebooks/          # Jupyter notebooks for analysis
├── models/             # Serialized models (.pkl)
├── app/                # Streamlit application
├── docs/               # Project documentation
├── visualizations/     # Generated plots
├── tests/              # Unit tests
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Results & Business Impact
- **Performance**: Achieved **0.7517 ROC-AUC**, meeting the rigorous success criteria.
- **Recall**: **75%** recall means the model captures 3 out of 4 churning customers.
- **ROI**: Estimated **121% ROI** on retention campaigns by targeting the top 30% risk segment.
- **Recommendation**: Deploy "Win-Back" campaigns for High-Risk customers (churn prob > 70%).
