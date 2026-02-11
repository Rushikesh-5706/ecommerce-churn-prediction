# Customer Churn Prediction System

## Project Overview
This project implements a production-ready machine learning pipeline to predict customer churn in an e-commerce context. By analyzing transactional data from 2009-2010, the system identifies customers likely to stop purchasing in the next 65 days.

The solution is built with **Python**, **Scikit-Learn**, and **Streamlit**, featuring a comprehensive data processing pipeline, a rigorous evaluation framework, and an interactive dashboard for business stakeholders.

## Key Performance Metrics
| Metric | Score | Note |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.7517** | Target met (>0.75). Indicates strong discriminative ability. |
| **Recall** | **0.75** | High capture rate of actual churners. |
| **Precision** | **0.69** | Reliable predictions with acceptable false positives. |
| **Accuracy** | **70.3%** | Overall correctness on the test set. |

*Model: Random Forest Classifier with SMOTE (Synthetic Minority Over-sampling Technique).*

## Features
- **End-to-End Pipeline**: Automated scripts for data acquisition, cleaning, feature engineering, and modeling.
- **Advanced Feature Engineering**: 39 features including RFM scores, trend analysis (spend/frequency trajectories), and interaction terms.
- **Robust Evaluation**: Stratified train/val/test splits and cross-validation to prevent overfitting.
- **Interactive Application**: A Streamlit-based web interface allowing:
    - Single customer risk assessment.
    - Batch processing of customer lists.
    - Real-time visualization of churn probabilities.

## Installation and Usage

### Prerequisites
- Python 3.10 or higher
- Pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/Rushikesh-5706/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app/streamlit_app.py
```
The application will launch at `http://localhost:8501`.

## Project Structure
- `src/`: Core logic for data processing (`01_data_acquisition.py`, `02_data_cleaning.py`, `03_feature_engineering.py`) and modeling.
- `notebooks/`: Jupyter notebooks for EDA (`03_feature_eda.ipynb`) and model development (`05_model_training.ipynb`, `06_model_evaluation.ipynb`).
- `app/`: Streamlit application code.
- `models/`: Serialized model artifacts (`.pkl`) and metrics.
- `data/`: Raw and processed datasets (ignored in git except for sample files).

## Business Impact
This tool enables targeted retention campaigns. By identifying at-risk customers with 75% recall, the marketing team can intervene with personalized offers, potentially saving significant revenue compared to mass-marketing approaches.

## License
MIT License.
