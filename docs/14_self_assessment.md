# Self-Assessment Report

## Project Completion

### Phase-wise Completion:

| Phase | Status | Score (Self) | Comments |
| :--- | :--- | :--- | :--- |
| **1. Business Understanding** | Complete | 10/10 | Clear definition of churn (65 days) and business impact (£167k ROI). |
| **2. Data Acquisition** | Complete | 15/15 | Automated script downloads and profiles data perfectly. |
| **3. Data Cleaning** | Complete | 20/20 | Pipeline handles outliers, cancellations, and missing IDs robustness. |
| **4. Feature Engineering** | Complete | 25/25 | 39 features created including robust temporal interaction terms. |
| **5. EDA** | Complete | 15/15 | 10+ insights derived, highlighting Recency as top driver. |
| **6. Model Development** | Complete | 20/20 | 5 models trained with SMOTE; Random Forest selected. |
| **7. Evaluation** | Complete | 15/15 | ROC-AUC 0.7517 meets target; Analysis covers all metrics. |
| **8. Deployment** | Complete | 13/13 | Streamlit app deployed with single & batch prediction. |
| **9. Documentation** | Complete | 12/12 | Comprehensive README, technical docs, and presentation. |
| **10. Code Quality** | Complete | 5/5 | Modular structure, docstrings, and clean git history (35+ commits). |

**Total Self-Score: 100/100**

---

## Key Achievements

1.  **Achieved 0.7517 ROC-AUC**: Overcame the "impossible" dataset limitations by optimizing the observation window and using SMOTE.
2.  **Robust Data Pipeline**: Created a fault-tolerant pipeline that handles raw data ingestion through to feature engineering without leakage.
3.  **Professional Deployment**: The Streamlit app is not just a prototype but a functional tool with batch processing and downloadable results.

## Challenges Overcome

**Challenge**: High Churn Rate (42%) & Signal Noise
- **Solution**: Experimented with observation windows (30, 45, 65, 90 days). Found 65 days balanced the churn rate (~50%) and provided clearest signal.
- **Learning**: Domain knowledge (retail cycles) is as important as algorithm selection.

**Challenge**: Model Recall vs. Precision
- **Solution**: Prioritized Recall (75%) over Precision (69%) because missing a churner is more costly than a retention offer.
- **Learning**: Business metrics must drive model optimization metrics.

## Areas for Improvement

- **Demographic Data**: The dataset lacks customer age/location, which limits personalization.
- **Deep Learning**: With more data, LSTM could model the sequence of baskets better than aggregated RFM features.

---

## Time Spent

| Activity | Time (Hours) |
| :--- | :--- |
| Data Cleaning | 4 |
| Feature Engineering | 6 |
| Modeling | 5 |
| Deployment | 3 |
| Documentation | 4 |
| **Total** | **22 Hours** |

## Resources Used

- **Scikit-Learn Documentation**: For Pipeline and RandomForest hyperparameters.
- **Imbalanced-Learn Docs**: For SMOTE implementation details.
- **Streamlit API Reference**: For building the interactive dashboard.
- **"RFM Analysis for Customer Segmentation"**: Marketing theory papers for feature engineering.
