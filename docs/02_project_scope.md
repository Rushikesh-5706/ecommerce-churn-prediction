# Project Scope Document

## Executive Summary

This document defines the boundaries of the Customer Churn Prediction System project, clearly outlining what is included and excluded from the current scope. The project focuses on building a production-ready ML system for predicting customer churn using historical transactional data.

## In Scope

### 1. Data Processing & Analysis
✅ **Data Acquisition**
- Download and load UCI Online Retail dataset
- Validate data quality and completeness
- Document data characteristics

✅ **Data Cleaning**
- Handle missing values (especially CustomerID)
- Remove cancelled transactions and returns
- Outlier detection and treatment
- Data type conversions and standardization

✅ **Feature Engineering**
- RFM (Recency, Frequency, Monetary) analysis
- Behavioral pattern features
- Temporal features (purchase velocity, recent activity)
- Product affinity features
- Customer segmentation

### 2. Machine Learning Development
✅ **Model Development**
- Implement 5 classification algorithms:
  - Logistic Regression (baseline)
  - Decision Tree
  - Random Forest
  - Gradient Boosting (XGBoost/LightGBM)
  - Neural Network (MLP)
- Model comparison and selection
- Hyperparameter tuning (optional)

✅ **Model Evaluation**
- Performance metrics: ROC-AUC, Precision, Recall, F1-Score
- Cross-validation (5-fold stratified)
- Confusion matrix analysis
- Feature importance analysis
- Error analysis and model diagnostics

✅ **Target Variable**
- Binary classification: Churned (1) vs Active (0)
- Churn definition: No purchase in 90-day observation window
- Temporal split methodology

### 3. Deployment & Application
✅ **Web Application**
- Interactive Streamlit dashboard
- Single customer prediction interface
- Batch prediction capability (CSV upload)
- Model performance visualization
- User-friendly documentation

✅ **Deployment**
- Cloud deployment on free platform (Streamlit Cloud)
- Public URL access
- Docker containerization
- API for predictions

### 4. Documentation & Deliverables
✅ **Technical Documentation**
- Complete README with setup instructions
- Data dictionary and feature documentation
- Model architecture documentation
- API reference
- Troubleshooting guide

✅ **Business Documentation**
- Business problem statement
- ROI analysis and business impact
- Implementation recommendations
- Presentation slides (10-12 slides)

✅ **Code Quality**
- Clean, well-organized code structure
- Docstrings for all functions
- Version control (Git) with ≥20 commits
- Proper .gitignore configuration

## Out of Scope

### 1. Advanced Features (Future Enhancements)
❌ **Real-Time Predictions**
- Current scope: Batch predictions only
- Real-time streaming data processing excluded
- Future: Could integrate with real-time transaction systems

❌ **Product Recommendation System**
- Focus is solely on churn prediction
- Product recommendations are a separate ML problem
- Future: Could combine churn prevention with personalized recommendations

❌ **Inventory Optimization**
- Not related to customer churn
- Separate business problem requiring different data
- Future: Could be part of broader analytics platform

❌ **Customer Acquisition Modeling**
- Scope limited to retention, not acquisition
- Different target variable and features required
- Future: Complementary model to churn prediction

### 2. Data Sources
❌ **External Data Integration**
- No social media data
- No web analytics (clickstream, session data)
- No demographic data beyond what's in transactions
- No email engagement metrics
- Reason: Limited to provided dataset only

❌ **Real-Time Data Feeds**
- No live transaction streaming
- Historical data only
- Future: Could integrate with production databases

### 3. Advanced ML Techniques
❌ **Deep Learning Models**
- No CNN, RNN, LSTM, or Transformer models
- Scope limited to traditional ML and basic neural networks
- Reason: Tabular data doesn't require deep learning complexity

❌ **Ensemble Stacking**
- No meta-models combining multiple base models
- Single best model selection only
- Future: Could improve performance by 1-2%

❌ **AutoML Frameworks**
- No automated model selection (H2O, Auto-sklearn)
- Manual model development for learning purposes
- Future: Could speed up experimentation

### 4. Deployment Features
❌ **Mobile Application**
- Web-only interface
- No iOS/Android apps
- Future: Could develop mobile-responsive PWA

❌ **Authentication & User Management**
- No login system
- Public access only
- Future: Required for production enterprise deployment

❌ **Database Integration**
- No persistent database for predictions
- File-based storage only
- Future: PostgreSQL/MongoDB for production

❌ **A/B Testing Framework**
- No built-in experimentation platform
- Manual campaign tracking only
- Future: Integrate with marketing automation tools

### 5. Business Operations
❌ **Automated Retention Campaigns**
- Model provides predictions only
- No automated email/SMS triggers
- Marketing team implements campaigns manually
- Future: Integration with CRM systems

❌ **Multi-Language Support**
- English only
- No internationalization (i18n)
- Future: Support for global markets

❌ **Advanced Analytics Dashboard**
- Basic visualizations only
- No drill-down capabilities or advanced BI features
- Future: Power BI/Tableau integration

## Timeline

**Total Duration**: 8 weeks (part-time) or 2-3 weeks (full-time)

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1-2: Setup & Data Acquisition | Week 1 | Business docs, raw data, initial EDA |
| Phase 3: Data Cleaning | Week 2 | Cleaned dataset, validation reports |
| Phase 4: Feature Engineering | Week 2-3 | Customer-level features, churn labels |
| Phase 5: EDA | Week 3 | Visualizations, statistical insights |
| Phase 6-7: Modeling & Evaluation | Week 4-5 | Trained models, performance reports |
| Phase 8: Deployment | Week 6 | Live web application |
| Phase 9-10: Documentation & Polish | Week 7-8 | Final docs, presentation, code cleanup |

## Constraints

### Technical Constraints
1. **Tools**: Free and open-source only
   - No paid APIs or services
   - No proprietary software licenses

2. **Compute Resources**: Local machine or free cloud tier
   - No GPU requirements
   - Models must train in reasonable time (<1 hour)

3. **Data Size**: Work with provided dataset only
   - ~500k transactions
   - ~4,000 customers
   - No additional data collection

### Business Constraints
1. **Budget**: £0 for tools and infrastructure
2. **Team**: Individual project (no team collaboration)
3. **Access**: Public deployment (no sensitive data)

### Regulatory Constraints
1. **Data Privacy**: Use anonymized customer IDs only
2. **Compliance**: No PII (Personally Identifiable Information)
3. **Ethics**: Fair and unbiased model predictions

## Assumptions

1. **Data Quality**: Dataset is representative of actual customer behavior
2. **Churn Definition**: 90-day window is appropriate for this business
3. **Feature Availability**: Transactional data alone is sufficient for prediction
4. **Stakeholder Buy-In**: Business teams will use model predictions
5. **Deployment Platform**: Streamlit Cloud remains free and available

## Dependencies

### External Dependencies
- UCI Machine Learning Repository (data source)
- Streamlit Cloud (deployment platform)
- GitHub (version control and deployment trigger)
- Python ecosystem (scikit-learn, pandas, etc.)

### Internal Dependencies
- Successful data cleaning (Phase 3) required for feature engineering (Phase 4)
- Feature engineering required for modeling (Phase 6)
- Model training required for deployment (Phase 8)

## Success Criteria

### Must-Have (Required for Project Completion)
- ✅ ROC-AUC ≥ 0.75 on test set
- ✅ All 10 phases completed
- ✅ Deployed application with public URL
- ✅ Complete documentation
- ✅ Docker configuration

### Should-Have (Highly Desirable)
- ✅ ROC-AUC ≥ 0.78 (target performance)
- ✅ Feature importance visualization
- ✅ Business impact analysis with ROI calculations
- ✅ Clean Git history with ≥20 commits

### Nice-to-Have (Bonus)
- ⭐ Hyperparameter tuning
- ⭐ Model explainability (SHAP values)
- ⭐ Advanced visualizations (interactive plots)
- ⭐ Comprehensive unit tests

## Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Data quality issues | High | Medium | Robust cleaning pipeline, validation checks |
| Model performance below target | High | Low | Multiple algorithms, feature engineering |
| Deployment platform unavailable | Medium | Low | Document local deployment alternative |
| Timeline overrun | Medium | Medium | Prioritize must-haves, defer nice-to-haves |
| Scope creep | Low | Medium | Strict adherence to this scope document |

## Change Management

Any changes to this scope must be:
1. Documented with justification
2. Assessed for impact on timeline and deliverables
3. Approved before implementation

**Scope Freeze Date**: After Phase 2 completion (Week 1)

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Approved By**: Project Lead
