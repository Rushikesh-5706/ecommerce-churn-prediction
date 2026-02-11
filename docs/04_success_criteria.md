# Success Criteria Matrix

## Performance Metrics Thresholds

This document defines three levels of success for the Customer Churn Prediction System: **Minimum** (acceptable), **Target** (expected), and **Stretch** (exceptional).

## Model Performance Metrics

| Metric | Minimum | Target | Stretch | Business Justification |
|--------|---------|--------|---------|------------------------|
| **ROC-AUC** | 0.75 | 0.80 | 0.85 | Primary metric for ranking customers by churn risk |
| **Precision** | 0.70 | 0.75 | 0.80 | Minimize wasted retention budget on false positives |
| **Recall** | 0.65 | 0.70 | 0.75 | Maximize revenue protection by catching churners |
| **F1-Score** | 0.67 | 0.72 | 0.77 | Balanced performance across precision and recall |

### Metric Definitions

**ROC-AUC (Area Under Receiver Operating Characteristic Curve)**
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Interpretation**: Probability that model ranks a random churner higher than a random active customer
- **Why Primary**: Threshold-independent, handles class imbalance well
- **Business Use**: Rank customers by churn risk for prioritized outreach

**Precision**
- **Formula**: TP / (TP + FP)
- **Interpretation**: Of customers predicted to churn, what % actually churn?
- **Business Impact**: High precision = efficient use of retention budget
- **Cost of Low Precision**: Wasted marketing spend, potential customer annoyance

**Recall (Sensitivity)**
- **Formula**: TP / (TP + FN)
- **Interpretation**: Of customers who actually churn, what % did we catch?
- **Business Impact**: High recall = maximum revenue protection
- **Cost of Low Recall**: Lost customers, missed revenue opportunities

**F1-Score**
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Interpretation**: Harmonic mean balancing precision and recall
- **Business Impact**: Ensures model doesn't sacrifice one metric for the other

## Business Impact Metrics

| Metric | Minimum | Target | Stretch | Measurement Period |
|--------|---------|--------|---------|-------------------|
| **Churn Rate Reduction** | 10% | 15% | 20% | 6 months post-deployment |
| **Revenue Protected** | £200,000 | £300,000 | £500,000 | Annual |
| **Campaign ROI** | 2:1 | 3:1 | 5:1 | Per campaign |
| **Customer Retention** | +5% | +10% | +15% | Year-over-year |

### Business Metric Calculations

**Churn Rate Reduction**
```
Baseline Churn Rate: 30%
Target Churn Rate: 25.5% (15% reduction)
Calculation: (30% - 25.5%) / 30% = 15%
```

**Revenue Protected**
```
Assumptions:
- Average Customer LTV: £500
- Customers saved from churn: 600/year (target)
- Revenue Protected: 600 × £500 = £300,000
```

**Campaign ROI**
```
Assumptions:
- Retention campaign cost: £10 per customer
- Customers targeted: 1,200 (30% of 4,000)
- Campaign cost: £12,000
- Success rate: 30% (360 customers retained)
- Revenue saved: 360 × £500 = £180,000
- ROI: £180,000 / £12,000 = 15:1 (exceeds stretch goal)

Conservative estimate with 20% success rate:
- Revenue saved: 240 × £500 = £120,000
- ROI: £120,000 / £12,000 = 10:1 (still exceeds stretch)

Realistic target with 10% success rate:
- Revenue saved: 120 × £500 = £60,000
- ROI: £60,000 / £12,000 = 5:1 (meets stretch goal)

Minimum viable with 5% success rate:
- Revenue saved: 60 × £500 = £30,000
- ROI: £30,000 / £12,000 = 2.5:1 (meets minimum)
```

## Technical Performance Criteria

| Criterion | Minimum | Target | Stretch |
|-----------|---------|--------|---------|
| **Model Training Time** | < 2 hours | < 1 hour | < 30 min |
| **Prediction Latency** | < 10 sec/1000 | < 5 sec/1000 | < 2 sec/1000 |
| **Model File Size** | < 200 MB | < 100 MB | < 50 MB |
| **Application Load Time** | < 5 seconds | < 3 seconds | < 2 seconds |

## Data Quality Criteria

| Criterion | Minimum | Target | Stretch |
|-----------|---------|--------|---------|
| **Data Retention Rate** | 50% | 65% | 75% |
| **Feature Count** | 20 | 28 | 35+ |
| **Churn Rate** | 20-40% | 25-35% | 28-32% |
| **Missing Values (Final)** | 0% | 0% | 0% |

### Data Quality Justification

**Data Retention Rate**
- **Minimum (50%)**: Acceptable but indicates aggressive cleaning
- **Target (65%)**: Balanced cleaning preserving valuable data
- **Stretch (75%)**: Excellent data quality with minimal loss

**Feature Count**
- **Minimum (20)**: Basic RFM + behavioral features
- **Target (28)**: Comprehensive feature set across all categories
- **Stretch (35+)**: Advanced feature engineering with interactions

**Churn Rate**
- **Why Range**: Too low (<20%) = insufficient positive examples; Too high (>40%) = data quality issues
- **Target**: 25-35% aligns with e-commerce industry benchmarks

## Model Validation Criteria

| Criterion | Minimum | Target | Stretch |
|-----------|---------|--------|---------|
| **Cross-Validation Stability** | CV std < 0.10 | CV std < 0.05 | CV std < 0.03 |
| **Train-Test Gap** | < 0.10 | < 0.05 | < 0.03 |
| **Feature Importance Concentration** | Top 10 explain 60% | Top 10 explain 70% | Top 10 explain 80% |

### Validation Justification

**Cross-Validation Stability**
- **Metric**: Standard deviation of ROC-AUC across 5 folds
- **Low Std**: Model is stable and not overfitting to specific data splits
- **High Std**: Model is unstable, may not generalize well

**Train-Test Gap**
- **Metric**: |Train ROC-AUC - Test ROC-AUC|
- **Low Gap**: Model generalizes well to unseen data
- **High Gap**: Overfitting, model memorized training data

**Feature Importance Concentration**
- **Metric**: Cumulative importance of top 10 features
- **High Concentration**: Few features drive predictions (simpler model)
- **Low Concentration**: Many features needed (complex patterns)

## Deployment Criteria

| Criterion | Minimum | Target | Stretch |
|-----------|---------|--------|---------|
| **Application Uptime** | 95% | 99% | 99.9% |
| **User Interface Responsiveness** | Functional | Intuitive | Delightful |
| **Documentation Completeness** | All required docs | Comprehensive | Exemplary |
| **Code Quality Score** | Pass all checks | Clean code | Production-ready |

## Acceptance Criteria

### Phase-by-Phase Acceptance

**Phase 1-2: Business Understanding & Data Acquisition**
- ✅ All 4 business documents completed
- ✅ Dataset loaded with 500k+ rows
- ✅ Data quality summary generated

**Phase 3: Data Cleaning**
- ✅ Retention rate ≥ 50%
- ✅ Zero missing values in final dataset
- ✅ All validation checks pass

**Phase 4: Feature Engineering**
- ✅ Churn rate between 20-40%
- ✅ Minimum 20 features created
- ✅ No data leakage (verified)

**Phase 5: EDA**
- ✅ Minimum 15 visualizations
- ✅ Statistical significance tests performed
- ✅ Key insights documented

**Phase 6-7: Modeling & Evaluation**
- ✅ 5 models implemented and compared
- ✅ Test set ROC-AUC ≥ 0.75
- ✅ Cross-validation performed
- ✅ Business impact analysis completed

**Phase 8: Deployment**
- ✅ Streamlit app deployed with public URL
- ✅ Single and batch predictions working
- ✅ All features functional

**Phase 9-10: Documentation & Code Quality**
- ✅ Complete README with setup instructions
- ✅ Technical documentation
- ✅ Presentation (10-12 slides)
- ✅ ≥20 Git commits
- ✅ Clean code with docstrings

## Scoring Rubric Alignment

| Phase | Points | Minimum to Pass | Target Score |
|-------|--------|----------------|--------------|
| Phase 1 | 10 | 7 | 9 |
| Phase 2 | 15 | 11 | 13 |
| Phase 3 | 20 | 15 | 18 |
| Phase 4 | 25 | 19 | 23 |
| Phase 5 | 15 | 11 | 13 |
| Phase 6 | 20 | 15 | 18 |
| Phase 7 | 15 | 11 | 13 |
| Phase 8 | 13 | 10 | 12 |
| Phase 9 | 12 | 9 | 11 |
| Phase 10 | 5 | 4 | 5 |
| **Total** | **150** | **112 (75%)** | **135 (90%)** |

**Note**: Rubric totals 150 points, but normalized to 100 points for final score.

## Risk Thresholds

| Risk Level | Condition | Action Required |
|------------|-----------|-----------------|
| 🟢 **Low** | All metrics meet target | Proceed to next phase |
| 🟡 **Medium** | Metrics between minimum and target | Review and improve if time permits |
| 🔴 **High** | Any metric below minimum | STOP - Debug and fix before proceeding |

### Critical Failure Conditions

**Immediate Stop and Debug Required**:
- ❌ Data retention < 50%
- ❌ Churn rate < 20% or > 40%
- ❌ Test ROC-AUC < 0.75
- ❌ Data leakage detected
- ❌ Deployment fails

## Success Celebration Criteria

**Minimum Success** (Pass Project):
- All phases completed
- All minimum thresholds met
- Deployed application working
- Complete documentation

**Target Success** (Strong Performance):
- All target thresholds met
- ROC-AUC ≥ 0.78
- Clean, well-documented code
- Insightful business analysis

**Stretch Success** (Exceptional Work):
- All stretch thresholds met
- ROC-AUC ≥ 0.85
- Production-ready code quality
- Innovative feature engineering
- Comprehensive model explainability

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Review Frequency**: After each phase completion
