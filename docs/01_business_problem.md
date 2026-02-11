# Business Problem Statement

## 1. Business Context

The e-commerce industry faces significant challenges in customer retention. Studies show that:
- Acquiring a new customer costs **5-25x more** than retaining an existing one
- A 5% increase in customer retention can increase profits by 25-95%
- The probability of selling to an existing customer is 60-70%, while for a new prospect it's only 5-20%

RetailCo Analytics operates in a highly competitive e-commerce market where customer churn directly impacts revenue and profitability. With increasing customer acquisition costs and market saturation, focusing on retention has become a strategic imperative.

### Current Business Pain Points

1. **Reactive Retention**: Currently, retention efforts are reactive rather than proactive
2. **Inefficient Marketing Spend**: Blanket marketing campaigns waste resources on customers who would stay anyway
3. **Lost Revenue**: High-value customers churning without warning
4. **Limited Insights**: Lack of understanding about which factors drive churn
5. **No Prioritization**: Unable to identify which customers to target first

## 2. Problem Definition

**Churn Definition**: A customer is considered "churned" if they have not made a purchase in the last **90 days** (3 months).

This definition is based on:
- Industry benchmarks for e-commerce repeat purchase cycles
- Business stakeholder input on acceptable customer dormancy periods
- Analysis of historical purchase patterns

**Primary Objective**: Build a predictive model to identify customers at high risk of churning in the next 3 months, enabling proactive retention interventions.

## 3. Stakeholders

### Marketing Team
- **Need**: Customer segments for targeted retention campaigns
- **Use Case**: Design personalized email campaigns, special offers, and loyalty programs
- **Success Metric**: Increase campaign ROI by 30%

### Sales Team
- **Need**: Churn probability scores for each customer
- **Use Case**: Prioritize outreach to high-value at-risk customers
- **Success Metric**: Reduce churn rate among top 20% customers by 15%

### Product Team
- **Need**: Insights into product preferences and purchasing patterns
- **Use Case**: Improve product recommendations and user experience
- **Success Metric**: Increase average order frequency by 10%

### Executive Team
- **Need**: ROI projections and business impact metrics
- **Use Case**: Strategic decision-making on retention budget allocation
- **Success Metric**: Demonstrate positive ROI within 6 months

## 4. Business Impact

### Expected Outcomes

**Churn Rate Reduction**: Target 15-20% reduction in overall churn rate
- Current estimated churn: 30-35%
- Target churn: 25-28%

**Revenue Impact**:
- Average customer lifetime value: £500
- If we prevent 200 customers from churning per quarter: £100,000 saved revenue
- Annual impact: £400,000

**Cost Savings**:
- Targeted retention campaigns cost £10 per customer
- Blanket campaigns cost £15 per customer
- With 4,000 customers, targeting only high-risk (30%) = £12,000 vs £60,000
- **Savings**: £48,000 per campaign

**Customer Lifetime Value Increase**:
- Retained customers have 2-3x higher LTV
- Expected LTV increase: 15-20%

## 5. Success Metrics

### Primary Metric
**ROC-AUC Score > 0.78**
- Measures model's ability to distinguish between churners and non-churners
- Industry benchmark for churn prediction: 0.75-0.85
- Our target: 0.78 (competitive performance)

### Secondary Metrics

**Precision > 0.75**
- Minimize false positives (customers incorrectly flagged as churners)
- Important because: Unnecessary retention efforts waste budget and may annoy loyal customers
- Business impact: Efficient use of marketing budget

**Recall > 0.70**
- Catch actual churners before they leave
- Important because: Missing a churner means lost revenue
- Business impact: Maximize revenue protection

**F1-Score > 0.72**
- Balanced measure of precision and recall
- Ensures model doesn't sacrifice one metric for the other

### Business Success Metrics

1. **Retention Campaign ROI**: Minimum 3:1 return
   - Cost of campaign: £10 per customer
   - Value of retained customer: £500 LTV
   - Break-even: 2% success rate

2. **Churn Rate Reduction**: 15-20% reduction within 6 months

3. **Revenue Protected**: Minimum £300,000 annually

4. **Customer Satisfaction**: No decrease in NPS scores (ensure retention efforts don't feel intrusive)

## 6. Project Constraints

- **Budget**: Limited to free/open-source tools only
- **Timeline**: 8 weeks to production deployment
- **Data**: Historical transactional data only (no demographic or behavioral web data)
- **Deployment**: Must be accessible via web interface for non-technical stakeholders
- **Privacy**: Must comply with data protection regulations (anonymized customer IDs)

## 7. Risk Assessment

### Technical Risks
- **Data Quality**: High percentage of missing customer IDs (~25%)
- **Class Imbalance**: Expected churn rate of 25-35% may require special handling
- **Feature Availability**: Limited to transactional data only

### Business Risks
- **Model Adoption**: Stakeholders may not trust ML predictions initially
- **False Positives**: Over-targeting loyal customers could damage relationships
- **False Negatives**: Missing high-value churners could be costly

### Mitigation Strategies
- Implement robust data cleaning and validation
- Use appropriate evaluation metrics for imbalanced data
- Provide model explainability features
- Start with pilot program on small customer segment
- Continuous monitoring and model retraining

## 8. Success Criteria Summary

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| ROC-AUC | 0.75 | 0.78 | 0.85 |
| Precision | 0.70 | 0.75 | 0.80 |
| Recall | 0.65 | 0.70 | 0.75 |
| F1-Score | 0.67 | 0.72 | 0.77 |
| Churn Reduction | 10% | 15% | 20% |
| Campaign ROI | 2:1 | 3:1 | 5:1 |

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Owner**: Data Science Team
