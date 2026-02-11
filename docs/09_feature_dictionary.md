# Feature Dictionary (Phase 4.4)

## Target Variable
| Feature | Type | Description | Example | Business Meaning |
| :--- | :--- | :--- | :--- | :--- |
| **Churn** | Binary | 1=Churned, 0=Active | 1 | Customer made no purchases in the 3-month observation period (Next 90 days) |

## RFM Features
| Feature | Type | Description | Range | Business Meaning |
| :--- | :--- | :--- | :--- | :--- |
| **Recency** | Integer | Days since last purchase | 0-365+ | **Lower** is better. Recent buyers are more engaged. |
| **Frequency** | Integer | Total count of unique invoices | 1-200+ | **Higher** is better. Frequent buyers are loyal. |
| **TotalSpent** | Float | Total monetary value (Revenue) | 0-50k+ | **Higher** is better. High Value Customers. |
| **AvgOrderValue** | Float | Average spend per transaction | 0-5k | Indicates purchasing power/basket size. |
| **UniqueProducts** | Integer | Count of unique StockCodes bought | 1-100+ | Variety of interest. |
| **TotalItems** | Integer | Sum of all quantities purchased | 1-10k+ | Volume of consumption. |
| **RFM_Score** | Integer | Composite score (Recency+Freq+Monetary) | 3-12 | Overall customer value score. |

## Behavioral Features
| Feature | Type | Description | Business Meaning |
| :--- | :--- | :--- | :--- |
| **AvgDaysBetweenPurchases** | Float | Mean days between transactions | Consistency of purchasing. Lower = more regular. |
| **AvgBasketSize** | Float | Mean items per invoice | Scale of typical purchase. |
| **StdBasketSize** | Float | Std Dev of basket size | Consistency of order size. |
| **PreferredDay** | Integer | Most frequent day of purchase (0-6) | Shopping habit timing (Weekend vs Weekday). |
| **PreferredHour** | Integer | Most frequent hour of purchase | Interaction time preference. |
| **CountryDiversity** | Integer | Count of unique countries shipped to | Cross-border activity (rare for most). |

## Temporal Features
| Feature | Type | Description | Business Meaning |
| :--- | :--- | :--- | :--- |
| **CustomerLifetimeDays** | Integer | Days between first and last purchase | Long-term loyalty measurement. |
| **PurchaseVelocity** | Float | Purchases per day of lifetime | Intensity of relationship. |
| **Purchases_Last30Days** | Integer | Count of invoices in last 30d | Short-term engagement trend. |
| **Purchases_Last60Days** | Integer | Count of invoices in last 60d | Mid-term engagement trend. |
| **Purchases_Last90Days** | Integer | Count of invoices in last 90d | Long-term engagement trend. |

## Product Features
| Feature | Type | Description | Business Meaning |
| :--- | :--- | :--- | :--- |
| **ProductDiversityScore** | Float | Unique Products / Total Items | 0-1. High = Explorer, Low = Bulk/Repetitive buyer. |
| **AvgPricePreference** | Float | Avg UnitPrice of items bought | Preference for premium vs cheap items. |

## Feature Engineering Decisions

### Why these features?
1.  **RFM (Recency, Frequency, Monetary)**: The standard in retail analytics. Proven to be the strongest predictors of churn.
2.  **Temporal Trends (Last 30/60/90 days)**: Churn is often preceded by a "cooling off" period. Capturing the slope of activity (e.g., high lifetime frequency but 0 in last 30 days) is critical.
3.  **Product Diversity**: Differentiates between customers who buy a single item in bulk (one-off) vs those who shop the catalog (engaged).

### Feature Interactions
- **Recency x Frequency**: High Frequency + High Recency (stopped buying recently) = **High Churn Risk (At Risk)**.
- **Recency x Monetary**: High Spender + High Recency = **High Value Loss Risk**.
