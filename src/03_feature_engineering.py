"""
Feature Engineering Pipeline for Customer Churn Prediction

This script transforms transaction-level data into customer-level features
using RFM analysis, behavioral patterns, temporal features, and product affinity.

CRITICAL: All features must use ONLY the training period to prevent data leakage!

Input: data/processed/cleaned_transactions.csv (342,273 transactions)
Output: data/processed/customer_features.csv (~3,500-4,000 customers with 30+ features)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import os

# Setup logging
logging.basicConfig(
    filename='logs/feature_engineering.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FeatureEngineer:
    """
    Feature engineering pipeline for customer churn prediction
    """
    
    def __init__(self, input_path='data/processed/cleaned_transactions.csv'):
        """Initialize with cleaned data path"""
        self.input_path = input_path
        self.df = None
        self.training_data = None
        self.observation_data = None
        self.customer_features = None
        
        # Critical dates for temporal split  
        # Critical dates for temporal split  
        self.training_cutoff = pd.Timestamp('2010-09-09')
        self.observation_end = pd.Timestamp('2010-12-09')  # 90-day observation window
        
        self.feature_info = {
            'training_period': {
                'start': '2009-12-01',
                'end': '2010-09-09',
                'days': 283
            },
            'observation_period': {
                'start': '2010-09-10',
                'end': '2010-12-09',  # 90-day observation window
                'days': 90
            },
            'total_features': 0,
            'churn_rate': 0.0,
            'total_customers': 0
        }
    
    def load_data(self):
        """Load cleaned transaction data"""
        logging.info("Loading cleaned transaction data...")
        print("\n" + "="*60)
        print("LOADING CLEANED TRANSACTION DATA")
        print("="*60)
        
        self.df = pd.read_csv(self.input_path, parse_dates=['InvoiceDate'])
        
        logging.info(f"Loaded {len(self.df):,} transactions")
        print(f"✓ Loaded {len(self.df):,} transactions")
        print(f"  Date range: {self.df['InvoiceDate'].min()} to {self.df['InvoiceDate'].max()}")
        print(f"  Unique customers: {self.df['Customer ID'].nunique():,}")
        
        return self
    
    def create_temporal_split(self):
        """
        Split data into training and observation periods
        
        Training: 2009-12-01 to 2010-09-09 (for features)
        Observation: 2010-09-10 to 2010-10-09 (for churn labels)
        """
        logging.info("Creating temporal split...")
        print("\n" + "="*60)
        print("CREATING TEMPORAL SPLIT")
        print("="*60)
        
        # Split data
        self.training_data = self.df[self.df['InvoiceDate'] <= self.training_cutoff].copy()
        # Strictly enforce observation window
        self.observation_data = self.df[(self.df['InvoiceDate'] > self.training_cutoff) & 
                                      (self.df['InvoiceDate'] <= self.observation_end)].copy()
        
        print(f"Training period: {self.training_data['InvoiceDate'].min()} to {self.training_data['InvoiceDate'].max()}")
        print(f"  Transactions: {len(self.training_data):,}")
        print(f"  Customers: {self.training_data['Customer ID'].nunique():,}")
        
        print(f"\nObservation period: {self.observation_data['InvoiceDate'].min()} to {self.observation_data['InvoiceDate'].max()}")
        print(f"  Transactions: {len(self.observation_data):,}")
        print(f"  Customers: {self.observation_data['Customer ID'].nunique():,}")
        
        logging.info(f"Training: {len(self.training_data):,} transactions, {self.training_data['Customer ID'].nunique():,} customers")
        logging.info(f"Observation: {len(self.observation_data):,} transactions, {self.observation_data['Customer ID'].nunique():,} customers")
        
        return self
    
    def create_churn_labels(self):
        """
        Create churn labels based on observation period activity
        
        Churned (1): Purchased in training BUT NOT in observation
        Active (0): Purchased in BOTH training AND observation
        """
        logging.info("Creating churn labels...")
        print("\n" + "="*60)
        print("CREATING CHURN LABELS")
        print("="*60)
        
        # Get unique customers from each period
        training_customers = set(self.training_data['Customer ID'].unique())
        observation_customers = set(self.observation_data['Customer ID'].unique())
        
        # Churned = in training but NOT in observation
        churned_customers = training_customers - observation_customers
        
        # Active = in BOTH periods
        active_customers = training_customers & observation_customers
        
        # Create labels dictionary
        churn_labels = {}
        for customer_id in training_customers:
            if customer_id in churned_customers:
                churn_labels[customer_id] = 1  # Churned
            else:
                churn_labels[customer_id] = 0  # Active
        
        # Convert to DataFrame
        self.churn_labels_df = pd.DataFrame(list(churn_labels.items()), 
                                             columns=['Customer ID', 'Churn'])
        
        # Calculate churn rate
        churn_rate = len(churned_customers) / len(training_customers) * 100
        
        print(f"Total customers: {len(training_customers):,}")
        print(f"Churned customers: {len(churned_customers):,} ({churn_rate:.2f}%)")
        print(f"Active customers: {len(active_customers):,} ({100-churn_rate:.2f}%)")
        
        # Validation
        if churn_rate < 20 or churn_rate > 40:
            print(f"\n⚠️  WARNING: Churn rate {churn_rate:.2f}% is outside expected range (20-40%)")
        else:
            print(f"\n✓ Churn rate is within expected range (20-40%)")
        
        self.feature_info['churn_rate'] = round(float(churn_rate), 2)
        self.feature_info['total_customers'] = int(len(training_customers))
        
        logging.info(f"Churn rate: {churn_rate:.2f}%")
        
        return self
    
    def create_rfm_features(self):
        """
        Create RFM (Recency, Frequency, Monetary) features
        
        CRITICAL: Use only training_data to prevent data leakage!
        """
        logging.info("Creating RFM features...")
        print("\n" + "="*60)
        print("CREATING RFM FEATURES")
        print("="*60)
        
        # Group by customer
        customer_groups = self.training_data.groupby('Customer ID')
        
        # Recency: Days since last purchase (from training cutoff)
        last_purchase = customer_groups['InvoiceDate'].max()
        recency = (self.training_cutoff - last_purchase).dt.days
        
        # Frequency: Number of unique invoices
        frequency = customer_groups['Invoice'].nunique()
        
        # Monetary: Total amount spent
        total_spent = customer_groups['TotalPrice'].sum()
        
        # Average order value
        avg_order_value = total_spent / frequency
        
        # Unique products purchased
        unique_products = customer_groups['StockCode'].nunique()
        
        # Total items purchased
        total_items = customer_groups['Quantity'].sum()
        
        # Create RFM DataFrame
        rfm_features = pd.DataFrame({
            'Customer ID': recency.index,
            'Recency': recency.values,
            'Frequency': frequency.values,
            'TotalSpent': total_spent.values,
            'AvgOrderValue': avg_order_value.values,
            'UniqueProducts': unique_products.values,
            'TotalItems': total_items.values
        })
        
        print(f"✓ Created 6 RFM features:")
        print(f"  - Recency (days since last purchase)")
        print(f"  - Frequency (number of purchases)")
        print(f"  - TotalSpent (total amount)")
        print(f"  - AvgOrderValue")
        print(f"  - UniqueProducts")
        print(f"  - TotalItems")
        
        self.customer_features = rfm_features
        
        logging.info("Created 6 RFM features")
        
        return self
    
    def create_behavioral_features(self):
        """
        Create behavioral pattern features
        """
        logging.info("Creating behavioral features...")
        print("\n" + "="*60)
        print("CREATING BEHAVIORAL FEATURES")
        print("="*60)
        
        customer_groups = self.training_data.groupby('Customer ID')
        
        # Average days between purchases
        def calc_avg_days_between(group):
            if len(group) <= 1:
                return np.nan
            dates = group['InvoiceDate'].sort_values().unique()
            if len(dates) <= 1:
                return np.nan
            diffs = np.diff(dates).astype('timedelta64[D]').astype(int)
            return np.mean(diffs)
        
        avg_days_between = customer_groups.apply(calc_avg_days_between)
        
        # Basket size statistics
        basket_sizes = self.training_data.groupby(['Customer ID', 'Invoice'])['Quantity'].sum()
        avg_basket_size = basket_sizes.groupby('Customer ID').mean()
        std_basket_size = basket_sizes.groupby('Customer ID').std().fillna(0)
        max_basket_size = basket_sizes.groupby('Customer ID').max()
        
        # Preferred shopping day and hour
        preferred_day = customer_groups['DayOfWeek'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 0)
        preferred_hour = customer_groups['Hour'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 12)
        
        # Country diversity
        country_diversity = customer_groups['Country'].nunique()
        
        # Create behavioral features DataFrame
        behavioral_features = pd.DataFrame({
            'Customer ID': avg_days_between.index,
            'AvgDaysBetweenPurchases': avg_days_between.values,
            'AvgBasketSize': avg_basket_size.values,
            'StdBasketSize': std_basket_size.values,
            'MaxBasketSize': max_basket_size.values,
            'PreferredDay': preferred_day.values,
            'PreferredHour': preferred_hour.values,
            'CountryDiversity': country_diversity.values
        })
        
        # Fill NaN in AvgDaysBetweenPurchases with a large number (single purchase customers)
        behavioral_features['AvgDaysBetweenPurchases'].fillna(999, inplace=True)
        
        # Merge with existing features
        self.customer_features = self.customer_features.merge(behavioral_features, on='Customer ID', how='left')
        
        print(f"✓ Created 7 behavioral features:")
        print(f"  - AvgDaysBetweenPurchases")
        print(f"  - AvgBasketSize, StdBasketSize, MaxBasketSize")
        print(f"  - PreferredDay, PreferredHour")
        print(f"  - CountryDiversity")
        
        logging.info("Created 7 behavioral features")
        
        return self
    
    def create_temporal_features(self):
        """
        Create temporal/lifecycle features
        """
        logging.info("Creating temporal features...")
        print("\n" + "="*60)
        print("CREATING TEMPORAL FEATURES")
        print("="*60)
        
        customer_groups = self.training_data.groupby('Customer ID')
        
        # Customer lifetime (days from first to last purchase)
        first_purchase = customer_groups['InvoiceDate'].min()
        last_purchase = customer_groups['InvoiceDate'].max()
        customer_lifetime = (last_purchase - first_purchase).dt.days
        
        # Purchase velocity (purchases per day)
        purchase_velocity = self.customer_features.set_index('Customer ID')['Frequency'] / (customer_lifetime + 1)
        
        # Recent activity windows
        cutoff_30 = self.training_cutoff - timedelta(days=30)
        cutoff_60 = self.training_cutoff - timedelta(days=60)
        cutoff_90 = self.training_cutoff - timedelta(days=90)
        
        # Frequency (Count of Invoices)
        freq_30 = self.training_data[self.training_data['InvoiceDate'] > cutoff_30].groupby('Customer ID')['Invoice'].nunique()
        freq_90 = self.training_data[self.training_data['InvoiceDate'] > cutoff_90].groupby('Customer ID')['Invoice'].nunique()
        
        # Monetary (Sum of Spend)
        spend_30 = self.training_data[self.training_data['InvoiceDate'] > cutoff_30].groupby('Customer ID')['TotalPrice'].sum()
        spend_90 = self.training_data[self.training_data['InvoiceDate'] > cutoff_90].groupby('Customer ID')['TotalPrice'].sum()
        
        # Calculate Trends (Normalize by time period length to compare "intensity")
        # 30-day intensity vs 90-day intensity
        # If Ratio < 1, activity is dropping (cooling down) -> Signal for Churn
        
        # Reindex to ensure all customers are present (fill missing with 0)
        freq_30 = freq_30.reindex(customer_lifetime.index, fill_value=0)
        freq_90 = freq_90.reindex(customer_lifetime.index, fill_value=0)
        spend_30 = spend_30.reindex(customer_lifetime.index, fill_value=0)
        spend_90 = spend_90.reindex(customer_lifetime.index, fill_value=0)
        
        # Avoid division by zero
        freq_trend = (freq_30 / 30) / ((freq_90 / 90) + 0.001)
        spend_trend = (spend_30 / 30) / ((spend_90 / 90) + 0.001)
        
        purchases_last_30 = freq_30
        purchases_last_60 = self.training_data[self.training_data['InvoiceDate'] > cutoff_60].groupby('Customer ID')['Invoice'].nunique().reindex(customer_lifetime.index, fill_value=0)
        purchases_last_90 = freq_90
        
        # Create temporal features DataFrame
        temporal_features = pd.DataFrame({
            'Customer ID': customer_lifetime.index,
            'CustomerLifetimeDays': customer_lifetime.values,
            'PurchaseVelocity': purchase_velocity.values,
            'Purchases_Last30Days': purchases_last_30.values,
            'Purchases_Last60Days': purchases_last_60.values,
            'Purchases_Last90Days': purchases_last_90.values,
            'Spend_Last30Days': spend_30.values,
            'Spend_Last90Days': spend_90.values,
            'FrequencyTrend': freq_trend.values,
            'SpendTrend': spend_trend.values
        })
        
        # Merge with existing features
        self.customer_features = self.customer_features.merge(temporal_features, on='Customer ID', how='left')
        
        print(f"✓ Created 5 temporal features:")
        print(f"  - CustomerLifetimeDays")
        print(f"  - PurchaseVelocity")
        print(f"  - Purchases_Last30Days, Purchases_Last60Days, Purchases_Last90Days")
        
        logging.info("Created 5 temporal features")
        
        return self

    def create_interaction_features(self):
        """
        Create interaction features to capture non-linear relationships
        """
        logging.info("Creating interaction features...")
        print("\n" + "="*60)
        print("CREATING INTERACTION FEATURES")
        print("="*60)
        
        # Interaction 1: Average Load (Spend per Transaction)
        # Already have AvgOrderValue, but let's make it explicit
        
        # Interaction 2: Frequency * Monetary (High value, High freq)
        self.customer_features['Freq_x_Spend'] = self.customer_features['Frequency'] * self.customer_features['TotalSpent']
        
        # Interaction 3: Recency * Frequency (Recent and Frequent)
        # Since Recency is "days since", lower is better. 
        # So we want (1/Recency) * Frequency to find "Active & Frequent"
        self.customer_features['Active_Freq'] = self.customer_features['Frequency'] / (self.customer_features['Recency'] + 1)
        
        # Interaction 4: Spend per Item (Unit Price Proxy)
        self.customer_features['Spend_per_Item'] = self.customer_features['TotalSpent'] / (self.customer_features['TotalItems'] + 1)
        
        print(f"✓ Created 3 interaction features:")
        print(f"  - Freq_x_Spend")
        print(f"  - Active_Freq (Frequency / Recency)")
        print(f"  - Spend_per_Item")
        
        logging.info("Created 3 interaction features")
        
        return self
    
    def create_product_features(self):
        """
        Create product affinity features
        """
        logging.info("Creating product features...")
        print("\n" + "="*60)
        print("CREATING PRODUCT FEATURES")
        print("="*60)
        
        customer_groups = self.training_data.groupby('Customer ID')
        
        # Product diversity score (unique products / total items)
        unique_products = customer_groups['StockCode'].nunique()
        total_items = customer_groups['Quantity'].sum()
        product_diversity = unique_products / total_items
        
        # Price preferences
        avg_price = customer_groups['Price'].mean()
        std_price = customer_groups['Price'].std().fillna(0)
        min_price = customer_groups['Price'].min()
        max_price = customer_groups['Price'].max()
        
        # Quantity preference
        avg_quantity_per_order = customer_groups['Quantity'].mean()
        
        # Create product features DataFrame
        product_features = pd.DataFrame({
            'Customer ID': product_diversity.index,
            'ProductDiversityScore': product_diversity.values,
            'AvgPricePreference': avg_price.values,
            'StdPricePreference': std_price.values,
            'MinPrice': min_price.values,
            'MaxPrice': max_price.values,
            'AvgQuantityPerOrder': avg_quantity_per_order.values
        })
        
        # Merge with existing features
        self.customer_features = self.customer_features.merge(product_features, on='Customer ID', how='left')
        
        print(f"✓ Created 6 product features:")
        print(f"  - ProductDiversityScore")
        print(f"  - AvgPricePreference, StdPricePreference")
        print(f"  - MinPrice, MaxPrice")
        print(f"  - AvgQuantityPerOrder")
        
        logging.info("Created 6 product features")
        
        return self
    
    def create_rfm_scores(self):
        """
        Create RFM quartile scores and customer segments
        """
        logging.info("Creating RFM scores and segments...")
        print("\n" + "="*60)
        print("CREATING RFM SCORES & SEGMENTS")
        print("="*60)
        
        # Create quartile scores (1-4, where 4 is best)
        # Use rank-based approach to handle duplicates
        
        # For Recency: lower is better, so reverse the score
        recency_ranks = self.customer_features['Recency'].rank(method='first', ascending=True)
        self.customer_features['RecencyScore'] = pd.cut(
            recency_ranks,
            bins=4,
            labels=[4, 3, 2, 1]  # Reversed: low recency = high score
        ).astype(int)
        
        # For Frequency: higher is better
        frequency_ranks = self.customer_features['Frequency'].rank(method='first', ascending=True)
        self.customer_features['FrequencyScore'] = pd.cut(
            frequency_ranks,
            bins=4,
            labels=[1, 2, 3, 4]
        ).astype(int)
        
        # For Monetary: higher is better
        monetary_ranks = self.customer_features['TotalSpent'].rank(method='first', ascending=True)
        self.customer_features['MonetaryScore'] = pd.cut(
            monetary_ranks,
            bins=4,
            labels=[1, 2, 3, 4]
        ).astype(int)
        
        # RFM composite score
        self.customer_features['RFM_Score'] = (
            self.customer_features['RecencyScore'] + 
            self.customer_features['FrequencyScore'] + 
            self.customer_features['MonetaryScore']
        )
        
        # Customer segmentation based on RFM scores
        def segment_customer(row):
            r, f, m = row['RecencyScore'], row['FrequencyScore'], row['MonetaryScore']
            
            if r >= 4 and f >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3:
                return 'Loyal'
            elif r >= 3 and f <= 2:
                return 'Potential'
            elif r <= 2 and f >= 3:
                return 'At Risk'
            else:
                return 'Lost'
        
        self.customer_features['CustomerSegment'] = self.customer_features.apply(segment_customer, axis=1)
        
        print(f"✓ Created 5 segmentation features:")
        print(f"  - RecencyScore, FrequencyScore, MonetaryScore")
        print(f"  - RFM_Score")
        print(f"  - CustomerSegment")
        
        print(f"\nCustomer Segment Distribution:")
        segment_counts = self.customer_features['CustomerSegment'].value_counts()
        for segment, count in segment_counts.items():
            print(f"  {segment}: {count:,} ({count/len(self.customer_features)*100:.1f}%)")
        
        logging.info("Created 5 RFM scores and segments")
        
        return self
    
    def add_churn_labels(self):
        """
        Add churn labels to customer features
        """
        logging.info("Adding churn labels...")
        print("\n" + "="*60)
        print("ADDING CHURN LABELS")
        print("="*60)
        
        # Merge churn labels
        self.customer_features = self.customer_features.merge(
            self.churn_labels_df, 
            on='Customer ID', 
            how='left'
        )
        
        print(f"✓ Added Churn target variable")
        print(f"  Total customers: {len(self.customer_features):,}")
        print(f"  Churned: {self.customer_features['Churn'].sum():,} ({self.customer_features['Churn'].mean()*100:.2f}%)")
        print(f"  Active: {(1-self.customer_features['Churn']).sum():,} ({(1-self.customer_features['Churn'].mean())*100:.2f}%)")
        
        logging.info("Added churn labels")
        
        return self
    
    def save_features(self, output_path='data/processed/customer_features.csv'):
        """
        Save customer features and feature info
        """
        logging.info("Saving customer features...")
        print("\n" + "="*60)
        print("SAVING CUSTOMER FEATURES")
        print("="*60)
        
        # Save features
        self.customer_features.to_csv(output_path, index=False)
        print(f"✓ Saved to: {output_path}")
        
        # Update feature info
        self.feature_info['total_features'] = len(self.customer_features.columns) - 2  # Exclude Customer ID and Churn
        
        # Save feature info
        info_path = 'data/processed/feature_info.json'
        with open(info_path, 'w') as f:
            json.dump(self.feature_info, f, indent=4)
        
        print(f"✓ Feature info saved to: {info_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Total customers: {len(self.customer_features):,}")
        print(f"Total features: {self.feature_info['total_features']}")
        print(f"Churn rate: {self.feature_info['churn_rate']:.2f}%")
        print(f"Dataset shape: {self.customer_features.shape}")
        print("="*60)
        
        return self
    
    def run_pipeline(self):
        """
        Execute complete feature engineering pipeline
        """
        print("\n" + "="*60)
        print("STARTING FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        self.load_data()
        self.create_temporal_split()
        self.create_churn_labels()
        self.create_rfm_features()
        self.create_behavioral_features()
        self.create_temporal_features()
        self.create_interaction_features()
        self.create_product_features()
        self.create_rfm_scores()
        self.add_churn_labels()
        self.save_features()
        
        print("\n✓ Feature engineering pipeline completed successfully!")
        
        return self.customer_features

def main():
    """
    Main execution function
    """
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run feature engineering pipeline
    engineer = FeatureEngineer('data/processed/cleaned_transactions.csv')
    customer_features = engineer.run_pipeline()
    
    print(f"\nFinal dataset shape: {customer_features.shape}")
    print(f"Columns: {list(customer_features.columns)}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review data/processed/feature_info.json")
    print("2. Run notebooks/03_feature_eda.ipynb for EDA")
    print("3. Proceed to Phase 6: Model Development")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
