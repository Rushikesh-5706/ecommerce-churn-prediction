"""
Data Cleaning Pipeline for Customer Churn Prediction System

This script implements a comprehensive data cleaning pipeline that transforms
the raw UCI Online Retail dataset into a clean, analysis-ready format.

Expected Input: data/raw/online_retail.csv (525,461 rows)
Expected Output: data/processed/cleaned_transactions.csv (~350k-400k rows)

Cleaning Steps:
1. Remove missing CustomerIDs
2. Handle cancelled invoices  
3. Remove negative quantities
4. Remove zero/negative prices
5. Handle missing descriptions
6. Remove outliers (IQR method)
7. Remove duplicates
8. Add derived columns
9. Convert data types
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import os

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/data_cleaning.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataCleaner:
    """
    Comprehensive data cleaning pipeline for Online Retail dataset
    """
    
    def __init__(self, input_path='data/raw/online_retail.csv'):
        """Initialize with raw data path"""
        self.input_path = input_path
        self.df = None
        self.cleaning_stats = {
            'original_rows': 0,
            'rows_after_cleaning': 0,
            'rows_removed': 0,
            'retention_rate': 0.0,
            'missing_values_before': {},
            'missing_values_after': {},
            'steps_applied': []
        }
    
    def load_data(self):
        """Load raw dataset with proper encoding"""
        logging.info("Loading raw dataset...")
        print("\n" + "="*60)
        print("LOADING RAW DATASET")
        print("="*60)
        
        try:
            # Try UTF-8 first
            self.df = pd.read_csv(
                self.input_path,
                encoding='utf-8',
                parse_dates=['InvoiceDate']
            )
        except UnicodeDecodeError:
            # Fallback to latin1
            logging.info("UTF-8 failed, using latin1 encoding")
            print("UTF-8 encoding failed, trying latin1...")
            self.df = pd.read_csv(
                self.input_path,
                encoding='latin1',
                parse_dates=['InvoiceDate']
            )
        
        self.cleaning_stats['original_rows'] = len(self.df)
        self.cleaning_stats['missing_values_before'] = self.df.isnull().sum().to_dict()
        
        logging.info(f"Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
        print(f"✓ Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")
        print(f"  Date range: {self.df['InvoiceDate'].min()} to {self.df['InvoiceDate'].max()}")
        
        return self
    
    def remove_missing_customer_ids(self):
        """
        Step 1: Remove rows with missing CustomerID
        
        Reasoning: CustomerID is essential for creating customer-level features.
        Cannot impute as it's a unique identifier.
        """
        logging.info("Step 1: Removing missing CustomerIDs...")
        print("\nStep 1: Removing missing CustomerIDs...")
        
        initial_rows = len(self.df)
        
        # Remove rows where Customer ID is null
        self.df = self.df[self.df['Customer ID'].notna()].copy()
        
        rows_removed = initial_rows - len(self.df)
        logging.info(f"Removed {rows_removed:,} rows with missing CustomerID")
        print(f"  Removed: {rows_removed:,} rows ({rows_removed/initial_rows*100:.2f}%)")
        print(f"  Remaining: {len(self.df):,} rows")
        
        self.cleaning_stats['steps_applied'].append({
            'step': 'remove_missing_customer_ids',
            'rows_removed': int(rows_removed),
            'percentage_removed': round(float(rows_removed/initial_rows*100), 2)
        })
        
        return self
    
    def handle_cancelled_invoices(self):
        """
        Step 2: Handle cancelled invoices (InvoiceNo starting with 'C')
        
        Decision: Remove all cancellations for simpler modeling
        """
        logging.info("Step 2: Handling cancelled invoices...")
        print("\nStep 2: Handling cancelled invoices...")
        
        initial_rows = len(self.df)
        
        # Check for invoices starting with 'C'
        cancelled_mask = self.df['Invoice'].astype(str).str.startswith('C')
        cancelled_count = cancelled_mask.sum()
        
        if cancelled_count > 0:
            self.df = self.df[~cancelled_mask].copy()
            rows_removed = initial_rows - len(self.df)
            logging.info(f"Removed {rows_removed:,} cancelled invoices")
            print(f"  Removed: {rows_removed:,} cancelled invoices")
        else:
            rows_removed = 0
            print(f"  No cancelled invoices found (Invoice starting with 'C')")
        
        print(f"  Remaining: {len(self.df):,} rows")
        
        self.cleaning_stats['steps_applied'].append({
            'step': 'handle_cancelled_invoices',
            'rows_removed': int(rows_removed)
        })
        
        return self
    
    def handle_negative_quantities(self):
        """
        Step 3: Handle negative quantities (returns)
        
        Decision: Remove negative quantities as they represent returns
        """
        logging.info("Step 3: Handling negative quantities...")
        print("\nStep 3: Handling negative quantities...")
        
        initial_rows = len(self.df)
        
        # Remove rows with Quantity <= 0
        self.df = self.df[self.df['Quantity'] > 0].copy()
        
        rows_removed = initial_rows - len(self.df)
        logging.info(f"Removed {rows_removed:,} rows with negative/zero quantities")
        print(f"  Removed: {rows_removed:,} rows with Quantity ≤ 0")
        print(f"  Remaining: {len(self.df):,} rows")
        
        self.cleaning_stats['steps_applied'].append({
            'step': 'handle_negative_quantities',
            'rows_removed': int(rows_removed)
        })
        
        return self
    
    def handle_zero_prices(self):
        """
        Step 4: Remove transactions with zero or negative prices
        
        Reasoning: Zero/negative prices are data errors or special cases
        """
        logging.info("Step 4: Removing zero/negative prices...")
        print("\nStep 4: Removing zero/negative prices...")
        
        initial_rows = len(self.df)
        
        # Remove rows with Price <= 0
        self.df = self.df[self.df['Price'] > 0].copy()
        
        rows_removed = initial_rows - len(self.df)
        logging.info(f"Removed {rows_removed:,} rows with invalid prices")
        print(f"  Removed: {rows_removed:,} rows with Price ≤ 0")
        print(f"  Remaining: {len(self.df):,} rows")
        
        self.cleaning_stats['steps_applied'].append({
            'step': 'handle_zero_prices',
            'rows_removed': int(rows_removed)
        })
        
        return self
    
    def handle_missing_descriptions(self):
        """
        Step 5: Handle missing product descriptions
        
        Approach: Remove rows with missing descriptions
        """
        logging.info("Step 5: Handling missing descriptions...")
        print("\nStep 5: Handling missing descriptions...")
        
        initial_rows = len(self.df)
        
        # Remove rows with missing Description
        self.df = self.df[self.df['Description'].notna()].copy()
        
        rows_removed = initial_rows - len(self.df)
        logging.info(f"Removed {rows_removed:,} rows with missing descriptions")
        print(f"  Removed: {rows_removed:,} rows with missing Description")
        print(f"  Remaining: {len(self.df):,} rows")
        
        self.cleaning_stats['steps_applied'].append({
            'step': 'handle_missing_descriptions',
            'rows_removed': int(rows_removed)
        })
        
        return self
    
    def remove_outliers(self):
        """
        Step 6: Remove statistical outliers using IQR method
        
        Method: IQR (Interquartile Range)
        Threshold: 1.5 × IQR
        """
        logging.info("Step 6: Removing outliers using IQR method...")
        print("\nStep 6: Removing outliers (IQR method)...")
        
        initial_rows = len(self.df)
        
        # Quantity outliers
        Q1_qty = self.df['Quantity'].quantile(0.25)
        Q3_qty = self.df['Quantity'].quantile(0.75)
        IQR_qty = Q3_qty - Q1_qty
        lower_qty = Q1_qty - 1.5 * IQR_qty
        upper_qty = Q3_qty + 1.5 * IQR_qty
        
        print(f"  Quantity IQR: Q1={Q1_qty}, Q3={Q3_qty}, IQR={IQR_qty:.2f}")
        print(f"  Quantity bounds: [{lower_qty:.2f}, {upper_qty:.2f}]")
        
        # Price outliers
        Q1_price = self.df['Price'].quantile(0.25)
        Q3_price = self.df['Price'].quantile(0.75)
        IQR_price = Q3_price - Q1_price
        lower_price = Q1_price - 1.5 * IQR_price
        upper_price = Q3_price + 1.5 * IQR_price
        
        print(f"  Price IQR: Q1={Q1_price:.2f}, Q3={Q3_price:.2f}, IQR={IQR_price:.2f}")
        print(f"  Price bounds: [£{lower_price:.2f}, £{upper_price:.2f}]")
        
        # Apply filters
        self.df = self.df[
            (self.df['Quantity'] >= lower_qty) & 
            (self.df['Quantity'] <= upper_qty) &
            (self.df['Price'] >= lower_price) & 
            (self.df['Price'] <= upper_price)
        ].copy()
        
        rows_removed = initial_rows - len(self.df)
        logging.info(f"Removed {rows_removed:,} outlier rows")
        print(f"  Removed: {rows_removed:,} outlier rows")
        print(f"  Remaining: {len(self.df):,} rows")
        
        self.cleaning_stats['steps_applied'].append({
            'step': 'remove_outliers',
            'rows_removed': int(rows_removed),
            'method': 'IQR',
            'threshold': 1.5
        })
        
        return self
    
    def remove_duplicates(self):
        """
        Step 7: Remove duplicate transactions
        """
        logging.info("Step 7: Removing duplicates...")
        print("\nStep 7: Removing duplicates...")
        
        initial_rows = len(self.df)
        
        # Remove exact duplicates
        self.df = self.df.drop_duplicates().copy()
        
        rows_removed = initial_rows - len(self.df)
        logging.info(f"Removed {rows_removed:,} duplicate rows")
        print(f"  Removed: {rows_removed:,} duplicate rows")
        print(f"  Remaining: {len(self.df):,} rows")
        
        self.cleaning_stats['steps_applied'].append({
            'step': 'remove_duplicates',
            'rows_removed': int(rows_removed)
        })
        
        return self
    
    def add_derived_columns(self):
        """
        Step 8: Add useful derived columns
        """
        logging.info("Step 8: Creating derived columns...")
        print("\nStep 8: Creating derived columns...")
        
        # TotalPrice
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['Price']
        print("  ✓ Created TotalPrice = Quantity × Price")
        
        # Date components
        self.df['Year'] = self.df['InvoiceDate'].dt.year
        self.df['Month'] = self.df['InvoiceDate'].dt.month
        self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
        self.df['Hour'] = self.df['InvoiceDate'].dt.hour
        
        print("  ✓ Created Year, Month, DayOfWeek, Hour from InvoiceDate")
        
        logging.info("Created derived columns: TotalPrice, Year, Month, DayOfWeek, Hour")
        
        self.cleaning_stats['steps_applied'].append({
            'step': 'add_derived_columns',
            'columns_added': ['TotalPrice', 'Year', 'Month', 'DayOfWeek', 'Hour']
        })
        
        return self
    
    def convert_data_types(self):
        """
        Step 9: Convert data types for efficiency
        """
        logging.info("Step 9: Converting data types...")
        print("\nStep 9: Converting data types...")
        
        # CustomerID to integer
        self.df['Customer ID'] = self.df['Customer ID'].astype(int)
        print("  ✓ Converted Customer ID to integer")
        
        # Categorical columns
        self.df['StockCode'] = self.df['StockCode'].astype('category')
        self.df['Country'] = self.df['Country'].astype('category')
        print("  ✓ Converted StockCode and Country to category")
        
        logging.info("Data type conversions completed")
        
        self.cleaning_stats['steps_applied'].append({
            'step': 'convert_data_types'
        })
        
        return self
    
    def save_cleaned_data(self, output_path='data/processed/cleaned_transactions.csv'):
        """
        Save cleaned dataset and cleaning statistics
        """
        logging.info("Saving cleaned data...")
        print("\n" + "="*60)
        print("SAVING CLEANED DATA")
        print("="*60)
        
        # Create directory if not exists
        os.makedirs('data/processed', exist_ok=True)
        
        # Save cleaned data
        self.df.to_csv(output_path, index=False)
        logging.info(f"Cleaned data saved to: {output_path}")
        print(f"✓ Saved to: {output_path}")
        
        # Update statistics
        self.cleaning_stats['rows_after_cleaning'] = len(self.df)
        self.cleaning_stats['rows_removed'] = (
            self.cleaning_stats['original_rows'] - 
            self.cleaning_stats['rows_after_cleaning']
        )
        self.cleaning_stats['retention_rate'] = round(
            (self.cleaning_stats['rows_after_cleaning'] / 
             self.cleaning_stats['original_rows'] * 100), 2
        )
        self.cleaning_stats['missing_values_after'] = self.df.isnull().sum().to_dict()
        
        # Save statistics
        stats_path = 'data/processed/cleaning_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(self.cleaning_stats, f, indent=4, default=str)
        
        logging.info("Cleaning statistics saved")
        print(f"✓ Statistics saved to: {stats_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("DATA CLEANING SUMMARY")
        print("="*60)
        print(f"Original rows: {self.cleaning_stats['original_rows']:,}")
        print(f"Cleaned rows: {self.cleaning_stats['rows_after_cleaning']:,}")
        print(f"Rows removed: {self.cleaning_stats['rows_removed']:,}")
        print(f"Retention rate: {self.cleaning_stats['retention_rate']:.2f}%")
        print("="*60)
        
        # Validation
        if self.cleaning_stats['retention_rate'] < 50:
            print("\n⚠️  WARNING: Retention rate < 50%")
            print("   Consider reviewing outlier thresholds")
        elif self.cleaning_stats['retention_rate'] > 80:
            print("\n⚠️  WARNING: Retention rate > 80%")
            print("   May not have removed enough outliers")
        else:
            print("\n✓ Retention rate is within expected range (50-80%)")
        
        return self
    
    def run_pipeline(self):
        """
        Execute complete cleaning pipeline
        """
        print("\n" + "="*60)
        print("STARTING DATA CLEANING PIPELINE")
        print("="*60)
        
        self.load_data()
        self.remove_missing_customer_ids()
        self.handle_cancelled_invoices()
        self.handle_negative_quantities()
        self.handle_zero_prices()
        self.handle_missing_descriptions()
        self.remove_outliers()
        self.remove_duplicates()
        self.add_derived_columns()
        self.convert_data_types()
        self.save_cleaned_data()
        
        print("\n✓ Data cleaning pipeline completed successfully!")
        
        return self.df

def main():
    """
    Main execution function
    """
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run cleaning pipeline
    cleaner = DataCleaner('data/raw/online_retail.csv')
    cleaned_df = cleaner.run_pipeline()
    
    print(f"\nCleaned dataset shape: {cleaned_df.shape}")
    print(f"Columns: {list(cleaned_df.columns)}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review data/processed/cleaning_statistics.json")
    print("2. Run notebooks/02_data_validation.ipynb")
    print("3. Proceed to Phase 4: Feature Engineering")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
