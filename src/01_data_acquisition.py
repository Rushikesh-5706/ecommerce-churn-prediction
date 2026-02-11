"""
Data Acquisition Script for Customer Churn Prediction System

This script downloads the UCI Online Retail dataset and performs initial data profiling.

Dataset: UCI Online Retail II
Source: http://archive.ics.uci.edu/ml/machine-learning-databases/00502/
Alternative: https://www.kaggle.com/datasets/carrie1/ecommerce-data

Expected Output:
- data/raw/online_retail.csv
- data/raw/data_quality_summary.json
- data/raw/data_profile.txt
"""

import pandas as pd
import requests
import os
import json
from datetime import datetime
import sys

def download_dataset():
    """
    Download the Online Retail dataset from UCI ML Repository
    Save to data/raw/online_retail.csv
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("="*60)
    print("DOWNLOADING UCI ONLINE RETAIL DATASET")
    print("="*60)
    
    # Create directory structure
    os.makedirs('data/raw', exist_ok=True)
    
    # Primary URL (UCI ML Repository)
    primary_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
    
    # Alternative URLs
    alternative_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx",
        # Note: Kaggle requires authentication, so we'll provide instructions if UCI fails
    ]
    
    output_path = 'data/raw/online_retail.xlsx'
    csv_path = 'data/raw/online_retail.csv'
    
    # Check if file already exists
    if os.path.exists(csv_path):
        print(f"✓ Dataset already exists at: {csv_path}")
        return True
    
    print(f"\nAttempting to download from UCI ML Repository...")
    print(f"URL: {primary_url}")
    print(f"File size: ~45MB (this may take 2-3 minutes)")
    
    try:
        # Try primary URL
        response = requests.get(primary_url, timeout=60)
        response.raise_for_status()
        
        # Save Excel file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded successfully to: {output_path}")
        
        # Convert to CSV for easier processing
        print("\nConverting Excel to CSV...")
        df = pd.read_excel(output_path, engine='openpyxl')
        df.to_csv(csv_path, index=False)
        print(f"✓ Converted to CSV: {csv_path}")
        
        # Remove Excel file to save space
        os.remove(output_path)
        
        print(f"\n✓ Dataset downloaded: {datetime.now()}")
        print(f"✓ Saved to: {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download from primary URL: {e}")
        
        # Try alternative URLs
        for alt_url in alternative_urls:
            try:
                print(f"\nTrying alternative URL: {alt_url}")
                response = requests.get(alt_url, timeout=60)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✓ Downloaded successfully")
                
                # Convert to CSV
                df = pd.read_excel(output_path, engine='openpyxl')
                df.to_csv(csv_path, index=False)
                os.remove(output_path)
                
                print(f"✓ Converted to CSV: {csv_path}")
                return True
                
            except Exception as alt_e:
                print(f"✗ Failed: {alt_e}")
                continue
        
        # If all URLs fail, provide manual instructions
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*60)
        print("\nAutomatic download failed. Please download manually:")
        print("\nOption 1: UCI ML Repository")
        print("  1. Visit: http://archive.ics.uci.edu/ml/datasets/Online+Retail+II")
        print("  2. Download 'online_retail_II.xlsx'")
        print(f"  3. Save to: {os.path.abspath(output_path)}")
        print("\nOption 2: Kaggle (requires account)")
        print("  1. Visit: https://www.kaggle.com/datasets/carrie1/ecommerce-data")
        print("  2. Download 'data.csv'")
        print(f"  3. Rename to 'online_retail.csv' and save to: {os.path.abspath('data/raw/')}")
        print("\nAfter manual download, run this script again.")
        print("="*60)
        
        return False

def load_raw_data():
    """
    Load the raw dataset and return DataFrame
    
    Returns:
        pd.DataFrame: Raw dataset
    """
    csv_path = 'data/raw/online_retail.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ Error: Dataset not found at {csv_path}")
        print("Please run download_dataset() first or download manually.")
        return None
    
    print("\nLoading dataset...")
    
    try:
        # Try reading with different encodings
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying latin1...")
            df = pd.read_csv(csv_path, encoding='latin1')
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None

def generate_data_profile(df):
    """
    Generate initial data profile and save to data/raw/data_profile.txt
    
    Args:
        df (pd.DataFrame): Raw dataset
    
    Include:
    - Number of rows and columns
    - Column names and types
    - Memory usage
    - First few rows preview
    """
    if df is None:
        print("✗ Cannot generate profile: DataFrame is None")
        return
    
    print("\nGenerating data profile...")
    
    profile_path = 'data/raw/data_profile.txt'
    
    with open(profile_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("UCI ONLINE RETAIL DATASET - DATA PROFILE\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*60 + "\n\n")
        
        # Dataset shape
        f.write("DATASET SHAPE\n")
        f.write("-"*60 + "\n")
        f.write(f"Rows: {df.shape[0]:,}\n")
        f.write(f"Columns: {df.shape[1]}\n\n")
        
        # Column information
        f.write("COLUMN INFORMATION\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Column Name':<20} {'Data Type':<15} {'Non-Null Count':<15}\n")
        f.write("-"*60 + "\n")
        for col in df.columns:
            f.write(f"{col:<20} {str(df[col].dtype):<15} {df[col].count():>14,}\n")
        f.write("\n")
        
        # Memory usage
        f.write("MEMORY USAGE\n")
        f.write("-"*60 + "\n")
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        f.write(f"Total: {memory_mb:.2f} MB\n\n")
        
        # First 10 rows
        f.write("FIRST 10 ROWS\n")
        f.write("-"*60 + "\n")
        f.write(df.head(10).to_string())
        f.write("\n\n")
        
        # Basic statistics
        f.write("BASIC STATISTICS\n")
        f.write("-"*60 + "\n")
        f.write(df.describe(include='all').to_string())
        f.write("\n")
    
    print(f"✓ Data profile saved to: {profile_path}")

def generate_data_quality_summary(df):
    """
    Generate data quality summary and save to JSON
    
    Args:
        df (pd.DataFrame): Raw dataset
    """
    if df is None:
        print("✗ Cannot generate quality summary: DataFrame is None")
        return
    
    print("\nGenerating data quality summary...")
    
    # Calculate statistics
    missing_values = df.isnull().sum().to_dict()
    
    # Detect cancelled invoices (InvoiceNo starting with 'C')
    if 'InvoiceNo' in df.columns:
        cancelled_invoices = df['InvoiceNo'].astype(str).str.startswith('C').sum()
    else:
        cancelled_invoices = 0
    
    # Detect negative quantities
    if 'Quantity' in df.columns:
        negative_quantities = (df['Quantity'] < 0).sum()
    else:
        negative_quantities = 0
    
    # Missing CustomerID percentage
    if 'CustomerID' in df.columns:
        missing_customer_ids = df['CustomerID'].isnull().sum()
        missing_customer_ids_pct = (missing_customer_ids / len(df)) * 100
    else:
        missing_customer_ids = 0
        missing_customer_ids_pct = 0
    
    # Date range
    if 'InvoiceDate' in df.columns:
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            date_range = {
                'start': df['InvoiceDate'].min().strftime('%Y-%m-%d'),
                'end': df['InvoiceDate'].max().strftime('%Y-%m-%d')
            }
        except:
            date_range = {'start': 'Unknown', 'end': 'Unknown'}
    else:
        date_range = {'start': 'Unknown', 'end': 'Unknown'}
    
    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    
    summary = {
        'total_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'missing_values': {k: int(v) for k, v in missing_values.items()},
        'duplicate_rows': int(duplicate_rows),
        'date_range': date_range,
        'negative_quantities': int(negative_quantities),
        'cancelled_invoices': int(cancelled_invoices),
        'missing_customer_ids': int(missing_customer_ids),
        'missing_customer_ids_percentage': round(float(missing_customer_ids_pct), 2)
    }
    
    # Save to JSON
    json_path = 'data/raw/data_quality_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"✓ Data quality summary saved to: {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATA QUALITY SUMMARY")
    print("="*60)
    print(f"Total Rows: {summary['total_rows']:,}")
    print(f"Total Columns: {summary['total_columns']}")
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Missing CustomerIDs: {summary['missing_customer_ids']:,} ({summary['missing_customer_ids_percentage']:.2f}%)")
    print(f"Cancelled Invoices: {summary['cancelled_invoices']:,}")
    print(f"Negative Quantities: {summary['negative_quantities']:,}")
    print(f"Duplicate Rows: {summary['duplicate_rows']:,}")
    print("="*60)

def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("CUSTOMER CHURN PREDICTION - DATA ACQUISITION")
    print("="*60)
    
    # Step 1: Download dataset
    success = download_dataset()
    
    if not success:
        print("\n✗ Data acquisition failed. Please download manually and run again.")
        sys.exit(1)
    
    # Step 2: Load dataset
    df = load_raw_data()
    
    if df is None:
        print("\n✗ Failed to load dataset.")
        sys.exit(1)
    
    # Step 3: Generate data profile
    generate_data_profile(df)
    
    # Step 4: Generate data quality summary
    generate_data_quality_summary(df)
    
    print("\n" + "="*60)
    print("DATA ACQUISITION COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nNext Steps:")
    print("1. Review data/raw/data_profile.txt")
    print("2. Review data/raw/data_quality_summary.json")
    print("3. Run notebooks/01_initial_data_exploration.ipynb")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
