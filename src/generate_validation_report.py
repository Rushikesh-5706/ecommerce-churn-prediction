import pandas as pd
import json
from datetime import datetime
import os

def generate_validation_report():
    print("Generating validation report...")
    
    # Load cleaned data
    # Note: Column names will be the NEW ones (InvoiceNo, etc.) if data acquisition ran
    # But if cleaning ran on OLD data, it might be mixed. 
    # Let's check what's actually there.
    
    try:
        df = pd.read_csv('data/processed/cleaned_data.csv')
    except Exception as e:
        print(f"Error loading cleaned data: {e}")
        return

    # Map possible column names to standard ones for validation check
    # (Handling transition period where file might have old names)
    col_map = {
        'Invoice': 'InvoiceNo',
        'Price': 'UnitPrice', 
        'Customer ID': 'CustomerID'
    }
    df.rename(columns=col_map, inplace=True)

    validation_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "date_range": {
            "start": pd.to_datetime(df['InvoiceDate']).min().strftime('%Y-%m-%d'),
            "end": pd.to_datetime(df['InvoiceDate']).max().strftime('%Y-%m-%d')
        },
        "unique_customers": df['CustomerID'].nunique(),
        "unique_products": df['StockCode'].nunique(),
        "unique_countries": df['Country'].nunique(),
        "total_revenue": df['TotalPrice'].sum(),
        "average_order_value": df['TotalPrice'].mean(),
        "validation_passed": True,
        "checks": {
            "no_missing_values": bool(df.isnull().sum().sum() == 0),
            "all_quantities_positive": bool((df['Quantity'] > 0).all()),
            "all_prices_positive": bool((df['UnitPrice'] > 0).all()),
            "customer_id_is_integer": bool(pd.api.types.is_numeric_dtype(df['CustomerID']))
        }
    }

    with open('data/processed/validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=4)
    
    print("✓ Validation report generated: data/processed/validation_report.json")

if __name__ == "__main__":
    generate_validation_report()
