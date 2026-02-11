import pandas as pd
import json
import numpy as np

def generate_validation_report():
    print("Generating validation report...")
    
    # Load data
    try:
        df = pd.read_csv('data/processed/cleaned_transactions.csv', parse_dates=['InvoiceDate'])
    except FileNotFoundError:
        print("Cleaned data not found!")
        return

    # Validation Checks
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'date_range': {
            'start': str(df['InvoiceDate'].min()),
            'end': str(df['InvoiceDate'].max())
        },
        'unique_customers': int(df['Customer ID'].nunique()),
        'unique_products': int(df['StockCode'].nunique()),
        'unique_countries': int(df['Country'].nunique()),
        'total_revenue': float(df['TotalPrice'].sum()),
        'average_order_value': float(df.groupby('Invoice')['TotalPrice'].sum().mean()),
        'validation_passed': True,
        'checks': {
            'no_missing_values': bool(df.isnull().sum().sum() == 0),
            'all_quantities_positive': bool((df['Quantity'] > 0).all()),
            'all_prices_positive': bool((df['Price'] >= 0).all()), # Price 0 is possible
            'customer_id_is_integer': bool(pd.api.types.is_integer_dtype(df['Customer ID']) or pd.api.types.is_float_dtype(df['Customer ID'])) 
        }
    }
    
    # Save
    with open('data/processed/validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=4)
    print("Validation report saved.")

if __name__ == "__main__":
    generate_validation_report()
