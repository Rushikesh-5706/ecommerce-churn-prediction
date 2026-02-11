"""
Prediction API Module

Provides functions for making churn predictions on new customer data.
Can be imported and used by other applications.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import json

class ChurnPredictor:
    """
    Churn prediction API wrapper
    
    Example:
        predictor = ChurnPredictor()
        prediction, probability = predictor.predict_single(customer_data)
    """
    
    def __init__(self, model_path='models/best_model.pkl', scaler_path='models/scaler.pkl'):
        """
        Initialize predictor with trained model and scaler
        
        Args:
            model_path: Path to saved model file
            scaler_path: Path to saved scaler file
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load feature names
        with open('data/processed/feature_names.json', 'r') as f:
            feature_info = json.load(f)
        self.feature_names = feature_info['feature_names']
    
    def predict_single(self, customer_data: Dict) -> Tuple[int, float]:
        """
        Predict churn for a single customer
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            (prediction, probability): Binary prediction (0/1) and churn probability (0-1)
        """
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in customer_data:
                customer_data[feature] = 0
        
        # Create DataFrame with correct feature order
        df = pd.DataFrame([customer_data])[self.feature_names]
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        
        # Predict
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0][1]
        
        return int(prediction), float(probability)
    
    def predict_batch(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn for multiple customers
        
        Args:
            customer_df: DataFrame with customer features
            
        Returns:
            DataFrame with added columns: Churn_Prediction, Churn_Probability, Risk_Level
        """
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in customer_df.columns:
                customer_df[feature] = 0
        
        # Scale and predict
        X = customer_df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Add results to dataframe
        result_df = customer_df.copy()
        result_df['Churn_Prediction'] = predictions
        result_df['Churn_Probability'] = probabilities
        result_df['Risk_Level'] = result_df['Churn_Probability'].apply(
            lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.4 else 'Low')
        )
        
        return result_df
    
    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get feature importances from the model
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = list(zip(self.feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            return feature_importance[:top_n]
        else:
            return []

def predict_customer_churn(recency: int, frequency: int, total_spent: float, 
                          avg_order_value: float, unique_products: int, 
                          total_items: int, **kwargs) -> Dict:
    """
    Convenience function for quick churn prediction
    
    Args:
        recency: Days since last purchase
        frequency: Number of purchases
        total_spent: Total amount spent
        avg_order_value: Average order value
        unique_products: Number of unique products
        total_items: Total items purchased
        **kwargs: Additional customer features
        
    Returns:
        Dictionary with prediction results
    """
    predictor = ChurnPredictor()
    
    customer_data = {
        'Recency': recency,
        'Frequency': frequency,
        'TotalSpent': total_spent,
        'AvgOrderValue': avg_order_value,
        'UniqueProducts': unique_products,
        'TotalItems': total_items,
        **kwargs
    }
    
    prediction, probability = predictor.predict_single(customer_data)
    
    return {
        'will_churn': bool(prediction),
        'churn_probability': probability,
        'risk_level': 'High' if probability > 0.7 else ('Medium' if probability > 0.4 else 'Low'),
        'recommendation': get_recommendation(prediction, probability)
    }

def get_recommendation(prediction: int, probability: float) -> str:
    """
    Get retention recommendation based on prediction
    
    Args:
        prediction: Binary churn prediction (0/1)
        probability: Churn probability (0-1)
        
    Returns:
        Recommendation string
    """
    if prediction == 1:
        if probability > 0.8:
            return "HIGH RISK: Immediate intervention required. Offer significant discount and personal contact."
        elif probability > 0.6:
            return "MEDIUM-HIGH RISK: Send re-engagement email with exclusive offer."
        else:
            return "MODERATE RISK: Monitor closely and encourage with loyalty rewards."
    else:
        if probability > 0.3:
            return "LOW RISK: Continue current engagement. Consider upsell opportunities."
        else:
            return "VERY LOW RISK: Excellent customer. Reward with VIP perks and referral incentives."

if __name__ == "__main__":
    # Example usage
    predictor = ChurnPredictor()
    
    # Single prediction
    customer = {
        'Recency': 45,
        'Frequency': 3,
        'TotalSpent': 250.0,
        'AvgOrderValue': 83.33,
        'UniqueProducts': 15,
        'TotalItems': 30
    }
    
    prediction, probability = predictor.predict_single(customer)
    print(f"Churn Prediction: {prediction}")
    print(f"Churn Probability: {probability:.2%}")
    print(f"Recommendation: {get_recommendation(prediction, probability)}")
