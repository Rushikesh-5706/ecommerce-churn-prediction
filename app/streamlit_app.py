"""
Customer Churn Prediction - Streamlit Web Application

This application provides:
1. Single customer churn prediction
2. Batch prediction via CSV upload
3. Interactive dashboard with visualizations
4. Model performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .prediction-low {
        color: #2ecc71;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler"""
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

@st.cache_data
def load_feature_names():
    """Load feature names"""
    import json
    with open('data/processed/feature_names.json', 'r') as f:
        feature_info = json.load(f)
    return feature_info['feature_names']

def predict_churn(customer_data, model, scaler, feature_names):
    """Predict churn for a single customer"""
    # Ensure all features are present
    for feature in feature_names:
        if feature not in customer_data:
            customer_data[feature] = 0
    
    # Create DataFrame with correct feature order
    df = pd.DataFrame([customer_data])[feature_names]
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    
    return prediction, probability

def main():
    # Load model
    model, scaler = load_model_and_scaler()
    feature_names = load_feature_names()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", "📈 Dashboard", "ℹ️ About"])
    
    # HOME PAGE
    if page == "🏠 Home":
        st.markdown('<h1 class="main-header">Customer Churn Prediction System</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Welcome to the Customer Churn Prediction Platform
        
        This application predicts which customers are likely to churn (stop purchasing) based on their behavioral patterns.
        
        **Features:**
        - 🔮 **Single Prediction**: Predict churn for individual customers
        - 📊 **Batch Prediction**: Upload CSV file for bulk predictions
        - 📈 **Dashboard**: Interactive visualizations and insights
        - 🎯 **High Accuracy**: Trained on 3,213 customers with 33 features
        
        **Model Performance:**
        - **ROC-AUC Score**: 0.7517
        - **Precision**: 0.69
        - **Recall**: 0.75
        
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers Analyzed", "2,249")
        with col2:
            st.metric("Features Used", "39")
        with col3:
            st.metric("Model Accuracy", "70.3%")
        
        st.info("👈 Use the sidebar to navigate between different sections")
    
    # SINGLE PREDICTION PAGE
    elif page == "🔮 Single Prediction":
        st.title("Single Customer Churn Prediction")
        st.markdown("Enter customer information to predict churn probability")
        
        with st.form("prediction_form"):
            st.subheader("RFM Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                recency = st.number_input("Recency (days since last purchase)", 0, 365, 30)
                frequency = st.number_input("Frequency (number of purchases)", 1, 100, 5)
            
            with col2:
                total_spent = st.number_input("Total Spent (£)", 0.0, 10000.0, 500.0)
                avg_order_value = st.number_input("Avg Order Value (£)", 0.0, 1000.0, 100.0)
            
            with col3:
                unique_products = st.number_input("Unique Products", 1, 200, 20)
                total_items = st.number_input("Total Items", 1, 1000, 50)
            
            st.subheader("Behavioral Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_days_between = st.number_input("Avg Days Between Purchases", 1, 365, 30)
                avg_basket_size = st.number_input("Avg Basket Size", 1, 100, 10)
            
            with col2:
                customer_lifetime = st.number_input("Customer Lifetime (days)", 1, 365, 180)
                purchase_velocity = st.number_input("Purchase Velocity", 0.0, 1.0, 0.1)
            
            with col3:
                purchases_last_30 = st.number_input("Purchases (Last 30 Days)", 0, 20, 1)
                segment = st.selectbox("Customer Segment", ["Champions", "Loyal", "Potential", "At Risk", "Lost"])
            
            submitted = st.form_submit_button("🔮 Predict Churn")
            
            if submitted:
                # Create customer data
                customer_data = {
                    'Recency': recency,
                    'Frequency': frequency,
                    'TotalSpent': total_spent,
                    'AvgOrderValue': avg_order_value,
                    'UniqueProducts': unique_products,
                    'TotalItems': total_items,
                    'AvgDaysBetweenPurchases': avg_days_between,
                    'AvgBasketSize': avg_basket_size,
                    'StdBasketSize': avg_basket_size * 0.3,
                    'MaxBasketSize': avg_basket_size * 2,
                    'PreferredDay': 3,
                    'PreferredHour': 14,
                    'CountryDiversity': 1,
                    'CustomerLifetimeDays': customer_lifetime,
                    'PurchaseVelocity': purchase_velocity,
                    'Purchases_Last30Days': purchases_last_30,
                    'Purchases_Last60Days': purchases_last_30 * 2,
                    'Purchases_Last90Days': purchases_last_30 * 3,
                    'ProductDiversityScore': unique_products / total_items if total_items > 0 else 0,
                    'AvgPricePreference': avg_order_value / avg_basket_size if avg_basket_size > 0 else 0,
                    'StdPricePreference': 5.0,
                    'MinPrice': 1.0,
                    'MaxPrice': 50.0,
                    'AvgQuantityPerOrder': avg_basket_size,
                    'RecencyScore': 3,
                    'FrequencyScore': 3,
                    'MonetaryScore': 3,
                    'RFM_Score': 9,
                    
                    # Missing Interaction/Trend Features (Crucial for correct prediction)
                    'FrequencyTrend': 1.0,  # Assume stable behavior
                    'SpendTrend': 1.0,      # Assume stable spending
                    'Freq_x_Spend': frequency * total_spent,
                    'Active_Freq': frequency / (recency + 1),
                    'Spend_per_Item': total_spent / (total_items + 1) if total_items > 0 else 0,
                }
                
                # Add segment one-hot encoding
                for seg in ["Champions", "Loyal", "Potential", "At Risk", "Lost"]:
                    customer_data[f'Segment_{seg}'] = 1 if seg == segment else 0
                
                # Predict
                prediction, probability = predict_churn(customer_data, model, scaler, feature_names)
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown('<p class="prediction-high">⚠️ HIGH CHURN RISK</p>', unsafe_allow_html=True)
                        st.error(f"This customer is likely to churn")
                    else:
                        st.markdown('<p class="prediction-low">✅ LOW CHURN RISK</p>', unsafe_allow_html=True)
                        st.success(f"This customer is likely to remain active")
                
                with col2:
                    st.metric("Churn Probability", f"{probability*100:.1f}%")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("📋 Recommendations")
                if prediction == 1:
                    st.warning("""
                    **Retention Strategies:**
                    - Send personalized re-engagement email
                    - Offer exclusive discount (10-15%)
                    - Provide loyalty rewards
                    - Conduct satisfaction survey
                    - Assign dedicated account manager
                    """)
                else:
                    st.info("""
                    **Engagement Strategies:**
                    - Continue current engagement level
                    - Introduce new products
                    - Request product reviews
                    - Offer referral incentives
                    """)
    
    # BATCH PREDICTION PAGE
    elif page == "📊 Batch Prediction":
        st.title("Batch Churn Prediction")
        st.markdown("Upload a CSV file with customer data for bulk predictions")
        
        st.info("""
        **Required CSV Format:**
        Your CSV should contain columns: `Recency`, `Frequency`, `TotalSpent`, `AvgOrderValue`, `UniqueProducts`, `TotalItems`, etc.
        
        [Download Sample CSV Template](#)
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✓ Loaded {len(df)} customers")
                
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🔮 Predict Churn for All Customers"):
                    with st.spinner("Making predictions..."):
                        # Ensure all required features are present
                        for feature in feature_names:
                            if feature not in df.columns:
                                df[feature] = 0
                        
                        # Scale and predict
                        X = df[feature_names]
                        X_scaled = scaler.transform(X)
                        
                        predictions = model.predict(X_scaled)
                        probabilities = model.predict_proba(X_scaled)[:, 1]
                        
                        df['Churn_Prediction'] = predictions
                        df['Churn_Probability'] = probabilities
                        df['Risk_Level'] = df['Churn_Probability'].apply(
                            lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.4 else 'Low')
                        )
                    
                    st.success("✓ Predictions completed!")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Customers", len(df))
                    with col2:
                        st.metric("Predicted Churners", int(predictions.sum()))
                    with col3:
                        st.metric("Churn Rate", f"{predictions.mean()*100:.1f}%")
                    with col4:
                        st.metric("High Risk", len(df[df['Risk_Level'] == 'High']))
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(df[['Churn_Prediction', 'Churn_Probability', 'Risk_Level']], use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    fig = px.histogram(df, x='Churn_Probability', nbins=30,
                                     title="Distribution of Churn Probabilities")
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # DASHBOARD PAGE
    elif page == "📈 Dashboard":
        st.title("Analytics Dashboard")
        
        # Load visualizations
        viz_dir = "visualizations"
        
        if os.path.exists(viz_dir):
            st.subheader("Model Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists(f"{viz_dir}/01_roc_curve.png"):
                    img = Image.open(f"{viz_dir}/01_roc_curve.png")
                    st.image(img, caption="ROC Curve", use_container_width=True)
            
            with col2:
                if os.path.exists(f"{viz_dir}/03_confusion_matrix.png"):
                    img = Image.open(f"{viz_dir}/03_confusion_matrix.png")
                    st.image(img, caption="Confusion Matrix", use_container_width=True)
            
            st.subheader("Feature Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists(f"{viz_dir}/04_feature_importance.png"):
                    img = Image.open(f"{viz_dir}/04_feature_importance.png")
                    st.image(img, caption="Feature Importance", use_container_width=True)
            
            with col2:
                if os.path.exists(f"{viz_dir}/05_model_comparison.png"):
                    img = Image.open(f"{viz_dir}/05_model_comparison.png")
                    st.image(img, caption="Model Comparison", use_container_width=True)
            
            st.subheader("Churn Analysis")
            
            if os.path.exists(f"{viz_dir}/06_churn_analysis.png"):
                img = Image.open(f"{viz_dir}/06_churn_analysis.png")
                st.image(img, caption="Churn Distribution Analysis", use_container_width=True)
        else:
            st.warning("Visualizations not found. Please run the visualization script first.")
    
    # ABOUT PAGE
    elif page == "ℹ️ About":
        st.title("About This Application")
        
        st.markdown("""
        ### Customer Churn Prediction System
        
        **Version:** 1.0  
        **Last Updated:** February 2026
        
        #### What is Customer Churn?
        Customer churn occurs when customers stop purchasing from a business. For this system, 
        churn is defined as no purchase activity in the last 90 days.
        
        #### How It Works
        1. **Data Collection**: Historical customer transaction data (2009-2010)
        2. **Feature Engineering**: 33 features created from RFM analysis, behavioral patterns, and temporal data
        3. **Model Training**: Ensemble of Random Forest, Gradient Boosting, and Neural Network
        4. **Prediction**: Real-time churn probability calculation
        
        #### Model Performance
        - **ROC-AUC Score**: 0.7517
        - **Accuracy**: 70.3%
        - **Precision**: 69%
        - **Recall**: 75%
        
        #### Dataset Statistics
        - **Training Customers**: 2,249
        - **Validation Customers**: 482
        - **Test Customers**: 482
        - **Features**: 33
        - **Churn Rate**: 41.9%
        
        #### Technology Stack
        - **ML Framework**: Scikit-learn
        - **Web Framework**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly, Matplotlib, Seaborn
        
        #### Contact & Support
        For questions or support, please contact the development team.
        
        ---
        
        *Built with ❤️ for customer retention*
        """)

if __name__ == "__main__":
    main()
