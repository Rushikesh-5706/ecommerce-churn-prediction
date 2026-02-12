"""
Model Evaluation and Visualization Script

Creates all required visualizations for model evaluation:
1. ROC Curve
2. Precision-Recall Curve
3. Confusion Matrix
4. Feature Importance
5. Learning Curves
6. Model Comparison Chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import learning_curve
from scipy import stats
import json
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def load_data_and_model():
    """Load test data and best model"""
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    # Use tuned Random Forest (best individual model)
    model = joblib.load('models/random_forest_tuned.pkl')
    
    return X_train, y_train, X_test, y_test, model

def create_roc_curve(model, X_test, y_test):
    """Create ROC Curve"""
    print("Creating ROC Curve...")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/01_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to visualizations/01_roc_curve.png")

def create_precision_recall_curve(model, X_test, y_test):
    """Create Precision-Recall Curve"""
    print("Creating Precision-Recall Curve...")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('visualizations/02_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to visualizations/02_precision_recall_curve.png")

def create_confusion_matrix(model, X_test, y_test):
    """Create Confusion Matrix"""
    print("Creating Confusion Matrix...")
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Active', 'Churned'],
                yticklabels=['Active', 'Churned'])
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.savefig('visualizations/03_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to visualizations/03_confusion_matrix.png")

def create_feature_importance(model, X_train):
    """Create Feature Importance Plot"""
    print("Creating Feature Importance Plot...")
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Top 20 features
    top_n = 20
    top_indices = indices[:top_n]
    top_features = [X_train.columns[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), top_importances[::-1], align='center', color='steelblue')
    plt.yticks(range(top_n), top_features[::-1])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('visualizations/04_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to visualizations/04_feature_importance.png")

def create_model_comparison():
    """Create Model Comparison Chart"""
    print("Creating Model Comparison Chart...")
    
    # Load comparison data
    comparison_df = pd.read_csv('models/model_comparison.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Val_ROC_AUC', 'Val_Precision', 'Val_Recall', 'Val_F1']
    titles = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        comparison_sorted = comparison_df.sort_values(metric)
        
        ax.barh(comparison_sorted['Model'], comparison_sorted[metric], color='teal')
        ax.set_xlabel(title, fontsize=11)
        ax.set_title(f'Model Comparison: {title}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(comparison_sorted[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/05_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to visualizations/05_model_comparison.png")

def create_churn_distribution():
    """Create Churn Distribution Visualization"""
    print("Creating Churn Distribution...")
    
    # Load customer features
    df = pd.read_csv('data/processed/customer_features.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Churn Distribution
    ax = axes[0, 0]
    churn_counts = df['Churn'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    ax.pie(churn_counts, labels=['Active', 'Churned'], autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title('Churn Distribution', fontsize=12, fontweight='bold')
    
    # 2. Recency by Churn
    ax = axes[0, 1]
    df.boxplot(column='Recency', by='Churn', ax=ax)
    ax.set_title('Recency by Churn Status', fontsize=12, fontweight='bold')
    ax.set_xlabel('Churn Status')
    ax.set_ylabel('Recency (days)')
    plt.sca(ax)
    plt.xticks([1, 2], ['Active', 'Churned'])
    
    # 3. CustomerSegment Distribution
    ax = axes[1, 0]
    segment_churn = pd.crosstab(df['CustomerSegment'], df['Churn'], normalize='index')
    segment_churn.plot(kind='bar', stacked=True, ax=ax, color=colors)
    ax.set_title('Churn Rate by Customer Segment', fontsize=12, fontweight='bold')
    ax.set_xlabel('Customer Segment')
    ax.set_ylabel('Proportion')
    ax.legend(['Active', 'Churned'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 4. RFM Score Distribution
    ax = axes[1, 1]
    churned = df[df['Churn'] == 1]['RFM_Score']
    active = df[df['Churn'] == 0]['RFM_Score']
    ax.hist([active, churned], bins=10, label=['Active', 'Churned'], 
            color=['#2ecc71', '#e74c3c'], alpha=0.7)
    ax.set_title('RFM Score Distribution by Churn', fontsize=12, fontweight='bold')
    ax.set_xlabel('RFM Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    plt.suptitle('')  # Remove auto title
    plt.tight_layout()
    plt.savefig('visualizations/06_churn_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to visualizations/06_churn_analysis.png")

def ensure_eda_dir():
    """Ensure EDA directory exists"""
    os.makedirs('visualizations/eda', exist_ok=True)


def create_recency_plot(df):
    """Create Recency Distribution Check"""
    print("Creating Recency Plot...")
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(x='Churn', y='Recency', data=df, palette=['#2ecc71', '#e74c3c'])
    plt.title('Recency Distribution by Churn Status', fontsize=14, fontweight='bold')
    plt.xlabel('Churn Status (0=Active, 1=Churned)', fontsize=12)
    plt.ylabel('Recency (Days since last purchase)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    ensure_eda_dir()
    plt.savefig('visualizations/eda/06_recency_by_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to visualizations/eda/06_recency_by_churn.png")

def create_frequency_plot(df):
    """Create Frequency Distribution Check"""
    print("Creating Frequency Plot...")
    
    # Cap outliers for better visualization if needed, but boxplot handles them well
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='Frequency', data=df, palette=['#2ecc71', '#e74c3c'])
    plt.yscale('log') # Log scale often helps with frequency/monetary data
    plt.title('Frequency Distribution by Churn Status (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Churn Status (0=Active, 1=Churned)', fontsize=12)
    plt.ylabel('Frequency (Number of purchases)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    ensure_eda_dir()
    plt.savefig('visualizations/eda/07_frequency_by_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to visualizations/eda/07_frequency_by_churn.png")
    
def create_monetary_plot(df):
    """Create Monetary Distribution Check"""
    print("Creating Monetary Plot...")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='TotalSpent', data=df, palette=['#2ecc71', '#e74c3c'])
    plt.yscale('log')
    plt.title('Monetary Value Distribution by Churn Status (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Churn Status (0=Active, 1=Churned)', fontsize=12)
    plt.ylabel('Total Spent ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    ensure_eda_dir()
    plt.savefig('visualizations/eda/08_monetary_by_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to visualizations/eda/08_monetary_by_churn.png")

def create_statistical_tests(df):
    """Create Statistical Significance Visualization"""
    print("Creating Statistical Test Visualization...")
    
    features = ['Recency', 'Frequency', 'TotalSpent', 'AvgOrderValue', 'UniqueProducts']
    p_values = []
    
    for feature in features:
        churned = df[df['Churn'] == 1][feature]
        active = df[df['Churn'] == 0][feature]
        
        # Perform t-test (independent samples)
        t_stat, p_val = stats.ttest_ind(churned, active, equal_var=False)
        p_values.append(p_val)
    
    # Create DataFrame for plotting
    stats_df = pd.DataFrame({
        'Feature': features,
        'P-Value': p_values
    })
    
    # Sort by significance (p-value, ascending)
    stats_df = stats_df.sort_values('P-Value')
    
    plt.figure(figsize=(12, 6))
    
    # Bar plot of -log10(p-value) to make small values visible and significant ones large
    # Using -log10(p) is a common way to visualize significance (Manhattan plot style)
    # But for simplicity, we can just plot the p-values and add a significance line at 0.05
    
    # Let's plot 1 - p_value to show "Confidence" or just bar chart of p-values
    # Better yet: Bar chart of p-values with a red line at 0.05
    
    bars = plt.bar(stats_df['Feature'], stats_df['P-Value'], color='skyblue')
    
    # Highlight significant features
    for i, bar in enumerate(bars):
        if stats_df.iloc[i]['P-Value'] < 0.05:
            bar.set_color('#2ecc71') # Green for significant
        else:
            bar.set_color('#95a5a6') # Grey for not significant
            
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (Alpha = 0.05)')
    
    plt.title('Statistical Significance of Features (T-Test P-Values)', fontsize=14, fontweight='bold')
    plt.ylabel('P-Value', fontsize=12)
    plt.xlabel('Feature', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(stats_df['P-Value']):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        
    ensure_eda_dir()
    plt.savefig('visualizations/eda/09_statistical_tests.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved to visualizations/eda/09_statistical_tests.png")

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Load data and model
    X_train, y_train, X_test, y_test, model = load_data_and_model()
    
    # Create all visualizations
    create_roc_curve(model, X_test, y_test)
    create_precision_recall_curve(model, X_test, y_test)
    create_confusion_matrix(model, X_test, y_test)
    create_feature_importance(model, X_train)
    create_model_comparison()
    create_churn_distribution()

    # Load full dataset for RFM and Stats analysis
    df = pd.read_csv('data/processed/customer_features.csv')
    create_recency_plot(df)
    create_frequency_plot(df)
    create_monetary_plot(df)
    create_statistical_tests(df)
    
    print("\n" + "="*60)
    print("VISUALIZATION CREATION COMPLETED")
    print("="*60)
    print("\n✓ 10 visualizations saved to visualizations/ directory (including eda/ subdirectory)")
    print("\nFiles created:")
    print("  1. 01_roc_curve.png")
    print("  2. 02_precision_recall_curve.png")
    print("  3. 03_confusion_matrix.png")
    print("  4. 04_feature_importance.png")
    print("  5. 05_model_comparison.png")
    print("  6. 06_churn_analysis.png")
    print("  7. eda/06_recency_by_churn.png")
    print("  8. eda/07_frequency_by_churn.png")
    print("  9. eda/08_monetary_by_churn.png")
    print("  10. eda/09_statistical_tests.png")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
