"""
Generate Additional EDA Visualizations (#10-#15)

Creates the missing EDA visualizations to meet the 15+ requirement.
Visualizations 1-9 already exist in visualizations/eda/.
This script adds visualizations 10-15.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os
import json

def generate_additional_eda_visualizations():
    """Generate visualizations #10 through #15."""
    
    # Load customer features
    features_path = 'data/processed/customer_features.csv'
    if not os.path.exists(features_path):
        print(f"ERROR: {features_path} not found. Run feature engineering first.")
        return
    
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} customers with {len(df.columns)} features")
    
    os.makedirs('visualizations/eda', exist_ok=True)
    
    # =========================================================================
    # VISUALIZATION 10: Purchase Velocity Distribution by Churn Status
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'PurchaseVelocity' in df.columns:
        vel_col = 'PurchaseVelocity'
    elif 'Frequency' in df.columns:
        vel_col = 'Frequency'
    else:
        vel_col = df.columns[2]  # fallback
    
    churned = df[df['Churn'] == 1][vel_col].dropna()
    active = df[df['Churn'] == 0][vel_col].dropna()
    
    ax.hist(active, bins=30, alpha=0.7, label='Active', color='#06A77D', edgecolor='black', linewidth=0.5)
    ax.hist(churned, bins=30, alpha=0.7, label='Churned', color='#E63946', edgecolor='black', linewidth=0.5)
    ax.set_xlabel(vel_col, fontsize=12, fontweight='bold')
    ax.set_ylabel('Customer Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{vel_col} Distribution by Churn Status', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/eda/10_purchase_velocity_by_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Visualization 10: Purchase Velocity Distribution by Churn')
    
    # =========================================================================
    # VISUALIZATION 11: Average Order Value Distribution
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'AvgOrderValue' in df.columns:
        aov_col = 'AvgOrderValue'
    elif 'AvgBasketValue' in df.columns:
        aov_col = 'AvgBasketValue'
    else:
        aov_col = 'TotalSpent'
    
    churned_aov = df[df['Churn'] == 1][aov_col].dropna()
    active_aov = df[df['Churn'] == 0][aov_col].dropna()
    
    # Clip extreme values for better visualization
    clip_upper = df[aov_col].quantile(0.95)
    
    ax.hist(active_aov.clip(upper=clip_upper), bins=30, alpha=0.7, label='Active', color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax.hist(churned_aov.clip(upper=clip_upper), bins=30, alpha=0.7, label='Churned', color='#F77F00', edgecolor='black', linewidth=0.5)
    ax.set_xlabel(f'{aov_col} (£)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Customer Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{aov_col} Distribution by Churn Status', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/eda/11_avg_order_value_by_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Visualization 11: Average Order Value Distribution by Churn')
    
    # =========================================================================
    # VISUALIZATION 12: Customer Lifetime Distribution
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'CustomerLifetimeDays' in df.columns:
        lt_col = 'CustomerLifetimeDays'
    elif 'DaysSinceFirst' in df.columns:
        lt_col = 'DaysSinceFirst'
    else:
        lt_col = 'Recency'
    
    churned_lt = df[df['Churn'] == 1][lt_col].dropna()
    active_lt = df[df['Churn'] == 0][lt_col].dropna()
    
    bp = ax.boxplot([active_lt, churned_lt], labels=['Active', 'Churned'], 
                    patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#06A77D')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#E63946')
    bp['boxes'][1].set_alpha(0.7)
    
    ax.set_ylabel(f'{lt_col}', fontsize=12, fontweight='bold')
    ax.set_title(f'{lt_col} by Churn Status', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean annotations
    for i, (data, label) in enumerate([(active_lt, 'Active'), (churned_lt, 'Churned')], 1):
        mean_val = data.mean()
        ax.annotate(f'Mean: {mean_val:.1f}', xy=(i, mean_val), fontsize=10,
                   fontweight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('visualizations/eda/12_customer_lifetime_by_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Visualization 12: Customer Lifetime Distribution by Churn')
    
    # =========================================================================
    # VISUALIZATION 13: RFM Scores Distribution
    # =========================================================================
    rfm_cols = []
    for col in ['RecencyScore', 'FrequencyScore', 'MonetaryScore']:
        if col in df.columns:
            rfm_cols.append(col)
    
    if len(rfm_cols) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors = ['#E63946', '#F77F00', '#06A77D']
        
        for idx, (col, color) in enumerate(zip(rfm_cols, colors)):
            axes[idx].hist(df[col], bins=4, color=color, edgecolor='black', alpha=0.8)
            axes[idx].set_xlabel(f'{col} (1-4)', fontweight='bold')
            axes[idx].set_ylabel('Customer Count', fontweight='bold')
            axes[idx].set_title(f'{col} Distribution', fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.suptitle('RFM Score Distributions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('visualizations/eda/13_rfm_scores_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Visualization 13: RFM Scores Distribution')
    else:
        print('⚠ Skipped Visualization 13: RFM score columns not found')
    
    # =========================================================================
    # VISUALIZATION 14: Churn Rate by Customer Segment
    # =========================================================================
    if 'CustomerSegment' in df.columns:
        segment_analysis = df.groupby('CustomerSegment').agg(
            Churned=('Churn', 'sum'),
            Total=('Churn', 'count'),
            ChurnRate=('Churn', 'mean')
        ).reset_index()
        segment_analysis = segment_analysis.sort_values('ChurnRate', ascending=False)
        
        colors = ['#E63946', '#F77F00', '#FCBF49', '#06A77D', '#118B6F']
        if len(segment_analysis) > len(colors):
            colors = colors * (len(segment_analysis) // len(colors) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(segment_analysis['CustomerSegment'], 
                     segment_analysis['ChurnRate'] * 100,
                     color=colors[:len(segment_analysis)], 
                     edgecolor='black', linewidth=1.5, alpha=0.85)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Churn Rate (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Customer Segment', fontsize=12, fontweight='bold')
        ax.set_title('Churn Rate by Customer Segment', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/eda/14_segment_churn_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Visualization 14: Churn Rate by Segment')
    else:
        print('⚠ Skipped Visualization 14: CustomerSegment column not found')
    
    # =========================================================================
    # VISUALIZATION 15: Recent Purchase Activity Comparison
    # =========================================================================
    recent_cols = []
    for col in ['Purchases_Last30Days', 'Purchases_Last60Days', 'Purchases_Last90Days']:
        if col in df.columns:
            recent_cols.append(col)
    
    if len(recent_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        x = np.arange(len(recent_cols))
        width = 0.35
        
        active_means = [df[df['Churn'] == 0][col].mean() for col in recent_cols]
        churned_means = [df[df['Churn'] == 1][col].mean() for col in recent_cols]
        
        bars1 = ax.bar(x - width/2, active_means, width, label='Active', 
                      color='#06A77D', edgecolor='black', linewidth=1.2, alpha=0.85)
        bars2 = ax.bar(x + width/2, churned_means, width, label='Churned',
                      color='#E63946', edgecolor='black', linewidth=1.2, alpha=0.85)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Time Window', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Purchases', fontsize=12, fontweight='bold')
        ax.set_title('Recent Purchase Activity: Active vs Churned Customers', fontsize=14, fontweight='bold')
        short_labels = [c.replace('Purchases_Last', 'Last ').replace('Days', ' Days') for c in recent_cols]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=11)
        ax.legend(fontsize=11, framealpha=0.95)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/eda/15_recent_purchase_activity.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ Visualization 15: Recent Purchase Activity Comparison')
    else:
        print('⚠ Skipped Visualization 15: Recent purchase columns not found')
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    eda_files = [f for f in os.listdir('visualizations/eda') if f.endswith('.png')]
    eval_files = [f for f in os.listdir('visualizations') if f.endswith('.png')]
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"  EDA visualizations: {len(eda_files)}")
    print(f"  Evaluation visualizations: {len(eval_files)}")
    print(f"  Total: {len(eda_files) + len(eval_files)}")
    print(f"  Requirement: 15+ → {'✓ MET' if len(eda_files) + len(eval_files) >= 15 else '✗ NOT MET'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_additional_eda_visualizations()
