import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_class_distribution_pie(df, save_path="graphs/class_distribution.png"):
    class_counts = df['Class'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#66b3ff', '#ff9999']
    labels = ['Normal (0)', 'Fraud (1)']
    explode = (0, 0.1)  # explode the fraud slice
    
    wedges, texts, autotexts = ax.pie(
        class_counts.values, 
        labels=labels,
        autopct='%1.2f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        shadow=True
    )
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.title('Class Distribution: Normal vs Fraud Transactions', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")

def create_correlation_heatmap(df, save_path="graphs/correlation_heatmap.png"):
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Create a larger figure
    plt.figure(figsize=(16, 12))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=False,  # Set to True if you want correlation values
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        fmt='.2f'
    )
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")

def create_precision_recall_curve(y_true, y_scores, save_path="graphs/precision_recall_curve.png"):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='b', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
    plt.plot([0, 1], [y_true.mean(), y_true.mean()], 
             color='gray', linestyle='--', label='Baseline (Random)')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")

def create_reconstruction_error_distribution(reconstruction_errors, labels, save_path="graphs/reconstruction_error_dist.png"):
    normal_errors = [err for err, label in zip(reconstruction_errors, labels) if label == 0]
    fraud_errors = [err for err, label in zip(reconstruction_errors, labels) if label == 1]
    
    plt.figure(figsize=(12, 6))
    
    plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal Transactions', color='blue', density=True)
    plt.hist(fraud_errors, bins=50, alpha=0.7, label='Fraud Transactions', color='red', density=True)
    
    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Reconstruction Errors: Normal vs Fraud', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")

def create_feature_importance_plot(df, save_path="graphs/feature_importance.png"):
    # Calculate variance for each feature
    feature_vars = df.drop(columns=['Class']).var().sort_values(ascending=False)
    
    # Take top 15 features
    top_features = feature_vars.head(15)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features.values, color='steelblue')
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Variance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Top 15 Features by Variance (Feature Importance Proxy)', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")

def create_threshold_optimization_plot(y_true, y_scores, save_path="graphs/threshold_optimization.png"):
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    thresholds = np.arange(0.1, 50, 0.5)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        if len(np.unique(y_pred)) > 1:  # Check if we have both classes
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
            f1_scores.append(f1_score(y_true, y_pred))
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
    
    # Find optimal threshold (max F1)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Threshold Optimization: Precision, Recall, and F1-Score', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")
    print(f"Optimal threshold: {optimal_threshold:.2f}")

def create_time_pattern_analysis(df, save_path="graphs/time_pattern_analysis.png"):
    if 'Hour_sin' in df.columns and 'Hour_cos' in df.columns:
        # Convert back to hour for visualization
        df['Hour'] = np.arctan2(df['Hour_sin'], df['Hour_cos']) * 24 / (2 * np.pi)
        df['Hour'] = (df['Hour'] + 24) % 24
        
        fraud_by_hour = df[df['Class'] == 1].groupby(df['Hour'].astype(int)).size()
        normal_by_hour = df[df['Class'] == 0].groupby(df['Hour'].astype(int)).size()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Fraud transactions by hour
        ax1.bar(fraud_by_hour.index, fraud_by_hour.values, color='red', alpha=0.7)
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Number of Fraud Transactions', fontsize=12)
        ax1.set_title('Fraud Transactions by Hour of Day', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(24))
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Normal transactions by hour
        ax2.bar(normal_by_hour.index, normal_by_hour.values, color='blue', alpha=0.7)
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Number of Normal Transactions', fontsize=12)
        ax2.set_title('Normal Transactions by Hour of Day', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(24))
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved to {save_path}")