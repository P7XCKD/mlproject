#!/usr/bin/env python3
"""
Smart File Type Prediction - Evaluate Results
Analyzes detection performance and generates visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

def load_results_and_logs():
    """Load detection results and modification logs."""
    results_path = 'reports/detection_report.csv'
    changes_log_path = None
    
    # Find changes log
    possible_paths = [
        'test_folder/modified/changes_log.csv',
        'modified/changes_log.csv',
        'changes_log.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            changes_log_path = path
            break
    
    if not os.path.exists(results_path):
        print(f"Error: Detection results not found at {results_path}")
        print("Please run 'detect_suspicious_files.py' first.")
        return None, None
    
    results_df = pd.read_csv(results_path)
    changes_df = None
    
    if changes_log_path and os.path.exists(changes_log_path):
        changes_df = pd.read_csv(changes_log_path)
        print(f"Loaded changes log from: {changes_log_path}")
    else:
        print("Changes log not found. Limited evaluation available.")
    
    print(f"Loaded detection results from: {results_path}")
    return results_df, changes_df

def evaluate_detection_performance(results_df, changes_df):
    """Evaluate how well the system detected suspicious files."""
    if changes_df is None:
        print("Cannot evaluate performance without changes log.")
        return None
    
    print(f"\n{'='*50}")
    print("PERFORMANCE EVALUATION")
    print(f"{'='*50}")
    
    # Merge results with ground truth
    merged_df = pd.merge(
        results_df, 
        changes_df, 
        left_on='file_name', 
        right_on='modified_file', 
        how='inner'
    )
    
    if len(merged_df) == 0:
        print("No matching files found between results and changes log.")
        return None
    
    # Ground truth: files with changed extensions should be flagged as suspicious
    merged_df['should_be_suspicious'] = merged_df['status_y'] == 'CHANGED'
    merged_df['correctly_detected'] = merged_df['is_suspicious'] == merged_df['should_be_suspicious']
    
    # Calculate metrics
    total_files = len(merged_df)
    correctly_detected = merged_df['correctly_detected'].sum()
    accuracy = correctly_detected / total_files
    
    # True/False positives/negatives
    tp = len(merged_df[(merged_df['should_be_suspicious']) & (merged_df['is_suspicious'])])  # Correctly flagged suspicious
    fp = len(merged_df[(~merged_df['should_be_suspicious']) & (merged_df['is_suspicious'])])  # Incorrectly flagged safe files
    tn = len(merged_df[(~merged_df['should_be_suspicious']) & (~merged_df['is_suspicious'])])  # Correctly identified safe
    fn = len(merged_df[(merged_df['should_be_suspicious']) & (~merged_df['is_suspicious'])])  # Missed suspicious files
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Total files evaluated: {total_files}")
    print(f"Overall accuracy: {accuracy:.3f} ({correctly_detected}/{total_files})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1_score:.3f}")
    print()
    print("Confusion Matrix:")
    print(f"True Positives (Suspicious correctly flagged): {tp}")
    print(f"False Positives (Safe incorrectly flagged): {fp}")
    print(f"True Negatives (Safe correctly identified): {tn}")
    print(f"False Negatives (Suspicious missed): {fn}")
    
    # Detailed breakdown
    print(f"\n{'='*30}")
    print("DETAILED BREAKDOWN")
    print(f"{'='*30}")
    
    if fn > 0:
        print(f"\n❌ MISSED SUSPICIOUS FILES ({fn}):")
        missed_files = merged_df[(merged_df['should_be_suspicious']) & (~merged_df['is_suspicious'])]
        for _, row in missed_files.iterrows():
            print(f"   {row['file_name']} (predicted: {row['predicted_type']}, confidence: {row['confidence']:.3f})")
    
    if fp > 0:
        print(f"\n⚠️  FALSE ALARMS ({fp}):")
        false_alarms = merged_df[(~merged_df['should_be_suspicious']) & (merged_df['is_suspicious'])]
        for _, row in false_alarms.iterrows():
            print(f"   {row['file_name']} (predicted: {row['predicted_type']}, confidence: {row['confidence']:.3f})")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'evaluation_df': merged_df
    }

def create_visualizations(results_df, changes_df, eval_results):
    """Create visualization charts."""
    print(f"\n{'='*40}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*40}")
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Detection Results Pie Chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Pie chart of detection results
    detection_counts = results_df['is_suspicious'].value_counts()
    axes[0, 0].pie(detection_counts.values, 
                   labels=['Safe Files', 'Suspicious Files'], 
                   autopct='%1.1f%%',
                   colors=['lightgreen', 'lightcoral'])
    axes[0, 0].set_title('Detection Results Distribution')
    
    # 2. Confidence Distribution
    axes[0, 1].hist(results_df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Prediction Confidence Distribution')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Number of Files')
    
    # 3. File Type Distribution
    if 'predicted_type' in results_df.columns:
        type_counts = results_df['predicted_type'].value_counts()
        axes[1, 0].bar(type_counts.index, type_counts.values, color='lightblue')
        axes[1, 0].set_title('Predicted File Types Distribution')
        axes[1, 0].set_xlabel('File Type')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Performance Metrics (if available)
    if eval_results:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [eval_results['accuracy'], eval_results['precision'], 
                 eval_results['recall'], eval_results['f1_score']]
        
        bars = axes[1, 1].bar(metrics, values, color=['gold', 'lightgreen', 'lightcoral', 'lightblue'])
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    else:
        axes[1, 1].text(0.5, 0.5, 'Performance metrics\nnot available\n(no changes log)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Performance Metrics')
    
    plt.tight_layout()
    plt.savefig('reports/detection_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: reports/detection_analysis.png")
    plt.close()
    
    # 5. Confusion Matrix (if available)
    if eval_results:
        plt.figure(figsize=(8, 6))
        cm = np.array([[eval_results['tn'], eval_results['fp']], 
                      [eval_results['fn'], eval_results['tp']]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Safe', 'Predicted Suspicious'],
                   yticklabels=['Actually Safe', 'Actually Suspicious'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved to: reports/confusion_matrix.png")
        plt.close()

def generate_summary_report(results_df, changes_df, eval_results):
    """Generate a comprehensive text summary report."""
    report_path = 'reports/summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("SMART FILE TYPE PREDICTION - SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write("DETECTION STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total files analyzed: {len(results_df)}\n")
        f.write(f"Files flagged as suspicious: {results_df['is_suspicious'].sum()}\n")
        f.write(f"Files marked as safe: {(~results_df['is_suspicious']).sum()}\n")
        f.write(f"Average confidence: {results_df['confidence'].mean():.3f}\n")
        f.write(f"Minimum confidence: {results_df['confidence'].min():.3f}\n")
        f.write(f"Maximum confidence: {results_df['confidence'].max():.3f}\n\n")
        
        # File type distribution
        f.write("PREDICTED FILE TYPE DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        type_dist = results_df['predicted_type'].value_counts()
        for file_type, count in type_dist.items():
            f.write(f"{file_type}: {count}\n")
        f.write("\n")
        
        # Performance metrics
        if eval_results:
            f.write("PERFORMANCE EVALUATION:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Accuracy: {eval_results['accuracy']:.3f}\n")
            f.write(f"Precision: {eval_results['precision']:.3f}\n")
            f.write(f"Recall: {eval_results['recall']:.3f}\n")
            f.write(f"F1-Score: {eval_results['f1_score']:.3f}\n\n")
            
            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 15 + "\n")
            f.write(f"True Positives: {eval_results['tp']}\n")
            f.write(f"False Positives: {eval_results['fp']}\n")
            f.write(f"True Negatives: {eval_results['tn']}\n")
            f.write(f"False Negatives: {eval_results['fn']}\n\n")
        
        # Suspicious files list
        suspicious_files = results_df[results_df['is_suspicious']]
        if len(suspicious_files) > 0:
            f.write("SUSPICIOUS FILES DETECTED:\n")
            f.write("-" * 25 + "\n")
            for _, row in suspicious_files.iterrows():
                f.write(f"File: {row['file_name']}\n")
                f.write(f"  Predicted type: {row['predicted_type']}\n")
                f.write(f"  File extension: {row['file_extension']}\n")
                f.write(f"  Confidence: {row['confidence']:.3f}\n\n")
    
    print(f"Summary report saved to: {report_path}")

def main():
    """Main evaluation function."""
    print("=== Smart File Type Prediction - Results Evaluation ===")
    
    # Load data
    results_df, changes_df = load_results_and_logs()
    if results_df is None:
        return
    
    # Evaluate performance
    eval_results = evaluate_detection_performance(results_df, changes_df)
    
    # Create visualizations
    create_visualizations(results_df, changes_df, eval_results)
    
    # Generate summary report
    generate_summary_report(results_df, changes_df, eval_results)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print("Results saved in 'reports' folder:")
    print("  - detection_analysis.png (charts)")
    print("  - confusion_matrix.png (if applicable)")
    print("  - summary_report.txt (detailed text report)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()