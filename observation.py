#!/usr/bin/env python3
"""
Smart File Type Prediction - Model Observation Tool
==================================================
This script analyzes the trained model and test results to explain
what the ML system learned and how it performs in simple terms.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import glob

def load_model_info():
    """Load information about the trained model"""
    try:
        with open('models/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model_info
    except FileNotFoundError:
        print("❌ No trained model found. Please run 'train_realistic_model.py' first.")
        return None

def load_training_data():
    """Load training data statistics"""
    try:
        df = pd.read_csv('models/training_data.csv')
        return df
    except FileNotFoundError:
        print("❌ No training data found. Please run 'train_realistic_model.py' first.")
        return None

def analyze_test_results():
    """Analyze all available test results"""
    results = {}
    
    # Look for unknown test results
    unknown_files = glob.glob('reports/unknown_test_results/*.csv')
    if unknown_files:
        latest_unknown = max(unknown_files, key=os.path.getctime)
        try:
            df = pd.read_csv(latest_unknown)
            results['unknown'] = df
        except:
            pass
    
    # Look for adversarial test results
    adversarial_files = glob.glob('reports/adversarial_test_results/*.csv')
    if adversarial_files:
        latest_adversarial = max(adversarial_files, key=os.path.getctime)
        try:
            df = pd.read_csv(latest_adversarial)
            results['adversarial'] = df
        except:
            pass
    
    return results

def explain_model_training(model_info, training_data):
    """Explain what the model learned during training"""
    print("\n" + "="*60)
    print("🧠 WHAT THE MODEL LEARNED DURING TRAINING")
    print("="*60)
    
    if model_info:
        print(f"📚 Training Summary:")
        print(f"   • Learned from {model_info.get('num_files', 'unknown')} different files")
        print(f"   • Can identify {len(model_info.get('file_types', []))} file types: {', '.join(model_info.get('file_types', []))}")
        print(f"   • Uses {model_info.get('num_features', 'unknown')} different patterns to make decisions")
        print(f"   • Training accuracy: {model_info.get('training_accuracy', 0)*100:.1f}%")
        
        if 'feature_components' in model_info:
            components = model_info['feature_components']
            print(f"\n🔍 How it analyzes files:")
            print(f"   • Raw bytes: {components.get('raw_bytes', 0)} patterns (file signatures)")
            print(f"   • Byte histogram: {components.get('histogram', 0)} patterns (byte frequency)")
            print(f"   • Statistics: {components.get('statistics', 0)} patterns (file characteristics)")
            print(f"   • Total patterns: {components.get('total', 0)}")
    
    if training_data is not None:
        print(f"\n📊 Training Data Analysis:")
        file_counts = training_data['true_type'].value_counts()
        for file_type, count in file_counts.items():
            print(f"   • {file_type.upper()}: {count} files ({count/len(training_data)*100:.1f}%)")
        
        print(f"\n💡 What this means:")
        print(f"   • The model learned to recognize patterns in file headers and content")
        print(f"   • It can distinguish between different file types by analyzing raw bytes")
        print(f"   • Higher training accuracy means it learned the patterns well")

def explain_real_world_performance(test_results):
    """Explain how the model performs on real data"""
    print("\n" + "="*60)
    print("🌍 REAL-WORLD PERFORMANCE ANALYSIS")
    print("="*60)
    
    if 'unknown' in test_results:
        df = test_results['unknown']
        
        # Calculate basic metrics
        correct_predictions = (df['predicted_type'] == df['true_type']).sum()
        total_predictions = len(df)
        accuracy = correct_predictions / total_predictions
        
        print(f"📈 Unknown Data Test Results:")
        print(f"   • Tested on {total_predictions} completely new files")
        print(f"   • Correctly identified {correct_predictions} files ({accuracy*100:.1f}%)")
        print(f"   • This represents real-world performance on fresh data")
        
        # Analyze by file type
        print(f"\n📂 Performance by File Type:")
        for file_type in df['true_type'].unique():
            type_df = df[df['true_type'] == file_type]
            type_correct = (type_df['predicted_type'] == type_df['true_type']).sum()
            type_total = len(type_df)
            type_accuracy = type_correct / type_total if type_total > 0 else 0
            print(f"   • {file_type.upper()}: {type_correct}/{type_total} correct ({type_accuracy*100:.1f}%)")
        
        # Confidence analysis
        avg_confidence = df['confidence'].mean()
        correct_confidence = df[df['predicted_type'] == df['true_type']]['confidence'].mean()
        incorrect_confidence = df[df['predicted_type'] != df['true_type']]['confidence'].mean()
        
        print(f"\n🎯 Confidence Analysis:")
        print(f"   • Average confidence: {avg_confidence*100:.1f}%")
        print(f"   • Confidence when correct: {correct_confidence*100:.1f}%")
        print(f"   • Confidence when wrong: {incorrect_confidence*100:.1f}%")
        
        print(f"\n💡 What this means:")
        if accuracy > 0.8:
            print(f"   • 🟢 EXCELLENT: Model performs very well on new files")
        elif accuracy > 0.6:
            print(f"   • 🟡 GOOD: Model performs reasonably well on new files")
        elif accuracy > 0.4:
            print(f"   • 🟠 FAIR: Model has moderate performance on new files")
        else:
            print(f"   • 🔴 POOR: Model struggles with new files")
        
        print(f"   • Higher confidence usually means more accurate predictions")
        print(f"   • This test shows how useful the model would be in practice")

def explain_security_robustness(test_results):
    """Explain how the model handles challenging/adversarial cases"""
    print("\n" + "="*60)
    print("🔒 SECURITY & ROBUSTNESS ANALYSIS")
    print("="*60)
    
    if 'adversarial' in test_results:
        df = test_results['adversarial']
        
        # Calculate basic metrics
        correct_predictions = (df['predicted_type'] == df['true_type']).sum()
        total_predictions = len(df)
        accuracy = correct_predictions / total_predictions
        
        print(f"🛡️ Adversarial Test Results:")
        print(f"   • Tested on {total_predictions} deliberately challenging files")
        print(f"   • Correctly identified {correct_predictions} files ({accuracy*100:.1f}%)")
        print(f"   • These files were designed to confuse the model")
        
        # Analyze by challenge type if available
        if 'challenge_type' in df.columns:
            print(f"\n⚔️ Performance by Challenge Type:")
            for challenge in df['challenge_type'].unique():
                challenge_df = df[df['challenge_type'] == challenge]
                challenge_correct = (challenge_df['predicted_type'] == challenge_df['true_type']).sum()
                challenge_total = len(challenge_df)
                challenge_accuracy = challenge_correct / challenge_total if challenge_total > 0 else 0
                print(f"   • {challenge}: {challenge_correct}/{challenge_total} correct ({challenge_accuracy*100:.1f}%)")
        
        print(f"\n💡 What this means:")
        if accuracy > 0.6:
            print(f"   • 🟢 ROBUST: Model is resistant to attacks and edge cases")
        elif accuracy > 0.4:
            print(f"   • 🟡 MODERATE: Model has some resistance to attacks")
        elif accuracy > 0.2:
            print(f"   • 🟠 VULNERABLE: Model can be confused by adversarial inputs")
        else:
            print(f"   • 🔴 WEAK: Model is easily fooled by adversarial inputs")
        
        print(f"   • Lower accuracy here is normal - these are very challenging tests")
        print(f"   • This test shows how the model handles malicious or corrupted files")

def provide_overall_assessment(model_info, test_results):
    """Provide an overall assessment and recommendations"""
    print("\n" + "="*60)
    print("📋 OVERALL MODEL ASSESSMENT")
    print("="*60)
    
    training_acc = model_info.get('training_accuracy', 0) if model_info else 0
    
    unknown_acc = 0
    adversarial_acc = 0
    
    if 'unknown' in test_results:
        df = test_results['unknown']
        unknown_acc = (df['predicted_type'] == df['true_type']).mean()
    
    if 'adversarial' in test_results:
        df = test_results['adversarial']
        adversarial_acc = (df['predicted_type'] == df['true_type']).mean()
    
    print(f"📊 Performance Summary:")
    print(f"   • Training Performance: {training_acc*100:.1f}%")
    if 'unknown' in test_results:
        print(f"   • Real-world Performance: {unknown_acc*100:.1f}%")
    if 'adversarial' in test_results:
        print(f"   • Security Robustness: {adversarial_acc*100:.1f}%")
    
    print(f"\n🎯 Strengths & Weaknesses:")
    
    # Training vs Real-world gap
    if model_info and 'unknown' in test_results:
        gap = training_acc - unknown_acc
        if gap < 0.1:
            print(f"   • ✅ Good generalization (small gap between training and real-world)")
        elif gap < 0.2:
            print(f"   • ⚠️ Moderate overfitting (some gap between training and real-world)")
        else:
            print(f"   • ❌ Significant overfitting (large gap between training and real-world)")
    
    # Overall recommendations
    print(f"\n💡 Recommendations:")
    
    if unknown_acc > 0.8:
        print(f"   • 🟢 Model is ready for production use")
    elif unknown_acc > 0.6:
        print(f"   • 🟡 Model is suitable for most use cases with monitoring")
        print(f"   • Consider collecting more training data to improve accuracy")
    elif unknown_acc > 0.4:
        print(f"   • 🟠 Model needs improvement before production use")
        print(f"   • Collect more diverse training data")
        print(f"   • Consider feature engineering improvements")
    else:
        print(f"   • 🔴 Model requires significant improvement")
        print(f"   • Re-evaluate training approach and data quality")
    
    if 'adversarial' in test_results and adversarial_acc < 0.3:
        print(f"   • ⚠️ Consider adversarial training to improve robustness")
    
    print(f"   • Regular retraining with new data will improve performance")

def main():
    """Main observation analysis function"""
    print("🔍 Smart File Type Prediction - Model Observation Analysis")
    print("=" * 65)
    print("This tool analyzes what your ML model learned and how it performs.")
    print(f"Analysis run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load all data
    model_info = load_model_info()
    training_data = load_training_data()
    test_results = analyze_test_results()
    
    if not model_info and not test_results:
        print("\n❌ No model or test results found!")
        print("Please run the following commands first:")
        print("1. python train_realistic_model.py")
        print("2. python test_unknown_data.py")
        return
    
    # Perform analysis
    if model_info or training_data is not None:
        explain_model_training(model_info, training_data)
    
    if test_results:
        if 'unknown' in test_results:
            explain_real_world_performance(test_results)
        
        if 'adversarial' in test_results:
            explain_security_robustness(test_results)
        
        provide_overall_assessment(model_info, test_results)
    
    print("\n" + "="*60)
    print("✅ OBSERVATION ANALYSIS COMPLETE!")
    print("="*60)
    print("This analysis helps you understand:")
    print("• What patterns your model learned")
    print("• How well it works on new, real data")
    print("• How robust it is against attacks")
    print("• Where improvements might be needed")
    print("\nRun this script again after retraining to see improvements!")

if __name__ == "__main__":
    main()