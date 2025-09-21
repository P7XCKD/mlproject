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
    print("🧠 WHAT DID THE AI LEARN?")
    print("="*60)
    
    if model_info:
        num_files = model_info.get('num_files', 'unknown')
        accuracy = model_info.get('training_accuracy', 0)*100
        
        print(f"📚 SIMPLE SUMMARY:")
        print(f"   • The AI looked at {num_files} example files")
        print(f"   • It learned to tell apart PDF, PNG, and TXT files")
        print(f"   • On those examples, it got {accuracy:.1f}% correct")
        
        print(f"\n🔍 HOW DOES IT WORK?")
        print(f"   • It looks at the first few bytes of each file")
        print(f"   • Different file types have different 'fingerprints'")
        print(f"   • Like how PDFs start with '%PDF' and PNGs start with special codes")
        print(f"   • It memorized these patterns from the examples")
    
    if training_data is not None:
        file_counts = training_data['true_type'].value_counts()
        print(f"\n📊 TRAINING DATA:")
        for file_type, count in file_counts.items():
            print(f"   • {file_type.upper()}: {count} example files")
        
        print(f"\n💡 IN SIMPLE TERMS:")
        print(f"   • Think of it like teaching a child to recognize animals")
        print(f"   • You show them many photos of cats, dogs, and birds")
        print(f"   • They learn the differences (fur, feathers, etc.)")
        print(f"   • Our AI learned file differences the same way")

def explain_real_world_performance(test_results):
    """Explain how the model performs on real data"""
    print("\n" + "="*60)
    print("🌍 HOW GOOD IS IT IN REAL LIFE?")
    print("="*60)
    
    if 'unknown' in test_results:
        df = test_results['unknown']
        
        # Calculate basic metrics
        correct_predictions = (df['predicted_type'] == df['true_type']).sum()
        total_predictions = len(df)
        accuracy = correct_predictions / total_predictions
        
        print(f"📈 THE REAL TEST:")
        print(f"   • We gave it {total_predictions} files it had NEVER seen before")
        print(f"   • It correctly guessed {correct_predictions} files")
        print(f"   • That's {accuracy*100:.1f}% success rate")
        print(f"   • This is like testing a student on completely new questions")
        
        # Analyze by file type
        print(f"\n📂 HOW GOOD IS IT WITH EACH FILE TYPE?")
        for file_type in df['true_type'].unique():
            type_df = df[df['true_type'] == file_type]
            type_correct = (type_df['predicted_type'] == type_df['true_type']).sum()
            type_total = len(type_df)
            type_accuracy = type_correct / type_total if type_total > 0 else 0
            
            if type_accuracy > 0.8:
                emoji = "🟢 EXCELLENT"
            elif type_accuracy > 0.6:
                emoji = "🟡 DECENT"
            elif type_accuracy > 0.3:
                emoji = "🟠 POOR"
            else:
                emoji = "🔴 TERRIBLE"
            
            print(f"   • {file_type.upper()}: {type_correct}/{type_total} correct ({type_accuracy*100:.1f}%) {emoji}")
        
        # Confidence analysis
        avg_confidence = df['confidence'].mean()
        print(f"\n🎯 HOW CONFIDENT IS IT?")
        print(f"   • Average confidence: {avg_confidence*100:.1f}%")
        print(f"   • Think of this like 'how sure am I?' percentage")
        
        print(f"\n💡 WHAT DOES THIS MEAN FOR YOU?")
        if accuracy > 0.8:
            print(f"   • 🟢 AMAZING: This AI is ready to use for real work!")
            print(f"   • It's very reliable and rarely makes mistakes")
        elif accuracy > 0.6:
            print(f"   • 🟡 OKAY: This AI is decent but needs some improvement")
            print(f"   • It works most of the time but might need human checking")
        elif accuracy > 0.4:
            print(f"   • 🟠 MEH: This AI needs more training")
            print(f"   • It's not reliable enough for important work yet")
        else:
            print(f"   • 🔴 BAD: This AI is basically guessing randomly")
            print(f"   • Don't trust it with anything important!")

def explain_security_robustness(test_results):
    """Explain how the model handles challenging/adversarial cases"""
    print("\n" + "="*60)
    print("🔒 CAN BAD GUYS FOOL IT?")
    print("="*60)
    
    if 'adversarial' in test_results:
        df = test_results['adversarial']
        
        # Calculate basic metrics
        correct_predictions = (df['predicted_type'] == df['true_type']).sum()
        total_predictions = len(df)
        accuracy = correct_predictions / total_predictions
        
        print(f"🛡️ THE HACKER TEST:")
        print(f"   • We tried {total_predictions} sneaky, tricky files designed to fool the AI")
        print(f"   • The AI still got {correct_predictions} correct")
        print(f"   • That's {accuracy*100:.1f}% success against attacks")
        print(f"   • This is like testing if someone can trick you with fake IDs")
        
        # Analyze by challenge type if available
        if 'challenge_type' in df.columns:
            print(f"\n⚔️ DIFFERENT TYPES OF TRICKS TESTED:")
            for challenge in df['challenge_type'].unique():
                challenge_df = df[df['challenge_type'] == challenge]
                challenge_correct = (challenge_df['predicted_type'] == challenge_df['true_type']).sum()
                challenge_total = len(challenge_df)
                challenge_accuracy = challenge_correct / challenge_total if challenge_total > 0 else 0
                
                if challenge_accuracy > 0.7:
                    emoji = "🟢 HARD TO FOOL"
                elif challenge_accuracy > 0.3:
                    emoji = "🟡 SOMETIMES FOOLED"
                else:
                    emoji = "🔴 EASILY FOOLED"
                
                print(f"   • {challenge}: {challenge_correct}/{challenge_total} correct ({challenge_accuracy*100:.1f}%) {emoji}")
        
        print(f"\n💡 SECURITY ASSESSMENT:")
        if accuracy > 0.6:
            print(f"   • 🟢 SECURE: Hard for hackers to fool this AI")
            print(f"   • It can spot most fake or corrupted files")
        elif accuracy > 0.4:
            print(f"   • 🟡 SOMEWHAT SECURE: Catches some tricks but not all")
            print(f"   • A skilled hacker might be able to fool it sometimes")
        elif accuracy > 0.2:
            print(f"   • 🟠 VULNERABLE: Easy to fool with the right tricks")
            print(f"   • Don't rely on this for security-critical tasks")
        else:
            print(f"   • 🔴 VERY VULNERABLE: Almost any trick works")
            print(f"   • This AI offers no real security protection")
        
        print(f"\n🤔 REMEMBER:")
        print(f"   • Even humans get fooled by really good fakes")
        print(f"   • The fact that it catches ANY tricks is actually pretty good")
        print(f"   • This test uses extremely difficult, deliberately deceptive files")

def provide_overall_assessment(model_info, test_results):
    """Provide an overall assessment and recommendations"""
    print("\n" + "="*60)
    print("📋 OVERALL: IS THIS AI ANY GOOD?")
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
    
    print(f"📊 REPORT CARD:")
    print(f"   • Learning from examples: {training_acc*100:.1f}% ✏️")
    if 'unknown' in test_results:
        print(f"   • Real-world performance: {unknown_acc*100:.1f}% 🌍")
    if 'adversarial' in test_results:
        print(f"   • Security against attacks: {adversarial_acc*100:.1f}% 🔒")
    
    print(f"\n🎯 BIGGEST PROBLEMS:")
    
    # Training vs Real-world gap
    if model_info and 'unknown' in test_results:
        gap = training_acc - unknown_acc
        if gap < 0.1:
            print(f"   • ✅ Good news: Works almost as well in real life as in training")
        elif gap < 0.2:
            print(f"   • ⚠️ Slight problem: A bit worse in real life than training")
        else:
            print(f"   • ❌ Big problem: Much worse in real life than training (overfitting)")
            print(f"     - Like a student who memorized answers but doesn't understand")
    
    # Overall recommendations
    print(f"\n💡 BOTTOM LINE:")
    
    if unknown_acc > 0.8:
        print(f"   • 🟢 THIS AI IS READY TO USE!")
        print(f"   • You can trust it for real work")
    elif unknown_acc > 0.6:
        print(f"   • 🟡 THIS AI IS OKAY BUT NEEDS WORK")
        print(f"   • Use it for non-critical tasks, double-check important stuff")
    elif unknown_acc > 0.4:
        print(f"   • 🟠 THIS AI NEEDS MORE TRAINING")
        print(f"   • Don't use it for anything important yet")
    else:
        print(f"   • 🔴 THIS AI IS NOT GOOD ENOUGH")
        print(f"   • It's basically guessing randomly")
    
    print(f"\n🔧 HOW TO MAKE IT BETTER:")
    print(f"   • Get more example files for training")
    print(f"   • Make sure training files are diverse and realistic")
    print(f"   • Test it more often on completely new files")
    
    if 'adversarial' in test_results and adversarial_acc < 0.3:
        print(f"   • Add some 'tricky' files to training to improve security")

def main():
    """Main observation analysis function"""
    print("🔍 AI FILE DETECTIVE - SIMPLE PERFORMANCE REPORT")
    print("=" * 65)
    print("Let's see how well your AI learned to identify file types!")
    print(f"Report created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load all data
    model_info = load_model_info()
    training_data = load_training_data()
    test_results = analyze_test_results()
    
    if not model_info and not test_results:
        print("\n❌ No AI model or test results found!")
        print("\nTo get a report, you need to:")
        print("1. Train the AI: python train_realistic_model.py")
        print("2. Test the AI: python test_unknown_data.py")
        print("3. Then run this report again: python observation.py")
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
    print("✅ REPORT COMPLETE!")
    print("="*60)
    print("This report shows you:")
    print("• How well your AI learned")
    print("• How good it is with new files")
    print("• How secure it is against attacks")
    print("• Whether it's ready to use")
    print("\nRun this again after improving your AI to see the changes!")

if __name__ == "__main__":
    main()