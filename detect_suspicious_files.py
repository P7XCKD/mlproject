#!/usr/bin/env python3
"""
Smart File Type Prediction - Detect Suspicious Files
Loads trained model and detects files with mismatched extensions.
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

def load_model_and_info():
    """Load the trained model and feature information."""
    try:
        model_path = 'models/file_classifier.pkl'
        info_path = 'models/model_info.pkl'
        
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found!")
            print("Please run 'train_model.py' first to train the model.")
            return None, None
        
        model = joblib.load(model_path)
        model_info = joblib.load(info_path)
        
        print(f"Model loaded successfully!")
        print(f"Trained on {model_info['num_files']} files")
        print(f"Recognizes {len(model_info['file_types'])} file types: {model_info['file_types']}")
        
        return model, model_info
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def extract_file_bytes(file_path, num_bytes=1024):
    """Extract first N bytes from a file."""
    try:
        with open(file_path, 'rb') as f:
            return f.read(num_bytes)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def bytes_to_features(file_bytes, num_bytes=1024):
    """Convert file bytes to feature vectors."""
    if not file_bytes:
        return np.zeros(num_bytes + 256)  # raw bytes + histogram
    
    # Pad or truncate to fixed size
    if len(file_bytes) < num_bytes:
        file_bytes += b'\x00' * (num_bytes - len(file_bytes))
    else:
        file_bytes = file_bytes[:num_bytes]
    
    # Raw bytes
    raw_features = np.frombuffer(file_bytes, dtype=np.uint8)
    
    # Histogram features
    histogram = np.zeros(256)
    for byte in file_bytes:
        histogram[byte] += 1
    
    # Normalize histogram
    if len(file_bytes) > 0:
        histogram = histogram / len(file_bytes)
    
    # Combine features
    return np.concatenate([raw_features, histogram])

def predict_file_type(model, file_path):
    """Predict the actual file type for a given file."""
    file_bytes = extract_file_bytes(file_path)
    if file_bytes is None:
        return None, None
    
    features = bytes_to_features(file_bytes)
    features = features.reshape(1, -1)  # Reshape for single prediction
    
    # Get prediction and confidence
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    return prediction, confidence

def get_file_extension(file_path):
    """Get the extension from file path."""
    return Path(file_path).suffix.lower().lstrip('.')

def scan_and_detect_suspicious_files(model, model_info, folder_path):
    """Scan folder and detect files with mismatched types."""
    print(f"\nScanning folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist!")
        return None
    
    results = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Skip log files
            if file.endswith('.csv') and 'log' in file.lower():
                continue
                
            file_path = os.path.join(root, file)
            file_extension = get_file_extension(file_path)
            
            print(f"Analyzing: {file}")
            
            # Predict actual file type
            predicted_type, confidence = predict_file_type(model, file_path)
            
            if predicted_type is None:
                print(f"  ❌ Could not analyze file")
                continue
            
            # Determine if suspicious
            is_suspicious = predicted_type != file_extension
            
            # Handle special cases
            if predicted_type == 'unknown':
                is_suspicious = False  # Can't determine, so don't flag
            
            # Map some extensions (e.g., jpeg vs jpg)
            extension_mappings = {
                'jpeg': 'jpg',
                'htm': 'html',
                'log': 'txt'
            }
            
            mapped_extension = extension_mappings.get(file_extension, file_extension)
            if predicted_type == mapped_extension:
                is_suspicious = False
            
            status = "🚨 SUSPICIOUS" if is_suspicious else "✅ SAFE"
            confidence_str = f"{confidence:.3f}"
            
            print(f"  {status} - Predicted: {predicted_type}, Extension: {file_extension}, Confidence: {confidence_str}")
            
            results.append({
                'file_name': file,
                'file_path': file_path,
                'predicted_type': predicted_type,
                'file_extension': file_extension,
                'confidence': confidence,
                'is_suspicious': is_suspicious,
                'status': status
            })
    
    return pd.DataFrame(results)

def generate_detection_report(results_df, output_folder):
    """Generate detailed detection report."""
    if results_df is None or len(results_df) == 0:
        print("No results to report.")
        return
    
    print(f"\n{'='*60}")
    print("DETECTION REPORT")
    print(f"{'='*60}")
    
    total_files = len(results_df)
    suspicious_files = len(results_df[results_df['is_suspicious']])
    safe_files = total_files - suspicious_files
    
    print(f"Total files analyzed: {total_files}")
    print(f"Safe files: {safe_files}")
    print(f"Suspicious files: {suspicious_files}")
    print(f"Suspicion rate: {suspicious_files/total_files*100:.1f}%")
    
    if suspicious_files > 0:
        print(f"\n🚨 SUSPICIOUS FILES DETECTED:")
        print("-" * 40)
        suspicious_df = results_df[results_df['is_suspicious']].sort_values('confidence', ascending=False)
        
        for _, row in suspicious_df.iterrows():
            print(f"📁 {row['file_name']}")
            print(f"   Predicted: {row['predicted_type']}")
            print(f"   Extension: {row['file_extension']}")
            print(f"   Confidence: {row['confidence']:.3f}")
            print(f"   Path: {row['file_path']}")
            print()
    
    # Save detailed report
    os.makedirs(output_folder, exist_ok=True)
    report_path = os.path.join(output_folder, 'detection_report.csv')
    results_df.to_csv(report_path, index=False)
    print(f"Detailed report saved to: {report_path}")
    
    return results_df

def analyze_specific_test_cases(model, modified_folder):
    """Analyze the specific test cases if they exist."""
    specific_folder = os.path.join(modified_folder, 'specific_tests')
    
    if not os.path.exists(specific_folder):
        return
    
    print(f"\n{'='*50}")
    print("SPECIFIC TEST CASES ANALYSIS")
    print(f"{'='*50}")
    
    test_log_path = os.path.join(specific_folder, 'specific_tests_log.csv')
    if os.path.exists(test_log_path):
        test_df = pd.read_csv(test_log_path)
        
        for _, row in test_df.iterrows():
            file_path = row['path']
            if os.path.exists(file_path):
                predicted_type, confidence = predict_file_type(model, file_path)
                
                true_ext = row['true_type'].lstrip('.')
                fake_ext = row['fake_type'].lstrip('.')
                
                detected = predicted_type == true_ext
                status = "✅ DETECTED" if detected else "❌ MISSED"
                
                print(f"{status} - {row['fake_file']}")
                print(f"   True type: {true_ext}")
                print(f"   Fake extension: {fake_ext}")
                print(f"   Predicted: {predicted_type}")
                print(f"   Confidence: {confidence:.3f}")
                print()

def main():
    """Main detection function."""
    print("=== Smart File Type Prediction - Suspicious File Detection ===")
    
    # Load model
    model, model_info = load_model_and_info()
    if model is None:
        return
    
    # Get folder path
    modified_folder = input("Enter path to 'modified' folder (or press Enter for default): ").strip()
    if not modified_folder:
        modified_folder = "test_folder/modified"
    
    # Detect suspicious files
    results_df = scan_and_detect_suspicious_files(model, model_info, modified_folder)
    
    if results_df is not None:
        # Generate report
        generate_detection_report(results_df, 'reports')
        
        # Analyze specific test cases
        analyze_specific_test_cases(model, modified_folder)
        
        print(f"\n{'='*60}")
        print("DETECTION COMPLETE!")
        print("Check the 'reports' folder for detailed results.")
        print(f"{'='*60}")
    else:
        print("Detection failed. Please check your folder path.")

if __name__ == "__main__":
    main()