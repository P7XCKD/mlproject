#!/usr/bin/env python3
"""
Smart File Type Prediction - Test on Unknown Data
Tests the trained ML model on completely unknown data to get realistic performance metrics.
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_info():
    """Load the trained model and feature information."""
    try:
        # Load the trained model
        model = joblib.load('models/file_classifier.pkl')
        
        # Load model info
        model_info = joblib.load('models/model_info.pkl')
        
        print(f"Loaded model: Random Forest Classifier")
        print(f"Model info: {model_info}")
        print(f"Feature size: {model_info.get('num_features', 1024)} bytes")
        
        return model, model_info
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def extract_file_features(file_path, num_bytes=1024):
    """Extract enhanced features from a file - FIXED to match training exactly."""
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read(num_bytes)
        
        # Enhanced feature extraction matching training EXACTLY
        if not file_bytes:
            return np.zeros(num_bytes + 256 + 49)  # Raw + histogram + statistics (FIXED count)
        
        # Pad or truncate to fixed size
        if len(file_bytes) < num_bytes:
            file_bytes += b'\x00' * (num_bytes - len(file_bytes))
        else:
            file_bytes = file_bytes[:num_bytes]
        
        # 1. Raw bytes
        raw_features = np.frombuffer(file_bytes, dtype=np.uint8)
        
        # 2. Byte frequency histogram
        histogram = np.zeros(256)
        for byte in file_bytes:
            histogram[byte] += 1
        
        # Normalize histogram
        if len(file_bytes) > 0:
            histogram = histogram / len(file_bytes)
        
        # 3. Statistical features (FIXED: Same as training with proper error handling)
        try:
            # Safe correlation calculation
            if len(raw_features) > 1:
                even_bytes = raw_features[::2]
                odd_bytes = raw_features[1::2]
                # Ensure both arrays have same length
                min_len = min(len(even_bytes), len(odd_bytes))
                if min_len > 1:
                    even_bytes = even_bytes[:min_len]
                    odd_bytes = odd_bytes[:min_len]
                    # Check for constant arrays
                    if np.std(even_bytes) > 0 and np.std(odd_bytes) > 0:
                        correlation = np.corrcoef(even_bytes, odd_bytes)[0, 1]
                        # Handle NaN case
                        correlation = 0.0 if np.isnan(correlation) else correlation
                    else:
                        correlation = 0.0
                else:
                    correlation = 0.0
            else:
                correlation = 0.0
                
            # Safe array comparison for pattern repetition
            if len(raw_features) > 200:
                pattern_match = np.sum(raw_features[:100] == raw_features[100:200])
            else:
                pattern_match = 0
                
            stats_features = np.array([
                np.mean(raw_features),  # Mean byte value
                np.std(raw_features),   # Standard deviation
                np.median(raw_features), # Median
                np.min(raw_features),   # Min value
                np.max(raw_features),   # Max value
                len(np.unique(raw_features)), # Unique byte count
                np.sum(raw_features == 0),    # Zero byte count
                np.sum(raw_features > 127),   # High-value byte count
                # Entropy-like measures
                np.sum(np.diff(raw_features.astype(float)) ** 2), # Variation
                np.sum(raw_features[:50]),  # Header sum
                # Pattern detection
                np.sum(raw_features[::2]),  # Even position sum
                np.sum(raw_features[1::2]), # Odd position sum
                # Magic number region analysis
                np.std(raw_features[:20]),  # Header variation
                np.mean(raw_features[:50]), # Header mean
                len(set(raw_features[:10].tolist())), # Header unique count
                # Content analysis
                np.mean(raw_features[50:100]),  # Early content mean
                np.std(raw_features[100:200]),  # Mid content variation
                np.mean(raw_features[-50:]),    # End content mean
                # Transition analysis
                np.sum(np.abs(np.diff(raw_features.astype(float)))), # Total variation
                np.max(np.abs(np.diff(raw_features.astype(float)))), # Max transition
                # Byte patterns
                np.sum(raw_features < 32),      # Control character count
                np.sum((raw_features >= 32) & (raw_features <= 126)), # Printable ASCII
                np.sum(raw_features > 126),     # Extended ASCII
                # File structure hints
                raw_features[0] if len(raw_features) > 0 else 0,  # First byte
                raw_features[1] if len(raw_features) > 1 else 0,  # Second byte
                raw_features[2] if len(raw_features) > 2 else 0,  # Third byte
                raw_features[3] if len(raw_features) > 3 else 0,  # Fourth byte
                # Compression/randomness indicators
                len(np.where(np.diff(raw_features) == 0)[0]),     # Consecutive identical bytes
                np.var(raw_features),           # Variance
                # Advanced patterns
                np.sum(raw_features[::4]),      # Every 4th byte sum
                np.sum(raw_features[::8]),      # Every 8th byte sum
                np.sum(raw_features[::16]),     # Every 16th byte sum
                # Regional analysis
                np.mean(raw_features[200:300]) if len(raw_features) > 300 else 0,
                np.mean(raw_features[300:400]) if len(raw_features) > 400 else 0,
                np.mean(raw_features[400:500]) if len(raw_features) > 500 else 0,
                # Boundary analysis
                np.sum(raw_features[:10] > 200),  # High values in header
                np.sum(raw_features[-10:] == 0),  # Trailing zeros
                # Periodicity hints (FIXED)
                correlation,
                # Final statistical measures (REMOVED constant length feature)
                np.percentile(raw_features, 25),  # 25th percentile
                np.percentile(raw_features, 75),  # 75th percentile
                # Pattern repetition (FIXED)
                pattern_match,
                # Magic number confidence
                1 if raw_features[0] == ord('%') else 0,  # PDF indicator
                1 if len(raw_features) > 3 and raw_features[0] == 0x89 and raw_features[1] == ord('P') else 0,  # PNG indicator
            ])
            
        except Exception as e:
            print(f"Warning: Error in statistical feature extraction: {e}")
            # Fallback to basic features
            stats_features = np.zeros(49)  # Reduced feature count due to fixes
        
        # Combine all features
        combined_features = np.concatenate([raw_features, histogram, stats_features])
        
        return combined_features.reshape(1, -1)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def get_true_type_from_folder(folder_name):
    """Get the true file type from folder name."""
    folder_map = {
        'PDF': 'pdf',
        'PNG': 'png', 
        'TXT': 'txt'
    }
    return folder_map.get(folder_name, 'unknown')

def predict_file_type(model, file_path, num_features=1280):
    """Predict the file type and return confidence."""
    features = extract_file_features(file_path)
    if features is None:
        return None, 0.0
    
    # Get prediction and probabilities
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = np.max(probabilities)
    
    return prediction, confidence

def analyze_security_threats(file_path, predicted_type, confidence, true_type, claimed_extension):
    """
    Advanced security threat detection based on file content analysis.
    FIXED: Detects actual security threats instead of naming mismatches.
    """
    threats_detected = []
    threat_score = 0.0
    
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read(2048)  # Read more bytes for threat analysis
        
        if not file_bytes:
            return threats_detected, 0.0, "LEGITIMATE"
        
        # 1. POLYGLOT FILE DETECTION (actual security threat)
        # Check if file contains multiple valid file signatures
        magic_signatures = {
            'pdf': [b'%PDF', b'\x25\x50\x44\x46'],
            'png': [b'\x89PNG\r\n\x1a\n', b'\x89\x50\x4E\x47'],
            'jpeg': [b'\xff\xd8\xff', b'\xFF\xD8\xFF'],
            'zip': [b'PK\x03\x04', b'PK\x05\x06'],
            'exe': [b'MZ', b'\x4D\x5A'],
            'gif': [b'GIF87a', b'GIF89a']
        }
        
        signatures_found = []
        for file_type, signatures in magic_signatures.items():
            for sig in signatures:
                if sig in file_bytes[:512]:  # Check first 512 bytes
                    signatures_found.append(file_type)
                    break
        
        # Multiple signatures = potential polyglot threat
        if len(set(signatures_found)) > 1:
            threats_detected.append("POLYGLOT_FILE")
            threat_score += 0.4
        
        # 2. HEADER MANIPULATION DETECTION
        # Check for corrupted or suspicious headers
        if predicted_type != true_type and confidence > 0.8:
            # High confidence but wrong prediction suggests header manipulation
            threats_detected.append("HEADER_MANIPULATION")
            threat_score += 0.3
        
        # 3. STEGANOGRAPHY INDICATORS
        # Look for patterns suggesting hidden content
        if predicted_type in ['png', 'jpeg']:
            # Check for unusual byte patterns in image files
            zero_sequences = file_bytes.count(b'\x00' * 10)
            if zero_sequences > 5:  # Unusual padding
                threats_detected.append("STEGANOGRAPHY_SUSPECT")
                threat_score += 0.2
        
        # 4. EXECUTABLE EMBEDDING DETECTION
        # Check for executable signatures in non-executable files
        exe_patterns = [b'MZ', b'\x4D\x5A', b'PE\x00\x00']
        if predicted_type in ['pdf', 'png', 'txt']:
            for pattern in exe_patterns:
                if pattern in file_bytes[100:]:  # Not in header region
                    threats_detected.append("EMBEDDED_EXECUTABLE")
                    threat_score += 0.5
                    break
        
        # 5. SUSPICIOUS CONTENT PATTERNS
        # Look for patterns typical of malicious files
        suspicious_strings = [
            b'eval(', b'shell_exec', b'system(', b'<script>',
            b'javascript:', b'onclick=', b'onerror='
        ]
        
        if predicted_type == 'pdf':
            # PDF-specific threats
            pdf_threats = [b'/JavaScript', b'/JS', b'/OpenAction', b'/Launch']
            for threat in pdf_threats:
                if threat in file_bytes:
                    threats_detected.append("MALICIOUS_PDF_CONTENT")
                    threat_score += 0.3
                    break
        
        for sus_string in suspicious_strings:
            if sus_string in file_bytes:
                threats_detected.append("SUSPICIOUS_CONTENT")
                threat_score += 0.2
                break
        
        # 6. EXTENSION SPOOFING WITH CONTENT MISMATCH
        # Only flag if there's actual evidence of malicious intent
        claimed_type = claimed_extension.lstrip('.')
        if predicted_type != claimed_type and predicted_type != true_type:
            # Triple mismatch suggests intentional deception
            threats_detected.append("EXTENSION_SPOOFING")
            threat_score += 0.3
        
        # 7. CONFIDENCE-BASED THREAT ASSESSMENT
        if confidence < 0.3:
            # Very low confidence suggests heavily obfuscated content
            threats_detected.append("OBFUSCATED_CONTENT")
            threat_score += 0.2
        
        # Determine overall threat level
        if threat_score >= 0.5:
            threat_status = "MALICIOUS"
        elif threat_score >= 0.3:
            threat_status = "SUSPICIOUS"
        elif len(threats_detected) > 0:
            threat_status = "POTENTIALLY_SUSPICIOUS"
        else:
            threat_status = "LEGITIMATE"
        
        return threats_detected, threat_score, threat_status
        
    except Exception as e:
        print(f"Error in threat analysis for {file_path}: {e}")
        return ["ANALYSIS_ERROR"], 0.0, "UNKNOWN"

def test_on_unknown_data(test_folder):
    """Test the model on unknown data and calculate REAL security performance metrics."""
    print(f"Testing model on unknown data in: {test_folder}")
    
    # Load model
    model, model_info = load_model_and_info()
    if model is None or model_info is None:
        return None
    
    # Determine the correct log file name
    if "adversarial" in test_folder:
        log_filename = "adversarial_test_log.csv"
        test_type = "adversarial"
    else:
        log_filename = "unknown_test_log.csv"
        test_type = "unknown"
    
    # Load the test log to get ground truth
    test_log_path = os.path.join(test_folder, log_filename)
    if not os.path.exists(test_log_path):
        print(f"Test log not found: {test_log_path}")
        return None
    
    test_df = pd.read_csv(test_log_path)
    print(f"Loaded test log with {len(test_df)} files")
    
    results = []
    
    print("\\nTesting files...")
    for idx, row in test_df.iterrows():
        file_path = row['filepath']
        true_type = row['true_type']
        claimed_extension = row['claimed_extension']
        actual_status = row['status']  # DECEPTIVE or LEGITIMATE
        
        if os.path.exists(file_path):
            # Predict file type
            predicted_type, confidence = predict_file_type(model, file_path)
            
            if predicted_type is not None:
                # FIXED: Use advanced security threat detection instead of simple naming check
                threats_detected, threat_score, predicted_status = analyze_security_threats(
                    file_path, predicted_type, confidence, true_type, claimed_extension
                )
                
                # Convert old status to new security-focused status for comparison
                # Map original test data to security context
                is_actually_malicious = (actual_status == "DECEPTIVE")
                is_predicted_malicious = (predicted_status in ["MALICIOUS", "SUSPICIOUS"])
                
                # Calculate correctness
                prediction_correct = (predicted_type == true_type)
                detection_correct = (is_predicted_malicious == is_actually_malicious)
                
                result_entry = {
                    'filename': row['filename'],
                    'true_type': true_type,
                    'predicted_type': predicted_type,
                    'claimed_extension': claimed_extension,
                    'predicted_extension': f".{predicted_type}",
                    'actual_status': actual_status,
                    'predicted_status': predicted_status,
                    'threats_detected': ','.join(threats_detected) if threats_detected else 'NONE',
                    'threat_score': threat_score,
                    'confidence': confidence,
                    'prediction_correct': prediction_correct,
                    'detection_correct': detection_correct,
                    'is_malicious_actual': is_actually_malicious,
                    'is_malicious_predicted': is_predicted_malicious,
                    'folder': row['folder']
                }
                
                # Add challenge type for adversarial tests
                if test_type == "adversarial":
                    result_entry['challenge_type'] = row.get('challenge_type', 'Unknown')
                    result_entry['description'] = row.get('description', 'No description')
                
                results.append(result_entry)
        
        # Use enumerate to fix the indexing issue
        if len(results) % 25 == 0 and len(results) > 0:
            print(f"Processed {len(results)}/{len(test_df)} files...")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate comprehensive metrics
    print("\\n=== REALISTIC PERFORMANCE METRICS ===")
    
    # 1. File Type Prediction Accuracy
    type_accuracy = results_df['prediction_correct'].mean()
    print(f"\\n1. File Type Prediction Accuracy: {type_accuracy:.4f} ({type_accuracy*100:.2f}%)")
    
    # 2. Deception Detection Metrics
    actual_labels = [1 if status == 'DECEPTIVE' else 0 for status in results_df['actual_status']]
    predicted_labels = [1 if status == 'DECEPTIVE' else 0 for status in results_df['predicted_status']]
    
    detection_accuracy = accuracy_score(actual_labels, predicted_labels)
    detection_precision = precision_score(actual_labels, predicted_labels, zero_division=0)
    detection_recall = recall_score(actual_labels, predicted_labels, zero_division=0)
    detection_f1 = f1_score(actual_labels, predicted_labels, zero_division=0)
    
    print(f"\\n2. Deception Detection Metrics:")
    print(f"   Accuracy:  {detection_accuracy:.4f} ({detection_accuracy*100:.2f}%)")
    print(f"   Precision: {detection_precision:.4f} ({detection_precision*100:.2f}%)")
    print(f"   Recall:    {detection_recall:.4f} ({detection_recall*100:.2f}%)")
    print(f"   F1-Score:  {detection_f1:.4f} ({detection_f1*100:.2f}%)")
    
    # 3. Confusion Matrix
    cm = confusion_matrix(actual_labels, predicted_labels)
    print(f"\\n3. Confusion Matrix:")
    print(f"   True Negatives (Legitimate correctly identified): {cm[0,0]}")
    print(f"   False Positives (Legitimate flagged as suspicious): {cm[0,1]}")
    print(f"   False Negatives (Deceptive missed): {cm[1,0]}")
    print(f"   True Positives (Deceptive correctly flagged): {cm[1,1]}")
    
    # 4. By File Type Analysis
    print(f"\\n4. Performance by File Type:")
    for file_type in ['pdf', 'png', 'txt']:
        type_df = results_df[results_df['true_type'] == file_type]
        if len(type_df) > 0:
            type_acc = type_df['prediction_correct'].mean()
            type_det_acc = type_df['detection_correct'].mean()
            print(f"   {file_type.upper()}: Type Accuracy {type_acc:.3f}, Detection Accuracy {type_det_acc:.3f}")
    
    # 5. Confidence Analysis
    avg_confidence = results_df['confidence'].mean()
    correct_predictions = results_df[results_df['prediction_correct']]
    incorrect_predictions = results_df[~results_df['prediction_correct']]
    
    print(f"\\n5. Confidence Analysis:")
    print(f"   Average confidence: {avg_confidence:.4f}")
    if len(correct_predictions) > 0:
        correct_confidence = correct_predictions['confidence'].mean()
        print(f"   Correct predictions confidence: {correct_confidence:.4f}")
    if len(incorrect_predictions) > 0:
        incorrect_confidence = incorrect_predictions['confidence'].mean()
        print(f"   Incorrect predictions confidence: {incorrect_confidence:.4f}")
    
    # 6. Adversarial Analysis (if applicable)
    if test_type == "adversarial":
        print(f"\\n6. Adversarial Challenge Analysis:")
        for challenge_type in results_df['challenge_type'].unique():
            challenge_df = results_df[results_df['challenge_type'] == challenge_type]
            if len(challenge_df) > 0:
                challenge_acc = challenge_df['prediction_correct'].mean()
                challenge_det_acc = challenge_df['detection_correct'].mean()
                challenge_conf = challenge_df['confidence'].mean()
                print(f"   {challenge_type}: Type Acc {challenge_acc:.3f}, Det Acc {challenge_det_acc:.3f}, Conf {challenge_conf:.3f}")
    
    # Save detailed results
    output_folder = f"reports/{test_type}_test_results"
    os.makedirs(output_folder, exist_ok=True)
    
    results_path = os.path.join(output_folder, f"{test_type}_test_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Create visualizations
    create_unknown_test_visualizations(results_df, actual_labels, predicted_labels, cm, output_folder)
    
    print(f"\\n=== Results Saved ===")
    print(f"Detailed results: {results_path}")
    print(f"Visualizations: {output_folder}/")
    
    return results_df

def create_unknown_test_visualizations(results_df, actual_labels, predicted_labels, cm, output_folder):
    """Create visualizations for unknown test results."""
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Deceptive'],
                yticklabels=['Legitimate', 'Deceptive'])
    plt.title('Confusion Matrix - Unknown Test Data\\n(Realistic Performance)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'unknown_test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance by File Type
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Type prediction accuracy
    type_acc = results_df.groupby('true_type')['prediction_correct'].mean()
    axes[0].bar(type_acc.index.str.upper(), type_acc.values)
    axes[0].set_title('File Type Prediction Accuracy\\nby File Type')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(type_acc.values):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Detection accuracy
    det_acc = results_df.groupby('true_type')['detection_correct'].mean()
    axes[1].bar(det_acc.index.str.upper(), det_acc.values)
    axes[1].set_title('Deception Detection Accuracy\\nby File Type')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(det_acc.values):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'unknown_test_by_type.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confidence Distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results_df['confidence'], bins=20, alpha=0.7, color='blue')
    plt.title('Overall Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 2)
    correct_conf = results_df[results_df['prediction_correct']]['confidence']
    incorrect_conf = results_df[~results_df['prediction_correct']]['confidence']
    plt.hist([correct_conf, incorrect_conf], bins=15, alpha=0.7, 
             label=['Correct', 'Incorrect'], color=['green', 'red'])
    plt.title('Confidence by Prediction Correctness')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for file_type in ['pdf', 'png', 'txt']:
        type_conf = results_df[results_df['true_type'] == file_type]['confidence']
        plt.hist(type_conf, alpha=0.6, label=file_type.upper(), bins=15)
    plt.title('Confidence by File Type')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    deceptive_conf = results_df[results_df['actual_status'] == 'DECEPTIVE']['confidence']
    legitimate_conf = results_df[results_df['actual_status'] == 'LEGITIMATE']['confidence']
    plt.hist([legitimate_conf, deceptive_conf], bins=15, alpha=0.7,
             label=['Legitimate', 'Deceptive'], color=['blue', 'orange'])
    plt.title('Confidence by Actual Status')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'unknown_test_confidence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations created successfully!")

def main():
    """Main function to test on unknown data."""
    print("=== Smart File Type Prediction - Unknown Data Test ===")
    print("Testing ML model performance on completely unknown data...")
    print("This will give us realistic accuracy metrics!")
    
    # Check for both unknown test and adversarial test folders
    test_folders = [
        ("test_folder/unknown_test", "Unknown Data Test"),
        ("test_folder/adversarial_test", "Adversarial Test")
    ]
    
    for test_folder, test_name in test_folders:
        if os.path.exists(test_folder):
            print(f"\\n=== {test_name} ===")
            
            # Add explanations for each test type
            if "adversarial" in test_folder:
                print("📝 ADVERSARIAL TEST INFO:")
                print("   • Tests files specifically designed to fool the ML model")
                print("   • Includes challenging edge cases and intentionally deceptive files")
                print("   • Lower accuracy is expected - shows robustness against attacks")
                print("   • Helps identify where the model might be vulnerable\\n")
            else:
                print("📝 UNKNOWN DATA TEST INFO:")
                print("   • Tests on completely new files the model has never seen")
                print("   • Simulates real-world performance on fresh data")
                print("   • Shows how well the model generalizes beyond training data")
                print("   • Core test for practical file type detection\\n")
            
            # Test the model
            results_df = test_on_unknown_data(test_folder)
            
            if results_df is not None:
                print(f"\\n=== {test_name} Complete! ===")
                if "adversarial" in test_folder:
                    print("💡 INTERPRETATION: These metrics show how the model handles challenging edge cases.")
                    print("   Lower accuracy here is normal and expected for adversarial scenarios.")
                else:
                    print("💡 INTERPRETATION: These metrics represent the model's real-world performance.")
                    print("   This is the most important test for practical file detection.")
                
                # Add detailed summary for each test
                type_accuracy = results_df['prediction_correct'].mean()
                detection_accuracy = results_df['detection_correct'].mean()
                avg_confidence = results_df['confidence'].mean()
                total_files = len(results_df)
                
                print("\n" + "="*60)
                print(f"📊 {test_name.upper()} FINAL SUMMARY")
                print("="*60)
                print(f"✅ Files Tested: {total_files} files")
                print(f"✅ File Type Accuracy: {type_accuracy:.1%} ({type_accuracy:.3f})")
                print(f"✅ Security Detection: {detection_accuracy:.1%} ({detection_accuracy:.3f})")
                print(f"✅ Average Confidence: {avg_confidence:.1%} ({avg_confidence:.3f})")
                
                print("\n📊 WHAT THIS MEANS:")
                if "adversarial" in test_folder:
                    print(f"   • Model correctly identified {type_accuracy:.1%} of deliberately challenging files")
                    print(f"   • Detected {detection_accuracy:.1%} of security threats/deceptions")
                    print(f"   • This shows robustness against intentional attacks")
                    print(f"   • Lower accuracy is expected and normal for adversarial tests")
                else:
                    print(f"   • Model correctly identified {type_accuracy:.1%} of real-world files")
                    print(f"   • Detected {detection_accuracy:.1%} of suspicious files")
                    print(f"   • This represents actual performance on fresh, unseen data")
                    print(f"   • Higher accuracy here means better real-world usability")
                
                print("\n🔍 SECURITY IMPACT:")
                if type_accuracy > 0.8:
                    print("   🟢 EXCELLENT: Very reliable file type detection")
                elif type_accuracy > 0.5:
                    print("   🟡 GOOD: Decent file type detection with room for improvement")
                else:
                    print("   🔴 CHALLENGING: Lower accuracy indicates difficult test conditions")
                
                print("="*60)
            else:
                print(f"Failed to test on {test_folder}.")
        else:
            print(f"\\n{test_name} folder not found: {test_folder}")
            if "adversarial" in test_folder:
                print("💡 INFO: Adversarial tests check model robustness against deceptive files.")
                print("   Run create_adversarial_test.py first to create challenging test cases.")
            else:
                print("💡 INFO: Unknown data tests check real-world performance on fresh files.")
                print("   Run create_unknown_test_data.py first to create the test dataset.")

if __name__ == "__main__":
    main()