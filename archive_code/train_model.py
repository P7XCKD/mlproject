#!/usr/bin/env python3
"""
Smart File Type Prediction - Training Script
Extracts byte patterns from files and trains ML classifier to detect true file types.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Magic number signatures for common file types
MAGIC_NUMBERS = {
    'jpg': [b'\xff\xd8\xff'],
    'png': [b'\x89PNG\r\n\x1a\n'],
    'gif': [b'GIF87a', b'GIF89a'],
    'pdf': [b'%PDF'],
    'zip': [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'],
    'exe': [b'MZ'],
    'mp3': [b'ID3', b'\xff\xfb', b'\xff\xfa'],
    'wav': [b'RIFF'],
    'txt': [],  # Text files don't have consistent magic numbers
    'xml': [b'<?xml'],
    'json': [],  # JSON files start with { or [
    'csv': [],   # CSV files are plain text
    'docx': [b'PK\x03\x04'],  # DOCX is actually a ZIP file
    'bmp': [b'BM'],
    'rar': [b'Rar!']
}

def extract_file_bytes(file_path, num_bytes=1024):
    """Extract first N bytes from a file."""
    try:
        with open(file_path, 'rb') as f:
            return f.read(num_bytes)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def detect_file_type_by_magic(file_bytes):
    """Detect file type using magic number signatures."""
    if not file_bytes:
        return 'unknown'
    
    # Check magic numbers
    for file_type, signatures in MAGIC_NUMBERS.items():
        for signature in signatures:
            if file_bytes.startswith(signature):
                return file_type
    
    # Special cases for text-based formats
    try:
        text_content = file_bytes.decode('utf-8', errors='ignore')
        if text_content.strip().startswith('{') or text_content.strip().startswith('['):
            return 'json'
        elif ',' in text_content and '\n' in text_content:
            return 'csv'
        elif all(ord(char) < 128 for char in text_content[:100]):  # ASCII text
            return 'txt'
    except:
        pass
    
    return 'unknown'

def bytes_to_features(file_bytes, num_bytes=1024):
    """Convert file bytes to feature vectors."""
    if not file_bytes:
        return np.zeros(num_bytes)
    
    # Pad or truncate to fixed size
    if len(file_bytes) < num_bytes:
        file_bytes += b'\x00' * (num_bytes - len(file_bytes))
    else:
        file_bytes = file_bytes[:num_bytes]
    
    # Convert to numpy array
    return np.frombuffer(file_bytes, dtype=np.uint8)

def create_histogram_features(file_bytes):
    """Create byte frequency histogram (256 features)."""
    if not file_bytes:
        return np.zeros(256)
    
    histogram = np.zeros(256)
    for byte in file_bytes:
        histogram[byte] += 1
    
    # Normalize by file length
    if len(file_bytes) > 0:
        histogram = histogram / len(file_bytes)
    
    return histogram

def scan_folder_and_extract_features(folder_path):
    """Scan folder and extract features from all files."""
    print(f"Scanning folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist!")
        return None, None, None

    file_data = []
    file_paths = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower().lstrip('.')
            
            print(f"Processing: {file}")
            
            # Extract bytes
            file_bytes = extract_file_bytes(file_path)
            if file_bytes is None:
                continue
            
            # Determine true file type from folder structure
            folder_name = os.path.basename(root).lower()
            if folder_name in ['pdf']:
                true_type = 'pdf'
            elif folder_name in ['png']:
                true_type = 'png'
            elif folder_name in ['txt']:
                true_type = 'txt'
            else:
                # Fallback to magic number detection for other cases
                true_type = detect_file_type_by_magic(file_bytes)
            
            # Create features
            raw_features = bytes_to_features(file_bytes)
            hist_features = create_histogram_features(file_bytes)
            
            # Combine features
            combined_features = np.concatenate([raw_features, hist_features])
            
            file_data.append({
                'file_path': file_path,
                'file_name': file,
                'extension': file_extension,
                'true_type': true_type,
                'features': combined_features,
                'file_size': len(file_bytes)
            })
            file_paths.append(file_path)
    
    if not file_data:
        print("No files found to process!")
        return None, None, None
    
    # Convert to arrays
    features = np.array([item['features'] for item in file_data])
    labels = np.array([item['true_type'] for item in file_data])
    
    # Create DataFrame for analysis
    df = pd.DataFrame([{
        'file_path': item['file_path'],
        'file_name': item['file_name'],
        'extension': item['extension'],
        'true_type': item['true_type'],
        'file_size': item['file_size']
    } for item in file_data])
    
    print(f"\nProcessed {len(file_data)} files")
    print(f"File types found: {df['true_type'].value_counts().to_dict()}")
    
    return features, labels, df

def train_models(X, y):
    """Train multiple ML models and return the best one."""
    print("\n=== Training ML Models ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    best_model = None
    best_score = 0
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.3f}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
    
    print(f"\nBest model accuracy: {best_score:.3f}")
    return best_model, results

def main():
    """Main training function."""
    print("=== Smart File Type Prediction - Training ===")
    
    # Set paths - automatically use our dataset
    original_folder = "test_folder/original"
    print(f"Using dataset folder: {original_folder}")
    
    # Extract features
    features, labels, df = scan_folder_and_extract_features(original_folder)
    
    if features is None:
        print("No data to train on. Please check your folder path and files.")
        return
    
    # Remove unknown files
    if labels is not None and df is not None:
        valid_mask = labels != 'unknown'
        features = features[valid_mask]
        labels = labels[valid_mask]
        df = df[df['true_type'] != 'unknown'].reset_index(drop=True)
    
    if len(features) == 0 or labels is None:
        print("No recognizable file types found. Please add more diverse files.")
        return
    
    print(f"\nTraining on {len(features)} files with {len(np.unique(labels))} file types")
    
    # Train models
    best_model, results = train_models(features, labels)
    
    # Save model and data
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/file_classifier.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    if df is not None:
        data_path = 'models/training_data.csv'
        df.to_csv(data_path, index=False)
        print(f"Training data saved to: {data_path}")
    
    # Save feature info
    feature_info = {
        'num_features': features.shape[1],
        'file_types': list(np.unique(labels)),
        'num_files': len(features)
    }
    
    info_path = 'models/model_info.pkl'
    joblib.dump(feature_info, info_path)
    print(f"Model info saved to: {info_path}")
    
    print("\n=== Training Complete! ===")
    print("You can now run 'create_modified_files.py' to generate test data.")

if __name__ == "__main__":
    main()