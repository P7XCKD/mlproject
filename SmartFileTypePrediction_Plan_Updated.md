# Smart File Type Prediction: Project Plan

## Overview
This project builds a machine learning system to predict the true type of a file by analyzing its raw byte data, helping to flag suspicious files that may have misleading extensions.

---

## Project Structure
```
test_folder/
├── original/          # Contains files with correct extensions
│   ├── document.pdf
│   ├── image.jpg
│   ├── music.mp3
│   ├── program.exe
│   └── ...
├── modified/          # Contains same files with wrong extensions (auto-generated)
│   ├── document.jpg   # PDF file disguised as JPG
│   ├── image.exe      # JPG file disguised as EXE
│   └── ...
└── models/            # Trained ML models and data
    ├── file_classifier.pkl
    └── training_data.csv
```

---

## Workflow Steps

### 1. Training Phase
- **Input**: `original/` folder with correctly labeled files
- **Process**: 
  - Extract first 1024 bytes from each file
  - Detect true file type using magic numbers and byte patterns
  - Label each file with its actual type (not extension)
  - Train ML classifier on byte patterns vs true file types
- **Output**: Trained model saved as `file_classifier.pkl`

### 2. Simulation Phase (Automated)
- **Process**:
  - Copy all files from `original/` to `modified/`
  - Randomly change file extensions to create "suspicious" files
  - Example: `document.pdf` → `document.jpg` (PDF content with JPG extension)
- **Output**: `modified/` folder with files having wrong extensions

### 3. Detection Phase
- **Input**: Files from `modified/` folder
- **Process**:
  - Extract first 1024 bytes from each file
  - Predict actual file type using trained ML model
  - Compare predicted type with file extension
  - Flag mismatches as suspicious
- **Output**: Detection results showing which files are suspicious

### 4. Evaluation & Results
- Show accuracy of detection system
- Display flagged files with their true vs claimed types
- Generate confusion matrix and performance metrics

---

## Implementation Scripts

### 1. `train_model.py`
- Processes `original/` folder
- Extracts byte features and trains ML model
- Saves trained model for later use

### 2. `create_modified_files.py`
- Copies files from `original/` to `modified/`
- Randomly changes extensions to simulate attacks
- Creates ground truth for testing

### 3. `detect_suspicious_files.py`
- Loads trained model
- Tests files in `modified/` folder
- Flags mismatches between predicted and claimed types

### 4. `evaluate_results.py`
- Analyzes detection performance
- Generates visualization and metrics
- Shows which files were correctly flagged

---

## Technical Details
- **Languages**: Python
- **Libraries**: numpy, pandas, scikit-learn, matplotlib, seaborn
- **Features**: Raw byte vectors, byte histograms, magic number patterns
- **Models**: Random Forest, Logistic Regression, optional CNN
- **Byte Size**: First 1024 bytes per file

---

## What You Provide
- `test_folder/original/` with sample files of various types

## What I Will Build
- Complete automation from training to detection
- All 4 Python scripts with full functionality
- Visualization of results and performance metrics

---

## Expected Output
```
=== File Type Prediction Results ===
✓ document.pdf → Predicted: PDF, Extension: PDF (SAFE)
⚠ document.jpg → Predicted: PDF, Extension: JPG (SUSPICIOUS!)
✓ image.png → Predicted: PNG, Extension: PNG (SAFE)
⚠ image.exe → Predicted: PNG, Extension: EXE (SUSPICIOUS!)

Detection Accuracy: 95%
Suspicious Files Found: 2/4
```