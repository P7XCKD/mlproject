Smart File Type Prediction
Problem:
Just checking a file’s name or extension isn’t safe, because someone can rename a harmful file (like a program .exe) to look like a picture .jpg.

ML Solution
The model looks inside the file’s raw data (its first bytes), learns the hidden patterns of each file type, and predicts what the file really is. If that doesn’t match the given extension, it’s flagged as suspicious.
***
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

]

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