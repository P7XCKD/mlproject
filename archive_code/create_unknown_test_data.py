#!/usr/bin/env python3
"""
Smart File Type Prediction - Create Unknown Test Data
Creates completely new test files that the ML model has never seen before.
This provides a realistic evaluation of model performance on unknown data.
"""

import os
import shutil
import random
import pandas as pd
from pathlib import Path
import time

# Extensions for creating deceptive files
DECEPTIVE_EXTENSIONS = [
    '.jpg', '.png', '.gif', '.pdf', '.txt', '.docx', 
    '.mp3', '.wav', '.zip', '.exe', '.csv', '.json',
    '.html', '.xml', '.ppt', '.xls', '.dll', '.bat'
]

def create_sample_files():
    """Create sample files with actual content of different types."""
    samples = {
        'pdf_samples': [
            b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF',
            b'%PDF-1.5\n%\xc4\xe5\xf2\xe5\xeb\xa7\xf3\xa0\xd0\xc4\xc6\n1 0 obj\n<<\n/Type /Catalog\n/Version /1.5\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 595.276 841.89]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Test Document) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000015 00000 n \n0000000066 00000 n \n0000000123 00000 n \n0000000213 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n308\n%%EOF',
        ],
        'png_samples': [
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x12IDATx\x9cc```b```\x02\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82',
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x02\x00\x00\x00\xfd\xd4\x9as\x00\x00\x00\x1fIDATx\x9cc\xf8\x0f\x00\x00\x00\x00\xff\xffc\xf8\x0f\x00\x00\x00\x00\xff\xff\x03\x00\x00\x06\x00\x05]\xc2\xd4/\x00\x00\x00\x00IEND\xaeB`\x82',
        ],
        'txt_samples': [
            b'This is a test document.\nIt contains multiple lines of text.\nThis file should be recognized as a text file.\n\nLine 1: Sample content\nLine 2: More content\nLine 3: Additional text\n\nEnd of document.',
            b'TECHNICAL REPORT\n================\n\nDate: 2025-09-21\nAuthor: Test System\nSubject: File Analysis\n\nThis document contains important information about file type detection.\nThe system should correctly identify this as a text document.\n\nData points:\n- Point 1: Text classification\n- Point 2: Binary analysis\n- Point 3: Pattern recognition\n\nConclusion: This is plaintext content.',
            b'Configuration File\n###################\n\n[Settings]\nmode=production\ndebug=false\nversion=2.1.0\n\n[Database]\nhost=localhost\nport=5432\nname=testdb\n\n[Security]\nencryption=enabled\nprotocol=TLS1.3\n\n# End of configuration',
        ]
    }
    return samples

def create_unknown_test_dataset(output_folder, num_files_per_type=100, deception_rate=0.6):
    """
    Create unknown test dataset with realistic files.
    
    Args:
        output_folder: Where to save the test files
        num_files_per_type: Number of files per type to generate
        deception_rate: Percentage of files with wrong extensions (0.0 to 1.0)
    """
    print(f"Creating unknown test dataset in: {output_folder}")
    print(f"Files per type: {num_files_per_type}")
    print(f"Deception rate: {deception_rate*100}%")
    
    # Create output structure
    for file_type in ['PDF', 'PNG', 'TXT']:
        os.makedirs(os.path.join(output_folder, file_type), exist_ok=True)
    
    # Get sample file content
    samples = create_sample_files()
    
    changes_log = []
    file_counter = 1
    
    # Generate PDF files
    print("\nGenerating PDF test files...")
    for i in range(num_files_per_type):
        original_content = random.choice(samples['pdf_samples'])
        filename_base = f"unknown_pdf_{file_counter:04d}"
        
        # Decide if this file should be deceptive
        is_deceptive = random.random() < deception_rate
        
        if is_deceptive:
            # Choose a wrong extension
            wrong_ext = random.choice([ext for ext in DECEPTIVE_EXTENSIONS if ext != '.pdf'])
            filename = f"{filename_base}{wrong_ext}"
            status = "DECEPTIVE"
        else:
            filename = f"{filename_base}.pdf"
            status = "LEGITIMATE"
        
        # Save file
        filepath = os.path.join(output_folder, 'PDF', filename)
        with open(filepath, 'wb') as f:
            f.write(original_content)
        
        changes_log.append({
            'file_id': file_counter,
            'filename': filename,
            'true_type': 'pdf',
            'claimed_extension': os.path.splitext(filename)[1],
            'status': status,
            'folder': 'PDF',
            'filepath': filepath
        })
        
        file_counter += 1
    
    # Generate PNG files
    print("Generating PNG test files...")
    for i in range(num_files_per_type):
        original_content = random.choice(samples['png_samples'])
        filename_base = f"unknown_png_{file_counter:04d}"
        
        is_deceptive = random.random() < deception_rate
        
        if is_deceptive:
            wrong_ext = random.choice([ext for ext in DECEPTIVE_EXTENSIONS if ext != '.png'])
            filename = f"{filename_base}{wrong_ext}"
            status = "DECEPTIVE"
        else:
            filename = f"{filename_base}.png"
            status = "LEGITIMATE"
        
        filepath = os.path.join(output_folder, 'PNG', filename)
        with open(filepath, 'wb') as f:
            f.write(original_content)
        
        changes_log.append({
            'file_id': file_counter,
            'filename': filename,
            'true_type': 'png',
            'claimed_extension': os.path.splitext(filename)[1],
            'status': status,
            'folder': 'PNG',
            'filepath': filepath
        })
        
        file_counter += 1
    
    # Generate TXT files
    print("Generating TXT test files...")
    for i in range(num_files_per_type):
        original_content = random.choice(samples['txt_samples'])
        # Add some randomization to text content
        random_suffix = f"\n\nFile ID: {file_counter}\nGenerated: 2025-09-21\nRandom seed: {random.randint(1000, 9999)}"
        full_content = original_content + random_suffix.encode('utf-8')
        
        filename_base = f"unknown_txt_{file_counter:04d}"
        
        is_deceptive = random.random() < deception_rate
        
        if is_deceptive:
            wrong_ext = random.choice([ext for ext in DECEPTIVE_EXTENSIONS if ext != '.txt'])
            filename = f"{filename_base}{wrong_ext}"
            status = "DECEPTIVE"
        else:
            filename = f"{filename_base}.txt"
            status = "LEGITIMATE"
        
        filepath = os.path.join(output_folder, 'TXT', filename)
        with open(filepath, 'wb') as f:
            f.write(full_content)
        
        changes_log.append({
            'file_id': file_counter,
            'filename': filename,
            'true_type': 'txt',
            'claimed_extension': os.path.splitext(filename)[1],
            'status': status,
            'folder': 'TXT',
            'filepath': filepath
        })
        
        file_counter += 1
    
    # Save test log
    df = pd.DataFrame(changes_log)
    log_path = os.path.join(output_folder, 'unknown_test_log.csv')
    df.to_csv(log_path, index=False)
    
    # Print summary
    total_files = len(df)
    deceptive_count = len(df[df['status'] == 'DECEPTIVE'])
    legitimate_count = len(df[df['status'] == 'LEGITIMATE'])
    
    print(f"\n=== Unknown Test Dataset Created ===")
    print(f"Total files: {total_files}")
    print(f"Deceptive files: {deceptive_count} ({deceptive_count/total_files*100:.1f}%)")
    print(f"Legitimate files: {legitimate_count} ({legitimate_count/total_files*100:.1f}%)")
    print(f"Test log saved: {log_path}")
    
    # Show breakdown by type
    print(f"\nBreakdown by true type:")
    for file_type in ['pdf', 'png', 'txt']:
        type_df = df[df['true_type'] == file_type]
        type_deceptive = len(type_df[type_df['status'] == 'DECEPTIVE'])
        print(f"  {file_type.upper()}: {len(type_df)} files ({type_deceptive} deceptive)")
    
    return df

def main():
    """Main function to create unknown test data."""
    print("=== Smart File Type Prediction - Create Unknown Test Data ===")
    
    # Set parameters
    output_folder = "test_folder/unknown_test"
    files_per_type = 50  # Smaller dataset for realistic testing
    deception_rate = 0.6  # 60% of files will have wrong extensions
    
    print(f"Output folder: {output_folder}")
    print(f"Files per type: {files_per_type}")
    print(f"Deception rate: {deception_rate*100}%")
    
    # Create the unknown test dataset
    test_df = create_unknown_test_dataset(output_folder, files_per_type, deception_rate)
    
    if test_df is not None:
        print(f"\n=== Unknown Test Dataset Ready! ===")
        print("This dataset contains completely new files that the ML model has never seen.")
        print("You can now test the model's real-world performance on truly unknown data.")
        print(f"Next step: Run detection on '{output_folder}' to get realistic accuracy metrics.")
    else:
        print("Failed to create unknown test dataset.")

if __name__ == "__main__":
    main()