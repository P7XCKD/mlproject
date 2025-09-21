#!/usr/bin/env python3
"""
Smart File Type Prediction - Create Adversarial Test Cases
Creates challenging test cases to test the limits of the ML model.
"""

import os
import random
import pandas as pd
from pathlib import Path

def create_adversarial_test_cases(output_folder):
    """Create adversarial test cases that are intentionally hard to detect."""
    print(f"Creating adversarial test cases in: {output_folder}")
    
    # Create output structure
    for file_type in ['PDF', 'PNG', 'TXT']:
        os.makedirs(os.path.join(output_folder, file_type), exist_ok=True)
    
    test_cases = []
    file_counter = 1
    
    # 1. Mixed content files (start with one type, end with another)
    print("Creating mixed content files...")
    
    # PDF header + PNG content
    pdf_header = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n'
    png_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
    mixed_content = pdf_header + b'\n' * 50 + png_content
    
    filepath = os.path.join(output_folder, 'PDF', f'adversarial_{file_counter:04d}_mixed.exe')
    with open(filepath, 'wb') as f:
        f.write(mixed_content)
    
    test_cases.append({
        'file_id': file_counter,
        'filename': f'adversarial_{file_counter:04d}_mixed.exe',
        'true_type': 'pdf',  # Starts with PDF
        'claimed_extension': '.exe',
        'status': 'DECEPTIVE',
        'folder': 'PDF',
        'filepath': filepath,
        'challenge_type': 'Mixed Content',
        'description': 'PDF header followed by PNG content'
    })
    file_counter += 1
    
    # 2. Minimal magic numbers with padding
    print("Creating minimal magic number files...")
    
    # Very minimal PDF
    minimal_pdf = b'%PDF' + b'\x00' * 1020
    filepath = os.path.join(output_folder, 'PDF', f'adversarial_{file_counter:04d}_minimal.jpg')
    with open(filepath, 'wb') as f:
        f.write(minimal_pdf)
    
    test_cases.append({
        'file_id': file_counter,
        'filename': f'adversarial_{file_counter:04d}_minimal.jpg',
        'true_type': 'pdf',
        'claimed_extension': '.jpg',
        'status': 'DECEPTIVE',
        'folder': 'PDF',
        'filepath': filepath,
        'challenge_type': 'Minimal Magic',
        'description': 'Minimal PDF signature with null padding'
    })
    file_counter += 1
    
    # 3. Corrupted/partial magic numbers
    print("Creating corrupted magic number files...")
    
    # Almost PNG but corrupted
    corrupted_png = b'\x89PN' + b'G\r\n\x1a\n' + b'corrupted_data' * 100
    filepath = os.path.join(output_folder, 'PNG', f'adversarial_{file_counter:04d}_corrupted.pdf')
    with open(filepath, 'wb') as f:
        f.write(corrupted_png)
    
    test_cases.append({
        'file_id': file_counter,
        'filename': f'adversarial_{file_counter:04d}_corrupted.pdf',
        'true_type': 'png',
        'claimed_extension': '.pdf',
        'status': 'DECEPTIVE',
        'folder': 'PNG',
        'filepath': filepath,
        'challenge_type': 'Corrupted Magic',
        'description': 'Slightly corrupted PNG signature'
    })
    file_counter += 1
    
    # 4. Text with binary-like patterns
    print("Creating text with binary patterns...")
    
    # Text that looks like binary
    binary_text = "\\x89\\x50\\x4E\\x47\\x0D\\x0A\\x1A\\x0A\\x00\\x00\\x00\\x0D\\x49\\x48\\x44\\x52" * 20
    binary_text += "\\nThis is actually a text file disguised as binary data.\\n"
    binary_text += "It contains escaped binary sequences but is pure text.\\n"
    
    filepath = os.path.join(output_folder, 'TXT', f'adversarial_{file_counter:04d}_binary_text.png')
    with open(filepath, 'wb') as f:
        f.write(binary_text.encode('utf-8'))
    
    test_cases.append({
        'file_id': file_counter,
        'filename': f'adversarial_{file_counter:04d}_binary_text.png',
        'true_type': 'txt',
        'claimed_extension': '.png',
        'status': 'DECEPTIVE',
        'folder': 'TXT',
        'filepath': filepath,
        'challenge_type': 'Binary-like Text',
        'description': 'Text file with escaped binary sequences'
    })
    file_counter += 1
    
    # 5. Empty and very small files
    print("Creating edge case files...")
    
    # Very small PDF
    tiny_pdf = b'%PDF-1.4'
    filepath = os.path.join(output_folder, 'PDF', f'adversarial_{file_counter:04d}_tiny.txt')
    with open(filepath, 'wb') as f:
        f.write(tiny_pdf)
    
    test_cases.append({
        'file_id': file_counter,
        'filename': f'adversarial_{file_counter:04d}_tiny.txt',
        'true_type': 'pdf',
        'claimed_extension': '.txt',
        'status': 'DECEPTIVE',
        'folder': 'PDF',
        'filepath': filepath,
        'challenge_type': 'Tiny File',
        'description': 'Very small PDF file'
    })
    file_counter += 1
    
    # 6. Polyglot files (valid in multiple formats)
    print("Creating polyglot files...")
    
    # File that's both valid text and has binary signature
    polyglot_content = b'%PDF-1.4\\nThis file has a PDF header but is also readable text.\\n'
    polyglot_content += b'Line 1: This is line one of the text content\\n'
    polyglot_content += b'Line 2: This is line two of the text content\\n'
    polyglot_content += b'Line 3: This could be interpreted as either PDF or text\\n'
    
    filepath = os.path.join(output_folder, 'PDF', f'adversarial_{file_counter:04d}_polyglot.mp3')
    with open(filepath, 'wb') as f:
        f.write(polyglot_content)
    
    test_cases.append({
        'file_id': file_counter,
        'filename': f'adversarial_{file_counter:04d}_polyglot.mp3',
        'true_type': 'pdf',  # PDF signature takes precedence
        'claimed_extension': '.mp3',
        'status': 'DECEPTIVE',
        'folder': 'PDF',
        'filepath': filepath,
        'challenge_type': 'Polyglot',
        'description': 'File valid as both PDF and text'
    })
    file_counter += 1
    
    # 7. Legitimate files with suspicious extensions
    print("Creating legitimate but suspicious files...")
    
    # Real PNG with misleading extension
    real_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x02\x00\x00\x00\xfd\xd4\x9as\x00\x00\x00\x1fIDATx\x9cc\xf8\x0f\x00\x00\x00\x00\xff\xffc\xf8\x0f\x00\x00\x00\x00\xff\xff\x03\x00\x00\x06\x00\x05]\xc2\xd4/\x00\x00\x00\x00IEND\xaeB`\x82'
    
    filepath = os.path.join(output_folder, 'PNG', f'adversarial_{file_counter:04d}_real.exe')
    with open(filepath, 'wb') as f:
        f.write(real_png)
    
    test_cases.append({
        'file_id': file_counter,
        'filename': f'adversarial_{file_counter:04d}_real.exe',
        'true_type': 'png',
        'claimed_extension': '.exe',
        'status': 'DECEPTIVE',
        'folder': 'PNG',
        'filepath': filepath,
        'challenge_type': 'Real File, Wrong Extension',
        'description': 'Legitimate PNG with .exe extension'
    })
    file_counter += 1
    
    # 8. Files with correct extensions (control group)
    print("Creating control group files...")
    
    # Normal text file with correct extension
    normal_text = b'This is a normal text file with proper content.\\n'
    normal_text += b'It should be correctly identified as text.\\n'
    normal_text += b'Extension matches content type.\\n'
    
    filepath = os.path.join(output_folder, 'TXT', f'adversarial_{file_counter:04d}_normal.txt')
    with open(filepath, 'wb') as f:
        f.write(normal_text)
    
    test_cases.append({
        'file_id': file_counter,
        'filename': f'adversarial_{file_counter:04d}_normal.txt',
        'true_type': 'txt',
        'claimed_extension': '.txt',
        'status': 'LEGITIMATE',
        'folder': 'TXT',
        'filepath': filepath,
        'challenge_type': 'Control - Normal',
        'description': 'Normal text file with correct extension'
    })
    file_counter += 1
    
    # Save test log
    df = pd.DataFrame(test_cases)
    log_path = os.path.join(output_folder, 'adversarial_test_log.csv')
    df.to_csv(log_path, index=False)
    
    # Print summary
    total_files = len(df)
    deceptive_count = len(df[df['status'] == 'DECEPTIVE'])
    legitimate_count = len(df[df['status'] == 'LEGITIMATE'])
    
    print(f"\\n=== Adversarial Test Dataset Created ===")
    print(f"Total files: {total_files}")
    print(f"Deceptive files: {deceptive_count} ({deceptive_count/total_files*100:.1f}%)")
    print(f"Legitimate files: {legitimate_count} ({legitimate_count/total_files*100:.1f}%)")
    print(f"Test log saved: {log_path}")
    
    # Show breakdown by challenge type
    print(f"\\nBreakdown by challenge type:")
    for challenge_type in df['challenge_type'].unique():
        count = len(df[df['challenge_type'] == challenge_type])
        print(f"  {challenge_type}: {count} files")
    
    return df

def main():
    """Main function to create adversarial test data."""
    print("=== Smart File Type Prediction - Create Adversarial Test Cases ===")
    
    output_folder = "test_folder/adversarial_test"
    
    print(f"Output folder: {output_folder}")
    print("Creating challenging test cases to push the model's limits...")
    
    # Create the adversarial test dataset
    test_df = create_adversarial_test_cases(output_folder)
    
    if test_df is not None:
        print(f"\\n=== Adversarial Test Dataset Ready! ===")
        print("This dataset contains challenging edge cases and adversarial examples.")
        print("These test cases will reveal the true limits of the ML model.")
        print(f"Next step: Run detection on '{output_folder}' to test model robustness.")
    else:
        print("Failed to create adversarial test dataset.")

if __name__ == "__main__":
    main()