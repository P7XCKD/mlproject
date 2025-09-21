#!/usr/bin/env python3
"""
Smart File Type Prediction - Create Modified Files
Copies files from 'original' folder and changes extensions to simulate suspicious files.
"""

import os
import shutil
import random
import pandas as pd
from pathlib import Path

# Common file extensions for disguising
DISGUISE_EXTENSIONS = [
    '.jpg', '.png', '.gif', '.pdf', '.txt', '.docx', 
    '.mp3', '.wav', '.zip', '.exe', '.csv', '.json'
]

def create_modified_folder(original_folder, modified_folder, change_probability=0.7):
    """
    Copy files from original folder and randomly change extensions while maintaining folder structure.
    
    Args:
        original_folder: Path to original files
        modified_folder: Path where modified files will be saved
        change_probability: Probability of changing an extension (0.0 to 1.0)
    """
    print(f"Creating modified files from: {original_folder}")
    print(f"Output folder: {modified_folder}")
    
    if not os.path.exists(original_folder):
        print(f"Error: Original folder {original_folder} does not exist!")
        return None
    
    # Create modified folder
    os.makedirs(modified_folder, exist_ok=True)
    
    # Track changes
    changes_log = []
    
    # Process all files while maintaining folder structure
    for root, dirs, files in os.walk(original_folder):
        for file in files:
            original_path = os.path.join(root, file)
            original_name, original_ext = os.path.splitext(file)
            
            # Get relative path to maintain folder structure
            relative_root = os.path.relpath(root, original_folder)
            if relative_root == '.':
                # Files in root directory
                target_subfolder = modified_folder
            else:
                # Files in subdirectories (PDF/, PNG/, TXT/)
                target_subfolder = os.path.join(modified_folder, relative_root)
                os.makedirs(target_subfolder, exist_ok=True)
            
            # Decide whether to change extension
            change_extension = random.random() < change_probability
            
            if change_extension:
                # Choose a random different extension
                available_extensions = [ext for ext in DISGUISE_EXTENSIONS if ext != original_ext.lower()]
                if available_extensions:
                    new_ext = random.choice(available_extensions)
                    new_filename = f"{original_name}{new_ext}"
                    status = "CHANGED"
                else:
                    new_filename = file
                    new_ext = original_ext
                    status = "UNCHANGED"
            else:
                new_filename = file
                new_ext = original_ext
                status = "UNCHANGED"
            
            # Copy file with new name to the appropriate subfolder
            modified_path = os.path.join(target_subfolder, new_filename)
            
            try:
                shutil.copy2(original_path, modified_path)
                
                changes_log.append({
                    'original_file': file,
                    'original_extension': original_ext,
                    'modified_file': new_filename,
                    'modified_extension': new_ext,
                    'status': status,
                    'original_path': original_path,
                    'modified_path': modified_path
                })
                
                print(f"{status}: {file} → {new_filename}")
                
            except Exception as e:
                print(f"Error copying {file}: {e}")
    
    # Save changes log
    if changes_log:
        df = pd.DataFrame(changes_log)
        log_path = os.path.join(modified_folder, 'changes_log.csv')
        df.to_csv(log_path, index=False)
        print(f"\nChanges log saved to: {log_path}")
        
        # Print summary
        changed_count = len(df[df['status'] == 'CHANGED'])
        unchanged_count = len(df[df['status'] == 'UNCHANGED'])
        
        print(f"\n=== Summary ===")
        print(f"Total files processed: {len(df)}")
        print(f"Extensions changed: {changed_count}")
        print(f"Extensions unchanged: {unchanged_count}")
        print(f"Change rate: {changed_count/len(df)*100:.1f}%")
        
        return df
    else:
        print("No files were processed.")
        return None

def create_specific_test_cases(original_folder, modified_folder):
    """Create specific test cases for demonstration while maintaining folder structure."""
    print("\n=== Creating Specific Test Cases ===")
    
    # Define specific suspicious combinations
    test_cases = [
        ('.pdf', '.jpg'),   # PDF disguised as image
        ('.exe', '.txt'),   # Executable disguised as text
        ('.zip', '.pdf'),   # Archive disguised as document
        ('.jpg', '.exe'),   # Image disguised as executable
    ]
    
    # Create specific_tests folder in each category
    created_tests = []
    
    for original_ext, fake_ext in test_cases:
        # Find a file with the original extension
        for root, dirs, files in os.walk(original_folder):
            for file in files:
                if file.lower().endswith(original_ext.lower()):
                    original_path = os.path.join(root, file)
                    original_name = os.path.splitext(file)[0]
                    
                    # Get relative path to maintain folder structure
                    relative_root = os.path.relpath(root, original_folder)
                    if relative_root == '.':
                        # Files in root directory
                        target_subfolder = os.path.join(modified_folder, 'specific_tests')
                    else:
                        # Files in subdirectories (PDF/, PNG/, TXT/)
                        target_subfolder = os.path.join(modified_folder, relative_root, 'specific_tests')
                    
                    os.makedirs(target_subfolder, exist_ok=True)
                    
                    # Create disguised version
                    fake_filename = f"{original_name}_disguised{fake_ext}"
                    fake_path = os.path.join(target_subfolder, fake_filename)
                    
                    try:
                        shutil.copy2(original_path, fake_path)
                        created_tests.append({
                            'original_file': file,
                            'fake_file': fake_filename,
                            'true_type': original_ext,
                            'fake_type': fake_ext,
                            'path': fake_path,
                            'folder': relative_root
                        })
                        print(f"Created in {relative_root}/: {file} → {fake_filename}")
                        break
                    except Exception as e:
                        print(f"Error creating test case {fake_filename}: {e}")
            else:
                continue
            break
    
    if created_tests:
        # Save overall test log
        test_df = pd.DataFrame(created_tests)
        test_log_path = os.path.join(modified_folder, 'specific_tests_log.csv')
        test_df.to_csv(test_log_path, index=False)
        print(f"Specific tests log saved to: {test_log_path}")
    
    return created_tests

def main():
    """Main function to create modified files."""
    print("=== Smart File Type Prediction - Create Modified Files ===")
    
    # Set folder paths automatically
    original_folder = "test_folder/original"
    modified_folder = "test_folder/modified"
    change_probability = 0.7  # 70% files will have wrong extensions
    
    print(f"Using original folder: {original_folder}")
    print(f"Using modified folder: {modified_folder}")
    print(f"Change probability: {change_probability} (70% suspicious, 30% legitimate)")
    
    # Create modified files
    changes_df = create_modified_folder(original_folder, modified_folder, change_probability)
    
    if changes_df is not None:
        # Create specific test cases
        test_cases = create_specific_test_cases(original_folder, modified_folder)
        
        print(f"\n=== Modified Files Created Successfully! ===")
        print(f"Modified folder: {modified_folder}")
        print("You can now run 'detect_suspicious_files.py' to test the detection system.")
    else:
        print("Failed to create modified files. Please check your folder paths.")

if __name__ == "__main__":
    main()