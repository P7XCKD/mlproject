#!/usr/bin/env python3
"""
Archive Extraction Script - Extract random text files from archive.zip
"""

import os
import zipfile
import random
import shutil
from pathlib import Path

def extract_random_txt_files(archive_path, output_folder, max_files=20):
    """
    Extract random text files from an archive to the output folder.
    
    Args:
        archive_path: Path to the archive file
        output_folder: Where to extract files
        max_files: Maximum number of text files to extract
    """
    print(f"Processing archive: {archive_path}")
    
    if not os.path.exists(archive_path):
        print(f"Error: Archive {archive_path} not found!")
        return []
    
    extracted_files = []
    
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Get all file names in the archive
            all_files = zip_ref.namelist()
            print(f"Total files in archive: {len(all_files)}")
            
            # Filter for text files (common text extensions)
            text_extensions = ['.txt', '.log', '.md', '.csv', '.json', '.xml', '.yaml', '.yml', '.ini', '.cfg', '.conf']
            text_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in text_extensions)]
            
            print(f"Text files found: {len(text_files)}")
            if len(text_files) == 0:
                print("No text files found in archive!")
                return []
            
            # Randomly select files to extract
            num_to_extract = min(max_files, len(text_files))
            selected_files = random.sample(text_files, num_to_extract)
            
            print(f"Extracting {num_to_extract} random text files...")
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            for file_path in selected_files:
                try:
                    # Get just the filename (remove directory structure)
                    filename = os.path.basename(file_path)
                    
                    # Handle duplicate filenames
                    counter = 1
                    original_filename = filename
                    while os.path.exists(os.path.join(output_folder, filename)):
                        name, ext = os.path.splitext(original_filename)
                        filename = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    # Extract file
                    source = zip_ref.open(file_path)
                    target_path = os.path.join(output_folder, filename)
                    
                    with open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    source.close()
                    
                    # Check file size (skip if too large or empty)
                    file_size = os.path.getsize(target_path)
                    if file_size > 1024 * 1024:  # Skip files larger than 1MB
                        os.remove(target_path)
                        print(f"  Skipped {filename} (too large: {file_size/1024:.1f}KB)")
                        continue
                    elif file_size == 0:  # Skip empty files
                        os.remove(target_path)
                        print(f"  Skipped {filename} (empty file)")
                        continue
                    
                    extracted_files.append(target_path)
                    print(f"  ✓ Extracted: {filename} ({file_size} bytes)")
                    
                except Exception as e:
                    print(f"  ❌ Failed to extract {file_path}: {e}")
            
    except zipfile.BadZipFile:
        print("Error: Invalid or corrupted zip file!")
        return []
    except Exception as e:
        print(f"Error processing archive: {e}")
        return []
    
    print(f"\nSuccessfully extracted {len(extracted_files)} text files to {output_folder}")
    return extracted_files

def main():
    """Main function to extract files from archive."""
    print("=== Archive Text File Extractor ===")
    
    # Set paths
    archive_path = "c:/Probz/MLT PROJECT/test_folder/original/archive.zip"
    output_folder = "c:/Probz/MLT PROJECT/test_folder/original"
    
    # Extract random text files (500 to balance with 582 PNG files)
    max_files = 500
    print(f"Extracting up to {max_files} text files to balance the dataset...")
    
    extracted_files = extract_random_txt_files(archive_path, output_folder, max_files)
    
    if extracted_files:
        print(f"\n=== Extraction Complete! ===")
        print(f"Extracted {len(extracted_files)} files:")
        for file_path in extracted_files:
            print(f"  - {os.path.basename(file_path)}")
        
        # Show updated file count
        total_files = len([f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))])
        print(f"\nTotal files in original folder: {total_files}")
    else:
        print("No files were extracted.")

if __name__ == "__main__":
    main()