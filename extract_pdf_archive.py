#!/usr/bin/env python3
"""
Extract PDF files from archive (1).zip and add to PDF folder
"""

import os
import zipfile
import random
import shutil
from pathlib import Path

def extract_pdf_files(archive_path, output_folder, max_files=500):
    """Extract PDF files from archive to the output folder."""
    print(f"Processing archive: {archive_path}")
    
    if not os.path.exists(archive_path):
        print(f"Error: Archive {archive_path} not found!")
        return []
    
    extracted_files = []
    
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            print(f"Total files in archive: {len(all_files)}")
            
            # Filter for PDF files
            pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
            print(f"PDF files found: {len(pdf_files)}")
            
            if len(pdf_files) == 0:
                print("No PDF files found in archive!")
                return []
            
            # Randomly select files to extract
            num_to_extract = min(max_files, len(pdf_files))
            selected_files = random.sample(pdf_files, num_to_extract)
            
            print(f"Extracting {num_to_extract} random PDF files...")
            
            for file_path in selected_files:
                try:
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
                    
                    # Check file size
                    file_size = os.path.getsize(target_path)
                    if file_size > 10 * 1024 * 1024:  # Skip files larger than 10MB
                        os.remove(target_path)
                        print(f"  Skipped {filename} (too large: {file_size/1024/1024:.1f}MB)")
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
    
    print(f"\nSuccessfully extracted {len(extracted_files)} PDF files")
    return extracted_files

def main():
    """Main function to extract PDF files."""
    print("=== PDF Archive Extractor ===")
    
    archive_path = "c:/Probz/MLT PROJECT/test_folder/original/PDF/archive (1).zip"
    output_folder = "c:/Probz/MLT PROJECT/test_folder/original/PDF"
    
    extracted_files = extract_pdf_files(archive_path, output_folder, 500)
    
    if extracted_files:
        print(f"\n=== Extraction Complete! ===")
        print(f"Extracted {len(extracted_files)} PDF files")
        
        # Show updated file count
        total_files = len([f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))])
        print(f"Total files in PDF folder: {total_files}")
    else:
        print("No PDF files were extracted.")

if __name__ == "__main__":
    main()