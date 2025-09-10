#!/usr/bin/env python3
"""
Jupyter Notebook Path Sanitizer

This script recursively processes all Jupyter notebook files (.ipynb) in a directory
and replaces file/folder paths with a generic placeholder ("....") to sanitize
notebooks for sharing while preserving code structure.

The script handles various path formats including:
- Google Colab paths (/content/drive/MyDrive/...)
- Local filesystem paths (/Users/..., /home/..., C:\...)
- Relative paths (./data/..., ../files/...)
- Model and data file paths with common extensions

Common use cases:
- Preparing notebooks for public sharing
- Removing sensitive path information
- Standardizing notebook examples

Usage:
    python3 replace_paths.py

Author: Generated for DrugDiscovery-ML-Code-Data-Figures-Notebooks project
Date: September 2025
"""

import os
import json
import re
from pathlib import Path

def replace_paths_in_notebook(notebook_path):
    """Replace file/folder paths with '..' in a Jupyter notebook."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = json.load(f)
        
        # Track if any changes were made
        changes_made = False
        
        # Iterate through all cells
        for cell in notebook_content.get('cells', []):
            if 'source' in cell:
                # Process each source line
                new_source = []
                for line in cell['source']:
                    original_line = line
                    
                    # Replace various path patterns with "...."
                    patterns_to_replace = [
                        # Google Colab drive paths
                        r"'/content/drive/MyDrive/[^']*'",
                        r'"/content/drive/MyDrive/[^"]*"',
                        r"'/content/drive'",
                        r'"/content/drive"',
                        
                        # Local file paths (starting with /)
                        r"'/[^']+\.(xlsx?|csv|txt|json|pkl|h5|hdf5|parquet)'",
                        r'"/[^"]+\.(xlsx?|csv|txt|json|pkl|h5|hdf5|parquet)"',
                        
                        # Windows paths
                        r"'[A-Z]:\\[^']*'",
                        r'"[A-Z]:\\[^"]*"',
                        
                        # Relative paths with common data file extensions
                        r"'[./\\][^']+\.(xlsx?|csv|txt|json|pkl|h5|hdf5|parquet)'",
                        r'"[./\\][^"]+\.(xlsx?|csv|txt|json|pkl|h5|hdf5|parquet)"',
                        
                        # Model file paths
                        r"'[^']+\.(model|joblib|pickle|pkl|pt|pth|ckpt|h5)'",
                        r'"[^"]+\.(model|joblib|pickle|pkl|pt|pth|ckpt|h5)"',
                    ]
                    
                    for pattern in patterns_to_replace:
                        line = re.sub(pattern, '"...."', line)
                    
                    # Special replacements for specific function calls
                    # Replace file paths in common pandas and file operations
                    function_patterns = [
                        (r"(pd\.read_excel\s*\(\s*)['\"][^'\"]+['\"]", r'\1"....."'),
                        (r"(pd\.read_csv\s*\(\s*)['\"][^'\"]+['\"]", r'\1"....."'),
                        (r"(\.to_excel\s*\(\s*)['\"][^'\"]+['\"]", r'\1"....."'),
                        (r"(\.to_csv\s*\(\s*)['\"][^'\"]+['\"]", r'\1"....."'),
                        (r"(open\s*\(\s*)['\"][^'\"]+['\"]", r'\1"....."'),
                        (r"(joblib\.load\s*\(\s*)['\"][^'\"]+['\"]", r'\1"....."'),
                        (r"(joblib\.dump\s*\([^,]+,\s*)['\"][^'\"]+['\"]", r'\1"....."'),
                        (r"(torch\.save\s*\([^,]+,\s*)['\"][^'\"]+['\"]", r'\1"....."'),
                        (r"(torch\.load\s*\(\s*)['\"][^'\"]+['\"]", r'\1"....."'),
                    ]
                    
                    for pattern, replacement in function_patterns:
                        line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)
                    
                    # Check if the line was modified
                    if line != original_line:
                        changes_made = True
                    
                    new_source.append(line)
                
                cell['source'] = new_source
        
        # Save the modified notebook if changes were made
        if changes_made:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_content, f, indent=1, ensure_ascii=False)
            print(f"✓ Updated: {notebook_path}")
            return True
        else:
            print(f"- No changes: {notebook_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {notebook_path}: {str(e)}")
        return False

def main():
    """Find and process all notebook files in the current directory."""
    # Get current working directory
    root_dir = Path('/Users/nb/DrugDiscovery-ML-Code-Data-Figures-Notebooks')
    
    # Find all .ipynb files
    notebook_files = list(root_dir.rglob('*.ipynb'))
    
    if not notebook_files:
        print("No notebook files found.")
        return
    
    print(f"Found {len(notebook_files)} notebook files to process...")
    print("=" * 60)
    
    updated_count = 0
    
    # Process each notebook
    for notebook_path in sorted(notebook_files):
        if replace_paths_in_notebook(notebook_path):
            updated_count += 1
    
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Total notebooks processed: {len(notebook_files)}")
    print(f"Notebooks updated: {updated_count}")
    print(f"Notebooks unchanged: {len(notebook_files) - updated_count}")

if __name__ == "__main__":
    main()
