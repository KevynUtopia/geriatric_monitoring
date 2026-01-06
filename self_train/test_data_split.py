#!/usr/bin/env python3
"""
Test script to verify the data splitting functionality works correctly.
This creates a small test dataset and runs the splitting process.
"""

import os
import shutil
import tempfile
from pathlib import Path
import sys

# Add the preprocessing directory to the path
sys.path.append(str(Path(__file__).parent / 'preprocessing'))

from data_split import DataSplitter

def create_test_dataset(test_dir, num_files=100):
    """Create a small test dataset for testing."""
    images_dir = test_dir / 'images'
    labels_dir = test_dir / 'labels'
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test files
    for i in range(num_files):
        base_name = f"test_file_{i:06d}"
        
        # Create dummy image file
        image_file = images_dir / f"{base_name}.jpg"
        with open(image_file, 'w') as f:
            f.write(f"dummy image {i}")
        
        # Create dummy label file
        label_file = labels_dir / f"{base_name}.txt"
        with open(label_file, 'w') as f:
            f.write(f"dummy label {i}")
    
    print(f"Created test dataset with {num_files} files")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    
    return images_dir, labels_dir

def verify_split(output_dir, expected_counts):
    """Verify the split was done correctly."""
    print("\nVerifying split...")
    
    success = True
    for split in ['train', 'val', 'test']:
        images_count = len(list((output_dir / 'images' / split).glob('*')))
        labels_count = len(list((output_dir / 'labels' / split).glob('*')))
        expected = expected_counts[split]
        
        print(f"{split}: {images_count} images, {labels_count} labels (expected: {expected})")
        
        if images_count != expected or labels_count != expected:
            print(f"  ❌ ERROR: Expected {expected} files each")
            success = False
        else:
            print(f"  ✅ OK")
    
    return success

def main():
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        print("=" * 60)
        print("Data Splitting Test")
        print("=" * 60)
        
        # Create test dataset
        num_files = 100
        images_dir, labels_dir = create_test_dataset(test_dir, num_files)
        
        # Output directory
        output_dir = test_dir / 'output'
        
        # Create splitter
        splitter = DataSplitter(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            output_dir=str(output_dir),
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2
        )
        
        # Run the splitting (Python phase only)
        print("\nRunning Python data splitting phase...")
        splitter.run_split(max_workers=4, seed=42)
        
        # Check that file lists were created
        file_lists_dir = output_dir / 'file_lists'
        if not file_lists_dir.exists():
            print("❌ ERROR: File lists directory not created")
            return False
        
        # Check that bash script was created
        copy_script = output_dir / 'copy_files.sh'
        if not copy_script.exists():
            print("❌ ERROR: Copy script not created")
            return False
        
        print("✅ Python phase completed successfully")
        print(f"File lists created: {file_lists_dir}")
        print(f"Copy script created: {copy_script}")
        
        # For testing, we'll simulate the bash phase by running the copy script
        print("\nRunning bash copying phase...")
        os.system(f"chmod +x {copy_script}")
        result = os.system(f"bash {copy_script}")
        
        if result != 0:
            print("❌ ERROR: Bash copying phase failed")
            return False
        
        # Verify the split
        expected_counts = {
            'train': int(num_files * 0.7),
            'val': int(num_files * 0.1),
            'test': num_files - int(num_files * 0.7) - int(num_files * 0.1)
        }
        
        success = verify_split(output_dir, expected_counts)
        
        if success:
            print("\n" + "=" * 60)
            print("✅ All tests passed! Data splitting works correctly.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ Some tests failed!")
            print("=" * 60)
            
        return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 