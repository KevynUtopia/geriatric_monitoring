import os
import random
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

from tqdm import tqdm
import time

class DataSplitter:
    def __init__(self, images_dir: str, labels_dir: str, output_dir: str, 
                 train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """
        Initialize the data splitter.
        
        Args:
            images_dir: Path to source images directory
            labels_dir: Path to source labels directory
            output_dir: Path to output directory where split data will be stored
            train_ratio: Ratio for training set (default 0.7)
            val_ratio: Ratio for validation set (default 0.1)
            test_ratio: Ratio for test set (default 0.2)
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Create output directory structure
        self.splits = ['train', 'val', 'test']
        self.setup_output_directories()
    
    def setup_output_directories(self):
        """Create the output directory structure."""
        print(f"Creating output directory structure in: {self.output_dir}")
        
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for split in self.splits:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        print("Directory structure created successfully!")
    
    def get_all_base_names(self) -> List[str]:
        """
        Get all base names from image files.
        
        Returns:
            List of base names (without extensions)
        """
        print("Scanning for image files to get base names...")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(f'*{ext}')))
            image_files.extend(list(self.images_dir.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_files)} image files")
        
        # Extract base names
        base_names = []
        for image_path in tqdm(image_files, desc="Extracting base names"):
            base_name = image_path.stem
            base_names.append(base_name)
        
        print(f"Extracted {len(base_names)} base names")
        return base_names
    
    def split_data(self, base_names: List[str]) -> Dict[str, List[str]]:
        """
        Split the base names into train/val/test sets.
        
        Args:
            base_names: List of base names (without extensions)
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys and corresponding base name lists
        """
        print("Splitting data into train/val/test sets...")
        
        # Shuffle the data
        random.shuffle(base_names)
        
        total_files = len(base_names)
        train_size = int(total_files * self.train_ratio)
        val_size = int(total_files * self.val_ratio)
        
        # Split the data
        train_names = base_names[:train_size]
        val_names = base_names[train_size:train_size + val_size]
        test_names = base_names[train_size + val_size:]
        
        splits = {
            'train': train_names,
            'val': val_names,
            'test': test_names
        }
        
        print(f"Split sizes:")
        print(f"  Train: {len(train_names)} files ({len(train_names)/total_files*100:.1f}%)")
        print(f"  Val: {len(val_names)} files ({len(val_names)/total_files*100:.1f}%)")
        print(f"  Test: {len(test_names)} files ({len(test_names)/total_files*100:.1f}%)")
        
        return splits
    

    
    def find_image_extension(self, base_name: str) -> str:
        """
        Find the actual image extension for a given base name.
        
        Args:
            base_name: Base name without extension
            
        Returns:
            The image extension (e.g., '.jpg') or empty string if not found
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for ext in image_extensions:
            if (self.images_dir / f"{base_name}{ext}").exists():
                return ext
            if (self.images_dir / f"{base_name}{ext.upper()}").exists():
                return ext.upper()
        
        return ""
    
    def generate_file_lists(self, base_names: List[str], split: str) -> Tuple[List[str], List[str]]:
        """
        Generate file lists for a given split.
        
        Args:
            base_names: List of base names (without extensions)
            split: Split name ('train', 'val', 'test')
            
        Returns:
            Tuple of (image_files, label_files) lists
        """
        image_files = []
        label_files = []
        
        for base_name in tqdm(base_names):
            # Find the actual image extension
            # image_ext = self.find_image_extension(base_name)
            # if not image_ext:
            #     print(f"Warning: Could not find image file for base name: {base_name}")
            #     continue
            image_ext = '.jpg'
            # Construct source paths
            src_image = self.images_dir / f"{base_name}{image_ext}"
            src_label = self.labels_dir / f"{base_name}.txt"
            
            image_files.append(str(src_image))
            label_files.append(str(src_label))
        
        return image_files, label_files
    
    def write_file_lists(self, splits: Dict[str, List[str]]) -> None:
        """
        Write file lists to text files for bash script to use.
        
        Args:
            splits: Dictionary with split names and corresponding base names
        """
        print("Generating file lists for bash copying...")
        
        for split in self.splits:
            if not splits[split]:
                continue
                
            print(f"Generating {split} file lists...")
            
            # Generate file lists
            image_files, label_files = self.generate_file_lists(splits[split], split)
            
            # Create lists directory
            lists_dir = self.output_dir / 'file_lists'
            lists_dir.mkdir(exist_ok=True)
            
            # Write image files list
            with open(lists_dir / f"{split}_images.txt", 'w') as f:
                for img_file in image_files:
                    f.write(f"{img_file}\n")
            
            # Write label files list
            with open(lists_dir / f"{split}_labels.txt", 'w') as f:
                for label_file in label_files:
                    f.write(f"{label_file}\n")
            
            print(f"  {split}: {len(image_files)} images, {len(label_files)} labels")
        
        print(f"File lists written to: {lists_dir}")
        
        # Create the bash copying script
        self.create_copy_script()
    
    def create_copy_script(self) -> None:
        """
        Create a bash script to copy files using rsync.
        """
        script_path = self.output_dir / 'copy_files.sh'
        
        with open(script_path, 'w') as f:
            f.write(f"""#!/bin/bash

# Auto-generated file copying script for data splitting
# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

set -e  # Exit on error

OUTPUT_DIR="{self.output_dir}"
LISTS_DIR="$OUTPUT_DIR/file_lists"

echo "========================================"
echo "File Copying Script for Data Splitting"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo "Lists directory: $LISTS_DIR"
echo ""

# Function to copy files from a list
copy_files() {{
    local split=$1
    local file_type=$2
    local file_list="$LISTS_DIR/${{split}}_${{file_type}}.txt"
    local dest_dir="$OUTPUT_DIR/$file_type/$split"
    
    if [[ ! -f "$file_list" ]]; then
        echo "Warning: File list not found: $file_list"
        return 1
    fi
    
    local total_files=$(wc -l < "$file_list")
    echo "Copying $total_files $file_type files to $split set..."
    
    # Create destination directory
    mkdir -p "$dest_dir"
    
    # Copy files with progress
    local count=0
    while IFS= read -r src_file; do
        if [[ -f "$src_file" ]]; then
            filename=$(basename "$src_file")
            cp "$src_file" "$dest_dir/$filename"
            ((count++))
            
            # Show progress every 1000 files
            if (( count % 1000 == 0 )); then
                echo "  Copied $count/$total_files $file_type files to $split..."
            fi
        else
            echo "Warning: Source file not found: $src_file"
        fi
    done < "$file_list"
    
    echo "  Completed: $count/$total_files $file_type files copied to $split"
    return 0
}}

# Function to copy files using efficient batch processing
copy_files_batch() {{
    local split=$1
    local file_type=$2
    local file_list="$LISTS_DIR/${{split}}_${{file_type}}.txt"
    local dest_dir="$OUTPUT_DIR/$file_type/$split"
    
    if [[ ! -f "$file_list" ]]; then
        echo "Warning: File list not found: $file_list"
        return 1
    fi
    
    local total_files=$(wc -l < "$file_list")
    echo "Copying $total_files $file_type files to $split set using batch processing..."
    
    # Create destination directory
    mkdir -p "$dest_dir"
    
    # Use xargs for efficient batch copying
    # This processes files in parallel batches for better performance
    cat "$file_list" | xargs -I {{}} -P 4 sh -c 'cp "$1" "$2/$(basename "$1")"' _ {{}} "$dest_dir" 2>/dev/null || {{
        echo "Batch copy failed, falling back to individual file copying..."
        copy_files "$split" "$file_type"
        return $?
    }}
    
    # Verify the copy
    local copied_count=$(find "$dest_dir" -type f | wc -l)
    echo "  Completed: $copied_count/$total_files $file_type files copied to $split"
    
    if [[ $copied_count -ne $total_files ]]; then
        echo "  Warning: Expected $total_files files but found $copied_count files"
    fi
    
    return 0
}}

# Main copying process
echo "Starting file copying process..."
echo "Using batch processing for efficient copying..."

# Copy train set
echo ""
echo "=== Copying TRAIN set ==="
copy_files_batch "train" "images" &
copy_files_batch "train" "labels" &
wait

# Copy validation set
echo ""
echo "=== Copying VALIDATION set ==="
copy_files_batch "val" "images" &
copy_files_batch "val" "labels" &
wait

# Copy test set
echo ""
echo "=== Copying TEST set ==="
copy_files_batch "test" "images" &
copy_files_batch "test" "labels" &
wait

echo ""
echo "========================================"
echo "File copying completed!"
echo "========================================"

# Verify the split
echo ""
echo "Verifying split..."
for split in train val test; do
    img_count=$(find "$OUTPUT_DIR/images/$split" -type f | wc -l)
    label_count=$(find "$OUTPUT_DIR/labels/$split" -type f | wc -l)
    echo "$split set: $img_count images, $label_count labels"
done

echo ""
echo "Data splitting and copying completed successfully!"
echo "Output directory: $OUTPUT_DIR"
""")
        
        # Make the script executable
        import stat
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
        
        print(f"Bash copying script created: {script_path}")
        print("Run this script after the Python splitting completes:")
    
    def run_split(self, max_workers: int = 4, seed: int = 42) -> None:
        """
        Run the complete data splitting process.
        
        Args:
            max_workers: Maximum number of worker threads for copying
            seed: Random seed for reproducible splits
        """
        start_time = time.time()
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        print(f"Starting data split with seed: {seed}")
        print(f"Source directories:")
        print(f"  Images: {self.images_dir}")
        print(f"  Labels: {self.labels_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Split ratios: {self.train_ratio:.1f}:{self.val_ratio:.1f}:{self.test_ratio:.1f}")
        print("-" * 50)
        
        # Step 1: Get all base names
        base_names = self.get_all_base_names()
        
        if not base_names:
            print("No base names found. Exiting.")
            return
        
        # Step 2: Split data
        splits = self.split_data(base_names)
        
        # Step 3: Generate file lists for bash copying
        self.write_file_lists(splits)
        
        # Summary
        end_time = time.time()
        duration = end_time - start_time
        
        print("-" * 50)
        print("Data splitting completed!")
        print(f"Total time: {duration:.2f} seconds")
        print(f"Total files processed: {len(base_names)} base names")
        print(f"Output directory: {self.output_dir}")
        
        # Instructions for next step
        print("\n" + "="*50)
        print("NEXT STEP: Run the bash copying script")
        print("="*50)
        copy_script_path = self.output_dir / 'copy_files.sh'
        print(f"Execute: {copy_script_path}")
        print("Or simply run: bash copy_files.sh")
        print("\nThis script will:")
        print("- Copy files efficiently using rsync")
        print("- Show progress for each split")
        print("- Verify the final split")
        print("- Handle errors gracefully")
        print("="*50)
    
    def verify_split(self) -> None:
        """Verify the split by counting files in each directory."""
        print("\nVerifying split...")
        
        for split in self.splits:
            images_count = len(list((self.output_dir / 'images' / split).glob('*')))
            labels_count = len(list((self.output_dir / 'labels' / split).glob('*')))
            
            print(f"{split.capitalize()} set: {images_count} images, {labels_count} labels")
            
            if images_count != labels_count:
                print(f"WARNING: Mismatch in {split} set!")


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to images directory')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Path to labels directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits (default: 42)')
    
    args = parser.parse_args()
    
    # Create splitter and run
    splitter = DataSplitter(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    splitter.run_split(max_workers=args.max_workers, seed=args.seed)


if __name__ == "__main__":
    # Example usage for the specific paths mentioned
    # Uncomment and modify as needed
    
    # images_dir = 'path_to_your_root/datasets/snh-pose/images'
    # labels_dir = 'path_to_your_root/datasets/snh-pose/labels'
    # output_dir = '/path/to/output/snh-pose-split'
    # 
    # splitter = DataSplitter(images_dir, labels_dir, output_dir)
    # splitter.run_split(max_workers=16)  # Use more workers for large datasets
    
    main()
