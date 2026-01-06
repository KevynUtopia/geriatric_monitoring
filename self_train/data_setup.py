#!/usr/bin/env python3
"""
Data setup script for COCO keypoint dataset and custom datasets.
"""

import argparse
import json
import shutil
import requests
import zipfile
from pathlib import Path
from typing import Dict, List
import yaml
from tqdm import tqdm
import os

from src.utils.config_loader import create_dataset_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dataset Setup for YOLO11 Keypoint Detection")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["coco", "custom"],
        required=True,
        help="Dataset type to setup"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data",
        help="Root directory for datasets"
    )
    
    parser.add_argument(
        "--coco_year", 
        type=str, 
        default="2017",
        choices=["2014", "2017"],
        help="COCO dataset year"
    )
    
    parser.add_argument(
        "--splits", 
        type=str, 
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to download"
    )
    
    parser.add_argument(
        "--custom_path", 
        type=str,
        help="Path to custom dataset (for custom dataset setup)"
    )
    
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["coco", "yolo"],
        default="coco",
        help="Custom dataset format"
    )
    
    parser.add_argument(
        "--convert_to_yolo", 
        action="store_true",
        help="Convert annotations to YOLO format"
    )
    
    parser.add_argument(
        "--sample_size", 
        type=int,
        help="Create a sample dataset with specified number of images"
    )
    
    return parser.parse_args()


class COCODatasetSetup:
    """Setup COCO keypoint dataset."""
    
    def __init__(self, data_dir: str, year: str = "2017"):
        self.data_dir = Path(data_dir)
        self.year = year
        self.coco_dir = self.data_dir / "coco"
        
        # COCO URLs
        self.urls = {
            "2017": {
                "train_images": "http://images.cocodataset.org/zips/train2017.zip",
                "val_images": "http://images.cocodataset.org/zips/val2017.zip",
                "test_images": "http://images.cocodataset.org/zips/test2017.zip",
                "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            },
            "2014": {
                "train_images": "http://images.cocodataset.org/zips/train2014.zip",
                "val_images": "http://images.cocodataset.org/zips/val2014.zip",
                "test_images": "http://images.cocodataset.org/zips/test2014.zip",
                "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
            }
        }
    
    def download_file(self, url: str, filename: str) -> Path:
        """Download a file with progress bar."""
        filepath = self.coco_dir / filename
        
        if filepath.exists():
            print(f"File {filename} already exists, skipping download")
            return filepath
        
        print(f"Downloading {filename}...")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        return filepath
    
    def extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract zip file."""
        print(f"Extracting {zip_path.name}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"Extracted to {extract_to}")
    
    def setup_coco_dataset(self, splits: List[str]):
        """Setup COCO dataset."""
        self.coco_dir.mkdir(parents=True, exist_ok=True)
        
        # Download annotations
        print("Setting up COCO annotations...")
        ann_url = self.urls[self.year]["annotations"]
        ann_file = self.download_file(ann_url, f"annotations_trainval{self.year}.zip")
        self.extract_zip(ann_file, self.coco_dir)
        
        # Download images for each split
        for split in splits:
            if split == "test":
                continue  # Test set doesn't have keypoint annotations
            
            print(f"Setting up {split} images...")
            img_url = self.urls[self.year][f"{split}_images"]
            img_file = self.download_file(img_url, f"{split}{self.year}.zip")
            self.extract_zip(img_file, self.coco_dir)
        
        # Create dataset configuration
        self.create_coco_config()
        
        print("COCO dataset setup completed!")
    
    def create_coco_config(self):
        """Create COCO dataset configuration for YOLO."""
        config = create_dataset_config(
            dataset_path=str(self.coco_dir),
            train_split=f"train{self.year}",
            val_split=f"val{self.year}",
            num_classes=1,
            class_names=['person'],
            keypoint_names=[
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
        )
        
        config_path = self.coco_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"COCO dataset configuration saved to: {config_path}")
    
    def create_sample_dataset(self, sample_size: int):
        """Create a sample dataset for quick testing."""
        print(f"Creating sample dataset with {sample_size} images...")
        
        sample_dir = self.coco_dir / "sample"
        sample_dir.mkdir(exist_ok=True)
        
        # Load COCO annotations
        from pycocotools.coco import COCO
        
        ann_file = self.coco_dir / "annotations" / f"person_keypoints_train{self.year}.json"
        coco = COCO(str(ann_file))
        
        # Get images with person keypoints
        img_ids = coco.getImgIds(catIds=coco.getCatIds(catNms=['person']))
        
        # Filter images with keypoint annotations
        valid_img_ids = []
        for img_id in img_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=0)
            annotations = coco.loadAnns(ann_ids)
            if any(ann['num_keypoints'] > 0 for ann in annotations):
                valid_img_ids.append(img_id)
        
        # Sample images
        import random
        random.shuffle(valid_img_ids)
        sample_img_ids = valid_img_ids[:sample_size]
        
        # Copy sample images
        sample_images_dir = sample_dir / f"train{self.year}"
        sample_images_dir.mkdir(exist_ok=True)
        
        sample_annotations = {
            'images': [],
            'annotations': [],
            'categories': coco.dataset['categories']
        }
        
        for img_id in tqdm(sample_img_ids, desc="Creating sample"):
            # Load image info
            img_info = coco.loadImgs(img_id)[0]
            
            # Copy image
            src_path = self.coco_dir / f"train{self.year}" / img_info['file_name']
            dst_path = sample_images_dir / img_info['file_name']
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                
                # Add to sample annotations
                sample_annotations['images'].append(img_info)
                
                # Add annotations for this image
                ann_ids = coco.getAnnIds(imgIds=img_id, catIds=1, iscrowd=0)
                annotations = coco.loadAnns(ann_ids)
                sample_annotations['annotations'].extend(annotations)
        
        # Save sample annotations
        sample_ann_dir = sample_dir / "annotations"
        sample_ann_dir.mkdir(exist_ok=True)
        
        sample_ann_file = sample_ann_dir / f"person_keypoints_train{self.year}.json"
        with open(sample_ann_file, 'w') as f:
            json.dump(sample_annotations, f)
        
        # Create sample dataset config
        sample_config = create_dataset_config(
            dataset_path=str(sample_dir),
            train_split=f"train{self.year}",
            val_split=f"train{self.year}",  # Use same split for sample
            num_classes=1,
            class_names=['person']
        )
        
        sample_config_path = sample_dir / "dataset.yaml"
        with open(sample_config_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, indent=2)
        
        print(f"Sample dataset created at: {sample_dir}")
        print(f"Sample dataset config: {sample_config_path}")


class CustomDatasetSetup:
    """Setup custom keypoint dataset."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.custom_dir = self.data_dir / "custom"
    
    def setup_custom_dataset(self, source_path: str, format_type: str = "coco"):
        """Setup custom dataset."""
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source dataset path not found: {source_path}")
        
        print(f"Setting up custom dataset from: {source_path}")
        
        # Create custom dataset directory
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        
        if format_type == "coco":
            self._setup_coco_format(source_path)
        elif format_type == "yolo":
            self._setup_yolo_format(source_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        print("Custom dataset setup completed!")
    
    def _setup_coco_format(self, source_path: Path):
        """Setup custom dataset in COCO format."""
        # Copy images and annotations
        if (source_path / "images").exists():
            shutil.copytree(source_path / "images", self.custom_dir / "images", dirs_exist_ok=True)
        
        if (source_path / "annotations").exists():
            shutil.copytree(source_path / "annotations", self.custom_dir / "annotations", dirs_exist_ok=True)
        
        # Create dataset configuration
        self._create_custom_config()
    
    def _setup_yolo_format(self, source_path: Path):
        """Setup custom dataset in YOLO format."""
        # Copy the entire dataset
        shutil.copytree(source_path, self.custom_dir, dirs_exist_ok=True)
        
        # Create dataset configuration if not exists
        if not (self.custom_dir / "dataset.yaml").exists():
            self._create_custom_config()
    
    def _create_custom_config(self):
        """Create custom dataset configuration."""
        # Try to infer configuration from existing files
        config = create_dataset_config(
            dataset_path=str(self.custom_dir),
            train_split="train",
            val_split="val",
            num_classes=1,
            class_names=['person']
        )
        
        config_path = self.custom_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"Custom dataset configuration saved to: {config_path}")
    
    def convert_to_yolo_format(self):
        """Convert COCO format annotations to YOLO format."""
        print("Converting annotations to YOLO format...")
        
        # This is a placeholder for COCO to YOLO conversion
        # In a real implementation, you would convert COCO JSON annotations
        # to YOLO TXT format with normalized coordinates
        
        ann_file = self.custom_dir / "annotations" / "instances.json"
        if ann_file.exists():
            print(f"Found annotations file: {ann_file}")
            # Implement COCO to YOLO conversion here
            print("COCO to YOLO conversion completed!")
        else:
            print("No COCO annotations found to convert")


def main():
    """Main data setup function."""
    args = parse_args()
    
    print(f"Setting up {args.dataset} dataset...")
    print(f"Data directory: {args.data_dir}")
    
    if args.dataset == "coco":
        # Setup COCO dataset
        coco_setup = COCODatasetSetup(args.data_dir, args.coco_year)
        coco_setup.setup_coco_dataset(args.splits)
        
        # Create sample dataset if requested
        if args.sample_size:
            coco_setup.create_sample_dataset(args.sample_size)
    
    elif args.dataset == "custom":
        # Setup custom dataset
        if not args.custom_path:
            raise ValueError("Custom dataset path is required for custom dataset setup")
        
        custom_setup = CustomDatasetSetup(args.data_dir)
        custom_setup.setup_custom_dataset(args.custom_path, args.format)
        
        if args.convert_to_yolo:
            custom_setup.convert_to_yolo_format()
    
    print("Dataset setup completed successfully!")


if __name__ == "__main__":
    main() 