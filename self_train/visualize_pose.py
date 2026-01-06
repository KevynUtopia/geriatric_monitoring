#!/usr/bin/env python3
"""
Simple pose visualization script.

Usage:
    python3 visualize_pose.py --image path/to/image.jpg --label path/to/label.txt
    python3 visualize_pose.py --images_dir path/to/images --labels_dir path/to/labels --out_dir path/to/output
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# COCO keypoint names and connections
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO skeleton connections (keypoint index pairs)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # legs
]

# Colors for different persons
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
]


def parse_yolo_label(label_path):
    """
    Parse YOLO format label file.
    
    Returns:
        List of detections, each containing:
        - bbox: [center_x, center_y, width, height] (normalized)
        - keypoints: [(x, y, visibility), ...] for 17 keypoints (normalized)
    """
    detections = []
    
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return detections
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = list(map(float, line.split()))
            
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            center_x, center_y, width, height = parts[1:5]
            
            # Parse keypoints (17 keypoints * 3 values each = 51 values)
            keypoints = []
            if len(parts) >= 56:  # 5 + 51 = 56
                for i in range(17):
                    idx = 5 + i * 3
                    if idx + 2 < len(parts):
                        x, y, vis = parts[idx:idx+3]
                        keypoints.append((x, y, vis))
                    else:
                        keypoints.append((0, 0, 0))
            
            detections.append({
                'class_id': class_id,
                'bbox': [center_x, center_y, width, height],
                'keypoints': keypoints
            })
    
    return detections


def draw_pose_on_image(image, detections):
    """
    Draw pose keypoints and skeleton on image.
    
    Args:
        image: OpenCV image (BGR format)
        detections: List of detection dictionaries
        
    Returns:
        Image with poses drawn
    """
    img_height, img_width = image.shape[:2]
    vis_image = image.copy()
    
    for person_idx, detection in enumerate(detections):
        color = COLORS[person_idx % len(COLORS)]
        
        # Draw bounding box
        bbox = detection['bbox']
        center_x, center_y, width, height = bbox
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int((center_x - width/2) * img_width)
        y1 = int((center_y - height/2) * img_height)
        x2 = int((center_x + width/2) * img_width)
        y2 = int((center_y + height/2) * img_height)
        
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw keypoints
        keypoints = detection['keypoints']
        pixel_keypoints = []
        
        for i, (x, y, vis) in enumerate(keypoints):
            if vis > 0:  # Visible keypoint
                px = int(x * img_width)
                py = int(y * img_height)
                pixel_keypoints.append((px, py))
                
                # Draw keypoint circle
                cv2.circle(vis_image, (px, py), 4, color, -1)
                
                # Add keypoint name
                cv2.putText(vis_image, str(i), (px+5, py-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            else:
                pixel_keypoints.append(None)
        
        # Draw skeleton connections
        for connection in COCO_SKELETON:
            pt1_idx, pt2_idx = connection
            if (pt1_idx < len(pixel_keypoints) and pt2_idx < len(pixel_keypoints) and
                pixel_keypoints[pt1_idx] is not None and pixel_keypoints[pt2_idx] is not None):
                
                pt1 = pixel_keypoints[pt1_idx]
                pt2 = pixel_keypoints[pt2_idx]
                cv2.line(vis_image, pt1, pt2, color, 2)
        
        # Add person ID
        cv2.putText(vis_image, f"Person {person_idx}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return vis_image


def visualize_single_image(image_path, label_path, save_path=None):
    """Visualize a single image with its pose labels."""
    # Load image
    if not os.path.exists(image_path):
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    # Parse labels
    detections = parse_yolo_label(label_path)
 
    if not detections:
        # Skip if no detections found
        return False
    elif len(detections) < 5:
        # Skip if fewer than 5 persons detected
        return False
    else:
        # Draw poses
        vis_image = draw_pose_on_image(image, detections)
        
        # Save visualized image if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, vis_image)
        
        return True


def process_single_image_worker(args):
    """Worker function for multiprocessing that processes a single image."""
    image_path, label_path, save_path = args
    return visualize_single_image(image_path, label_path, save_path)


def visualize_directory(images_dir, labels_dir, out_dir, vis_list_dir):
    """Visualize all images in a directory with their corresponding labels and save to output directory."""
    # get all images from images_dir
    image_files = os.listdir(images_dir)
    output_path = Path(out_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare tasks for multiprocessing
    tasks = []
    task_to_filename = {}  # Map task to filename for recording
    for image_file in image_files:
        # Convert string to Path object to use .stem
        image_path = Path(image_file)
        # Construct full image path
        full_image_path = Path(images_dir) / image_file
        label_file = Path(labels_dir) / (image_path.stem + '.txt')
        
        # Create output path maintaining the same structure
        output_file = output_path / image_file
        
        # Add task to list
        task = (str(full_image_path), str(label_file), str(output_file))
        tasks.append(task)
        task_to_filename[task] = image_file
    
    # Use multiprocessing to process images in parallel
    # max_workers = min(multiprocessing.cpu_count(), len(tasks))
    max_workers = 30
    print(f"Processing {len(tasks)} images using {max_workers} workers...")
    
    processed_count = 0
    
    # Create the visualized list file and write header
    visualized_list_path = Path(vis_list_dir) / "visualized_images.txt"
    os.makedirs(Path(vis_list_dir), exist_ok=True)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_single_image_worker, task): task for task in tasks}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing images"):
            try:
                success = future.result()
                task = future_to_task[future]
                if success:
                    processed_count += 1
                    # Write the filename to the list file immediately
                    with open(visualized_list_path, 'a') as f:
                        f.write(f"{task_to_filename[task]}\n")
            except Exception as e:
                task = future_to_task[future]
                print(f"Error processing {task[0]}: {e}")
    
    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed {processed_count}/{len(tasks)} images")
    print(f"Output saved to: {out_dir}")
    print(f"List of visualized images saved to: {visualized_list_path}")


def backup_visualized_images(output_dir, vis_list_dir):
    """
    Backup function: Keep only images that exist in the visualized images list.
    Delete images that are not in the list.
    
    Args:
        output_dir: Path to directory containing output images
        vis_list_dir: Path to directory containing visualized_images.txt
    """
    output_path = Path(output_dir)
    vis_list_path = Path(vis_list_dir) / "visualized_images.txt"
    
    # Check if the list file exists
    if not vis_list_path.exists():
        print(f"Visualized images list not found: {vis_list_path}")
        return
    
    # Read the list of visualized images
    with open(vis_list_path, 'r') as f:
        visualized_images = set(line.strip() for line in f.readlines())
    
    print(f"Found {len(visualized_images)} images in the visualized list")
    
    # Get all image files in the output directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    output_images = []
    for ext in image_extensions:
        output_images.extend(output_path.glob(f"*{ext}"))
        output_images.extend(output_path.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(output_images)} images in output directory")
    
    # Check each image and delete if not in the list
    deleted_count = 0
    kept_count = 0
    
    for image_file in output_images:
        if image_file.name in visualized_images:
            kept_count += 1
        else:
            # Delete the image if it's not in the visualized list
            try:
                image_file.unlink()
                deleted_count += 1
                print(f"Deleted: {image_file.name}")
            except Exception as e:
                print(f"Error deleting {image_file.name}: {e}")
    
    print(f"\n--- Backup Complete ---")
    print(f"Kept: {kept_count} images")
    print(f"Deleted: {deleted_count} images")
    print(f"Total processed: {kept_count + deleted_count} images")


def main():
    parser = argparse.ArgumentParser(description="Visualize pose keypoints from YOLO labels")
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--image', type=str, help='Path to single image file')
    group.add_argument('--images_dir', type=str, help='Path to images directory')
    
    parser.add_argument('--label', type=str, help='Path to single label file (required with --image)')
    parser.add_argument('--labels_dir', type=str, help='Path to labels directory (required with --images_dir)')
    parser.add_argument('--out_dir', type=str, help='Path to output directory (required with --images_dir)')

    parser.add_argument('--vis_list_dir', type=str, help='Path to directory to save list of visualized images')
    parser.add_argument('--backup', action='store_true', help='Run backup function to clean output directory based on visualized list')
    
    args = parser.parse_args()
    
    if args.backup:
        if not args.out_dir:
            parser.error("--out_dir is required when using --backup")
        if not args.vis_list_dir:
            parser.error("--vis_list_dir is required when using --backup")
        backup_visualized_images(args.out_dir, args.vis_list_dir)
    elif args.image:
        if not args.label:
            parser.error("--label is required when using --image")
        visualize_single_image(args.image, args.label)
        
    elif args.images_dir:
        if not args.labels_dir:
            parser.error("--labels_dir is required when using --images_dir")
        if not args.out_dir:
            parser.error("--out_dir is required when using --images_dir")
        visualize_directory(args.images_dir, args.labels_dir, args.out_dir, args.vis_list_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 