#!/usr/bin/env python3
"""
Debug script to test the saving functionality.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from video_engine import save_yolo_labels


def create_test_data():
    """Create test image and fake detection data."""
    # Create a simple test image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (100, 150, 200)  # Fill with a color
    
    # Draw a simple stick figure
    cv2.circle(image, (320, 100), 20, (255, 255, 255), -1)  # head
    cv2.line(image, (320, 120), (320, 300), (255, 255, 255), 3)  # body
    cv2.line(image, (320, 180), (280, 240), (255, 255, 255), 3)  # left arm
    cv2.line(image, (320, 180), (360, 240), (255, 255, 255), 3)  # right arm
    cv2.line(image, (320, 300), (280, 400), (255, 255, 255), 3)  # left leg
    cv2.line(image, (320, 300), (360, 400), (255, 255, 255), 3)  # right leg
    
    # Create fake detection data
    img_height, img_width = image.shape[:2]
    
    # Fake bounding box (around the stick figure)
    boxes = np.array([[250, 80, 390, 420]])  # [x1, y1, x2, y2]
    confs = np.array([0.95])
    
    # Fake keypoints (17 COCO keypoints)
    keypoints = np.array([[[
        320, 100,  # nose
        310, 95,   # left_eye
        330, 95,   # right_eye
        300, 100,  # left_ear
        340, 100,  # right_ear
        280, 180,  # left_shoulder
        360, 180,  # right_shoulder
        280, 240,  # left_elbow
        360, 240,  # right_elbow
        280, 300,  # left_wrist
        360, 300,  # right_wrist
        300, 300,  # left_hip
        340, 300,  # right_hip
        280, 350,  # left_knee
        360, 350,  # right_knee
        280, 400,  # left_ankle
        360, 400   # right_ankle
    ]]])
    
    keypoint_confs = np.array([[
        0.9, 0.8, 0.8, 0.7, 0.7,  # head
        0.9, 0.9, 0.8, 0.8, 0.7, 0.7,  # arms
        0.9, 0.9, 0.8, 0.8, 0.7, 0.7   # legs
    ]])
    
    return image, boxes, confs, keypoints, keypoint_confs, img_width, img_height


def test_saving():
    """Test the saving functionality."""
    print("Creating test data...")
    image, boxes, confs, keypoints, keypoint_confs, img_width, img_height = create_test_data()
    
    # Create output directories
    output_dir = Path("debug_output")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directories created:")
    print(f"  Images: {images_dir.absolute()}")
    print(f"  Labels: {labels_dir.absolute()}")
    
    # Test filenames
    test_filename = "test_day_camera1_video1_000000000001"
    
    # Save test image
    image_path = images_dir / f"{test_filename}.jpg"
    print(f"\nSaving test image to: {image_path}")
    success = cv2.imwrite(str(image_path), image)
    print(f"Image save success: {success}")
    
    if image_path.exists():
        print(f"✅ Image file created: {image_path} ({image_path.stat().st_size} bytes)")
    else:
        print(f"❌ Image file NOT created: {image_path}")
    
    # Save test label
    label_path = labels_dir / f"{test_filename}.txt"
    print(f"\nSaving test label to: {label_path}")
    
    try:
        save_yolo_labels(
            label_path,
            boxes,
            confs, 
            keypoints,
            keypoint_confs,
            img_width,
            img_height
        )
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                content = f.read()
            print(f"✅ Label file created: {label_path} ({label_path.stat().st_size} bytes)")
            print(f"Label content preview: {content[:100]}...")
        else:
            print(f"❌ Label file NOT created: {label_path}")
            
    except Exception as e:
        print(f"❌ Error saving label: {e}")
    
    print(f"\nTest completed. Check the debug_output directory.")
    print(f"You can visualize the result with:")
    print(f"python3 visualize_pose.py --image {image_path} --label {label_path}")


if __name__ == "__main__":
    test_saving() 