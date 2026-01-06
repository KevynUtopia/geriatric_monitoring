import os
import cv2
import time
import numpy as np
import pickle
import torch
import pandas as pd
from pathlib import Path


def process_one_iteration(args, input_video, filename, pose_detector):
    """
    Process one video iteration to extract frames and generate pose labels.
    
    Args:
        args: Command line arguments
        input_video: Path to input video file
        filename: Output filename prefix
        pose_detector: YOLO pose detection model
    """
    print(f"Processing video: {input_video}")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video}")
    
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"Video codec (FourCC): {fourcc_str}, FPS: {FPS}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video dimensions: {frame_width}x{frame_height}, Total frames: {num_frames}")

    # Create main output directories
    images_dir = Path(args.out_path) / "images"
    labels_dir = Path(args.out_path) / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created output directories:")
    print(f"   Images: {images_dir.absolute()}")
    print(f"   Labels: {labels_dir.absolute()}")
    
    # Create filename prefix from day_cam_videoname
    filename_prefix = filename.replace("/", "_").replace("\\", "_")
    print(f"📂 Filename prefix: '{filename_prefix}'")

    frame_count = 0
    saved_frame_count = 0
    missing_frame = 0
    total_start = time.time()

    while cap.isOpened():
        frame_count += 1
        ret, image = cap.read()
        
        # Check if frame is valid
        if not ret or image is None:
            missing_frame += 1
            if frame_count > num_frames:
                break
            else:
                continue
                
        # Skip frames based on interval
        if frame_count % args.frame_interval != 1:
            continue

        try:
            # Step 1: Detect poses using YOLO
            results = pose_detector(image, conf=args.confidence_threshold, verbose=False)
            result = results[0]
            
            # Extract detection results
            if result.boxes is not None and len(result.boxes) > 0:
                human_boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                human_confs = result.boxes.conf.cpu().numpy()
                human_cls = result.boxes.cls.cpu().numpy()
                
                if frame_count % 1000 == 1:  # Debug print every 1000 frames
                    print(f"🔍 Frame {frame_count}: Found {len(human_boxes)} detections")
                    print(f"   Classes: {human_cls}")
                    print(f"   Confidences: {human_confs}")
                
                # Extract keypoints if available
                if result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()  # [N, 17, 2]
                    keypoint_confs = result.keypoints.conf.cpu().numpy()  # [N, 17]
                    if frame_count % 1000 == 1:
                        print(f"   Keypoints shape: {keypoints.shape}")
                else:
                    keypoints = None
                    keypoint_confs = None
                    if frame_count % 1000 == 1:
                        print(f"   No keypoints detected")
                    
                # Filter for person class (class 0 in COCO)
                person_mask = human_cls == 0
                if np.any(person_mask):
                    # Save frame and labels
                    frame_filename = f"{filename_prefix}_{saved_frame_count:012d}"
                    
                    print(f"💾 Saving frame {frame_count} as {frame_filename}")
                    print(f"   Found {np.sum(person_mask)} person(s)")
                    
                    # Save image
                    image_path = images_dir / f"{frame_filename}.jpg"
                    success = cv2.imwrite(str(image_path), image)
                    if success:
                        print(f"   ✅ Image saved: {image_path}")
                    else:
                        print(f"   ❌ Failed to save image: {image_path}")
                    
                    # Save label
                    label_path = labels_dir / f"{frame_filename}.txt"
                    try:
                        save_yolo_labels(
                            label_path, 
                            human_boxes[person_mask], 
                            human_confs[person_mask],
                            keypoints[person_mask] if keypoints is not None else None,
                            keypoint_confs[person_mask] if keypoint_confs is not None else None,
                            frame_width, 
                            frame_height
                        )
                        print(f"   ✅ Label saved: {label_path}")
                    except Exception as e:
                        print(f"   ❌ Failed to save label: {e}")
                    
                    saved_frame_count += 1
                else:
                    if frame_count % 1000 == 1:
                        print(f"⏭️  Frame {frame_count}: No persons detected")
            else:
                if frame_count % 1000 == 1:
                    print(f"⏭️  Frame {frame_count}: No detections at all")
            
            # Print progress
            if frame_count % 1000 == 0:
                elapsed = time.time() - total_start
                print(f"Processed {frame_count} frames, saved {saved_frame_count} frames, "
                      f"missing {missing_frame} frames, elapsed: {elapsed:.2f}s")
                
        except Exception as e:
            print(f"Error processing frame {frame_count}: {str(e)}")
            continue

    # Cleanup
    cap.release()
    
    elapsed = time.time() - total_start
    print(f"Finished processing. Total frames: {frame_count}, "
          f"Saved frames: {saved_frame_count}, Missing frames: {missing_frame}, "
          f"Total time: {elapsed:.2f}s")


def save_yolo_labels(label_path, boxes, confs, keypoints, keypoint_confs, img_width, img_height):
    """
    Save detection results in YOLO format.
    
    Args:
        label_path: Path to save label file
        boxes: Bounding boxes in [x1, y1, x2, y2] format
        confs: Confidence scores
        keypoints: Keypoints in [N, 17, 2] format (x, y coordinates)
        keypoint_confs: Keypoint confidence scores [N, 17]
        img_width: Image width
        img_height: Image height
    """
    with open(label_path, 'w') as f:
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = box
            
            # Convert to YOLO format (center_x, center_y, width, height) normalized
            center_x = ((x1 + x2) / 2) / img_width
            center_y = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # Start with class and bounding box
            line = f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            
            # Add keypoints if available
            if keypoints is not None and i < len(keypoints):
                kpts = keypoints[i]  # [17, 2]
                kpt_confs = keypoint_confs[i] if keypoint_confs is not None else np.ones(17)
                
                for j, ((kx, ky), kconf) in enumerate(zip(kpts, kpt_confs)):
                    # Normalize keypoint coordinates
                    norm_x = kx / img_width
                    norm_y = ky / img_height
                    
                    # YOLO keypoint format: x, y, visibility
                    # visibility: 0=not visible, 1=visible but occluded, 2=visible
                    visibility = 2.0 if kconf > 0.5 else 0.0
                    
                    line += f" {norm_x:.6f} {norm_y:.6f} {visibility:.6f}"
            
            f.write(line + '\n')


