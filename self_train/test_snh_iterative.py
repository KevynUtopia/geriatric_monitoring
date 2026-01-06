from ultralytics import YOLO
import os
import cv2
from pathlib import Path
import glob
import sys

# Add the preprocessing directory to the path to import the function
sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocessing'))
from random_pick import annotate_and_save_image

dynamic_modle_dir = "dynamics_v2"
model_dir = f"/home/wzhangbu/self_train/runs/pose/{dynamic_modle_dir}/train_2/last.pt"
candidcate_image_dir = "path_to_your_root/datasets/snh-pose-split/cohens_test/set_a"
source_image_dir = "path_to_your_root/datasets/snh-pose-split/images/test"
output_dir =  "path_to_your_root/datasets/snh-pose-split/cohens_test/adapted_set_a"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_dir)

# Get all image files from candidate directory
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
candidate_images = []
for ext in image_extensions:
    candidate_images.extend(glob.glob(os.path.join(candidcate_image_dir, ext)))
    candidate_images.extend(glob.glob(os.path.join(candidcate_image_dir, ext.upper())))

print(f"Found {len(candidate_images)} images in candidate directory")


# Process each image
for candidate_image_path in candidate_images:
    # Get the image filename
    image_filename = os.path.basename(candidate_image_path)
    
    # Construct the source image path
    source_image_path = os.path.join(source_image_dir, image_filename)
    
    # Check if source image exists
    if not os.path.exists(source_image_path):
        print(f"Warning: Source image {source_image_path} not found, skipping...")
        continue
    
    print(f"Processing: {image_filename}")
    
    try:
        # Use the annotate_and_save_image function for consistent visualization
        source_path = Path(source_image_path)
        output_path = Path(output_dir)
        
        # Run annotation and save with the same style as random_pick.py
        person_count = annotate_and_save_image(
            image_path=source_path,
            destination_dir=output_path,
            model=model,
            conf=0.25  # Confidence threshold
        )
        
        print(f"Saved visualization to: {output_path / image_filename} (detected {person_count} people)")
        
    except Exception as e:
        print(f"Error processing {image_filename}: {str(e)}")
        continue

print("Processing completed!")