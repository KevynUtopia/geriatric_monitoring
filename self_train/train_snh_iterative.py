"""
Iterative training script with dynamic dataset management.
SAFETY FEATURE: The dynamic_dataset directory is never deleted, only archived and repopulated.
Each epoch archives the current dataset before processing, ensuring data integrity.
The dynamic_dataset name is always maintained for the current active dataset.
"""

from ultralytics import YOLO
import os
from pathlib import Path
import numpy as np
import cv2
import shutil
from tqdm import tqdm

def ensure_directory(directory_path: Path) -> None:
    """Create directory if it does not exist."""
    directory_path.mkdir(parents=True, exist_ok=True)


def read_image_names(list_file: Path) -> list[str]:
    """Read image names from a text file, stripping whitespace and ignoring empty/comment lines."""
    image_names: list[str] = []
    if not list_file.exists():
        print(f"List file not found: {list_file}")
        return image_names

    with list_file.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            # If a full/relative path is included, use only the filename
            name = os.path.basename(name)
            image_names.append(name)
    return image_names


def populate_dynamic_dataset(list_file: str, source_image_dir: str, source_label_dir: str,
                             dynamic_dataset: str, dynamic_label: str) -> None:

    print("Learn from previous annotated data")
    # first read list_file txt file
    with open(list_file, 'r') as f:
        image_names = f.readlines()


    populate_label_only = False
    # first roughly check if target image already exists by checking numbers
    num_length_images = len(image_names)
    num_target_images = len(os.listdir(dynamic_dataset))
    if num_length_images == num_target_images:
        populate_label_only = True
        print("Only populating labels, images already exist in dynamic_dataset")

    # then copy the image from source_dir to target_dir
    error_count = 0
    for image_name in tqdm(image_names, desc="Copying images and labels"):
        image_name = image_name.strip('\n').strip()

        image_path = os.path.join(source_image_dir, image_name)
        label_path = os.path.join(source_label_dir, image_name.replace(".jpg", ".txt"))
        
        target_image_path = os.path.join(dynamic_dataset, image_name)
        target_label_path = os.path.join(dynamic_label, image_name.replace(".jpg", ".txt"))

        try:
            if not populate_label_only:
                # move image_path to target_image_path
                shutil.copy(image_path, target_image_path)
            # move label_path to target_label_path
            shutil.copy(label_path, target_label_path)
        except Exception as e:
            error_count += 1
            print(f"Error copying image and label: {image_name}")
    print(f"Error count: {error_count}")


def populate_dynamic_dataset_with_teacher(list_file: str, source_image_dir: str, source_label_dir: str,
                                         dynamic_dataset: str, dynamic_label: str, teacher_model: YOLO) -> None:
    """
    Populate dynamic dataset with images and generate labels using teacher model.
    First copies images, then uses teacher model to infer and save labels.
    """

    print("Learn from teecher model")
    
    # first read list_file txt file
    with open(list_file, 'r') as f:
        image_names = f.readlines()

    populate_label_only = False
    # first roughly check if target image already exists by checking numbers
    num_length_images = len(image_names)
    num_target_images = len(os.listdir(dynamic_dataset))
    if num_length_images == num_target_images:
        populate_label_only = True
        print("Only populating labels, images already exist in dynamic_dataset")

    # then copy the image from source_dir to target_dir
    error_count = 0
    for image_name in tqdm(image_names, desc="Copying images"):  
        image_name = image_name.strip('\n').strip()

        image_path = os.path.join(source_image_dir, image_name)
        target_image_path = os.path.join(dynamic_dataset, image_name)

        try:
            if not populate_label_only:
                # copy image_path to target_image_path
                shutil.copy(image_path, target_image_path)
        except Exception as e:
            error_count += 1
            print(f"Error copying image: {image_name}")
    print(f"Image copy error count: {error_count}")

    # Now use teacher model to generate labels for all images in dynamic_dataset
    print("Using teacher model to generate initial labels...")
    for image_name in tqdm(os.listdir(dynamic_dataset), desc="Generating labels with teacher model"):
        image_path = os.path.join(dynamic_dataset, image_name)
        label_path = os.path.join(dynamic_label, image_name.replace(".jpg", ".txt"))
        
        # Use teacher model to infer and save labels
        infer_image_and_update(image_path, teacher_model, 0.5, label_path)  # Using 0.5 as initial confidence threshold

   
    

# ============================
# Inference/annotation helpers
# ============================
def save_yolo_labels(label_path: Path,
                     boxes: np.ndarray,
                     confs: np.ndarray,
                     keypoints: np.ndarray | None,
                     keypoint_confs: np.ndarray | None,
                     img_width: int,
                     img_height: int) -> None:

    with open(label_path, 'w') as f:
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = box
            center_x = ((x1 + x2) / 2) / img_width
            center_y = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            line = f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            if keypoints is not None and i < len(keypoints):
                kpts = keypoints[i]
                kpt_confs = keypoint_confs[i] if keypoint_confs is not None else np.ones(17)
                for (kx, ky), kconf in zip(kpts, kpt_confs):
                    norm_x = kx / img_width
                    norm_y = ky / img_height
                    visibility = 2.0 if float(kconf) > 0.5 else 0.0
                    line += f" {norm_x:.6f} {norm_y:.6f} {visibility:.6f}"
            f.write(line + '\n')


def infer_image_and_update(image_path: Path, model: YOLO, conf_thres: float, label_path: Path) -> int:
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    h, w = img.shape[:2]
    results = model(img, conf=conf_thres, verbose=False)
    result = results[0]
    if result.boxes is None or len(result.boxes) < 3:
        # skip updating if detected people < 3.
        print(f"Skipping updating: {image_path}")
        return 0
    human_boxes = result.boxes.xyxy.cpu().numpy()
    human_confs = result.boxes.conf.cpu().numpy()
    human_cls = result.boxes.cls.cpu().numpy()
    if result.keypoints is not None:
        keypoints = result.keypoints.xy.cpu().numpy()
        keypoint_confs = result.keypoints.conf.cpu().numpy()
    else:
        keypoints = None
        keypoint_confs = None
    person_mask = human_cls == 0
    person_count = int(np.sum(person_mask))


    save_yolo_labels(
        label_path,
        human_boxes[person_mask] if person_count > 0 else np.empty((0, 4)),
        human_confs[person_mask] if person_count > 0 else np.empty((0,)),
        keypoints[person_mask] if (keypoints is not None and person_count > 0) else None,
        keypoint_confs[person_mask] if (keypoint_confs is not None and person_count > 0) else None,
        w,
        h,
    )
    return person_count




if __name__ == "__main__":

    # Predefined variables
    dynamic_path = "/home/wzhangbu/self_train/runs/pose/dynamics"
    initial_images = "path_to_your_root/datasets/snh-pose-split/visualize/visualized_images.txt"
    source_image_dir = "path_to_your_root/datasets/snh-pose-split/images/train"
    source_label_dir = "path_to_your_root/datasets/snh-pose-split/labels/train"

    dynamic_dataset = "path_to_your_root/datasets/snh-pose-split/images/train_dynamic"
    dynamic_label =  "path_to_your_root/datasets/snh-pose-split/labels/train_dynamic"

    teacher_model = "/home/wzhangbu/elderlycare/weights/yolo11x-pose.pt"
    initial_model = "/home/wzhangbu/elderlycare/weights/yolo11n-pose.pt"
    update_model = ""  # updated trained weights path after each iteration


    latest = "/home/wzhangbu/self_train/runs/pose/train_st/weights/last.pt"

    threshold_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75]  # confidence thresholds for each epoch


    YOLO_teacher = YOLO(teacher_model)
    # Ensure dynamic dataset exists (create symlinks for the initial list if desired)
    populate_dynamic_dataset_with_teacher(initial_images, source_image_dir, source_label_dir, dynamic_dataset, dynamic_label, YOLO_teacher)


    n_epochs = 30

    for n in range(n_epochs):
        print(f"=== Iteration {n + 1}/{n_epochs} ===")
        
        # For the first epoch, use initial_model; for subsequent epochs, use the updated model
        model = YOLO(initial_model) if update_model == "" else YOLO(update_model)

        # Train the model
        results = model.train(data="config/snh-pose.yaml",
                              epochs=1, imgsz=640, batch=64,
                              device=[0], workers=14, verbose=True,
                              name=f"train_st", exist_ok=True, val=False)

        # Save the trained model to a new epoch folder
        new_folder = f"{dynamic_path}/train_{n+1}"
        os.makedirs(new_folder, exist_ok=True)
        shutil.copy(latest, new_folder)
        update_model = f"{new_folder}/last.pt"

        # Inference over dynamic_dataset: update annotations and prune images with < 3 persons
        conf_thres = threshold_list[min(n, len(threshold_list) - 1)]
        
        # For the first epoch, use teacher model; for subsequent epochs, use the trained model
        if n < 2:
            # First and second epoch: use teacher model to update labels
            print("First epoch: Using teacher model to update labels...")
            infer_model = YOLO_teacher
        else:
            # Subsequent epochs: use the trained model to update labels
            print(f"Epoch {n+1}: Using trained model to update labels...")
            infer_model = YOLO(update_model)

        # use infer_model to infer all images in dynamic_dataset, and use the detected results to update the labels in dynamic_label
        for image_name in tqdm(os.listdir(dynamic_dataset), desc=f"Inferring images using thres {conf_thres}"):
            image_path = os.path.join(dynamic_dataset, image_name)
            label_path = os.path.join(dynamic_label, image_name.replace(".jpg", ".txt"))
            infer_image_and_update(image_path, infer_model, conf_thres, label_path)

        

