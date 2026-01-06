#!/usr/bin/env python3

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import csv


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"
}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def find_images(source_dir: Path) -> List[Path]:
    return [p for p in sorted(source_dir.iterdir()) if is_image_file(p)]


def choose_images(
    all_images: List[Path],
    total_pick: int,
    shared_count: int,
    unique_per_folder: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    if len(all_images) < total_pick:
        raise ValueError(
            f"Not enough images to pick from: required {total_pick}, found {len(all_images)}"
        )

    # Sample total pool first
    pool = random.sample(all_images, total_pick)

    # Pick shared
    shared = random.sample(pool, shared_count)
    remaining = [p for p in pool if p not in set(shared)]

    required_remaining = unique_per_folder * 2
    if len(remaining) < required_remaining:
        raise ValueError(
            f"After selecting shared={shared_count}, need {required_remaining} unique images,"
            f" but only {len(remaining)} remain."
        )

    # Split remaining into unique sets
    unique_a = random.sample(remaining, unique_per_folder)
    remaining_after_a = [p for p in remaining if p not in set(unique_a)]
    unique_b = random.sample(remaining_after_a, unique_per_folder)

    return shared, unique_a, unique_b


def annotate_and_save_image(
    image_path: Path, destination_dir: Path, model: YOLO, conf: float
) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    results = model(img, conf=conf, verbose=False)
    result = results[0]
    person_count = 0
    if result.boxes is not None and result.boxes.cls is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(classes[idx])
            if cls_id == 0:
                person_count += 1
                # Draw bounding box
                cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                if confs is not None:
                    label = f"person {confs[idx]:.2f}"
                    cv2.putText(
                        img,
                        label,
                        (x1i, max(0, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
    # Save annotated image to destination with same filename
    out_path = destination_dir / image_path.name
    cv2.imwrite(str(out_path), img)
    return person_count


def copy_and_annotate_images(
    images: List[Path], destination_dir: Path, model: YOLO, conf: float
) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for src in images:
        annotate_and_save_image(src, destination_dir, model=model, conf=conf)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly pick images and copy into two folders: each gets 100 unique + 50 shared (default)."
        )
    )
    parser.add_argument(
        "--src",
        required=True,
        type=str,
        help="Source directory containing images.",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=str,
        help="Destination directory where two subfolders will be created.",
    )
    parser.add_argument(
        "--name-a",
        default="set_a",
        type=str,
        help="Name of the first destination subfolder (default: set_a).",
    )
    parser.add_argument(
        "--name-b",
        default="set_b",
        type=str,
        help="Name of the second destination subfolder (default: set_b).",
    )
    parser.add_argument(
        "--total-pick",
        default=250,
        type=int,
        help="Total number of images to sample from source before splitting (default: 250).",
    )
    parser.add_argument(
        "--shared",
        default=50,
        type=int,
        help="Number of images shared between both folders (default: 50).",
    )
    parser.add_argument(
        "--unique-per-folder",
        default=100,
        type=int,
        help="Number of unique images per folder (default: 100).",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Random seed for reproducibility (default: None).",
    )
    parser.add_argument(
        "--yolo-weights",
        default="/home/wzhangbu/elderlycare/weights/yolo11n-pose.pt",
        type=str,
        help="Path to YOLO weights for people counting (pose or detect model).",
    )
    parser.add_argument(
        "--min-people",
        default=4,
        type=int,
        help="Minimum number of people required in an image to be eligible (default: 5).",
    )
    parser.add_argument(
        "--yolo-conf",
        default=0.5,
        type=float,
        help="YOLO confidence threshold (default: 0.5).",
    )
    return parser


def validate_counts(total_pick: int, shared: int, unique_per_folder: int) -> None:
    if shared < 0 or unique_per_folder < 0 or total_pick <= 0:
        raise ValueError("Counts must be positive (shared and unique_per_folder can be zero or more).")

    per_folder = shared + unique_per_folder
    if per_folder <= 0:
        raise ValueError("Each folder must receive at least one image.")

    required = shared + unique_per_folder * 2
    if total_pick < required:
        raise ValueError(
            f"total_pick must be at least shared + 2*unique_per_folder = {required}, got {total_pick}"
        )


def yolo_detection(model=None, image_path: Path = None, conf: float = 0.5) -> int:
    if image_path is None:
        return 0
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    results = model(img, conf=conf, verbose=False)
    result = results[0]
    if result.boxes is None or result.boxes.cls is None:
        return 0
    classes = result.boxes.cls.cpu().numpy()
    # class 0 is person for COCO-trained YOLO models
    person_mask = classes == 0
    return int(np.sum(person_mask))


def filter_images_by_people(
    model: YOLO, image_paths: List[Path], min_people: int, conf: float
) -> List[Path]:
    eligible: List[Path] = []
    for p in image_paths:
        try:
            count = yolo_detection(model=model, image_path=p, conf=conf)
            if count >= min_people:
                eligible.append(p)
        except Exception:
            # Skip problematic images silently
            continue
    return eligible


def write_folder_csv(folder: Path, model: YOLO, conf: float) -> None:
    rows = []
    image_paths = [p for p in sorted(folder.iterdir()) if is_image_file(p)]
    for p in image_paths:
        total = yolo_detection(model=model, image_path=p, conf=conf)
        # Placeholders for type 1..4
        rows.append([p.name, total, 0, 0, 0, 0])
    csv_path = folder / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "total", "type 1", "type 2", "type 3", "type 4"])
        # Already sorted by name due to image_paths sorting
        writer.writerows(rows)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model = YOLO(args.yolo_weights)

    if args.seed is not None:
        random.seed(args.seed)

    source_dir = Path(args.src).expanduser().resolve()
    dest_dir = Path(args.dst).expanduser().resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist or is not a directory: {source_dir}")

    validate_counts(args.total_pick, args.shared, args.unique_per_folder)

    print(f"Finding images in {source_dir}")
    all_images = find_images(source_dir)
    if not all_images:
        raise SystemExit(f"No image files found in source: {source_dir}")

    # Filter images by person count
    print(f"Filtering images by person count with {args.yolo_weights} and conf {args.yolo_conf}")
    eligible_images = filter_images_by_people(
        model=model,
        image_paths=all_images,
        min_people=args.min_people,
        conf=args.yolo_conf,
    )
    if len(eligible_images) < (args.shared + 2 * args.unique_per_folder):
        raise SystemExit(
            f"Not enough eligible images with >= {args.min_people} people."
            f" Needed at least {args.shared + 2 * args.unique_per_folder}, found {len(eligible_images)}."
        )

    print(f"Choosing images with {args.total_pick} total, {args.shared} shared, and {args.unique_per_folder} unique per folder")
    shared, unique_a, unique_b = choose_images(
        all_images=eligible_images,
        total_pick=args.total_pick,
        shared_count=args.shared,
        unique_per_folder=args.unique_per_folder,
    )

    set_a = list(shared) + list(unique_a)
    set_b = list(shared) + list(unique_b)

    # Create destination subfolders
    dest_a = dest_dir / args.name_a
    dest_b = dest_dir / args.name_b
    dest_a.mkdir(parents=True, exist_ok=True)
    dest_b.mkdir(parents=True, exist_ok=True)

    # Annotate with YOLO and save; do not copy originals
    copy_and_annotate_images(set_a, dest_a, model=model, conf=args.yolo_conf)
    copy_and_annotate_images(set_b, dest_b, model=model, conf=args.yolo_conf)

    print("Completed copying images.")
    print(f"Source: {source_dir}")
    print(f"Destination A: {dest_a} -> {len(set_a)} images (unique {len(unique_a)} + shared {len(shared)})")
    print(f"Destination B: {dest_b} -> {len(set_b)} images (unique {len(unique_b)} + shared {len(shared)})")
    print(
        f"Totals: picked {args.total_pick} from source, copied {len(set_a) + len(set_b)} files (with shared duplicated)"
    )

    # Generate CSV summaries
    write_folder_csv(dest_a, model=model, conf=args.yolo_conf)
    write_folder_csv(dest_b, model=model, conf=args.yolo_conf)


if __name__ == "__main__":
    '''
        python preprocessing/random_pick.py --src path_to_your_root/datasets/snh-pose-split/images/test --dst path_to_your_root/datasets/snh-pose-split/cohens_test
    '''
    main()


