import os
import numpy as np
import torch
import cv2
import easyocr

from PIL import Image, ImageDraw

COCO = {80: 'face', 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog',
        17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
        38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
        45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
        52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
        67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

OUTPUT_FOLDER = ''

def update_global(new_value):
    global GLOBAL_VAR  # Required if modifying inside a function
    GLOBAL_VAR = new_value

def initializeState(states, person):
    states[person] = {}

    # Blink/Sleep detector states =============================
    states[person]["counters"] = {"open": 0, "closed": 0}
    states[person]["is_closed"] = False
    states[person]["close_frame"] = 0
    states[person]["last_found_frame"] = 0
    # =========================================================
    # Yawn detector states ====================================
    states[person]["window"] = []
    states[person]["current_yawning"] = False
    states[person]["prev_yawned"] = False
    states[person]["prev_ar"] = 0
    states[person]["increasing_wider"] = 0
    states[person]["last_found_frame"] = 0
    return states
    # =========================================================


def seed_torch(device, seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def fromarray(im):
    """Update self.im from a numpy array."""
    im = im if isinstance(im, Image.Image) else Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    return draw


def kpts(im, kpts, shape=(640, 640), radius=None, kpt_line=True, conf_thres=0.25, kpt_color=None,
         pil=False, lw=2, skeleton=None, limb_color=None):
    """
    Plot keypoints on the image.

    Args:
        kpts (torch.Tensor): Keypoints, shape [17, 3] (x, y, confidence).
        shape (tuple, optional): Image shape (h, w). Defaults to (640, 640).
        radius (int, optional): Keypoint radius. Defaults to 5.
        kpt_line (bool, optional): Draw lines between keypoints. Defaults to True.
        conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
        kpt_color (tuple, optional): Keypoint color (B, G, R). Defaults to None.

    Note:
        - `kpt_line=True` currently only supports human pose plotting.
        - Modifies self.im in-place.
        - If self.pil is True, converts image to numpy array and back to PIL.
    """
    radius = radius if radius is not None else lw
    if pil:
        # Convert to numpy first
        im = np.asarray(im).copy()
    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim in {2, 3}
    kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
    for i, k in enumerate(kpts):
        color_k = kpt_color or (kpt_color[i].tolist() if is_pose else colors(i))
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < conf_thres:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    if kpt_line:
        ndim = kpts.shape[-1]
        for i, sk in enumerate(skeleton):
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
            if ndim == 3:
                conf1 = kpts[(sk[0] - 1), 2]
                conf2 = kpts[(sk[1] - 1), 2]
                if conf1 < conf_thres or conf2 < conf_thres:
                    continue
            if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue
            cv2.line(
                im,
                pos1,
                pos2,
                kpt_color or limb_color[i].tolist(),
                thickness=int(np.ceil(lw / 2)),
                lineType=cv2.LINE_AA,
            )
    if pil:
        # Convert im back to PIL and update draw
        draw = fromarray(im)


def extract_timestamp(image):
    reader = easyocr.Reader(['en'])
    timestamp = reader.readtext(image[50:200, 850:1150], detail=0)[0]

    timestamp = ''.join([char for char in timestamp if char.isdigit()])
    if len(timestamp) == 6:
        return timestamp
    return None


def draw_reid(image_rgb, reid_out):
    out = []

    for reid in reid_out:
        box = reid['box_int']
        name = f'PID: {reid["name"]}'

        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # Add the name label with a background box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.1
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        box_color = (255, 0, 0)  # Blue background box

        # Get the size of the text
        (text_width, text_height), _ = cv2.getTextSize(name, font, font_scale, font_thickness)

        # Calculate the position of the text background box
        text_x = x_min
        text_y = y_min - 5  # Position the text slightly above the bounding box
        box_coords = ((text_x, text_y), (text_x + text_width, text_y - text_height))

        # Draw the background box
        cv2.rectangle(image_rgb, box_coords[0], box_coords[1], box_color, -1)  # -1 fills the rectangle

        # Put the text on top of the background box
        cv2.putText(image_rgb, name, (text_x, text_y), font, font_scale, text_color, font_thickness)
        out.append(name)
    return image_rgb, out


def draw_action(image_rgb, action_labels, bounding_boxes, action_reid):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 0, 255)  # White color
    thickness = 2
    out = []
    for action_label, box, reid in zip(action_labels, bounding_boxes, action_reid):
        x_min, y_min, x_max, y_max = box

        text, prob = '', 0.
        # ## skip the detection for those unknown identities
        # if reid['name'] == 'unknown':
        #     continue
        each_identity = []
        for idx, (text, prob) in enumerate(action_label):
            # Get the size of the text
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = x_min + 2  # 5 pixels from the left edge
            y = y_min - (text_height+2) * (idx + 1) - 25  #
            cap = f'{text}||{prob:.3f}'
            cv2.putText(image_rgb, cap, (int(x), int(y)), font, font_scale, font_color, thickness)
            each_identity.append(cap)
        out.append(each_identity)
    # Put the text on the image
    return image_rgb, out


def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def update_time(start_time: str, second: float) -> str:
    # Parse hours, minutes, seconds from start_time
    hh = int(start_time[0:2])
    mm = int(start_time[2:4])
    ss = int(start_time[4:6])

    # Convert everything to total seconds and add the offset
    total_seconds = hh * 3600 + mm * 60 + ss + second

    # Calculate new hh, mm, ss (handle overflow)
    new_hh = int(total_seconds // 3600) % 24  # Wrap around after 24 hours
    remaining_seconds = total_seconds % 3600
    new_mm = int(remaining_seconds // 60)
    new_ss = int(remaining_seconds % 60)

    # Format with leading zeros (e.g., 1 → "01")
    updated_time = f"{new_hh:02d}{new_mm:02d}{new_ss:02d}"

    return updated_time