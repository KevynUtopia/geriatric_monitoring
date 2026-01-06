import torch
import os
import pickle
import numpy as np
import torchreid
from people_profile import People
from ultralytics.engine.results import Results
from utils import initializeState, draw_reid, draw_action, extract_timestamp, \
    load_label_map, COCO, update_time
import mmcv
from collections import deque


import mmengine
# from mmaction.apis import (detection_inference, inference_skeleton,
#                            init_recognizer, pose_inference)
from mmengine.runner import load_checkpoint
from ultralytics import YOLO as YOLO_ultralytics
from mmaction.registry import MODELS
from pathlib import Path
import boxmot
from boxmot import BotSort, BoostTrack

def get_previous_features(original):
    # Split into two parts
    prefix, start, end = original.split('-')

    # Convert to integers and subtract 120
    new_start = int(start) - 120
    new_end = int(end) - 120

    # Format back to 5-digit strings with leading zeros
    formatted_start = f"{new_start:05d}"
    formatted_end = f"{new_end:05d}"

    # Combine into the final string
    result = f"{prefix}-{formatted_start}-{formatted_end}"
    return result




class BaseDetectionEngine:
    def __init__(self, recog_tolerance) -> None:
        self.progress = {}
        self.frame = 0
        self.image = None
        self.timestamp = ''
        self.recog_tolerance = recog_tolerance

        # initialize an empty numpy array
        self.names = COCO

        self.boxes = torch.tensor([])
        self.probs = torch.tensor([])
        self.cls = torch.tensor([])
        self.keypoints = torch.tensor([])
        self.track_ids = torch.tensor([])
        self.people = None
        # self.reig_model = torchreid.utils.FeatureExtractor(
        #     model_name='osnet_x1_0',
        #     model_path='/home/wzhangbu/elderlycare/weights/osnet_x0_25_msmt17.pth',
        #     device='cuda'
        # )
        self.reig_model = None

        self.skeleton = []
        self.clip_action_bag = []
        self.clip_reid_bag = []
        self.clip = deque(maxlen=None)  # Will be set to args.clip_len in initialize_models
        self.action_label = []
        self.action_proposal = []
        self.action_reid = []
        self.update_action_reid = []

        # self.known_features = {}

        self.config = None
        self.label_map = None
        self.action_recognizer = None
        self.face_detector = None
        self.pose_detector = None
        self.generic_detector = None
        self.tracker = None


    def reset(self):
        self.boxes = torch.tensor([])
        self.probs = torch.tensor([])
        self.cls = torch.tensor([])
        self.keypoints = torch.tensor([])
        self.track_ids = torch.tensor([])

    def reset_action_detector(self):
        self.skeleton = []
        self.clip_action_bag = []
        # self.clip.clear()  # Clear the deque instead of reassigning
        self.clip_reid_bag = []
        # self.action_reid = []
        # self.action_reid = self.update_action_reid
        # self.update_action_reid = []

    def initialize_models(self, args):
        '''
        Initialize the models,
        action_recognizer: the action recognition model using mmaction
        face_detector: the face detection model using YOLOface from finetuned YOLO11
        pose_detector: the pose detection model using YOLOpose from YOLO11
        '''
        self.config = mmengine.Config.fromfile(args.config)
        self.config.merge_from_dict(args.cfg_options)
        config = self.config
        # Load label_map
        label_map = load_label_map(args.label_map)
        try:
            if config['data']['train']['custom_classes'] is not None:
                label_map = {
                    id + 1: label_map[cls]
                    for id, cls in enumerate(config['data']['train']
                                             ['custom_classes'])
                }
        except KeyError:
            pass
        self.label_map = label_map

        # Set the maxlen of self.clip to args.clip_len
        self.clip = deque(maxlen=args.clip_len)

        self.action_recognizer = MODELS.build(config.model)
        # print("before")
        # load check point from /home/wzhangbu/.cache/torch/hub/checkpoints/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-bf93c9ea.pth
        # and then interpolate from [1, 1568, 1024] to [1, 588, 1024]
        # checkpoint = torch.load('/home/wzhangbu/.cache/torch/hub/checkpoints/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-bf93c9ea.pth', map_location='cpu')
        # pos_embed = checkpoint['state_dict']['backbone.pos_embed']
        # pos_embed = pos_embed.reshape(1, 1568, 1024)
        # # Add channel dimension and rearrange for interpolation
        # pos_embed = pos_embed.unsqueeze(1)  # Add channel dimension
        # pos_embed = pos_embed.permute(0, 1, 3, 2)  # Rearrange to (N, C, H, W) format
        # pos_embed = torch.nn.functional.interpolate(pos_embed, size=(1024, 588), mode='bilinear', align_corners=False)
        # pos_embed = pos_embed.permute(0, 1, 3, 2)  # Rearrange back
        # pos_embed = pos_embed.squeeze(1)  # Remove channel dimension
        # pos_embed = pos_embed.reshape(1, 588, 1024)
        # checkpoint['state_dict']['backbone.pos_embed'] = pos_embed
        # # then save to original path
        # torch.save(checkpoint, '/home/wzhangbu/.cache/torch/hub/checkpoints/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-bf93c9ea.pth')
        # exit(0)
        
        load_checkpoint(self.action_recognizer, args.checkpoint, map_location='cpu')
        # print("after")
        # exit(0)
        self.action_recognizer.to('cuda')
        self.action_recognizer.eval()
        self.config = config

        # self.face_detector = YOLO_ultralytics(os.path.join(args.weights_path, args.yolo_face_path), verbose=False)
        self.face_detector = None
        self.pose_detector = YOLO_ultralytics(os.path.join(args.weights_path, args.yolo_pose_path), verbose=False)
        # self.generic_detector = YOLO_ultralytics(os.path.join(args.weights_path, args.yolo_path), verbose=False)
        self.generic_detector = None

        # boxmot-10.0.30
        self.tracker = BotSort(
            # reid_weights=Path('weights/osnet_x1_0_market1501.pt'),  # ReID model to use
            reid_weights=Path('weights/osnet_x1_0_ours.pth.tar'),  # ReID model to use
            device=0,
            half=False,
            # with_reid=True,
            # use_rich_s = True,
            # use_sb = True,
            # use_vt = True,
            track_buffer=500,
        )

        try:
            self.reig_model = self.tracker.model
        except:
            self.reig_model = self.tracker.reid_model

    def initialize_progress(self, filename):
        self.new_progress = {"reid": {}, "action": {}, "skeleton": {}}

    def initialize_people(self, profile_folder):
        self.people = People(self.reig_model, profile_folder)
        print("People initialized with {} people".format(self.people.get_num_people()))

    def get_progress(self, key=None):
        return self.progress[key]

    def take_frame(self, image, frame):
        image = image.astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.frame = frame

    def update_timestamp(self, filename):
        new_timestamp = extract_timestamp(self.image)
        self.timestamp = new_timestamp if new_timestamp else self.timestamp
        if self.timestamp != '':
            self.timestamp = update_time(self.timestamp, self.frame//25*-1)

    def update_detection(self, detected_out):
        self.boxes = torch.cat((self.boxes, detected_out['box']), dim=0)
        self.probs = torch.cat((self.probs, detected_out['prob']), dim=0)
        self.cls = torch.cat((self.cls, detected_out['cls']), dim=0)
        if 'track_ids' in detected_out:
            self.track_ids = torch.cat((self.track_ids, detected_out['track_ids']), dim=0)
        # self.names.update(detected_out['names'])
        if 'keypoints' in detected_out:
            self.keypoints = torch.cat((self.keypoints, detected_out['keypoints']), axis=0)

    def draw(self, reid_out):
        boxes = torch.cat((self.boxes, self.probs.unsqueeze(1), self.cls.unsqueeze(1)), dim=1)
        results = Results(self.image, path='', names=self.names, boxes=boxes,
                          keypoints=self.keypoints)
        # results.boxes.is_track = True
        detected_results = results.plot(save=False, filename="result_new.jpg", line_width=2,labels=False)
        detected_results, reid_out = draw_reid(detected_results, reid_out)
        ## only detect the actions for those whoe are NOT unknown
        # detected_results = draw_action(detected_results, self.action_label, self.action_proposal, self.action_reid)
        detected_results, action_out = draw_action(detected_results, self.action_label,
                                                   self.action_proposal, self.action_reid)
        self.reset()
        return detected_results, reid_out, action_out

    def save_progress(self, reid_out, action_out):
        self.new_progress["reid"][self.frame] = self.action_reid
        self.new_progress["action"][self.frame] = action_out
        self.new_progress["skeleton"][self.frame] = self.skeleton

    def save_results(self, args, filename):
        """
        Save detection results to pickle files in the specified output directory.
        
        Args:
            args: Configuration arguments containing output path
            filename: Base filename for the output files
        """
        reid = self.new_progress["reid"]
        action = self.new_progress["action"]
        skeleton = self.new_progress["skeleton"]
        timestamp = self.timestamp if self.timestamp != '' else '000000'

        # Create output directory if it doesn't exist
        os.makedirs(args.out_path, exist_ok=True)

        # Save results to pickle files
        with open(os.path.join(args.out_path, f"{filename}-{timestamp}-reid.pkl"), "wb") as f:
            pickle.dump(reid, f)
        with open(os.path.join(args.out_path, f"{filename}-action.pkl"), "wb") as f:
            pickle.dump(action, f)
        with open(os.path.join(args.out_path, f"{filename}-skeleton.pkl"), "wb") as f:
            pickle.dump(skeleton, f)

