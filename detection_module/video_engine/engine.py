import lap
import time
import pickle
import cv2
import os
import numpy as np
import torch

import torchreid
from .base_engine import BaseDetectionEngine
from ultralytics.engine.results import Results

import mmcv
from mmaction.structures import ActionDataSample
from mmengine.structures import InstanceData
from boxmot.trackers.boosttrack.assoc import iou_batch

from modules import extract_timestamp
from people_profile import People

from utils import initializeState, draw_reid, draw_action, load_label_map
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FactorAnalysis,PCA

class DetectionEngine(BaseDetectionEngine):
    def __init__(self, args):
        super(DetectionEngine, self).__init__(
            recog_tolerance=args.recog_tolerance)
        # yolo_model = YOLO(args.yolo_model_path, args.yolo_classes_path, args.yolo_anchors_path, args.yolo_iou, args.yolo_score)
        self.initialize_models(args)
        self.known_id = []
        self.id_mapping = {}
        self.feature_mapping = {}
        self.selected_scores = []
        self.factor_analysis = FactorAnalysis(n_components=3, rotation='varimax')
        self.pca = PCA()


    def face_detect(self):
        result = self.face_detector(self.image, conf=0.50)[0]
        face_boxes = result.boxes
        face_probs = face_boxes.conf.cpu()
        face_cls = face_boxes.cls.cpu()+91 # out of COCO label index
        face_boxes = face_boxes.xyxy.cpu()
        # face_keypoints = result.keypoints

        detected_out = {'box':face_boxes, 'prob':face_probs, 'cls':face_cls,
                        'names':result.names}
        self.update_detection(detected_out)
        # out = result.plot(save=True, filename="result.jpg", line_width=2)

        return detected_out



    def track_reid(self, detected_human):
        """
        Track and re-identify detected humans using BotSort tracker and ReID features.
        
        Args:
            detected_human (dict): Dictionary containing detection boxes, probabilities, and classes
            
        Returns:
            list: List of dictionaries containing tracking results with names and bounding boxes
        """
        # Prepare detections for tracking
        detections = torch.cat((
            detected_human['box'], 
            detected_human['prob'].unsqueeze(1),
            detected_human['cls'].unsqueeze(1)
        ), dim=1).cpu().numpy()
        
        num_detections = detections.shape[0]
        tracks, trackers = self.tracker.update(detections, self.image)
        tracking_results = []
        missing_detections = []

        if len(tracks) == 0 or len(trackers) == 0:
            return tracking_results

        # Find detections that weren't matched to any track
        if detections.shape[0] != tracks.shape[0]:
            iou_matrix = iou_batch(detections[:, :4], tracks[:, :4]) > 0.95
            missing_detections = list(np.where(np.sum(iou_matrix, axis=1) == 0)[0])

        # Extract features from trackers
        track_features = np.stack([t.smooth_feat for t in trackers])
        known_features = self.people.all_representation

        # Match tracks to known identities using Hungarian algorithm
        cost_matrix = cdist(track_features, np.stack(known_features), "cosine")
        cost, track_indices, known_indices = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=0.15)
        matched_names = [self.people.all_names[i] for i in track_indices]

        # Process each detection
        track_idx = 0
        for detection_idx in range(num_detections):
            if detection_idx in missing_detections:
                tracking_results.append({
                    'name': 'unknown',
                    'box_int': detections[detection_idx, :4]
                })
                continue

            # Extract track information
            track = tracks[track_idx]
            x1, y1, x2, y2, track_id = track[:5].astype('int')
            bounding_box = [x1, y1, x2, y2]

            # Map track to identity
            try:
                if matched_names[track_idx] not in list(self.id_mapping.values()):
                    self.id_mapping[track_id] = matched_names[track_idx]
                tracking_results.append({
                    'name': self.id_mapping[track_id],
                    'box_int': bounding_box
                })
            except Exception:
                tracking_results.append({
                    'name': 'unknown',
                    'box_int': bounding_box
                })
            
            track_idx += 1

        self.clip_reid_bag.append(tracking_results)
        return tracking_results


    def pose_detect(self):
        # from mmdet.apis import init_detector, inference_detector
        # from mmdet.apis import DetInferencer
        # inferencer = DetInferencer('retinanet_x101-64x4d_fpn_2x_coco', device='cuda')
        # inferencer = DetInferencer('fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_coco', device='cuda')
        # inferencer = DetInferencer('deformable-detr-refine-twostage_r50_16xb2-50e_coco', device='cuda')
        # inferencer = DetInferencer('dino-5scale_swin-l_8xb2-36e_coco', device='cuda')
        # inferencer = DetInferencer('sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco', device='cuda')
        # inferencer(self.image, show=False)
        # exit(0)

        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # with torch.no_grad():
        #     for _ in range(10):
        #         _ =  self.pose_detector(self.image, conf=0.50)[0]
        # repetitions = 30
        # timings = np.zeros((repetitions, 1))
        # for rep in range(repetitions):
        #     starter.record()
        #     result = self.pose_detector(self.image, conf=0.50)[0]
        #     ender.record()
        #     torch.cuda.synchronize()
        #     curr_time = starter.elapsed_time(ender)
        #     timings[rep] = curr_time
        # mean_time = np.mean(timings)
        # std_time = np.std(timings)
        # print("Mean Inference Time: {:.4f} ms".format(mean_time))
        # print("Standard Deviation: {:.4f} ms".format(std_time))
        # exit(0)


        result = self.pose_detector(self.image, conf=0.50)[0]

        # result = self.pose_detector.track(self.image, conf=0.50, persist=True)[0]

        human_boxes = result.boxes
        # track_ids = human_boxes.id.int().cpu()
        try:
            human_probs = human_boxes.conf.cpu()
            human_cls = human_boxes.cls.cpu()
            human_boxes = human_boxes.xyxy.cpu()
            human_keypoints = result.keypoints.xy.cpu()
            keypoints_scores = result.keypoints.conf.cpu()
        except:
            return None

        eye_landmarks = human_keypoints[:, 0:5, :]

        names = result.names
        detected_out = {'box':human_boxes, 'prob':human_probs, 'cls':human_cls,
                        'names':names, 'keypoints':human_keypoints,}
                        # 'keypoints_scores':keypoints_scores}
        self.skeleton.append({'keypoints':human_keypoints, 'keypoint_scores':keypoints_scores})
        self.clip_action_bag.append(human_boxes)
        self.clip.append(self.image.astype(np.float32))  # deque will automatically handle the queue size
        self.update_detection(detected_out)
        return detected_out


    def object_detect(self, img):
        result = self.generic_detector(self.image, conf=0.70)[0]
        out = result.plot(save=False, filename="result.jpg", img=img, line_width=1, labels=True)
        # self.names.update(result.names)
        boxes = result.boxes

        # remove 'person' class using torch select
        probs = boxes.conf.cpu()
        cls = boxes.cls.cpu()
        boxes = boxes.xyxy.cpu()

        mask = (cls != 0.)
        boxes = boxes[mask]
        cls = cls[mask]
        probs = probs[mask]

        return out


    def action_recognition(self, args):
        # torch.cuda.empty_cache()

        # resize frames to shortside
        w, h, _ = self.image.shape
        # short_side = min(w, h)
        short_side = 512
        new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
        frames = [mmcv.imresize(img, (new_w, new_h)) for img in self.clip]  # This will still work with deque
        print(new_w, new_h)
        w_ratio, h_ratio = new_w / w, new_h / h

        human_detections = self.clip_action_bag
        for i in range(len(human_detections)):
            det = human_detections[i].to('cuda')
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            human_detections[i] = det[:, :4]

        img_norm_cfg = dict(
            mean=np.array(self.config.model.data_preprocessor.mean),
            std=np.array(self.config.model.data_preprocessor.std),
            to_rgb=False)

        # take the key frame at the center of the clip for action detection
        assert len(human_detections) == len(self.clip_reid_bag), \
            "The number of human detections and reid features should match."
        center_ind = len(human_detections) // 2
        proposal, reid = human_detections[center_ind], self.clip_reid_bag[center_ind]

        imgs = [f.astype(np.float32) for f in frames]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to('cuda')

        datasample = ActionDataSample()
        print(proposal.shape)
        datasample.proposals = InstanceData(bboxes=proposal)
        datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
        with (torch.no_grad()):
            # print the source code of self.action_recognizer
            print(input_tensor.shape)
            print(self.action_recognizer)
            print(input_tensor.shape)
            result = self.action_recognizer(input_tensor, [datasample], mode='predict')
            exit(0)
            scores = result[0].pred_instances.scores

            self.selected_scores.append(scores[center_ind].cpu().numpy())

            if len(self.selected_scores) > 5:
                selected_score = np.array(self.selected_scores)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(selected_score)
                self.factor_analysis.fit(X_scaled)
                self.pca.fit(X_scaled)

                factor_scores = self.factor_analysis.transform(X_scaled)
                weights = self.pca.explained_variance_ratio_[:3]
                composite_Score = np.dot(factor_scores, weights)
                print("composite_Score", composite_Score)
                # self.selected_scores = []



            # with np.printoptions(precision=6, suppress=True):
            #     print(scores.cpu().numpy())

            prediction = []

            # N proposals
            for i in range(proposal.shape[0]):
                prediction.append([])
            for i in range(1, scores.shape[1]+1):
                if i not in self.label_map:
                    continue
                row_all = []
                for j in range(proposal.shape[0]):
                    # if scores[j, i] > args.action_score_thr:
                    # if scores[j, i] > args.action_score_thr \
                    #     and i not in [8, 27, 29, 11, 12]:
                    #     try:
                    #         action = self.label_map[i]
                    #     except:
                    #         action = 'unknown'
                    #     prediction[j].append((action, scores[j, i].item()))
                    if i not in [8, 27, 29, 11, 12]:
                        try:
                            action = self.label_map[i]
                        except:
                            action = 'unknown'
                        prediction[j].append((action, scores[j, i-1].item()))
                    elif i == 8:
                        action = 'sleep?'
                        prediction[j].append((action, scores[j, i-1].item()))
                    elif i == 27:
                        action = 'drink?'
                        prediction[j].append((action, scores[j, i-1].item()))
                    elif i == 29:
                        action = 'eat?'
                        prediction[j].append((action, scores[j, i-1].item()))
                    elif i == 11:
                        action = 'sit?'
                        prediction[j].append((action, scores[j, i-1].item()))
                    elif i == 12:
                        action = 'stand?'
                        prediction[j].append((action, scores[j, i-1].item()))

            self.action_label = prediction
            self.action_proposal = proposal
            self.action_reid = reid
            # the first valid prediction of each clip is enough for analysis
            if sum(len(act) for act in prediction) > 0:
                return
                

    def reset_person(self, frame_not_found_reset):
        states = self.progress["states"]
        for person in list(states.keys()):
            if self.frame - states[person]["last_found_frame"] >= frame_not_found_reset:
                del states[person]
        self.progress["states"] =  states



