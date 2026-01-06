import argparse
from mmengine import DictAction
import os
import sys

# reformulate defineFlags() to argparse
def parse_args():
    home_path = os.path.expanduser('~')
    sys.path.append(os.path.join(home_path, 'elderlycare'))

    parser = argparse.ArgumentParser(description='Configurations')

    # Basic video processing arguments
    parser.add_argument("--input_path", type=str, default="", help="path to input video")
    parser.add_argument("--out_path", type=str, default="", help="path to output video")
    parser.add_argument("--run_name", type=str, default="default_run", help="unique identifier for this processing run")
    parser.add_argument("--debug", action='store_true', default=False, help="debug level")
    parser.add_argument("--debug_save_pic", action='store_true', default=False, help="debug save picture")
    parser.add_argument("--visualize_video", action='store_true', default=False, help="visualize video to .mov")
    parser.add_argument("--resume", action='store_true', default=False, help="resume from previous run")
    parser.add_argument("--re_encode_video", type=str, default='', help="re-encode the video with specific bitrate: e.g., '10000k'")
    
    # Frame sampling and processing
    # parser.add_argument("--sampel_rate", type=int, default=20, help="sample rate for video")
    parser.add_argument("--frame_interval", type=int, default=20, help="sample rate for video")
    # parser.add_argument("--temporal_sample_rate", type=int, default=6, help="frame for each spatiol-temporal analysis")
    parser.add_argument("--clip_len", type=int, default=6, help="frame for each spatiol-temporal analysis")
    parser.add_argument("--frame_not_found_reset", type=int, default=100, help="duration of frames not found before resetting state of person")
    parser.add_argument("--predict_stepsize", type=int, default=60, help="stepsize for prediction, default 20")
    # Model paths and configurations
    parser.add_argument("--weights_path", type=str, default="/home/wzhangbu/elderlycare/weights", help="path to model weights")
    parser.add_argument("--yolo_face_path", type=str, default="yolov11l-face.pt", help="path to YOLO face model weights")
    parser.add_argument("--yolo_pose_path", type=str, default="yolo11l-pose.pt", help="path to YOLO pose model weights")
    parser.add_argument("--yolo_path", type=str, default="yolo11l.pt", help="path to YOLO model weights")
    parser.add_argument("--profile_folder", type=str, default="/home/wzhangbu/elderlycare/people_profile/folder", help="profile_folder for reid")
    
    # Action recognition arguments
    parser.add_argument("--config", type=str, default="configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py", help="skeleton model config file path")
    parser.add_argument("--checkpoint", type=str, default="https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth", help="skeleton model checkpoint file/url")
    parser.add_argument("--label_map", type=str, default="tools/data/skeleton/label_map_ntu60.txt", help="label map file")
    parser.add_argument("--action_score_thr", type=float, default=0.5, help="the threshold of human action score")
    parser.add_argument("--recog_tolerance", type=float, default=0.8, help="threshold for unknown face recognition")
    
    # Configuration overrides
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    
    # Experiment tracking
    parser.add_argument("--project", type=str, default="smart nursing home", help="wandb project name")
    parser.add_argument("--exp_code", type=str, default="001", help="experiment code")
    
    # Random seed
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')

    args = parser.parse_args()
    return args


def analyzor_parse_args():
    parser = argparse.ArgumentParser(description='Configurations for analyzor')
    parser.add_argument("--input_path", type=str, default="/nfs/d/home/wzhangbutaset/pathology/elderlycare/results/", help="path to input video")
    parser.add_argument("--output_path", type=str, default="path_to_your_root/results/", help="path to input video")
    args = parser.parse_args()
    return args


def identifier_parse_args():
    parser = argparse.ArgumentParser(description='Configurations for analyzor')
    parser.add_argument("--input_path", type=str, default="path_to_your_root/results/", help="path to input video")
    parser.add_argument("--out_path", type=str, default="path_to_your_root/results/", help="path to input video")
    parser.add_argument("--weights_path", type=str, default="/home/wzhangbu/elderlycare/weights", help="path to model weights")
    parser.add_argument("--yolo_pose_path", type=str, default="yolo11l-pose.pt", help="path to YOLO pose model weights")
    parser.add_argument("--sampel_rate", type=int, default=15, help="sample rate for video")
    parser.add_argument("--video_sampel_rate", type=int, default=15, help="sample rate for video")
    args = parser.parse_args()
    return args


def alignment_parse_args():
    parser = argparse.ArgumentParser(description='Configurations for analyzor')
    parser.add_argument("--input_path", type=str, default="path_to_your_root/results", help="path to all cameras from one day")
    parser.add_argument("--output_path", type=str, default="path_to_your_root/results", help="path to input video")
    parser.add_argument("--task", type=str, default="alignment", help="path to input video")
    parser.add_argument("--soft", action='store_true', default=False, help="soft")
    args = parser.parse_args()
    return args


def ts_parse_args():
    parser = argparse.ArgumentParser(description='Configurations for time-series modelling')
    parser.add_argument("--input_path", type=str, default="path_to_your_root/results", help="path to all cameras from one day")
    parser.add_argument("--output_path", type=str, default="path_to_your_root/results", help="path to input video")
    args = parser.parse_args()
    return args