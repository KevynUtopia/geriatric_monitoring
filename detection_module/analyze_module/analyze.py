import os
import sys
home_path = os.path.expanduser('~')
sys.path.append(os.path.join(home_path, 'elderlycare'))
import pickle
import glob
from analyze_module import Analyzor
from get_args import analyzor_parse_args
from tqdm import tqdm
import json


def main(args, analyzor, video_name, start_time='000000'):
    action = f"{video_name}-action.pkl"

    # find the start time, which is the postfix of the reid file
    reid = glob.glob(f"{video_name}-*-reid.pkl")
    if len(reid) == 0:
        return
    assert len(start_time) > 0, f"files for {video_name} error"

    reid = reid[0]
    start_time = reid.split('-')[-2]

    skeleton = f"{video_name}-skeleton.pkl"
    with open(action, "rb") as f:
        action = pickle.load(f)
        action_time_key = list(action.keys())

    with open(reid, "rb") as f:
        reid = pickle.load(f)
        reid_time_key = list(reid.keys())
        # print(reid_key)

    with open(skeleton, "rb") as f:
        skeleton = pickle.load(f)
        skeleton_time_key = list(skeleton.keys())

    assert action_time_key == reid_time_key == skeleton_time_key, "keys are different"


    analyzor.update_states(reid, action, skeleton, keys=action_time_key, start_time=start_time)
    # analyzor.analyze()


if __name__ == '__main__':
    args = analyzor_parse_args()

    analyzor = Analyzor()

    all_days = os.listdir(args.input_path)
    # remove files in all_days and only retain folders
    all_days = [day for day in all_days if os.path.isdir(os.path.join(args.input_path, day))]
    all_days.sort()

    for day in all_days:
        if 'recording_' not in day:
            continue
        input_day = os.path.join(args.input_path, day)
        all_cameras = os.listdir(input_day)
        all_cameras.sort()

        # everyday, traverse all cameras
        for camera in all_cameras:
            input_camera = os.path.join(input_day, camera)

            # DEBUG
            # input_camera = 'path_to_your_root/results_v2/recording_2019_06_22_9_20_am/cam_10'
            all_files = os.path.join(input_camera, 'list.json')

            # read the json file
            with open(all_files, 'r') as f:
                video_dict = json.load(f)
                # video is the key of the dict
                video = list(video_dict.keys())
                for line in video:
                    video = line.strip().split('.')[0]
                    main(args, analyzor, os.path.join(input_camera, video))

            analyzor.save_results(out_dir=os.path.join(args.output_path, day, camera))



