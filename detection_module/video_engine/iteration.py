import os
import cv2
import time
import numpy as np
from video_engine.engine import DetectionEngine
from moviepy.editor import VideoFileClip
import math
import pickle
import torch
import pandas as pd

def train_one_iteration(args, input_video, filename, ALL_PROPOSAL_COUNT=0):

    engine = DetectionEngine(args=args)
    engine.initialize_progress(filename)
    engine.initialize_people(args.profile_folder)

    cap = cv2.VideoCapture(input_video)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"视频的压缩格式（FourCC）: {fourcc_str}, 帧率: {FPS}")

    # Create output directory structure
    date, cam, recording = input_video.split("/")[-3:]
    recording = recording.split(".")[0]

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = [frame_width, frame_height, frame_width, frame_height]
    window_size = args.clip_len * args.frame_interval
    
    # output
    if args.visualize_video:
        fps = 5
        # Use the input video name with a suffix for the output
        output_video_name = os.path.splitext(os.path.basename(input_video))[0] + "_processed.mp4"
        output_video_path = os.path.join(args.out_path, output_video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'X264' for H.264 codec
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    else:
        out = None

    num_frames = int(cap.get(7))
    print(f"视频总帧数: {num_frames}, 帧宽: {frame_width}, 帧高: {frame_height}")

    frame, missing_frame = 0, 0
    engine.reset_action_detector()
    total_start = time.time()

    precise_current_sec = []
    while cap.isOpened():

        # step 1 ~ read frame
        # ===================== read frame =====================
        frame += 1
        start = time.time()
        ret, image = cap.read()
        # skip broken frames (decoded from .mov) since HOP video is not well encoded
        if not ret or image is None:
            missing_frame += 1
            if frame > num_frames:
                break
            else:
                continue
        if frame % args.frame_interval != 1:
            continue

        engine.take_frame(image, frame)
        # get precise_current_second using opencv in-built function
        precise_current_sec.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES) / FPS))
        if engine.timestamp == '':
            engine.update_timestamp(filename)
        # ===================== end of reading =====================

        # Step 2 ~ Detect human and their poses using YOLO
        detected_human = engine.pose_detect()
        if detected_human is None:
            engine.save_progress(None, None)
            continue

        # Step 3 (preliminary) ~ Detect faces using YOLO

        # Step 4 ~ body-based reid (rather than face-recognition-based)
        reid_out = engine.track_reid(detected_human)
        # engine.track_reid(detected_human)

        # Step 5 ~ sptial-temporal action detection for every #clip_len of frames
        if frame % args.predict_stepsize == 1 and len(engine.clip) == args.clip_len:
            engine.action_recognition(args)
            time_checkpoint = time.time()
            print("Spatial-Temporal Action Detection, ~ {:.3f}s".format(start - time_checkpoint))
            detected_results, reid_out, action_out = engine.draw(reid_out)

            # Step 7 ~ save analysis results for every clip
            # includes reids, corresponding actions, and keypoints (for downstrean analysis)
            engine.save_progress(reid_out, action_out)

            # reset the detector for the next clip
            engine.reset_action_detector()
            precise_current_sec = []
        else:
            detected_results, reid_out, action_out = engine.draw(reid_out)
            time_checkpoint = time.time()
            print("Frame {} with {} missing, ~ {:.3f}s".format(frame, missing_frame, start - time_checkpoint))
        # detected_results, reid_out, action_out = engine.draw(reid_out)

        # Step 8 ~ save the video
        if args.visualize_video:
            out.write(detected_results)


    # end of processing
    cap.release()
    out.release() if out else None
    total_end = time.time()
    print("Frame ended at {} with {} missing, time cost: {:.3f}s".format(frame, missing_frame, total_end - total_start))

    engine.save_results(args, filename)

    if args.visualize_video and args.re_encode_video != '':
        # modifying bitrage
        try:
            clip = VideoFileClip(output_video_path)
            target_bitrate = args.re_encode_video
            output_video_path = output_video_path.replace(".mov", "_processed.mov")
            clip.write_videofile(output_video_path, codec='libx264', bitrate=target_bitrate)
        except Exception as e:
            print("Error in re-encoding video")