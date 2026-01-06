import os
import sys
home_path = os.path.expanduser('~')
sys.path.append(os.path.join(home_path, 'elderlycare'))
import pickle
import glob
import argparse
from analyze_module import Analyzor
from analyze_module.alignment import Alignment_Worker
from tqdm import tqdm
import json


def combined_parse_args():
    """Combined argument parser for both analysis and alignment phases"""
    parser = argparse.ArgumentParser(description='Combined Analysis and Alignment Pipeline')
    
    # Analysis phase arguments
    parser.add_argument("--input_path", type=str, required=True, 
                       help="Path to input data (containing recording folders)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to output results")
    
    # Alignment phase arguments
    parser.add_argument("--alignment_task", type=str, default="alignment", 
                       help="Task name for alignment output folder")
    parser.add_argument("--soft", action='store_true', default=False, 
                       help="Use soft alignment (mean) instead of mode")
    
    # Optional arguments
    parser.add_argument("--skip_analysis", action='store_true', default=False,
                       help="Skip analysis phase if results already exist")
    parser.add_argument("--skip_alignment", action='store_true', default=False,
                       help="Skip alignment phase")
    
    args = parser.parse_args()
    return args


def analyze_phase(input_path, output_path, skip_existing=False):
    """
    Execute the analysis phase - process video data and save per-camera results
    """
    print("=" * 60)
    print("STARTING ANALYSIS PHASE")
    print("=" * 60)
    
    analyzor = Analyzor()
    
    all_days = os.listdir(input_path)
    # Remove files and only retain folders
    all_days = [day for day in all_days if os.path.isdir(os.path.join(input_path, day))]
    all_days.sort()
    
    processed_days = []
    
    for day in tqdm(all_days, desc="Processing days"):
        if 'recording_' not in day:
            continue
            
        input_day = os.path.join(input_path, day)
        output_day = os.path.join(output_path, day)
        
        # Check if day already processed
        if skip_existing and os.path.exists(output_day):
            print(f"Skipping {day} - already processed")
            processed_days.append(day)
            continue
            
        all_cameras = os.listdir(input_day)
        all_cameras.sort()
        
        # Process each camera for this day
        for camera in tqdm(all_cameras, desc=f"Cameras in {day}", leave=False):
            input_camera = os.path.join(input_day, camera)
            output_camera = os.path.join(output_day, camera)
            
            # Check if this camera already processed
            if skip_existing and os.path.exists(output_camera):
                continue
            
            # Check for list.json file
            all_files = os.path.join(input_camera, 'list.json')
            if not os.path.exists(all_files):
                print(f"Warning: {all_files} not found, skipping {camera}")
                continue
            
            # Read the json file and process videos
            with open(all_files, 'r') as f:
                video_dict = json.load(f)
                videos = list(video_dict.keys())
                
                for video_line in tqdm(videos):
                    video = video_line.strip().split('.')[0]
                    video_path = os.path.join(input_camera, video)
                    
                    # Process this video
                    success = process_video(analyzor, video_path)
                    if not success:
                        print(f"Warning: Failed to process video {video_path}")
            
            # Save results for this camera
            analyzor.save_results(out_dir=output_camera)
        
        processed_days.append(day)
        print(f"Completed analysis for {day}")
    
    print(f"Analysis phase completed for {len(processed_days)} days")
    return processed_days


def process_video(analyzor, video_name, start_time='000000'):
    """
    Process a single video - load reid, action, skeleton data and update states
    """
    # print(f"Processing video: {video_name}")
    action_file = f"{video_name}-action.pkl"
    skeleton_file = f"{video_name}-skeleton.pkl"
    
    # Find the reid file with timestamp
    reid_files = glob.glob(f"{video_name}-*-reid.pkl")
    if len(reid_files) == 0:
        return False
    
    reid_file = reid_files[0]
    start_time = reid_file.split('-')[-2]
    
    # Check if all required files exist
    required_files = [action_file, reid_file, skeleton_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Warning: Required file {file_path} not found")
            return False
    
    try:
        # Load data
        with open(action_file, "rb") as f:
            action = pickle.load(f)
            action_time_key = list(action.keys())
        
        with open(reid_file, "rb") as f:
            reid = pickle.load(f)
            reid_time_key = list(reid.keys())
        
        with open(skeleton_file, "rb") as f:
            skeleton = pickle.load(f)
            skeleton_time_key = list(skeleton.keys())
        
        # Verify keys match
        if not (action_time_key == reid_time_key == skeleton_time_key):
            print(f"Warning: Mismatched keys for video {video_name}")
            return False
        
        # Update analyzer states
        analyzor.update_states(reid, action, skeleton, keys=action_time_key, start_time=start_time)
        return True
        
    except Exception as e:
        print(f"Error processing video {video_name}: {str(e)}")
        return False


def alignment_phase(output_path, alignment_task="alignment", soft=False):
    """
    Execute the alignment phase - align results across multiple camera views
    """
    print("=" * 60)
    print("STARTING ALIGNMENT PHASE")
    print("=" * 60)
    
    # Get all processed days
    all_days = os.listdir(output_path)
    all_days = [day for day in all_days if os.path.isdir(os.path.join(output_path, day)) and 'recording_' in day]
    all_days.sort()
    
    processed_alignments = []
    
    for day in tqdm(all_days, desc="Aligning days"):
        day_path = os.path.join(output_path, day)
        
        # Check if alignment already exists
        alignment_output = os.path.join(day_path, alignment_task)
        if os.path.exists(alignment_output):
            print(f"Alignment for {day} already exists, skipping...")
            continue
        
        # Get all cameras for this day
        all_cam = os.listdir(day_path)
        all_cam = [cam for cam in all_cam if cam.startswith('cam_')]
        
        if len(all_cam) < 2:
            print(f"Warning: Only {len(all_cam)} cameras found for {day}, skipping alignment")
            continue
            
        all_cam.sort()
        
        # Get all person candidates from all cameras
        all_cam_candidates = {}
        for cam in all_cam:
            cam_path = os.path.join(day_path, cam)
            if not os.path.exists(cam_path):
                continue
                
            all_files = os.listdir(cam_path)
            # Find all CSV files (person results)
            candidates = [os.path.splitext(file)[0] for file in all_files if file.endswith('.csv')]
            candidates.sort()
            all_cam_candidates[cam] = candidates
        
        if not all_cam_candidates:
            print(f"Warning: No candidates found for {day}")
            continue
        
        # Get union of all candidates
        all_candidates = set()
        for cam_candidates in all_cam_candidates.values():
            all_candidates.update(cam_candidates)
        all_candidates = list(all_candidates)
        all_candidates.sort()
        
        if not all_candidates:
            print(f"Warning: No person candidates found for {day}")
            continue
        
        print(f"Aligning {len(all_candidates)} people across {len(all_cam)} cameras for {day}")
        
        # Initialize and run alignment
        try:
            align = Alignment_Worker(all_cam=all_cam, all_candidates=all_candidates)
            align.initialize(input_path=day_path)
            align.align()
            align.post_process(output_path=day_path, task=alignment_task, soft=soft)
            
            processed_alignments.append(day)
            print(f"Completed alignment for {day}")
            
        except Exception as e:
            print(f"Error during alignment for {day}: {str(e)}")
            continue
    
    print(f"Alignment phase completed for {len(processed_alignments)} days")
    return processed_alignments


def main():
    """Main function to run the combined analysis and alignment pipeline"""
    args = combined_parse_args()
    
    print("Starting Combined Analysis and Alignment Pipeline")
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Skip analysis: {args.skip_analysis}")
    print(f"Skip alignment: {args.skip_alignment}")
    print(f"Soft alignment: {args.soft}")
    
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    
    processed_days = []
    
    # Phase 1: Analysis
    if not args.skip_analysis:
        processed_days = analyze_phase(
            input_path=args.input_path,
            output_path=args.output_path,
            skip_existing=False
        )
    else:
        print("Skipping analysis phase")
        # Get already processed days
        all_days = os.listdir(args.output_path)
        processed_days = [day for day in all_days if os.path.isdir(os.path.join(args.output_path, day))]
    
    # Phase 2: Alignment
    if not args.skip_alignment and processed_days:
        alignment_results = alignment_phase(
            output_path=args.output_path,
            alignment_task=args.alignment_task,
            soft=args.soft
        )
        print(f"Pipeline completed successfully!")
        print(f"Processed {len(processed_days)} days, aligned {len(alignment_results)} days")
    else:
        if args.skip_alignment:
            print("Skipping alignment phase")
        else:
            print("No processed days found for alignment")
    
    print("Combined pipeline finished!")


if __name__ == '__main__':
    main() 