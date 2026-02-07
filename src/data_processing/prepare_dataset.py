import json
import os
import torch
from torchvision.transforms import ToTensor
import shutil
import pandas as pd
from datasets import Dataset, Features, Value, Sequence, List
from pathlib import Path
import glob
# Added imports for Parquet generation (required for tasks.parquet)
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
import traceback
from datetime import datetime
from episode_data import EpisodeData
import numpy as np
from lerobot.datasets.compute_stats import compute_episode_stats, aggregate_stats
from lerobot.datasets.utils import load_info, write_stats
import tempfile
from episode_data import EpisodeData, CameraData
import re



def create_tasks_parquet(root_dir: Path, task_description: str):
    """
    Write meta/tasks.parquet in the format LeRobot expects:
    - DataFrame index = task string(s)
    - Column 'task_index' = integer index 0..N-1
    """
    tasks = [task_description]

    # Build DataFrame with task strings as index and task_index as column
    df = pd.DataFrame({"task_index": list(range(len(tasks)))}, index=tasks)
    df.index.name = None  # optional, but keeps index as pure strings

    tasks_dir = root_dir / "meta"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    tasks_parquet_path = tasks_dir / "tasks.parquet"

    # Use pandas to_parquet so the index is preserved in a way pd.read_parquet will restore it
    df.to_parquet(tasks_parquet_path)
    

def create_episodes_parquet(root_dir: Path):
    """
    Reads the data from episodes.jsonl and saves it as nested Parquet files
    in the format LeRobot expects.
    """
    episodes_jsonl_path = root_dir / "meta" / "episodes.jsonl"
    episodes_parquet_dir = root_dir / "meta" / "episodes"
    
    if not episodes_jsonl_path.exists():
        print(f"❌ WARNING: {episodes_jsonl_path} not found. Skipping episodes index creation.")
        return

    
    # 1. Read the JSONL file line by line
    with open(episodes_jsonl_path, 'r') as f:
        episode_lines = [json.loads(line) for line in f]
    
    if not episode_lines:
        print("❌ WARNING: episodes.jsonl is empty. Skipping episodes index creation.")
        return

    
    # 3. Create DataFrame and convert to Arrow Table (only for current episode)
    df = pd.DataFrame(episode_lines)
    table = pa.Table.from_pandas(df)

  
    # 4. Create the nested directory structure LeRobot expects
    # This creates a subdirectory with multiple Parquet files
    data_subdir = episodes_parquet_dir / f"chunk-000"
    data_subdir.mkdir(exist_ok=True, parents=True)
    
    # Write multiple Parquet files (LeRobot expects this structure)
    pq.write_table(table, data_subdir / f"file-000.parquet")


def generate_data_files(output_dir: Path, episode_data: EpisodeData, json_data: dict, last_frames_to_chop: int, first_frames_to_chop: int = 0, mode="diff"):
    """
    Generates the data files for a single episode.
    """
    
    
    num_joints = len(json_data["joint_names"])
    original_num_frames = len(json_data["frames"])
    effective_num_frames = original_num_frames - last_frames_to_chop - first_frames_to_chop
    delta_scale = 40  # Assuming joint positions are already in correct scale
    gripper_scale = 10  # Scale factor for gripper DOF if needed
    
    if effective_num_frames <= 0:
        print(f"❌ ERROR: Chopping {last_frames_to_chop} frames from {original_num_frames} results in 0 or fewer frames. Skipping data generation.")
        return 0

    # Get joint positions starting from the first_frames_to_chop index
    joint_positions = [frame["joint_positions"] for frame in json_data["frames"][first_frames_to_chop:]]
    
    lerobot_frames = []
    timestamp_base = 0.0
    
    for i in range(effective_num_frames):
        current_state_scaled = [pos for pos in joint_positions[i]]
        next_state_scaled = [pos for pos in joint_positions[i + 1]] if i + 1 < effective_num_frames else [pos for pos in joint_positions[i]]
        
        # Scale up the 7th DOF (index 6) by 50 times
        if len(current_state_scaled) > 6:
            current_state_scaled[6] *= gripper_scale
        if len(next_state_scaled) > 6:
            next_state_scaled[6] *= gripper_scale
        
        if mode == "diff":
            delta = [next_pos - current_pos for next_pos, current_pos in zip(next_state_scaled, current_state_scaled)]
            action = [d * delta_scale for d in delta]
        else:
            action = next_state_scaled
            

        is_done = (i == effective_num_frames - 1)
        
        # 🟢 CORRECTION: Correct calculation of the global 'index'
        # The global index is the offset + the current frame's index (i)
        global_index = episode_data.global_index_offset + i
        
        # Create frame data with observation images for each camera
        # Adjust frame_index to account for skipped frames
        frame_data = {
            "observation.state": action,
            "action": current_state_scaled,  # not too sure why the action comes before the state in lerobot visualization tool, so we manually change the order here to make it more intuitive for visualization
            "timestamp": timestamp_base,
            "episode_index": episode_data.episode_index,
            "frame_index": json_data["frames"][i + first_frames_to_chop]["frame_index"] - first_frames_to_chop,  # 局部索引 (0, 1, 2, ...)
            "index": global_index,  # 全局索引 (0, 1, 2, ..., N)
            "next.done": is_done,
            "next.reward": 1.0 if is_done else 0.0,
            "task_index": 0,
            #"task_description": episode_data.task_description
        }        
                
        lerobot_frames.append(frame_data)
        timestamp_base += 0.1
        
    # [Rest of the function remains the same, writing to Parquet]
    hf_dataset = Dataset.from_pandas(pd.DataFrame(lerobot_frames))
    
    # Build feature configuration including camera features
    feature_config_dict = {
        "observation.state": Sequence(Value("float32"), length=num_joints),
        "action": Sequence(Value("float32"), length=num_joints),
        "timestamp": Value("float64"),
        "episode_index": Value("int64"),
        "frame_index": Value("int64"),
        "index": Value("int64"),
        "next.done": Value("bool"),
        "next.reward": Value("float32"),
        "task_index": Value("int64"),
        #"task_description": Value("string"),
    }    
        
    feature_config = Features(feature_config_dict)
    hf_dataset = hf_dataset.cast(feature_config)


    # Export to Parquet with new directory structure
    chunk_dir = output_dir / "data" / f"chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = chunk_dir / f"file-000.parquet"
    
    # Check if parquet file already exists and append if it does
    if parquet_path.exists():
        # Read existing data
        existing_dataset = pd.read_parquet(parquet_path)
        # Concatenate new data with existing data
        combined_df = pd.concat([existing_dataset, pd.DataFrame(lerobot_frames)], ignore_index=True)
        # Convert back to dataset and cast features
        combined_dataset = Dataset.from_pandas(combined_df)
        combined_dataset = combined_dataset.cast(feature_config)
        # Write combined data back to parquet
        combined_dataset.to_parquet(parquet_path)
    else:
        # Write new data to parquet
        hf_dataset.to_parquet(parquet_path)
    
    return effective_num_frames

# Modified signature to accept last_frames_to_chop and first_frames_to_chop
def generate_meta_files(output_dir: Path, episode_data: EpisodeData, json_data: dict, is_first_episode: bool = False, last_frames_to_chop: int = 0, first_frames_to_chop: int = 0):
    
    
    # [File path definitions and checks remain the same...]
    data_path = "data/chunk-{episode_index:03d}/file-{episode_index:03d}.parquet"
    video_path = "videos/{video_key}/chunk-{chunk_index:03d}/file-{chunk_index:03d}.mp4"
    info_json_path = output_dir / "meta" / "info.json"
    
    num_joints = 7
    
    # 🔴 CORE CHANGE 1: Use the effective number of frames
    original_num_frames = len(json_data["frames"])
    effective_num_frames = original_num_frames - last_frames_to_chop - first_frames_to_chop
    
    if effective_num_frames <= 0:
        print(f"❌ ERROR: Effective frame count is 0 or less ({effective_num_frames}). Skipping meta generation.")
        return
        
    # --- Update info.json ---
    
    if is_first_episode or not info_json_path.exists():
        # Create base info_json structure
        info_json = {
            "codebase_version": "v3.0", 
            "fps": round(episode_data.fps, 2),
            "total_episodes": 1,
            "total_frames": effective_num_frames,
            "total_tasks": 1,
            "data_path": data_path,
            "video_path": video_path,
            "features": {
                "timestamp": {"dtype": "float64", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "next.done": {"dtype": "bool", "shape": [1]},
                "next.reward": {"dtype": "float32", "shape": [1]},
                "observation.state": {
                    "shape": [num_joints],
                    "dtype": "float32"
                },
                "action": {
                    "shape": [num_joints],
                    "dtype": "float32"
                },
                # "task_description": {  # 新增：任务字段定义
                #     "shape": [1],
                #     "dtype": "string"
                # }
            }
        }

        
        # Add camera features dynamically
        for camera_data in episode_data.cameras:
            camera_name = camera_data.camera
            feature_key = f"observation.images.{camera_name}"
            channel = 3
            codec = "av1"
            pix_fmt = "yuv420p"
                        
            info_json["features"][feature_key] = {
                "shape": [400, 640, channel],  # Adjust based on your actual video dimensions
                "dtype": "video",
                "names": [
                    "height",
                    "width",
                    "channel"
                ],
                "info": {
                    "video.fps": round(episode_data.fps, 2),
                    "video.codec": codec,
                    "video.pix_fmt": pix_fmt,
                    "video.is_depth_map": 'depth' in camera_name.lower(),
                    "has_audio": False
                }
            }
        
        with open(info_json_path, "w") as f:
            json.dump(info_json, f, indent=2)
    else:
        # Read existing info.json and update totals
        with open(info_json_path, "r") as f:
            info_json = json.load(f)
        
        # 🔴 CORE CHANGE 2: Update totals using effective_num_frames
        info_json["total_frames"] = info_json.get("total_frames", 0) + effective_num_frames
        info_json["total_episodes"] = info_json.get("total_episodes", 0) + 1
        
        with open(info_json_path, "w") as f:
            json.dump(info_json, f, indent=2)
            
    # --- episodes.jsonl (Index) ---
    
    # Calculate global index boundaries
    dataset_from_index = episode_data.global_index_offset 
    # 🔴 CORE CHANGE 3: dataset_to_index is now based on effective_num_frames
    dataset_to_index = dataset_from_index + effective_num_frames - 1
    
    # Create base episodes_jsonl structure
    episodes_jsonl = {
        "episode_index": episode_data.episode_index,
        "task_index": 0,
        "frame_index_offset": 0, # Index offset from the start of the Parquet file (always 0 here)
        "num_frames": effective_num_frames, # 🔴 CORE CHANGE 4: Report effective number of frames
        "dataset_from_index": dataset_from_index,
        "dataset_to_index": dataset_to_index,
    }
    
    # [Rest of episodes.jsonl writing and parquet index creation remains the same...]
    for camera_data in episode_data.cameras:
        camera_name = camera_data.camera
        episodes_jsonl[f"videos/observation.images.{camera_name}/from_timestamp"] = dataset_from_index * (1.0 / episode_data.fps)
        episodes_jsonl[f"videos/observation.images.{camera_name}/to_timestamp"] = dataset_to_index * (1.0 / episode_data.fps)
        episodes_jsonl[f"videos/observation.images.{camera_name}/chunk_index"] = 0
        episodes_jsonl[f"videos/observation.images.{camera_name}/file_index"] = 0
        #episodes_jsonl[f"videos/observation.images.{camera_name}/frame_index_offset"] = 0
        #episodes_jsonl[f"data/chunk_index"] = episode_data.episode_index
        episodes_jsonl[f"data/file_index"] = 0
        
    
    with open(output_dir / "meta" / "episodes.jsonl", "a") as f:
        f.write(json.dumps(episodes_jsonl) + "\n")
        
    if is_first_episode:
        create_tasks_parquet(output_dir, episode_data.task_description)
    


def generate_video_files(output_dir: Path, episode_data: EpisodeData, json_data: dict, last_frames_to_chop: int = 0, first_frames_to_chop: int = 0):
    """
    Generate video files, properly handling frame chopping to match tabular data.
    
    Args:
        output_dir (Path): Output directory for the dataset
        episode_data (EpisodeData): Episode data containing camera information
        json_data (dict): JSON data containing frame information
        last_frames_to_chop (int): Number of frames to chop from the end (global value)
        first_frames_to_chop (int): Number of frames to chop from the beginning (global value, but overridden by camera-specific values)
    """
    
    # Calculate effective number of frames using global values
    original_num_frames = len(json_data["frames"])
    effective_num_frames = original_num_frames - last_frames_to_chop - first_frames_to_chop
    
    if effective_num_frames <= 0:
        print(f"❌ ERROR: Effective frame count is 0 or less ({effective_num_frames}). Skipping video generation.")
        return
        
    for camera_data in episode_data.cameras:
        camera_name = camera_data.camera
        cam_folder = f"observation.images.{camera_name}"
    
        # Create the new directory structure for videos
        video_chunk_dir = output_dir / "videos" / cam_folder / f"chunk-{episode_data.episode_index:03d}"
        video_chunk_dir.mkdir(parents=True, exist_ok=True)
        output_video_name = f"file-{episode_data.episode_index:03d}.mp4"
        output_video_path = video_chunk_dir / output_video_name
    
        # Use camera-specific first_frames_to_chop if available, otherwise use global value
        camera_first_frames_to_chop = getattr(camera_data, 'first_frames_to_chop', first_frames_to_chop)
        
        # Recalculate effective number of frames for this camera
        camera_effective_num_frames = original_num_frames - last_frames_to_chop - camera_first_frames_to_chop
        
        if camera_effective_num_frames <= 0:
            print(f"❌ ERROR: Effective frame count for camera {camera_name} is 0 or less ({camera_effective_num_frames}). Skipping video generation.")
            continue
    
        # Handle video processing based on source type
        video_file_path = Path(camera_data.video_path)
        if video_file_path.exists() and video_file_path.suffix.lower() == '.mp4':
            # If it's a video file, we need to chop it to match the effective frame count
            # If first_frames_to_chop is specified, use the enhanced chop function
            if camera_first_frames_to_chop > 0:
                chop_video_to_frame_count(video_file_path, output_video_path, camera_effective_num_frames, episode_data.fps, camera_first_frames_to_chop)
            else:
                chop_video_to_frame_count(video_file_path, output_video_path, camera_effective_num_frames, episode_data.fps)
        elif video_file_path.exists():
            # For other file types, just copy
            shutil.copy(video_file_path, output_video_path)
        else:
            print(f"⚠️ WARNING: Video file not found at {camera_data.video_path}. Skipping video copy.")


def chop_video_to_frame_count(input_video_path: Path, output_video_path: Path, target_frame_count: int, fps: float, first_num_frames: int = 0):
    """
    Chop a video to contain exactly target_frame_count frames, skipping first_num_frames if specified.
    """
    try:
        # First, check if the input video has 1 channel (grayscale)
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=pix_fmt",
            "-of", "csv=p=0",
            str(input_video_path)
        ]
        
        import subprocess
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        is_grayscale = False
        if probe_result.returncode == 0:
            pixel_format = probe_result.stdout.strip()
            # Common grayscale formats
            grayscale_formats = ['gray', 'gray8', 'gray16le', 'gray16be']
            is_grayscale = pixel_format in grayscale_formats
        
        # Use ffmpeg to extract only the required number of frames, skipping first_num_frames if needed
        cmd = [
            "ffmpeg",
            "-i", str(input_video_path)
        ]
        
        # If first_num_frames is specified, skip those frames
        if first_num_frames > 0:
            if is_grayscale:
                # For grayscale videos, convert to rgb during frame selection
                cmd.extend(["-vf", f"select=gte(n\\,{first_num_frames}),setpts=N/TB/{fps},format=rgb24"])
            else:
                cmd.extend(["-vf", f"select=gte(n\\,{first_num_frames}),setpts=N/TB/{fps}"])
        elif is_grayscale:
            # For grayscale videos, convert to rgb
            cmd.extend(["-vf", "format=rgb24"])
        
        cmd.extend([
            "-vframes", str(target_frame_count),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-y",
            str(output_video_path)
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ WARNING: Failed to chop video {input_video_path} to {target_frame_count} frames (skipping {first_num_frames} frames): {result.stderr}")
            # Fallback: copy the original video (with conversion if needed)
            if is_grayscale:
                # Convert grayscale to rgb during copy
                convert_cmd = [
                    "ffmpeg",
                    "-i", str(input_video_path),
                    "-vf", "format=rgb24",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-y",
                    str(output_video_path)
                ]
                convert_result = subprocess.run(convert_cmd, capture_output=True, text=True)
                if convert_result.returncode != 0:
                    print(f"⚠️ WARNING: Failed to convert grayscale video {input_video_path}: {convert_result.stderr}")
                    shutil.copy(input_video_path, output_video_path)
            else:
                shutil.copy(input_video_path, output_video_path)
    except Exception as e:
        print(f"⚠️ WARNING: Error chopping video {input_video_path}: {e}")
        # Fallback: copy the original video
        shutil.copy(input_video_path, output_video_path)


def chop_first_frames_from_video(input_video_path: Path, output_video_path: Path, first_num_frames: int, fps: float):
    """
    Chop the first first_num_frames from a video.
    
    Args:
        input_video_path (Path): Path to the input video file
        output_video_path (Path): Path where the output video will be saved
        first_num_frames (int): Number of frames to chop from the beginning
        fps (float): Frames per second of the video
    """
    try:
        # First, check if the input video has 1 channel (grayscale)
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=pix_fmt",
            "-of", "csv=p=0",
            str(input_video_path)
        ]
        
        import subprocess
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        is_grayscale = False
        if probe_result.returncode == 0:
            pixel_format = probe_result.stdout.strip()
            # Common grayscale formats
            grayscale_formats = ['gray', 'gray8', 'gray16le', 'gray16be']
            is_grayscale = pixel_format in grayscale_formats
        
        # Use ffmpeg to skip the first first_num_frames
        cmd = [
            "ffmpeg",
            "-i", str(input_video_path)
        ]
        
        if is_grayscale:
            # For grayscale videos, convert to rgb during frame selection
            cmd.extend(["-vf", f"select=gt(n\\,{first_num_frames}),setpts=N/TB/{fps},format=rgb24"])
        else:
            cmd.extend(["-vf", f"select=gt(n\\,{first_num_frames}),setpts=N/TB/{fps}"])
        
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-y",
            str(output_video_path)
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ WARNING: Failed to chop first {first_num_frames} frames from video {input_video_path}: {result.stderr}")
            # Fallback: copy the original video (with conversion if needed)
            if is_grayscale:
                # Convert grayscale to rgb during copy
                convert_cmd = [
                    "ffmpeg",
                    "-i", str(input_video_path),
                    "-vf", "format=rgb24",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-y",
                    str(output_video_path)
                ]
                convert_result = subprocess.run(convert_cmd, capture_output=True, text=True)
                if convert_result.returncode != 0:
                    print(f"⚠️ WARNING: Failed to convert grayscale video {input_video_path}: {convert_result.stderr}")
                    shutil.copy(input_video_path, output_video_path)
            else:
                shutil.copy(input_video_path, output_video_path)
    except Exception as e:
        print(f"⚠️ WARNING: Error chopping first frames from video {input_video_path}: {e}")
        # Fallback: copy the original video
        shutil.copy(input_video_path, output_video_path)


def extract_video_frames_to_temp_dir(video_path: Path, temp_dir: Path) -> list[Path]:
    """
    Extract frames from a video file to a temporary directory.
    
    Args:
        video_path (Path): Path to the video file
        temp_dir (Path): Temporary directory to store extracted frames
        
    Returns:
        list[Path]: List of paths to extracted frame images
    """
    # Check if video file exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create temporary directory for frames
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_paths = []
    frame_index = 0
    
    # Extract frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame as PNG
        frame_path = frames_dir / f"frame_{frame_index:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        frame_index += 1
    
    # Release video capture
    cap.release()
    
    
    return frame_paths

def update_total_frames_from_episodes(output_dir: Path):
    """
    Reads episodes.jsonl file and calculates the total sum of num_frames,
    then updates info.json with the calculated sum.
    """
    
    
    # Path to episodes.jsonl and info.json
    episodes_jsonl_path = output_dir / "meta" / "episodes.jsonl"
    info_json_path = output_dir / "meta" / "info.json"
    
    # Check if episodes.jsonl exists
    if not episodes_jsonl_path.exists():
        print(f"❌ WARNING: {episodes_jsonl_path} not found. Skipping total frames update.")
        return
    
    # Read episodes.jsonl and sum num_frames
    total_frames = 0
    total_episodes = 0
    try:
        with open(episodes_jsonl_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    episode_data = json.loads(line)
                    total_frames += episode_data.get("num_frames", 0)
                    total_episodes += 1
        
        
        # Read existing info.json
        if not info_json_path.exists():
            print(f"❌ WARNING: {info_json_path} not found. Skipping total frames update.")
            return
            
        with open(info_json_path, 'r') as f:
            info_data = json.load(f)
        
        # Update total_frames
        info_data["total_frames"] = total_frames
        info_data["total_episodes"] = total_episodes
        
        # Write updated info.json
        with open(info_json_path, 'w') as f:
            json.dump(info_data, f, indent=2)
            
        
    except Exception as e:
        print(f"❌ ERROR: Failed to update total frames: {e}")



def load_video_to_numpy(video_path: Path):
    """Decodes video into (Frames, Height, Width, Channels) [0, 1]."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
            
        # Check number of channels
        if len(frame.shape) == 2:
            # Grayscale image, convert to 3 channels by repeating
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 1:
            # Single channel image, convert to 3 channels by repeating
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            # Assume BGR format and convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        frame = frame.astype(np.float32) / 255.0
        # Keep as HWC (Height, Width, Channels)
        frames.append(frame)
    cap.release()
    return np.stack(frames) if frames else None

def compute_and_save_dataset_stats(output_dir: Path):
    output_dir = Path(output_dir)
    info = load_info(output_dir)
    features = info["features"]
    
    numerical_dict = {}
    numerical_features = {}
    visual_results = {}
    
    # Read all parquet files
    data_dir = output_dir / "data"
    all_data_frames = []
    
    for chunk_dir in data_dir.glob("chunk-*"):
        for parquet_file in chunk_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            all_data_frames.append(df)
    
    if not all_data_frames:
        raise ValueError("No data files found for stats computation")
    
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    total_frames = len(combined_df)
    
    print(f"📊 Total frames in dataset: {total_frames}")

    # Process visual features with simple fixed stats instead of loading all data
    for f_name, f_info in features.items():
        if f_info["dtype"] in ["image", "video"]:
            print(f"\n🎥 Processing video feature: {f_name}")
            
            # Get shape info from features
            shape = f_info["shape"]  # Should be [400, 640, channels]
            height, width, channels = shape
            is_depth = 'depth' in f_name.lower()
            
            # Use simple fixed statistics for RGB/RGBD data
            # For RGB images, use min=0, max=1 (typical for normalized images)
            # For depth images, also use min=0, max=1 (can be adjusted if needed)
            if channels == 3:  # RGB
                means = [0.5, 0.5, 0.5]  # Simple average values
                stds = [0.5, 0.5, 0.5]   # Simple std values
                mins = [0.0, 0.0, 0.0]   # Min values for RGB
                maxs = [1.0, 1.0, 1.0]   # Max values for RGB
            else:  # Grayscale or depth
                means = [0.5]            # Simple average value
                stds = [0.5]             # Simple std value
                mins = [0.0]             # Min value
                maxs = [1.0]             # Max value

            # Format for LeRobot
            visual_results[f_name] = {
                "min":  [[[m]] for m in mins],
                "max":  [[[m]] for m in maxs],
                "mean": [[[m]] for m in means],
                "std":  [[[m]] for m in stds],
            }
            
            print(f"  ✅ Using fixed stats for {f_name} - Channels: {channels}")
            
        else:
            # Numerical feature handling
            if f_name in combined_df.columns:
                val = combined_df[f_name].values
                if len(val) > 0 and hasattr(val[0], '__len__'):
                    numerical_dict[f_name] = np.stack(val.tolist())
                else:
                    numerical_dict[f_name] = val
                numerical_features[f_name] = f_info

    # Compute numerical stats manually without using compute_episode_stats
    print("\n🧮 Computing numerical stats manually...")
    final_stats = {}
    
    # Process each numerical feature
    for f_name, data in numerical_dict.items():
        f_info = numerical_features[f_name]
        
        # Convert to numpy array if not already
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Skip boolean data types as they can't be used for numerical stats
        if data.dtype == np.bool_:
            print(f"  ⚠️ Skipping boolean feature '{f_name}' for numerical stats")
            continue
            
        # Handle different data shapes
        if data.ndim == 1:
            # Single dimensional data (e.g., scalar values)
            # Filter out non-numerical values
            numeric_mask = np.isfinite(data) if np.issubdtype(data.dtype, np.number) else np.ones(len(data), dtype=bool)
            filtered_data = data[numeric_mask] if np.issubdtype(data.dtype, np.number) else data.astype(float)
            
            if len(filtered_data) == 0:
                print(f"  ⚠️ No valid numerical data for feature '{f_name}'")
                continue
                
            final_stats[f_name] = {
                "min": np.array([np.min(filtered_data)]),
                "max": np.array([np.max(filtered_data)]),
                "mean": np.array([np.mean(filtered_data)]),
                "std": np.array([np.std(filtered_data)]),
                "count": np.array([len(filtered_data)])
            }
        elif data.ndim == 2:
            # Two dimensional data (e.g., vectors)
            # Compute stats for each column/feature
            num_features = data.shape[1]
            min_vals = []
            max_vals = []
            mean_vals = []
            std_vals = []
            
            for i in range(num_features):
                column_data = data[:, i]
                # Filter out non-numerical values
                numeric_mask = np.isfinite(column_data) if np.issubdtype(column_data.dtype, np.number) else np.ones(len(column_data), dtype=bool)
                filtered_column_data = column_data[numeric_mask] if np.issubdtype(column_data.dtype, np.number) else column_data.astype(float)
                
                if len(filtered_column_data) == 0:
                    min_vals.append(0.0)
                    max_vals.append(0.0)
                    mean_vals.append(0.0)
                    std_vals.append(0.0)
                    continue
                    
                min_vals.append(np.min(filtered_column_data))
                max_vals.append(np.max(filtered_column_data))
                mean_vals.append(np.mean(filtered_column_data))
                std_vals.append(np.std(filtered_column_data))
            
            final_stats[f_name] = {
                "min": np.array(min_vals),
                "max": np.array(max_vals),
                "mean": np.array(mean_vals),
                "std": np.array(std_vals),
                "count": np.array([len(data)])
            }
        else:
            # Higher dimensional data
            # Flatten and compute stats
            flattened_data = data.reshape(-1, data.shape[-1])
            num_features = flattened_data.shape[1]
            min_vals = []
            max_vals = []
            mean_vals = []
            std_vals = []
            
            for i in range(num_features):
                column_data = flattened_data[:, i]
                # Filter out non-numerical values
                numeric_mask = np.isfinite(column_data) if np.issubdtype(column_data.dtype, np.number) else np.ones(len(column_data), dtype=bool)
                filtered_column_data = column_data[numeric_mask] if np.issubdtype(column_data.dtype, np.number) else column_data.astype(float)
                
                if len(filtered_column_data) == 0:
                    min_vals.append(0.0)
                    max_vals.append(0.0)
                    mean_vals.append(0.0)
                    std_vals.append(0.0)
                    continue
                    
                min_vals.append(np.min(filtered_column_data))
                max_vals.append(np.max(filtered_column_data))
                mean_vals.append(np.mean(filtered_column_data))
                std_vals.append(np.std(filtered_column_data))
            
            final_stats[f_name] = {
                "min": np.array(min_vals),
                "max": np.array(max_vals),
                "mean": np.array(mean_vals),
                "std": np.array(std_vals),
                "count": np.array([len(flattened_data)])
            }

    # Merge visual stats
    for f_name, stats_val in visual_results.items():
        final_stats[f_name] = stats_val

    # Save stats
    write_stats(final_stats, output_dir)
    stats_path = output_dir / 'meta' / 'stats.json'
    print(f"\n✅ Stats saved to {stats_path}")
    
    # Print summary
    print("\n📊 Stats Summary:")
    for f_name in features:
        if f_name in final_stats:
            if f_name in visual_results:
                print(f"  {f_name}: Fixed visual stats used (min=0, max=1)")
            else:
                print(f"  {f_name}: Numerical stats computed")
        else:
            print(f"  ⚠️ {f_name}: No stats computed")
    
    return final_stats

def process_session(episode_data: EpisodeData, output_dir: Path, is_first_episode: bool = False, last_frames_to_chop: int = 10, first_frames_to_chop: int = 15, mode="diff"):
    """
    Main function to process one episode.
    
    last_frames_to_chop (int): The number of final frames to exclude from the dataset.
    first_frames_to_chop (int): The number of initial frames to exclude from the dataset.
    """
    # Create directories
    os.makedirs(output_dir / "meta", exist_ok=True)
    os.makedirs(output_dir / "data", exist_ok=True)
    os.makedirs(output_dir / "videos", exist_ok=True)
           
    with open(episode_data.joint_data_json_path, 'r') as f:
        json_data = json.load(f)

    # Pass the chop value to meta file generator
    generate_meta_files(output_dir, episode_data, json_data, is_first_episode, last_frames_to_chop, first_frames_to_chop)
    
    generate_video_files(output_dir, episode_data, json_data, last_frames_to_chop, first_frames_to_chop)
        

    # Pass the chop value to data file generator
    return generate_data_files(output_dir, episode_data, json_data, last_frames_to_chop, first_frames_to_chop, mode=mode)

def find_episode_folders(root_folder):
    """Find all episode folders with naming convention episode1, episode2, etc."""
    episode_folders = []
    pattern = re.compile(r'^episode(\d+)$', re.IGNORECASE)
    
    for item in root_folder.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                episode_folders.append((item, int(match.group(1))))
    
    # Sort by episode number
    episode_folders.sort(key=lambda x: x[1])
    return episode_folders

def find_json_and_videos(episode_folder):
    """Find JSON file and video files in the episode folder."""
    json_files = list(episode_folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {episode_folder}")
    if len(json_files) > 1:
        print(f"Warning: Multiple JSON files found in {episode_folder}, using {json_files[0]}")
    
    json_path = json_files[0]
    
    # Find video files (assuming common video extensions)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(episode_folder.glob(f"*{ext}"))
    
    return json_path, video_files

def get_camera_name_from_video_path(video_path):
    """Determine camera name based on video filename content."""
    filename = video_path.stem.lower()
    if 'depth' in filename:
        return 'depth'
    elif 'right' in filename:
        return 'right'
    elif 'gripper' in filename:
        return 'gripper'
    elif 'front' in filename:
        return 'front'
    else:
        # Fallback: use the last part of filename after underscore
        return video_path.stem.split('_')[-1]


def pad_video_to_target_frames(video_path: Path, target_frames: int, fps: float) -> bool:
    """
    Pad a video with duplicated frames to reach the target frame count.
    
    Args:
        video_path (Path): Path to the video file to pad
        target_frames (int): Target number of frames
        fps (float): Frames per second of the video
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # First, get the current number of frames in the video
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(video_path)
        ]
        
        import subprocess
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ WARNING: Error probing video {video_path}: {result.stderr}")
            return False
            
        current_frames = int(result.stdout.strip())
        frames_to_add = target_frames - current_frames
        
        if frames_to_add <= 0:
            # No padding needed
            return True
            
        print(f"  Padding video: current frames={current_frames}, target={target_frames}, adding {frames_to_add} frames")
        
        # Use ffmpeg to pad the video by duplicating the last frame
        padded_video_path = video_path.with_suffix('.padded.mp4')
        
        # Calculate duration to add based on frames and fps
        duration_to_add = frames_to_add / fps
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"tpad=stop_mode=clone:stop_duration={duration_to_add:.3f}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-y",
            str(padded_video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ WARNING: Error padding video {video_path}: {result.stderr}")
            return False
            
        # Replace the original video with the padded one
        import os
        os.replace(padded_video_path, video_path)
        print(f"  ✅ Successfully padded video to {target_frames} frames")
        return True
    except Exception as e:
        print(f"⚠️ WARNING: Exception while padding video {video_path}: {e}")
        return False


def combine_video_chunks_for_cameras(output_dir: Path, all_episodes_data: list, fps: float):
    """
    Combine all video chunks for each camera into one continuous video.
    
    Note: This function works with video chunks that have already been processed
    with frame chopping (first_frames_to_chop and last_frames_to_chop), so the
    combined video will only contain the frames that were kept after chopping.
    
    Args:
        output_dir (Path): Output directory for the dataset
        all_episodes_data (list): List of all EpisodeData objects
        fps (float): Frames per second for the combined videos
    """
    print("Combining video chunks for all cameras...")
    
    # Group video chunks by camera
    camera_chunks = {}
    
    # Collect all video chunks for each camera
    for episode in all_episodes_data:
        for camera_data in episode.cameras:
            camera_name = camera_data.camera
            # Source video path (these are already chopped videos)
            src_video_path = output_dir / "videos" / f"observation.images.{camera_name}" / f"chunk-{episode.episode_index:03d}" / f"file-{episode.episode_index:03d}.mp4"
            
            if camera_name not in camera_chunks:
                camera_chunks[camera_name] = []
            
            if src_video_path.exists():
                camera_chunks[camera_name].append({
                    'path': src_video_path,
                    'episode_index': episode.episode_index
                })
            else:
                print(f"⚠️ WARNING: Video chunk not found for camera {camera_name} in episode {episode.episode_index}")
                print(f"    Expected path: {src_video_path}")
    
    print(f"Found cameras: {list(camera_chunks.keys())}")
    
        
    # Combine chunks for each camera
    for camera_name, chunks in camera_chunks.items():
        if not chunks:
            raise FileNotFoundError(f"No video chunks found for camera {camera_name}")
            
        print(f"Combining {len(chunks)} video chunks for camera {camera_name}...")
        
        # Create directory for the combined video
        combined_video_dir = output_dir / "videos" / f"observation.images.{camera_name}" / "chunk-000-combined"
        combined_video_dir.mkdir(parents=True, exist_ok=True)
        combined_video_path = combined_video_dir / "file-000.mp4"
        
        # Check if we have any valid video chunks
        valid_chunks = [chunk for chunk in chunks if chunk['path'].exists()]
        if not valid_chunks:
            raise FileNotFoundError(f"No valid video chunks for camera {camera_name}")
            
        # Validate that all chunks are valid video files
        import subprocess
        for chunk in valid_chunks:
            # Check if the file is a valid video file using ffprobe
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1",
                str(chunk['path'])
            ]
            try:
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                if probe_result.returncode != 0:
                    print(f"⚠️ WARNING: Chunk {chunk['path']} is not a valid video file")
                    print(f"    ffprobe stderr: {probe_result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"⚠️ WARNING: Timeout checking validity of chunk {chunk['path']}")
            except Exception as e:
                print(f"⚠️ WARNING: Error checking validity of chunk {chunk['path']}: {e}")
        
        # Create a temporary file listing all video chunks with absolute paths
        import tempfile
        list_file_path = None
        try:
            # Create a fresh temporary file for each camera
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                list_file_path = f.name
                for chunk in valid_chunks:
                    # Use absolute path and escape special characters
                    abs_path = chunk['path'].resolve()
                    f.write(f"file '{abs_path}'\n")
            
            
            # Use ffmpeg to concatenate videos
            # Add -fflags +genpts to regenerate timestamps which can help with concatenation issues
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file_path,
                "-c", "copy",
                "-fflags", "+genpts",
                "-y",
                str(combined_video_path)
            ]
            
            print(f"Debug: Running ffmpeg command for camera {camera_name}: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ ERROR: Failed to combine video chunks for camera {camera_name}: {result.stderr}")
                # Also print stdout for additional debugging
                if result.stdout:
                    print(f"FFmpeg stdout: {result.stdout}")
                continue
                
            print(f"✅ Successfully combined video chunks for camera {camera_name}")
            
            # Verify the output file was created and has reasonable size
            if combined_video_path.exists():
                file_size = combined_video_path.stat().st_size
                print(f"✅ Verified output file for camera {camera_name}: {combined_video_path} ({file_size} bytes)")
                if file_size < 1000:  # Less than 1KB, likely empty or corrupt
                    print(f"⚠️ WARNING: Output file for camera {camera_name} is very small ({file_size} bytes), may be corrupt")
            else:
                print(f"❌ ERROR: Output file not found for camera {camera_name}: {combined_video_path}")
                continue
            
            # Remove individual video chunks after successful combination
            print(f"🗑️ Removing individual video chunks for camera {camera_name}...")
            removed_count = 0
            for chunk in valid_chunks:
                try:
                    if chunk['path'].exists():
                        os.remove(chunk['path'])
                        removed_count += 1
                        # Also remove the parent directory if it's empty
                        parent_dir = chunk['path'].parent
                        try:
                            parent_dir.rmdir()  # This will only remove if directory is empty
                        except OSError:
                            # Directory not empty or other error, which is fine
                            pass
                except Exception as e:
                    print(f"⚠️ WARNING: Failed to remove chunk {chunk['path']}: {e}")
            
            print(f"🗑️ Removed {removed_count} individual video chunks for camera {camera_name}")
            
            # Rename the combined video directory to remove the -combined suffix
            final_video_dir = output_dir / "videos" / f"observation.images.{camera_name}" / "chunk-000"
            if combined_video_dir.exists():
                # Remove the final directory if it already exists
                if final_video_dir.exists():
                    shutil.rmtree(final_video_dir)
                # Rename the combined directory to the final name
                combined_video_dir.rename(final_video_dir)
                print(f"✅ Renamed {combined_video_dir} to {final_video_dir}")
            
        except Exception as e:
            print(f"❌ ERROR: Exception occurred while combining video chunks for camera {camera_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up the temporary file
            if list_file_path and os.path.exists(list_file_path):
                try:
                    os.unlink(list_file_path)
                except Exception as e:
                    print(f"⚠️ WARNING: Failed to clean up temporary file {list_file_path}: {e}")


def update_metadata_for_aggregated_episode(output_dir: Path, all_episodes_data: list):
    """
    Update metadata files to reflect the single aggregated episode.
    
    Args:
        output_dir (Path): Output directory for the dataset
        all_episodes_data (list): List of all EpisodeData objects
    """
    print("Updating metadata for aggregated episode...")
    
    # Calculate total frames across all episodes
    total_frames = sum(episode.num_of_frames for episode in all_episodes_data)
    
    # Update info.json
    info_json_path = output_dir / "meta" / "info.json"
    if info_json_path.exists():
        with open(info_json_path, "r") as f:
            info_json = json.load(f)
        
        # Update totals for single aggregated episode
        info_json["total_episodes"] = len(all_episodes_data)
        info_json["total_frames"] = total_frames
        
        with open(info_json_path, "w") as f:
            json.dump(info_json, f, indent=2)
        print("✅ Updated info.json for aggregated episode")

def pad_videos(output_dir: Path):
    """
    Pad all videos in the dataset to reach the target frame count.
    
    Args:
        output_dir (Path): Output directory for the dataset
        target_frames (int): Target number of frames for each video 
        fps (float): Frames per second of the videos
    """    
    # Pad all videos to match the target frame count from info.json
    videos_dir = output_dir / "videos"
    
    # Check if videos directory exists
    if not videos_dir.exists():
        print(f"⚠️ Videos directory not found: {videos_dir}")
        return
    
    info_json_path = output_dir / "meta" / "info.json"
    if info_json_path.exists():
        try:
            with open(info_json_path, 'r') as f:
                info_data = json.load(f)
            target_frames = info_data.get("total_frames", 0)
            fps = info_data.get("fps", 10)
            video_fps = info_data.get("fps", fps)
            
            if target_frames > 0:
                # Iterate through all camera directories
                for camera_dir in videos_dir.iterdir():
                    if camera_dir.is_dir() and camera_dir.name.startswith("observation.images."):
                        print(f"Processing camera: {camera_dir.name}")
                        video_paths = []
                        
                        # Check all chunk directories for this camera
                        for chunk_dir in camera_dir.glob("chunk-*"):
                            for video_file in chunk_dir.glob("*.mp4"):
                                video_paths.append(video_file)
                        
                        # Pad all videos for this camera
                        for p in video_paths:
                            print(f"  Padding video to {target_frames} frames: {p}")
                            if pad_video_to_target_frames(p, target_frames, video_fps):
                                print(f"  ✅ Video padding completed for {p}")
                            else:
                                print(f"  ❌ Failed to pad video for {p}")
            else:
                print("⚠️ Target frame count is 0 or invalid. Skipping video padding.")
        except Exception as e:
            print(f"  ⚠️ Error reading info.json or padding videos: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Info.json not found: {info_json_path}. Cannot determine target frame count.")
    
    
def main():
    print('Starting dataset preparation...')
    # --- CONFIGURATION ---
    ROOT_FOLDER = Path("data/piper_training_data/")  # Root folder containing episode subfolders
    OUTPUT_FOLDER = Path("output/")  # Output folder for processed dataset
    REPO_ID = "ISDept/piper_arm"  # Your desired Hugging Face repo ID
    AGGREGATE_EPISODES = True  # New flag to control aggregation behavior
    MODE = "full"  # Processing mode: "diff" or "full"
    # ---------------------
    
    # Find all episode folders
    episode_folders = find_episode_folders(ROOT_FOLDER)
    
    if not episode_folders:
        print(f"No episode folders found in {ROOT_FOLDER}")
        return
    
    print(f"Found {len(episode_folders)} episode folders")
    
    # First, collect all episode data without processing to compute global statistics
    print("Collecting episode data for processing...")
    all_episodes_data = []
    
    # Store episode-specific parameters
    episode_params = {}
    
    global_index_offset = 0
    
    last_frames_to_chop = 5  # Default value
    for episode_folder, episode_idx in episode_folders:    
        
        # Store parameters for this episode
        episode_params[episode_idx] = {
            'last_frames_to_chop': last_frames_to_chop
        }
        
        try:
            # Create episode data object (without processing yet)
            json_path, video_files = find_json_and_videos(episode_folder)
            
            # Create CameraData objects from video files
            cameras_list = []
            for video_path in video_files:
                # Extract camera name from filename
                camera_name = get_camera_name_from_video_path(video_path)
                # Set first_frames_to_chop based on camera type
                first_frames_to_chop = 10  # Default value
                if 'gripper' in camera_name.lower():
                    if episode_idx == 1:
                        first_frames_to_chop = 12  # First episode has extra lag
                    else: 
                        first_frames_to_chop = 10  # Gripper camera lags by 0.1 second (1 frame at 10fps)
                cameras_list.append(CameraData(video_path=str(video_path), camera=camera_name, first_frames_to_chop=first_frames_to_chop))
            
            episode_data = EpisodeData(
                joint_data_json_path=str(json_path), 
                episode_index=episode_idx, 
                fps=10, 
                global_index_offset=0,  # Will be updated during processing
                cameras=cameras_list,
                folder = episode_folder,
                task_description = "Pick up the cube and place it into the container.",
                last_frames_to_chop = last_frames_to_chop
            )
            
            all_episodes_data.append(episode_data)
            
        except Exception as e:
            print(f"Error collecting data from episode {episode_idx}: {e}")
            traceback.print_exc()
            continue
    
    # Now process all episodes
    print("Processing episodes...")
    global_index_offset = 0
    
    # Sort episodes by index to ensure proper ordering
    all_episodes_data.sort(key=lambda x: x.episode_index)
    
    for episode in all_episodes_data:
        episode_folder = episode.folder
        episode_idx = episode.episode_index
        last_frames_to_chop = episode.last_frames_to_chop
        
        try:
            # Update global index offset before processing
            episode.global_index_offset = global_index_offset
            
            # Process the first episode differently to create initial files
            is_first_episode = (episode_idx == min(e.episode_index for e in all_episodes_data))
            num_of_frames = process_session(episode, OUTPUT_FOLDER, is_first_episode, last_frames_to_chop, first_frames_to_chop=10, mode=MODE)
            episode.num_of_frames = num_of_frames
            
            # Update global index offset for the next episode
            global_index_offset += episode.num_of_frames
            
        except Exception as e:
            print(f"Error processing episode {episode_idx}: {e}")
            traceback.print_exc()
            continue
    
    # Handle aggregation if requested
    if AGGREGATE_EPISODES and all_episodes_data:
        print("Aggregating all episodes into one giant episode...")
        
        # Combine all video chunks for each camera
        fps = all_episodes_data[0].fps if all_episodes_data else 10
        combine_video_chunks_for_cameras(OUTPUT_FOLDER, all_episodes_data, fps)
                
        # Update metadata for the single aggregated episode
        update_metadata_for_aggregated_episode(OUTPUT_FOLDER, all_episodes_data)
        
        # Update total frames in info.json
        update_total_frames_from_episodes(OUTPUT_FOLDER)
        
        create_episodes_parquet(OUTPUT_FOLDER)
        
        pad_videos(OUTPUT_FOLDER)
        
        # Compute and save dataset statistics for the aggregated episode
        compute_and_save_dataset_stats(OUTPUT_FOLDER)
        
        print("Dataset preparation with episode aggregation completed successfully!")
    

if __name__ == "__main__":
    main()
