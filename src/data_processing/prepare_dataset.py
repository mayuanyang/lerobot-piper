import json
import os
import shutil
import pandas as pd
from datasets import Dataset, Features, Value, Sequence
from pathlib import Path
import glob
# Added imports for Parquet generation (required for tasks.parquet)
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
from datetime import datetime
from .episode_data import EpisodeData
import numpy as np
from lerobot.datasets.compute_stats import compute_episode_stats, aggregate_stats
from lerobot.datasets.utils import load_info, write_stats



def create_tasks_parquet(root_dir: Path, task_title: str):
    """
    Generates the required meta/tasks.parquet file for LeRobot.
    This file defines the available tasks in the dataset (which is mandatory).
    """
    print("--- Creating meta/tasks.parquet ---")
    
    # The task index (0) must match the 'task_index' used in episodes.jsonl
    task_data = {
        'task_index': [0],
        'task_title': [task_title],
        'description': [f"Teleoperation dataset for the {task_title} task."]
    }
    
    # Convert to Arrow Table and write Parquet file
    df = pd.DataFrame(task_data)
    table = pa.Table.from_pandas(df)

    tasks_dir = root_dir / "meta"
    tasks_dir.mkdir(exist_ok=True, parents=True)
    tasks_parquet_path = tasks_dir / "tasks.parquet"

    pq.write_table(table, tasks_parquet_path)
    print(f"✅ Successfully created tasks.parquet at: {tasks_parquet_path}")

def create_episodes_parquet_index(root_dir: Path, episode_index: int):
    """
    Reads the data from episodes.jsonl and saves it as nested Parquet files
    in the format LeRobot expects.
    """
    episodes_jsonl_path = root_dir / "meta" / "episodes.jsonl"
    episodes_parquet_dir = root_dir / "meta" / "episodes"
    
    if not episodes_jsonl_path.exists():
        print(f"❌ WARNING: {episodes_jsonl_path} not found. Skipping episodes index creation.")
        return

    print("\n--- Creating meta/episodes/ index dataset ---")
    
    # 1. Read the JSONL file line by line
    with open(episodes_jsonl_path, 'r') as f:
        episode_lines = [json.loads(line) for line in f]
    
    if not episode_lines:
        print("❌ WARNING: episodes.jsonl is empty. Skipping episodes index creation.")
        return

    # 2. Create DataFrame and convert to Arrow Table
    df = pd.DataFrame(episode_lines)
    table = pa.Table.from_pandas(df)

    # 3. Create the nested directory structure LeRobot expects
    # This creates a subdirectory with multiple Parquet files
    data_subdir = episodes_parquet_dir / f"episode-{episode_index:03d}"
    data_subdir.mkdir(exist_ok=True, parents=True)
    
    # Write multiple Parquet files (LeRobot expects this structure)
    pq.write_table(table, data_subdir / f"file-{episode_index:03d}.parquet")
    
    print(f"✅ Successfully created episodes index dataset at: {episodes_parquet_dir}")

def generate_data_files(output_dir: Path, episode_data: EpisodeData, json_data: dict, last_frames_to_chop: int):
    print("\n--- Generating data files (chopping last frames) ---")
    
    num_joints = len(json_data["joint_names"])
    original_num_frames = len(json_data["frames"])
    effective_num_frames = original_num_frames - last_frames_to_chop
    
    if effective_num_frames <= 0:
        print(f"❌ ERROR: Chopping {last_frames_to_chop} frames from {original_num_frames} results in 0 or fewer frames. Skipping data generation.")
        return 0

    joint_positions = [frame["joint_positions"] for frame in json_data["frames"]]
    
    lerobot_frames = []
    timestamp_base = 0.0
    
    for i in range(effective_num_frames):
        # Determine the action for the current frame
        if i < effective_num_frames - 1:
             action = joint_positions[i + 1]
        else:
             # For the last frame, use the current position (or zero/no-op action)
             action = joint_positions[i] 

        is_done = (i == effective_num_frames - 1)
        
        # 🟢 CORRECTION: Correct calculation of the global 'index'
        # The global index is the offset + the current frame's index (i)
        global_index = episode_data.global_index_offset + i
        
        lerobot_frames.append({
            "observation.state": joint_positions[i],
            "action": action,
            "timestamp": timestamp_base,
            "episode_index": episode_data.episode_index,
            "frame_index": json_data["frames"][i]["frame_index"], # Local index (0, 1, 2, ...)
            "index": global_index, # Global index (0, 1, 2, ..., N)
            "next.done": is_done,
            "next.reward": 1.0 if is_done else 0.0,
            "task_index": 0,
        })
        timestamp_base += 0.1
        
    # [Rest of the function remains the same, writing to Parquet]
    hf_dataset = Dataset.from_pandas(pd.DataFrame(lerobot_frames))
    
    feature_config = Features({
        "observation.state": Sequence(Value("float32"), length=num_joints),
        "action": Sequence(Value("float32"), length=num_joints),
        "timestamp": Value("float64"),
        "episode_index": Value("int64"),
        "frame_index": Value("int64"),
        "index": Value("int64"),
        "next.done": Value("bool"),
        "next.reward": Value("float32"),
        "task_index": Value("int64"),
    })
    hf_dataset = hf_dataset.cast(feature_config)


    # Export to Parquet with new directory structure
    chunk_dir = output_dir / "data" / f"chunk-{episode_data.episode_index:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = chunk_dir / f"file-{episode_data.episode_index:03d}.parquet"
    hf_dataset.to_parquet(parquet_path)
    
    return effective_num_frames

# Modified signature to accept last_frames_to_chop
def generate_meta_files(output_dir: Path, task_title: str, episode_data: EpisodeData, json_data: dict, is_first_episode: bool = False, last_frames_to_chop: int = 0):
    print("\n--- Generating meta ---")
    
    # [File path definitions and checks remain the same...]
    data_path = "data/chunk-{episode_index:03d}/file-{episode_index:03d}.parquet"
    video_path = "videos/{video_key}/chunk-{chunk_index:03d}/episode_{chunk_index:03d}.mp4"
    info_json_path = output_dir / "meta" / "info.json"
    
    num_joints = len(json_data["joint_names"])
    
    # 🔴 CORE CHANGE 1: Use the effective number of frames
    original_num_frames = len(json_data["frames"])
    effective_num_frames = original_num_frames - last_frames_to_chop
    
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
                }
            }
        }
        
        # Add camera features dynamically
        for camera_data in episode_data.cameras:
            camera_name = camera_data.camera
            feature_key = f"observation.images.{camera_name}"
            
            info_json["features"][feature_key] = {
                "shape": [400, 640, 3],  # Adjust based on your actual video dimensions
                "dtype": "video",
                "names": [
                    "height",
                    "width",
                    "channel"
                ],
                "video_info": {
                    "video.fps": round(episode_data.fps, 2),
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
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
        "start_time": json_data["start_time"],
        "end_time": json_data["end_time"],
        # [Video metadata for each camera remains the same...]
    }
    
    # [Rest of episodes.jsonl writing and parquet index creation remains the same...]
    for camera_data in episode_data.cameras:
        camera_name = camera_data.camera
        episodes_jsonl[f"videos/observation.images.{camera_name}/from_timestamp"] = 0.0
        episodes_jsonl[f"videos/observation.images.{camera_name}/chunk_index"] = episode_data.episode_index
        episodes_jsonl[f"videos/observation.images.{camera_name}/file_index"] = episode_data.episode_index
        episodes_jsonl[f"videos/observation.images.{camera_name}/frame_index_offset"] = 0
        episodes_jsonl[f"data/chunk_index"] = episode_data.episode_index
        episodes_jsonl[f"data/file_index"] = episode_data.episode_index
        
    
    with open(output_dir / "meta" / "episodes.jsonl", "a") as f:
        f.write(json.dumps(episodes_jsonl) + "\n")
        
    if is_first_episode:
        create_tasks_parquet(output_dir, task_title)
    create_episodes_parquet_index(output_dir, episode_data.episode_index)


def generate_video_files(output_dir: Path, episode_data: EpisodeData, json_data: dict):
        
    for camera_data in episode_data.cameras:
        camera_name = camera_data.camera
        cam_folder = f"observation.images.{camera_name}"
        #os.makedirs(output_dir / cam_folder, exist_ok=True)
    
        # Create the new directory structure for videos
        video_chunk_dir = output_dir / "videos" / cam_folder / f"chunk-{episode_data.episode_index:03d}"
        video_chunk_dir.mkdir(parents=True, exist_ok=True)
        output_video_name = f"episode_{episode_data.episode_index:03d}.mp4"
    
        # Copy video file to the new directory structure
        video_file_path = Path(camera_data.video_path)
        if video_file_path.exists():
          if not 'depth' in camera_name.lower():
            shutil.copy(video_file_path, video_chunk_dir / output_video_name)
        else:
            print(f"⚠️ WARNING: Video file not found at {camera_data.video_path}. Skipping video copy.")
    
    # Generate video from images from depth images
    video_chunk_dir = output_dir / "videos" / "observation.images.depth" / f"chunk-{episode_data.episode_index:03d}"
    video_chunk_dir.mkdir(parents=True, exist_ok=True)
    output_video_name = f"episode_{episode_data.episode_index:03d}.mp4"
    
    # Get the only subfolder in episode_data.folder
    folder_path = Path(episode_data.folder)
    subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
    
    if len(subfolders) == 1:
        image_folder = subfolders[0]
        create_video_from_images(image_folder, video_chunk_dir / output_video_name, fps=episode_data.fps)
    elif len(subfolders) == 0:
        print(f"⚠️ WARNING: No subfolders found in {folder_path}. Skipping video creation.")
    else:
        print(f"⚠️ WARNING: Multiple subfolders found in {folder_path}. Expected exactly one. Skipping video creation.")


def create_video_from_images(image_folder, output_video_path, fps=30):
    """
    Creates a video from a sequence of PNG images in a folder.
    
    Parameters:
    image_folder (str): Path to the folder containing PNG images
    output_video_path (str): Path where the output video will be saved (e.g., 'output_video.mp4')
    fps (int): Frames per second for the output video (default: 30)
    """
    
    # Get all PNG files in the folder and sort them
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    if not image_files:
        print("No PNG images found in the specified folder.")
        return
    
    print(f"Found {len(image_files)} PNG images")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Could not read the first image: {image_files[0]}")
        return
    
    height, width, layers = first_image.shape
    size = (width, height)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    
    # Process each image
    for i, image_path in enumerate(image_files):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        # Write the frame to the video
        out.write(img)
        
        # Print progress every 100 frames
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    # Release everything when done
    out.release()
    print(f"Video successfully created: {output_video_path}")
    

def update_total_frames_from_episodes(output_dir: Path):
    """
    Reads episodes.jsonl file and calculates the total sum of num_frames,
    then updates info.json with the calculated sum.
    """
    print("\n--- Updating total frames from episodes ---")
    
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
        
        print(f"Calculated total frames: {total_frames}")
        
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
            
        print(f"✅ Successfully updated total_frames in info.json to: {total_frames}")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to update total frames: {e}")


def compute_and_save_dataset_stats(output_dir: Path):
    """
    Compute dataset statistics for all episodes and save them to stats.json.
    """
    print("\n--- Computing and saving dataset statistics ---")
    
    # Load dataset info
    try:
        info = load_info(output_dir)
        features = info["features"]
        print("✅ Successfully loaded dataset info.", features)
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset info: {e}")
        return
    
    # Load episodes data
    episodes_jsonl_path = output_dir / "meta" / "episodes.jsonl"
    if not episodes_jsonl_path.exists():
        print(f"❌ WARNING: {episodes_jsonl_path} not found. Skipping stats computation.")
        return
    
    # Read all episodes
    episodes_data = []
    try:
        with open(episodes_jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    episodes_data.append(json.loads(line))
    except Exception as e:
        print(f"❌ ERROR: Failed to read episodes data: {e}")
        return
    
    if not episodes_data:
        print("❌ WARNING: No episodes data found. Skipping stats computation.")
        return
    
    # Collect statistics for all episodes
    all_episode_stats = []
    
    for episode_info in episodes_data:
        episode_index = episode_info["episode_index"]
        
        # Load the parquet data for this episode
        chunk_index = episode_info.get("data/chunk_index", episode_index)
        file_index = episode_info.get("data/file_index", episode_index)
        parquet_path = output_dir / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        
        if not parquet_path.exists():
            print(f"⚠️ WARNING: Parquet file not found for episode {episode_index}: {parquet_path}")
            continue
            
        try:
            # Load parquet data
            episode_dataset = pd.read_parquet(parquet_path)
            
            # Convert to the format expected by compute_episode_stats
            episode_dict = {}
            for column in episode_dataset.columns:
                if column in features:
                    if features[column]["dtype"] in ["image", "video"]:
                        # For image/video features, collect the paths to the video files
                        # Extract camera name from the column name (e.g., "observation.images.rgb" -> "rgb")
                        camera_name = column.split(".")[-1]
                        
                        # Construct the video file path based on the episodes.jsonl info
                        chunk_index = episode_info.get(f"videos/observation.images.{camera_name}/chunk_index", episode_index)
                        video_path = output_dir / "videos" / f"observation.images.{camera_name}" / f"chunk-{chunk_index:03d}" / f"episode_{chunk_index:03d}.mp4"
                        
                        # Create a list of video paths for all frames in this episode
                        num_frames = episode_info["num_frames"]
                        video_paths = [str(video_path)] * num_frames
                        
                        episode_dict[column] = video_paths
                    else:
                        # Extract values and ensure they're in the right format
                        column_data = episode_dataset[column].values
                        
                        # Handle fixed-size list data (common in parquet files)
                        if len(column_data) > 0:
                            # Check if we have array-like data that needs special handling
                            first_element = column_data[0]
                            # Use a very safe check to avoid boolean ambiguity errors
                            is_array_like = False
                            try:
                                # Only check for __len__ if it's safe to do so
                                if not isinstance(first_element, (str, bytes)):
                                    is_array_like = hasattr(first_element, '__len__')
                            except:
                                # If any check fails, assume it's not array-like
                                is_array_like = False
                            
                            if is_array_like:
                                # For fixed-size lists, convert to proper 2D numpy array
                                try:
                                    # Convert to list first to handle pandas arrays properly
                                    if hasattr(column_data, 'tolist'):
                                        temp_list = column_data.tolist()
                                    else:
                                        temp_list = list(column_data)
                                    
                                    # Convert directly to numpy array with float32 dtype
                                    # This should handle most cases correctly
                                    column_data = np.array(temp_list, dtype=np.float32)
                                except Exception as e:
                                    print(f"Warning: Could not convert {column} data to 2D array: {e}")
                                    # Fallback to object array to prevent issues
                                    try:
                                        column_data = np.array(temp_list, dtype=object)
                                    except:
                                        # Last resort: keep original data
                                        pass
                            else:
                                # For scalar data, ensure it's a proper numpy array
                                if not isinstance(column_data, np.ndarray):
                                    try:
                                        column_data = np.array(column_data)
                                    except:
                                        pass  # Keep as-is
                        
                        episode_dict[column] = column_data
            
            # Compute stats for this episode
            try:
                episode_stats = compute_episode_stats(episode_dict, features)
                all_episode_stats.append(episode_stats)
                print(f"✅ Computed stats for episode {episode_index}")
            except Exception as e:
                # Check if this is the specific error we're trying to fix
                error_msg = str(e)
                if "truth value of an array with more than one element is ambiguous" in error_msg:
                    print(f"❌ ERROR: Ambiguous array truth value error for episode {episode_index}")
                    print(f"Column data types for episode {episode_index}:")
                    for col, data in episode_dict.items():
                        print(f"  {col}: type={type(data)}, shape={getattr(data, 'shape', 'N/A')}")
                        if hasattr(data, '__len__') and len(data) > 0:
                            print(f"    first element: {type(data[0])}")
                            if hasattr(data[0], '__len__'):
                                print(f"    first element shape: {getattr(data[0], 'shape', 'N/A')}")
                raise  # Re-raise the exception after logging
            
        except Exception as e:
            print(f"❌ ERROR: Failed to compute stats for episode {episode_index}: {e}")
            continue
    
    if not all_episode_stats:
        print("❌ ERROR: No episode statistics computed.")
        return
    
    # Aggregate statistics across all episodes
    try:
        aggregated_stats = aggregate_stats(all_episode_stats)
        
        # Save statistics to stats.json
        write_stats(aggregated_stats, output_dir)
        print(f"✅ Successfully computed and saved dataset statistics to: {output_dir / 'meta' / 'stats.json'}")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to aggregate or save statistics: {e}")


def process_session(episode_data: EpisodeData, output_dir: Path, is_first_episode: bool = False, last_frames_to_chop: int = 10):
    """
    Main function to process one episode.
    
    last_frames_to_chop (int): The number of final frames to exclude from the dataset.
    """
    # Create directories
    os.makedirs(output_dir / "meta", exist_ok=True)
    os.makedirs(output_dir / "data", exist_ok=True)
    os.makedirs(output_dir / "videos", exist_ok=True)
           
    with open(episode_data.joint_data_json_path, 'r') as f:
        json_data = json.load(f)

    # Note: We assume generate_video_files either handles chopping internally 
    # (if source is a video) or is called BEFORE chopping the images/joints.
    # For now, we assume video files contain all frames and only the tabular data is chopped.
    generate_video_files(output_dir, episode_data, json_data)
        
    # Pass the chop value to meta file generator
    generate_meta_files(output_dir, "Piper Arm Teleoperation", episode_data, json_data, is_first_episode, last_frames_to_chop)   
   
    # Pass the chop value to data file generator
    return generate_data_files(output_dir, episode_data, json_data, last_frames_to_chop)
