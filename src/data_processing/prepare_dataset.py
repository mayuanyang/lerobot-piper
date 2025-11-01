import json
import os
import shutil
import pandas as pd
from datasets import Dataset, Features, Value, Sequence
from pathlib import Path
# Added imports for Parquet generation (required for tasks.parquet)
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from .episode_data import EpisodeData



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

def generate_data_files(output_dir: Path, episode_data: EpisodeData, json_data: dict):
    print("\n--- Generating data files ---")
    num_joints = len(json_data["joint_names"])
    num_frames = len(json_data["frames"])
    joint_positions = [frame["joint_positions"] for frame in json_data["frames"]]
    
    
    
    # Create frames with state observations (image references are handled separately)
    lerobot_frames = []
    timestamp_base = 0.0
    for i in range(num_frames):
        lerobot_frames.append({
            "observation.state": joint_positions[i],
            "action": joint_positions[i + 1] if i < num_frames - 1 else joint_positions[i],
            "timestamp": timestamp_base,
            "episode_index": episode_data.episode_index,
            "frame_index": json_data["frames"][i]["frame_index"],
            "index": episode_data.episode_index * num_frames + i,
            "next.done": i == num_frames - 1,
            "next.reward": 0.0,
            "task_index": 0,
        })
        timestamp_base += 0.1
        
    # Create dataset with proper features including image
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

def generate_meta_files(output_dir: Path, task_title: str, episode_data: EpisodeData, json_data: dict):
    print("\n--- Generating meta ---")
    # Get data path and video path
    data_path = "data/episode-{episode_index:03d}/file-{episode_index:03d}.parquet"
    video_path = "videos/{video_key}/chunk-{chunk_index:03d}/episode_{chunk_index:03d}.mp4"
    
    num_joints = len(json_data["joint_names"])
    num_frames = len(json_data["frames"])
    joint_positions = [frame["joint_positions"] for frame in json_data["frames"]]
    
    # Create base info_json structure
    info_json = {
        "codebase_version": "v3.0", 
        "fps": round(episode_data.fps, 2),
        "total_episodes": 1,
        "total_frames": num_frames,
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
            "shape": [480, 640, 3],  # Adjust based on your actual video dimensions
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
                "video.is_depth_map": False,
                "has_audio": False
            }
        }
    
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info_json, f, indent=2)
        
    # Define global index bounds
    dataset_from_index = episode_data.global_index_offset
    dataset_to_index = episode_data.global_index_offset + num_frames
    
    # Create base episodes_jsonl structure
    episodes_jsonl = {
        "episode_index": episode_data.episode_index,
        "task_index": 0,
        "frame_index_offset": 0,
        "num_frames": num_frames,
        "dataset_from_index": dataset_from_index,
        "dataset_to_index": dataset_to_index,
        "start_time": json_data["start_time"],
        "end_time": json_data["end_time"],
        # Add video metadata for each camera
    }
    
    # Add video metadata for each camera
    for camera_data in episode_data.cameras:
        camera_name = camera_data.camera
        episodes_jsonl[f"videos/observation.images.{camera_name}/from_timestamp"] = 0.0
        episodes_jsonl[f"videos/observation.images.{camera_name}/chunk_index"] = episode_data.episode_index
        episodes_jsonl[f"videos/observation.images.{camera_name}/file_index"] = episode_data.episode_index
        episodes_jsonl[f"videos/observation.images.{camera_name}/frame_index_offset"] = 0
        
    
    with open(output_dir / "meta" / "episodes.jsonl", "a") as f:
        f.write(json.dumps(episodes_jsonl) + "\n")
        
    create_tasks_parquet(output_dir, task_title)
    create_episodes_parquet_index(output_dir, episode_data.episode_index)


def generate_video_files(output_dir: Path, episode_data: EpisodeData, json_data: dict):
    print("\n--- Generating videos ---")
    input_videos_path = json_data.get("video_files", ["path/to/placeholder.mp4"])
    
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
            shutil.copy(video_file_path, video_chunk_dir / output_video_name)
        else:
            print(f"⚠️ WARNING: Video file not found at {camera_data.video_path}. Skipping video copy.")

def process_session(episode_data: EpisodeData, output_dir: Path):
    # Create directories
    os.makedirs(output_dir / "meta", exist_ok=True)
    os.makedirs(output_dir / "data", exist_ok=True)
    os.makedirs(output_dir / "videos", exist_ok=True)
           
    with open(episode_data.joint_data_json_path, 'r') as f:
        json_data = json.load(f)

    generate_video_files(output_dir, episode_data, json_data)
        
    generate_meta_files(output_dir, "Piper Arm Teleoperation", episode_data, json_data)   
   
    generate_data_files(output_dir, episode_data, json_data)
        
    print(f"✅ Successfully processed episode {episode_data.episode_index}")
