import json
import os
import shutil
import pandas as pd
from datasets import Dataset, Features, Value, Sequence
from pathlib import Path
# Added imports for Parquet generation (required for tasks.parquet)
import pyarrow as pa
import pyarrow.parquet as pq

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

def create_episodes_parquet_index(root_dir: Path):
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
    data_subdir = episodes_parquet_dir / "data"
    data_subdir.mkdir(exist_ok=True, parents=True)
    
    # Write multiple Parquet files (LeRobot expects this structure)
    pq.write_table(table, data_subdir / "0000.parquet")
    
    print(f"✅ Successfully created episodes index dataset at: {episodes_parquet_dir}")

def process_session(json_path: Path, root_dir: Path, episode_index: int):
    """Converts a single session (JSON + MP4) into LeRobot format."""
    
    # Ensure json_path is a Path object
    json_path = Path(json_path)
    
    with open(json_path, 'r') as f:
        session_data = json.load(f)

    input_video_path = Path(session_data.get("video_file", "path/to/placeholder.mp4"))
    # Create the new directory structure for videos
    video_chunk_dir = root_dir / "videos" / "observation.image" / f"chunk-{episode_index}"
    video_chunk_dir.mkdir(parents=True, exist_ok=True)
    output_video_name = f"episode_{episode_index}_front_camera.mp4"
    
    num_joints = len(session_data["joint_names"])
    num_frames = len(session_data["frames"])
    joint_positions = [frame["joint_positions"] for frame in session_data["frames"]]
    
    # Create frames with both state and image observations
    lerobot_frames = []
    for i in range(num_frames):
        lerobot_frames.append({
            "observation.state": joint_positions[i],
            "observation.image": f"videos/observation.image/chunk-{episode_index}/episode_{episode_index}_front_camera.mp4",  # Video reference
            "action": joint_positions[i + 1] if i < num_frames - 1 else joint_positions[i],
            "timestamp": session_data["frames"][i]["timestamp"],
            "episode_index": episode_index,
            "frame_index": session_data["frames"][i]["frame_index"],
            "index": episode_index * num_frames + i,
            "next.done": i == num_frames - 1,
            "next.reward": 0.0,
            "task_index": 0,
        })
        
    # Create dataset with proper features including image
    hf_dataset = Dataset.from_pandas(pd.DataFrame(lerobot_frames))
    
    feature_config = Features({
        "observation.state": Sequence(Value("float32"), length=num_joints),
        "observation.image": Value("string"),  # Path to video file
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

    # Create directories
    os.makedirs(root_dir / "meta", exist_ok=True)
    os.makedirs(root_dir / "data", exist_ok=True)
    os.makedirs(root_dir / "videos", exist_ok=True)

    # Export to Parquet
    parquet_path = f"{root_dir}/data/episode_{episode_index}.parquet"
    hf_dataset.to_parquet(parquet_path)
    
    # Copy video file to the new directory structure
    if input_video_path.exists():
        shutil.copy(input_video_path, video_chunk_dir / output_video_name)
    else:
        print(f"⚠️ WARNING: Video file not found at {input_video_path}. Skipping video copy.")

    # Update info.json to properly configure image observations
    duration_s = session_data["frames"][-1]["timestamp"] - session_data["frames"][0]["timestamp"]
    estimated_fps = num_frames / duration_s if duration_s > 0 else 30.0
    
    # Get data path and video path
    data_path = f"{root_dir}/data/episode_{episode_index}.parquet"
    video_path = f"videos/observation.image/chunk-{episode_index}/episode_{episode_index}_front_camera.mp4"
    
    info_json = {
        "codebase_version": "v3.0", 
        "fps": round(estimated_fps, 2),
        "num_episodes": 1,
        "num_frames": num_frames,
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
            "observation.image": {
                "shape": [480, 640, 3],  # Adjust based on your actual video dimensions
                "dtype": "uint8",
                "path_template": "videos/observation.image/chunk-{episode_index}/episode_{episode_index}_front_camera.mp4",
                "fps": round(estimated_fps, 2)
            },
            "action": {
                "shape": [num_joints],
                "dtype": "float32"
            }
        }
    }
    
    with open(root_dir / "meta" / "info.json", "w") as f:
        json.dump(info_json, f, indent=2)
        
    # Append to episodes.jsonl
    episodes_jsonl = {
        "episode_index": episode_index,
        "task_index": 0,
        "frame_index_offset": 0,
        "num_frames": num_frames,
        "start_time": session_data["start_time"],
        "end_time": session_data["end_time"],
    }
    
    with open(root_dir / "meta" / "episodes.jsonl", "a") as f:
        f.write(json.dumps(episodes_jsonl) + "\n")
        
    print(f"✅ Successfully processed episode {episode_index}")
