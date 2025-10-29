import pandas as pd
from pathlib import Path
import json


def prepare_dataset(arg_json_path, arg_output_root):
  # === Step 1: Load JSON File ===
  json_path = Path(arg_json_path)

  if not json_path.exists():
      raise FileNotFoundError(f"JSON file not found: {arg_json_path}")

  with open(json_path, 'r') as f:
      raw_data = json.load(f)

  print(f"✅ Loaded data from {json_path}")

  # === Step 2: Extract Session Info ===
  session_id = raw_data.get("session_id", json_path.stem)
  task_name = raw_data.get("task_name", f"task_{session_id}")

  # Create output directory for this episode
  output_dir = arg_output_root / session_id
  output_dir.mkdir(parents=True, exist_ok=True)

  parquet_path = output_dir / f"{session_id}.parquet"

  print(f"Output will be saved to: {parquet_path}")

  # === Step 3: Build DataFrame from Frames ===
  rows = []

  for frame_idx, frame in enumerate(raw_data["frames"]):
      row = {
          'timestamp': frame['timestamp'],
          'frame_index': frame['frame_index'],
          'episode_index': session_id,
          'task_index': task_name,
          'index': len(rows),  # global index within dataset
          'next.done': False,
          'next.reward': 0.0,  # Placeholder — replace with real reward if available
      }

      # Current robot state (input)
      row['observation.state'] = frame['joint_positions']

      # Image path (can be video path + frame index or extracted image path)
      video_file = raw_data.get("video_file", None)
      if video_file:
          # Use video file with frame index (e.g., "video.mp4#frame_5")
          row['observation.image'] = f"frame_{frame_idx}"
      else:
          row['observation.image'] = None  # or use placeholder like "no_image"

      rows.append(row)

  df = pd.DataFrame(rows)

  print(f"📊 Created DataFrame with {len(df)} frames")

  # === Step 4: Compute Action as NEXT Frame's Joint Positions ===

  # Shift forward by one — action is target joint positions at next time step
  df['action'] = df['observation.state'].shift(-1)

  print('rows before dropping', len(df))

  # Drop last row since it has no next action (NaN in action)
  df.dropna(subset=['action'], inplace=True)
  df.reset_index(drop=True, inplace=True)

  print(f"🎯 Final dataset size after dropping last frame: {len(df)}")

  # === Step 5: Save to Parquet (LeRobot Format) ===

  df.to_parquet(parquet_path)

  print(f"✅ Successfully saved LeRobot dataset to:")
  print(f"   → {parquet_path.resolve()}")

  print("\n🎉 Dataset ready for use with Metis AI training via `lerobot train`!")
