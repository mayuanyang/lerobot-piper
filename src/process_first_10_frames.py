#!/usr/bin/env python3
"""
Script to process the first 10 frames from the parquet dataset:
1. Read the parquet file
2. Extract the first 10 frames with bounding box data
3. Extract corresponding frames from videos
4. Apply bounding boxes to frames
5. Save annotated frames to temp directory
"""

import pyarrow.parquet as pq
import cv2
import numpy as np
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw


def read_parquet_data(parquet_path, num_frames=10):
    """
    Read the first num_frames from the parquet file.
    
    Args:
        parquet_path: Path to the parquet file
        num_frames: Number of frames to read (default: 10)
        
    Returns:
        List of dictionaries containing frame data
    """
    print(f"Reading first {num_frames} frames from {parquet_path}")
    
    # Read the parquet file
    table = pq.read_table(parquet_path)
    
    # Get the first num_frames rows
    subset = table.slice(0, num_frames)
    
    # Extract data for each frame
    frames_data = []
    for i in range(num_frames):
        frame_data = {
            'frame_index': subset['frame_index'][i].as_py(),
            'episode_index': subset['episode_index'][i].as_py(),
            'bounding_boxes': subset['observation.box'][i].as_py(),
            'timestamp': subset['timestamp'][i].as_py()
        }
        frames_data.append(frame_data)
    
    print(f"Successfully extracted {len(frames_data)} frames")
    return frames_data


def get_video_paths(episode_index):
    """
    Get paths to video files for a given episode.
    
    Args:
        episode_index: Index of the episode
        
    Returns:
        Dictionary mapping camera names to video paths
    """
    episode_dir = f"src/data/piper_training_data/episode{episode_index}"
    
    if not os.path.exists(episode_dir):
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")
    
    # Look for MP4 files in the episode directory
    video_files = list(Path(episode_dir).glob("*.mp4"))
    
    # Map camera names to video paths
    video_paths = {}
    for video_file in video_files:
        filename = video_file.name
        # Extract camera name from filename (e.g., camera_front_20260205_202754.mp4 -> front)
        if "camera_front" in filename:
            video_paths["front"] = str(video_file)
        elif "camera_gripper" in filename:
            video_paths["gripper"] = str(video_file)
        elif "camera_right" in filename:
            video_paths["right"] = str(video_file)
        else:
            # Use filename without extension as key
            key = filename.split(".mp4")[0]
            video_paths[key] = str(video_file)
    
    return video_paths


def extract_frame_from_video(video_path, frame_index):
    """
    Extract a specific frame from a video file.
    
    Args:
        video_path: Path to the video file
        frame_index: Index of the frame to extract
        
    Returns:
        Frame as numpy array (BGR format) or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Release the video capture
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_index} from {video_path}")
        return None
    
    return frame


def draw_bounding_boxes_on_frame(frame, bounding_boxes):
    """
    Draw bounding boxes on a frame.
    
    Args:
        frame: numpy array (H, W, C) in BGR format
        bounding_boxes: List of bounding boxes (each box is a list of coordinates)
        
    Returns:
        Frame with bounding boxes drawn
    """
    h, w = frame.shape[:2]
    
    # Draw each bounding box
    print(f"Drawing {len(bounding_boxes)} bounding boxes on frame of size ({w}x{h})")
    for i, box in enumerate(bounding_boxes):
        if len(box) >= 4:
            # Box format is [x1, y1, x2, y2] in normalized coordinates [0, 1000]
            x1, y1, x2, y2 = box[:4]
            
            # Convert normalized coordinates [0, 1000] to pixel coordinates
            x1_px = int(x1 * w / 1000)
            y1_px = int(y1 * h / 1000)
            x2_px = int(x2 * w / 1000)
            y2_px = int(y2 * h / 1000)
            
            # Generate random color for each box
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Draw rectangle using OpenCV
            cv2.rectangle(frame, (x1_px, y1_px), (x2_px, y2_px), color, 2)
            
            # Add label
            label = f"Object {i}"
            cv2.putText(frame, label, (x1_px, y1_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame


def save_frame_to_temp(frame, frame_index, episode_index, camera_name, temp_dir="temp"):
    """
    Save a frame to the temp directory.
    
    Args:
        frame: Frame as numpy array
        frame_index: Index of the frame
        episode_index: Index of the episode
        camera_name: Name of the camera
        temp_dir: Directory to save frames (default: "temp")
    """
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create filename
    filename = f"episode_{episode_index}_frame_{frame_index}_{camera_name}.jpg"
    filepath = os.path.join(temp_dir, filename)
    
    # Save frame
    success = cv2.imwrite(filepath, frame)
    
    if success:
        print(f"Saved frame to {filepath}")
    else:
        print(f"Failed to save frame to {filepath}")


def process_first_10_frames():
    """
    Main function to process the first 10 frames.
    """
    # Path to the parquet file
    parquet_path = "src/output/data/chunk-000/file-000.parquet"
    
    # Read the first 10 frames from parquet
    frames_data = read_parquet_data(parquet_path, num_frames=10)
    
    # Process each frame
    for frame_data in frames_data:
        frame_index = frame_data['frame_index']
        episode_index = frame_data['episode_index']
        bounding_boxes = frame_data['bounding_boxes']
        
        print(f"\nProcessing frame {frame_index} from episode {episode_index}")
        print(f"Bounding boxes: {bounding_boxes}")
        
        try:
            # Get video paths for this episode
            video_paths = get_video_paths(episode_index)
            print(f"Found video paths: {video_paths}")
            
            # Process each camera view
            for camera_name, video_path in video_paths.items():
                print(f"  Processing {camera_name} camera...")
                
                # Extract frame from video
                frame = extract_frame_from_video(video_path, frame_index)
                if frame is None:
                    print(f"    Failed to extract frame from {video_path}")
                    continue
                
                # Select bounding boxes based on camera name
                selected_boxes = []
                if "gripper" in camera_name and len(bounding_boxes) >= 2:
                    # For gripper camera: use the first 2 boxes
                    selected_boxes = bounding_boxes[:2]
                elif "front" in camera_name and len(bounding_boxes) >= 4:
                    # For front camera: use the 3rd and 4th boxes
                    selected_boxes = bounding_boxes[2:4]
                elif "right" in camera_name and len(bounding_boxes) >= 6:
                    # For right camera: use the 5th and 6th boxes
                    selected_boxes = bounding_boxes[4:6]
                else:
                    # Default: use all boxes
                    selected_boxes = bounding_boxes
                
                print(f"    Using boxes: {selected_boxes}")
                
                # Apply bounding boxes to frame
                if selected_boxes:
                    annotated_frame = draw_bounding_boxes_on_frame(frame, selected_boxes)
                else:
                    annotated_frame = frame
                    print(f"    No bounding boxes for this frame")
                
                # Save annotated frame to temp directory
                save_frame_to_temp(annotated_frame, frame_index, episode_index, camera_name)
                
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            continue
        except Exception as e:
            print(f"  Unexpected error: {e}")
            continue
    
    print("\nProcessing complete! Check the 'temp' directory for annotated frames.")


if __name__ == "__main__":
    process_first_10_frames()