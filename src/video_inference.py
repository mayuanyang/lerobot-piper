"""
Video-based inference for LeRobot policies.
This script demonstrates how to process a video file and use it for inference.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import time

import collections
from typing import List, Dict, Any, Tuple

HISTORY_LENGTH = 4

try:
    from .lerobot_inference import LeRobotInference
except ImportError:
    # Fallback for when running as a script directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from lerobot_inference import LeRobotInference

class VideoInference:
    """A class to handle video-based inference with trained LeRobot policies."""
    
    def __init__(self, model_path: str, dataset_id: str = "ISdept/piper_arm"):
        self.model_path = Path(model_path)
        self.dataset_id = dataset_id
        self.inference_engine = LeRobotInference(model_path, dataset_id)
        self.device = self.inference_engine.device
        
    def load_model(self) -> bool:
        print(f"Loading model from: {self.model_path}")
        return self.inference_engine.load_model()
    
    def is_model_loaded(self) -> bool:
        return self.inference_engine.policy is not None
    
    def load_video(self, video_path: str) -> Optional[cv2.VideoCapture]:
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video file: {video_path}")
                return None
            return cap
        except Exception as e:
            print(f"Error loading video: {e}")
            return None
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (400, 640)) -> np.ndarray:
        # Resize to expected input size
        frame_resized = cv2.resize(frame, (target_size[1], target_size[0]))  # width, height
        
        # Convert from [H, W, C] to [C, H, W] format (channels first)
        frame_channels_first = np.transpose(frame_resized, (2, 0, 1))
        
        return frame_channels_first
    


    def process_video(self, rgb_video_path: str, gripper_video_path: str, depth_video_path: str, 
                        joint_states: List[np.ndarray], 
                        target_size: Tuple[int, int] = (400, 640)) -> List[Dict[str, Any]]:
        
        # Load videos
        rgb_cap = self.load_video(rgb_video_path)
        if rgb_cap is None:
            return []
            
        gripper_cap = self.load_video(gripper_video_path)
        if gripper_cap is None:
            rgb_cap.release()
            return []
            
        depth_cap = self.load_video(depth_video_path)
        if depth_cap is None:
            rgb_cap.release()
            gripper_cap.release()
            return []
        
        # Initialize temporal buffers (Deques)
        # The deque size is set to HISTORY_LENGTH (4) to store the current frame and 3 past frames.
        rgb_history = collections.deque(maxlen=HISTORY_LENGTH)
        gripper_history = collections.deque(maxlen=HISTORY_LENGTH)
        depth_history = collections.deque(maxlen=HISTORY_LENGTH)
        state_history = collections.deque(maxlen=HISTORY_LENGTH)
        
        results = []
        frame_count = 0
        
        try:
            print(f"Processing RGB video: {rgb_video_path}")
            print(f"Processing gripper video: {gripper_video_path}")
            print(f"Processing depth video: {depth_video_path}")
            print(f"Using a {HISTORY_LENGTH}-step temporal window for inference.")
            print("Press 'q' to stop processing early")
            
            while True:
                # Read frames from all videos
                rgb_ret, rgb_frame = rgb_cap.read()
                gripper_ret, gripper_frame = gripper_cap.read()
                depth_ret, depth_frame = depth_cap.read()
                
                # Break if any video ends
                if not (rgb_ret and gripper_ret and depth_ret):
                    break
                
                # Preprocess frames
                processed_rgb_frame = self.preprocess_frame(rgb_frame, target_size)
                processed_gripper_frame = self.preprocess_frame(gripper_frame, target_size)
                processed_depth_frame = self.preprocess_frame(depth_frame, target_size)
                
                # --- Temporal Buffering ---
                # Append current frames to the history deques
                rgb_history.append(processed_rgb_frame)
                gripper_history.append(processed_gripper_frame)
                depth_history.append(processed_depth_frame)

                # Get the joint state for the current frame
                current_joint_state = None
                if frame_count < len(joint_states):
                    current_joint_state = joint_states[frame_count]
                elif len(joint_states) > 0:
                    # Use last available joint state if video is longer than joint_states list
                    current_joint_state = joint_states[-1]
                
                if current_joint_state is not None:
                    state_history.append(current_joint_state)
                
                # --- Inference Check ---
                # Only start inference once we have enough frames (HISTORY_LENGTH)
                if len(rgb_history) < HISTORY_LENGTH:
                    frame_count += 1
                    
                    # Display frames while buffering (optional)
                    cv2.imshow('RGB Video Input', rgb_frame)
                    cv2.imshow('Gripper Video Input', gripper_frame)
                    cv2.imshow('Depth Video Input', depth_frame)
                    
                    # Break on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing stopped by user during buffering")
                        break
                    
                    continue # Skip inference until buffer is full
                
                # --- Prepare Observation for Inference ---
                # The observation keys must now contain the HISTORY_LENGTH frames/states
                # np.stack creates an array of shape (HISTORY_LENGTH, H, W, C) for images,
                # and (HISTORY_LENGTH, D) for the state vector.
                observation = {
                    "observation.images.rgb": np.stack(rgb_history),
                    "observation.images.gripper": np.stack(gripper_history),
                    "observation.images.depth": np.stack(depth_history)
                }
                
                # Add stacked joint state history if available
                if len(state_history) == HISTORY_LENGTH:
                    observation["observation.state"] = np.stack(state_history)
                elif len(joint_states) > 0:
                    # Fallback in case of mismatch, but this path is less desirable
                    print(f"Warning: State history length {len(state_history)} != {HISTORY_LENGTH}")
                
                # Run inference
                result = self.inference_engine.run_inference(observation)
                result["frame_index"] = frame_count
                results.append(result)
                
                # Display frames (using RGB frame as main display)
                cv2.imshow('RGB Video Input', rgb_frame)
                cv2.imshow('Gripper Video Input', gripper_frame)
                cv2.imshow('Depth Video Input', depth_frame)
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    print(f"Processed frame {frame_count} (Inference Started at frame {HISTORY_LENGTH - 1})")
                
                frame_count += 1
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing stopped by user")
                    break
                    
        except Exception as e:
            print(f"Error during video processing: {e}")
        finally:
            # Clean up
            rgb_cap.release()
            gripper_cap.release()
            depth_cap.release()
            cv2.destroyAllWindows()
        
        print(f"Finished processing {frame_count} frames, with {len(results)} inference results")
        return results

    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        try:
            # Convert results to a serializable format
            serializable_results = []
            for result in results:
                serializable_result = result.copy()
                if "result" in serializable_result and "action" in serializable_result["result"]:
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_result["result"]["action"] = serializable_result["result"]["action"].tolist()
                serializable_results.append(serializable_result)
            
            # Save to JSON file
            import json
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")

def create_sample_joint_states(num_frames: int) -> List[np.ndarray]:
    """Create sample joint states for testing."""
    # Create a sequence of joint states that gradually change
    joint_states = []
    for i in range(num_frames):
        # Create a sine wave pattern for joint positions
        phase = i * 0.1
        joints = np.array([
            0.1 + 0.05 * np.sin(phase),
            0.2 + 0.05 * np.sin(phase * 1.2),
            0.3 + 0.05 * np.sin(phase * 1.4),
            0.4 + 0.05 * np.sin(phase * 1.6),
            0.5 + 0.05 * np.sin(phase * 1.8),
            0.6 + 0.05 * np.sin(phase * 2.0),
            0.7 + 0.05 * np.sin(phase * 2.2)
        ], dtype=np.float32)
        joint_states.append(joints)
    return joint_states

def main():
    """Main function demonstrating video inference usage."""
    print("Video LeRobot Inference Demo")
    print("=" * 35)
    
    # Initialize video inference engine
    inference_engine = VideoInference("./model_output")
    
    
    # Try to load model
    print("Loading model...")
    if not inference_engine.load_model():
        print("Could not load trained model. Using demo mode.")
        print("\nTo use with a real trained model:")
        print("1. Train a model using train.py")
        print("2. Ensure the model is saved to ./model_output")
        print("3. Make sure lerobot is installed")
        return
    
    print("\n" + "=" * 35)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 35)
    
    
    print("Processing video files")
    rgb_video_path = "input/robot_session_rgb_20251113_080958.mp4"  # Replace with actual RGB video path
    gripper_video_path = "input/robot_session_gripper_20251113_080958.mp4"  # Replace with actual gripper video path
    depth_video_path = "input/depth_video_20251113_080958.mp4"  # Replace with actual depth video path
    joint_states = create_sample_joint_states(100)  # Sample joint states
    results = inference_engine.process_video(rgb_video_path, gripper_video_path, depth_video_path, joint_states)
    inference_engine.save_results(results, "inference_results.json")
    print(f"Processed {results} frames")
    

    
if __name__ == "__main__":
    main()
