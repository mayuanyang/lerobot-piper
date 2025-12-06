import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import collections
import traceback
import random
from collections import deque

# Define temporal window to match train.py configuration
fps = 10
frame_time = 1 / fps  # 0.1 seconds
obs_temporal_window = [
    -1 * frame_time,
    0.0    
]
HISTORY_LENGTH = len(obs_temporal_window)

# --- Assuming lerobot_inference import is correct from original context ---
try:
    from .lerobot_inference import LeRobotInference
except (ImportError, ValueError):
    # Fallback for when running as a script directly
    import sys
    sys.path.append(str(Path(__file__).parent))
    from lerobot_inference_smolvla import LeRobotInference
# ------------------------------------------------------------------------

class VideoInference:
    """A class to handle video-based inference with trained LeRobot policies."""
    
    def __init__(self, model_id: str, dataset_id: str = "ISdept/piper_arm", closed_loop: bool = False):
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.closed_loop = closed_loop  # Flag for closed-loop prediction
        self.inference_engine = LeRobotInference(model_id, dataset_id)
        self.device = self.inference_engine.device
        
        # Use the size specified by the preprocessor if available
        # Default to the size from dataset info (400, 640) based on info.json
        self.target_size: Tuple[int, int] = self._get_image_size()

    def _get_image_size(self) -> Tuple[int, int]:
        return (400, 640) # (Height, Width)

    def load_model(self) -> bool:
        print(f"Loading model from: {self.model_id}")
        return self.inference_engine.load_model()
    
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
    
    # --- REFACTORED: Ensuring Correct (H, W, C) shape for single-channel depth ---
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resizes the frame and ensures it is in (H, W, C) format (Channels Last).
        """
        target_h, target_w = self.target_size
        
        # 1. Resize (cv2 expects (W, H))
        frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to float32 and normalize to [0,1] range
        frame_processed = frame_resized.astype(np.float32) / 255.0

            
        # At this point, all frames (RGB, Gripper, Depth) must be 3D (H, W, C)
        # (Where C is 3 for RGB/Gripper, and 1 for grayscale Depth)
        return frame_processed


    def process_video(self, rgb_video_path: str, gripper_video_path: str, depth_video_path: str, 
                        joint_states: deque) -> List[Dict[str, Any]]:
        
        print(f"Expected Image Size: {self.target_size}")
        print(f"Closed-loop mode: {self.closed_loop}")

        # Load videos (omitted boilerplate for brevity)
        rgb_cap = self.load_video(rgb_video_path);
        if rgb_cap is None: return []
        gripper_cap = self.load_video(gripper_video_path);
        if gripper_cap is None: rgb_cap.release(); return []
        depth_cap = self.load_video(depth_video_path);
        if depth_cap is None: rgb_cap.release(); gripper_cap.release(); return []
        
        # Initialize temporal buffers (Deques)
        rgb_history = collections.deque(maxlen=HISTORY_LENGTH)
        gripper_history = collections.deque(maxlen=HISTORY_LENGTH)
        depth_history = collections.deque(maxlen=HISTORY_LENGTH)
        state_history = collections.deque(maxlen=HISTORY_LENGTH)
        predicted_action = None  # Track the most recent predicted action for closed-loop
        
        results = []
        frame_count = 0
        
        state_scale = 100000.0
        action_diff_scale = 10000.0
        
        try:
            print(f"Processing videos with {HISTORY_LENGTH}-step temporal window.")
            
            while True:
                # Read frames from all videos
                rgb_ret, rgb_frame = rgb_cap.read()
                gripper_ret, gripper_frame = gripper_cap.read()
                depth_ret, depth_frame = depth_cap.read()
                
                
                if not (rgb_ret and gripper_ret and depth_ret):
                    break
                
                # Preprocess frames (returns H, W, C)
                processed_rgb_frame = self.preprocess_frame(rgb_frame)
                processed_gripper_frame = self.preprocess_frame(gripper_frame)
                processed_depth_frame = self.preprocess_frame(depth_frame)
                
                # --- Temporal Buffering ---
                rgb_history.append(processed_rgb_frame)
                gripper_history.append(processed_gripper_frame)
                depth_history.append(processed_depth_frame)
                
                current_joint_state = joint_states.popleft()
                # Divide every value in current_joint_state by 100000
                current_joint_state = current_joint_state / state_scale
                
                noise = np.random.uniform(-0.1, 0.1, size=current_joint_state.shape)

                # Add noise to current_joint_state
                current_joint_state = current_joint_state + noise

                state_history.append(current_joint_state)

                                
                # --- Inference Check ---
                if len(rgb_history) < HISTORY_LENGTH:
                    print('No enough observation.state shape:', len(state_history), len(rgb_history), len(gripper_history), len(depth_history))
                    frame_count += 1
                    cv2.imshow('RGB Video Input', rgb_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing stopped by user during buffering")
                        break
                    continue
                
                
                # Stack the temporal history and prepare tensors in the correct format
                # Expected format for diffusion policy: [B, T, C, H, W] where B=1 (batch size)
                rgb_stacked = np.stack(rgb_history).astype(np.float32)  # [T, H, W, C]
                rgb_tensor = torch.from_numpy(rgb_stacked).permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, C, H, W]
                
                gripper_stacked = np.stack(gripper_history).astype(np.float32)  # [T, H, W, C]
                gripper_tensor = torch.from_numpy(gripper_stacked).permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, C, H, W]
                
                depth_stacked = np.stack(depth_history).astype(np.float32)  # [T, H, W, C]
                depth_tensor = torch.from_numpy(depth_stacked).permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, C, H, W]

                observation = {
                    "observation.images.rgb": rgb_tensor,
                    "observation.images.gripper": gripper_tensor,
                    "observation.images.depth": depth_tensor,
                }
                
                
                # Add stacked joint state history (T, D)
                # This includes the actual joint states for all frames
                if len(state_history) == HISTORY_LENGTH:
                    # Keep all 7 joints as the dataset contains 7-DOF joint positions
                    state_stack = np.stack(state_history).astype(np.float32)
                    observation["observation.state"] = torch.from_numpy(state_stack).unsqueeze(0)  # [1, T, D]
                
                # Run inference
                result = self.inference_engine.run_inference(observation)
                result["frame_index"] = frame_count
                results.append(result)
                
                
                # Process the predicted action for autoregressive prediction
                if result["success"] and "result" in result and "action" in result["result"]:
                    predicted_actions = result["result"]["action"]
                    print(f'The predicted action diff for frame {frame_count}:', predicted_actions[0])
                    
                    # Ensure the action is in the correct format (numpy array)
                    if isinstance(predicted_actions, np.ndarray):
                        # For diffusion policies, we typically use the first action in the sequence
                        if len(predicted_actions.shape) > 1:
                            predicted_action_diff = predicted_actions[0]
                    else:
                        # If it's not already a numpy array, convert it
                        if isinstance(predicted_actions, (list, tuple)):
                            predicted_actions = np.array(predicted_actions, dtype=np.float32)
                            if len(predicted_actions.shape) > 1:
                                predicted_action_diff = predicted_actions[0]
                        else:
                            predicted_action_diff = np.array([predicted_actions], dtype=np.float32)
                    
                    # Ensure the action diff is the correct shape (7 DOF for joint positions)
                    if predicted_action_diff.shape != (7,):
                        print(f"Warning: Unexpected action diff shape {predicted_action_diff.shape}, expected (7,)")
                    else:
                        # Apply the action diff to the current state to get the next state
                        predicted_action = current_joint_state * state_scale + predicted_action_diff * action_diff_scale
                        print(f'Applied action diff to get next state: {predicted_action}')
                    
                    # Set the accumulated action back to the result
                    print('The final predicted_action', predicted_action)
                    result["result"]["action"][0] = predicted_action  # Commented out to avoid tensor dimension mismatch
                    
                
                
                # Display and progress updates (omitted boilerplate for brevity)
                cv2.imshow('RGB Video Input', rgb_frame)
                cv2.imshow('Gripper Video Input', gripper_frame)
                cv2.imshow('Depth Video Input', depth_frame)
                
                if frame_count % 30 == 0:
                    print(f"Processed frame {frame_count}")
                
                frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing stopped by user")
                    break
                    
        except Exception as e:
            print(f"Error during video processing: {e}")
            traceback.print_exc()
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
            import json
            serializable_results = []
            for result in results:
                serializable_result = result.copy()
                if "result" in serializable_result and "action" in serializable_result["result"]:
                    serializable_result["result"]["action"] = serializable_result["result"]["action"].tolist()
                serializable_results.append(serializable_result)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")

def create_sample_joint_states() -> deque:
    import json
    from pathlib import Path
    
    # Load joint positions from metadata JSON file
    metadata_path = Path(__file__).parent / "temp" / "data_20251128_095915.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract all joint positions from the metadata
    frames = metadata["frames"]
    joint_states_queue = deque()
    
    # Process all frames in the metadata and add to queue
    for frame in frames:
        joint_positions = frame["joint_positions"]
        # Keep all 7 joints as the dataset contains 7-DOF joint positions
        joint_state = np.array(joint_positions, dtype=np.float32)
        joint_states_queue.append(joint_state)
    
    return joint_states_queue

def main(closed_loop=False):
    """Main function demonstrating video inference usage."""
    print("Video LeRobot Inference Demo")
    print("=" * 35)
    
    inference_engine = VideoInference("ISdept/smolvla-piper", closed_loop=closed_loop)
    
    print("Loading model...")
    if not inference_engine.load_model():
        print("Could not load trained model. Exiting.")
        return
    
    print("\n" + "=" * 35)
    print("MODEL LOADED SUCCESSFULLY")
    print(f"Inferred target image size: {inference_engine.target_size}")
    print("=" * 35)
    
    
    print("Processing video files")
    rgb_video_path = "input/episode_001_rgb.mp4"
    gripper_video_path = "input/episode_001_gripper.mp4"
    depth_video_path = "input/episode_001_depth.mp4"
    
    joint_states = create_sample_joint_states()
    
    print('the length of joint_states', len(joint_states))
    
    results = inference_engine.process_video(rgb_video_path, gripper_video_path, depth_video_path, joint_states)
    inference_engine.save_results(results, "temp/inference_results.json")
    print(f"Processed {len(results)} frames")
    

    
if __name__ == "__main__":
    import sys
    # Check if closed-loop mode is requested
    closed_loop = "--closed-loop" in sys.argv
    main(True)
