"""
Video-based inference for LeRobot policies.
This script demonstrates how to process a video file and use it for inference.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import collections

# Define temporal window to match train.py configuration
fps = 10
frame_time = 1 / fps  # 0.1 seconds
obs_temporal_window = [
    -3 * frame_time,  # Previous 3rd step
    -2 * frame_time,  # Previous 2nd step
    -1 * frame_time,  # Previous 1st step
    0.0               # Current step
]
HISTORY_LENGTH = len(obs_temporal_window)

# --- Assuming lerobot_inference import is correct from original context ---
try:
    from .lerobot_inference import LeRobotInference
except ImportError:
    # Fallback for when running as a script directly
    import sys
    sys.path.append(str(Path(__file__).parent))
    from lerobot_inference import LeRobotInference
# ------------------------------------------------------------------------

class VideoInference:
    """A class to handle video-based inference with trained LeRobot policies."""
    
    def __init__(self, model_path: str, dataset_id: str = "ISdept/piper_arm"):
        self.model_path = Path(model_path)
        self.dataset_id = dataset_id
        self.inference_engine = LeRobotInference(model_path, dataset_id)
        self.device = self.inference_engine.device
        
        # Use the size specified by the preprocessor if available
        self.target_size: Tuple[int, int] = self._get_image_size()

    def _get_image_size(self) -> Tuple[int, int]:
        """Tries to determine the expected image size from the preprocessor config."""
        try:
            pp = self.inference_engine.preprocessor
            if hasattr(pp.config, 'image_size') and pp.config.image_size:
                # LeRobot format is usually (H, W)
                return tuple(pp.config.image_size)
            elif hasattr(pp.config, 'input_shape') and len(pp.config.input_shape) >= 3:
                # Fallback to general input shape, assuming (C, H, W)
                _, h, w = pp.config.input_shape
                return (h, w)
        except Exception:
            pass # Use default fallback
        
        # Default size based on the error message [..., 84, 84]
        print("Warning: Could not auto-detect image size. Using default (84, 84).")
        return (84, 84) # (Height, Width)

    def load_model(self) -> bool:
        print(f"Loading model from: {self.model_path}")
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
        
        # 2. Convert to float32 and normalize
        frame_processed = frame_resized.astype(np.float32)
        

        # 3. CRITICAL: Ensure frame has a channel dimension (H, W, C)
        # If the frame is (H, W) (2 dimensions), we need to add a channel dimension (H, W, 1)
        if len(frame_processed.shape) == 2:
            frame_processed = frame_processed[:, :, np.newaxis]
            
        # At this point, all frames (RGB, Gripper, Depth) must be 3D (H, W, C)
        # (Where C is 3 for RGB/Gripper, and 1 for grayscale Depth)
        return frame_processed


    def process_video(self, rgb_video_path: str, gripper_video_path: str, depth_video_path: str, 
                        joint_states: List[np.ndarray]) -> List[Dict[str, Any]]:
        
        print(f"Expected Image Size: {self.target_size}")

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
        
        results = []
        frame_count = 0
        
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

                # ... (Joint state handling remains the same) ...
                current_joint_state = None
                if frame_count < len(joint_states):
                    current_joint_state = joint_states[frame_count]
                elif len(joint_states) > 0:
                    current_joint_state = joint_states[-1]
                
                if current_joint_state is not None:
                    state_history.append(current_joint_state)
                
                # --- Inference Check ---
                if len(rgb_history) < HISTORY_LENGTH:
                    frame_count += 1
                    cv2.imshow('RGB Video Input', rgb_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing stopped by user during buffering")
                        break
                    continue
                
                
                # --- Prepare Observation for Inference ---
                rgb_stacked = np.stack(rgb_history).astype(np.float32)
                gripper_stacked = np.stack(gripper_history).astype(np.float32)
                depth_stacked = np.stack(depth_history).astype(np.float32)


                observation = {
                    "observation.images.rgb": rgb_stacked,
                    "observation.images.gripper": gripper_stacked,
                    "observation.images.depth": depth_stacked
                }
                
                # Add stacked joint state history (T, D)
                if len(state_history) == HISTORY_LENGTH:
                    observation["observation.state"] = np.stack(state_history).astype(np.float32)
                
                # Run inference
                result = self.inference_engine.run_inference(observation)
                result["frame_index"] = frame_count
                results.append(result)
                
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
        finally:
            # Clean up
            rgb_cap.release()
            gripper_cap.release()
            depth_cap.release()
            cv2.destroyAllWindows()
        
        print(f"Finished processing {frame_count} frames, with {len(results)} inference results")
        return results

    
    # ... (save_results and create_sample_joint_states remain the same) ...

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

def create_sample_joint_states() -> List[np.ndarray]:
    joint_states = []
    
        
    joints1 = np.array([
        -0.07135841252559343,
        1.6254133016870895,
        -1.5040269480126633,
        0.0,
        -0.5187594912348668,
        -1.9740392433296308,
        0.06600000010803342
    ], dtype=np.float32)
    
    joints2 = np.array([
        -0.07135841252559343,
        1.6254133016870895,
        -1.5040269480126633,
        0.0,
        -0.5126108640939633,
        -1.9740392433296308,
        0.06600000010803342
    ], dtype=np.float32)
    
    joints3 = np.array([
        -0.07646108431757542,
        1.6254133016870895,
        -1.5040269480126633,
        0.0,
        -0.5199573625901166,
        -1.9740392433296308,
        0.06600000010803342
    ], dtype=np.float32)
    
    joints4 = np.array([
        -0.07646108431757542,
        1.6254133016870895,
        -1.5040269480126633,
        0.0,
        -0.5149285091397431,
        -1.9740392433296308,
        0.06600000010803342
    ], dtype=np.float32)
    
    joint_states.append(joints1)
    joint_states.append(joints2)
    joint_states.append(joints3)
    joint_states.append(joints4)
    
    return joint_states

def main():
    """Main function demonstrating video inference usage."""
    print("Video LeRobot Inference Demo")
    print("=" * 35)
    
    inference_engine = VideoInference("./model_output")
    
    print("Loading model...")
    if not inference_engine.load_model():
        print("Could not load trained model. Exiting.")
        return
    
    print("\n" + "=" * 35)
    print("MODEL LOADED SUCCESSFULLY")
    print(f"Inferred target image size: {inference_engine.target_size}")
    print("=" * 35)
    
    
    print("Processing video files")
    rgb_video_path = "input/robot_session_rgb_20251113_080958.mp4"
    gripper_video_path = "input/robot_session_gripper_20251113_080958.mp4"
    depth_video_path = "input/episode_001.mp4"
    joint_states = create_sample_joint_states()
    
    results = inference_engine.process_video(rgb_video_path, gripper_video_path, depth_video_path, joint_states)
    inference_engine.save_results(results, "temp/inference_results.json")
    print(f"Processed {len(results)} frames")
    

    
if __name__ == "__main__":
    main()