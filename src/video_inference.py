import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import collections
import traceback

# Define temporal window to match train.py configuration
fps = 10
frame_time = 1 / fps  # 0.1 seconds
obs_temporal_window = [
    -9 * frame_time,
    -8 * frame_time,
    -7 * frame_time,
    -6 * frame_time,
    -5 * frame_time,
    -4 * frame_time,
    -3 * frame_time,
    -2 * frame_time,
    -1 * frame_time,
    0.0    
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
    
    def __init__(self, model_id: str, dataset_id: str = "ISdept/piper_arm"):
        self.model_id = model_id
        self.dataset_id = dataset_id
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
        #frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to float32 and normalize to [0,1] range
        frame_processed = frame.astype(np.float32) / 255.0
                        

        # 3. CRITICAL: Ensure frame has a channel dimension (H, W, C)
        # If the frame is (H, W) (2 dimensions), we need to add a channel dimension (H, W, 1)
        if len(frame_processed.shape) == 2:
            frame_processed = frame_processed[:, :, np.newaxis]
            
        # At this point, all frames (RGB, Gripper, Depth) must be 3D (H, W, C)
        # (Where C is 3 for RGB/Gripper, and 1 for grayscale Depth)
        return frame_processed


    def process_video(self, rgb_video_path: str, gripper_video_path: str, depth_video_path: str, 
                        joint_states: List[np.ndarray], frames_to_skip: int) -> List[Dict[str, Any]]:
        
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
        predicted_action = None
        
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
                
                if frame_count < frames_to_skip:
                    frame_count += 1
                    continue
                # Preprocess frames (returns H, W, C)
                processed_rgb_frame = self.preprocess_frame(rgb_frame)
                processed_gripper_frame = self.preprocess_frame(gripper_frame)
                processed_depth_frame = self.preprocess_frame(depth_frame)
                
                # --- Temporal Buffering ---
                rgb_history.append(processed_rgb_frame)
                gripper_history.append(processed_gripper_frame)
                depth_history.append(processed_depth_frame)

                # Handle joint state: use provided joint states for initial frames,
                # then use predicted actions for autoregressive prediction
                current_joint_state = None
                if frame_count < len(joint_states):
                    current_joint_state = joint_states[frame_count]
                elif predicted_action is not None:
                    # For autoregressive prediction, use the last predicted action
                    current_joint_state = predicted_action
                else:
                    # Fallback to last known joint state
                    if len(joint_states) > 0:
                        current_joint_state = joint_states[-1]
                
                
                # Add current joint state to history
                if current_joint_state is not None:
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
                # This includes both actual joint states (initial frames) and
                # predicted actions (subsequent frames) for autoregressive prediction
                if len(state_history) == HISTORY_LENGTH:
                    observation["observation.state"] = np.expand_dims(np.stack(state_history).astype(np.float32), axis=0)
                
                # Run inference
                result = self.inference_engine.run_inference(observation)
                result["frame_index"] = frame_count
                results.append(result)
                
                
                # Process the predicted action for autoregressive prediction
                if result["success"] and "result" in result and "action" in result["result"]:
                    predicted_actions = result["result"]["action"]
                    print('The predicted actions:', predicted_actions[0])
                    
                    # Ensure the action is in the correct format (numpy array)
                    if isinstance(predicted_actions, np.ndarray):
                        # For diffusion policies, we typically use the first action in the sequence
                        if len(predicted_actions.shape) > 1:
                            predicted_action = predicted_actions[0]
                    else:
                        # If it's not already a numpy array, convert it
                        if isinstance(predicted_actions, (list, tuple)):
                            predicted_actions = np.array(predicted_actions, dtype=np.float32)
                            if len(predicted_actions.shape) > 1:
                                predicted_action = predicted_actions[0]
                        else:
                            predicted_action = np.array([predicted_actions], dtype=np.float32)
                    
                    # Ensure the action is the correct shape (7 DOF for joint positions)
                    if predicted_action.shape != (7,):
                        print(f"Warning: Unexpected action shape {predicted_action.shape}, expected (7,)")
                
                
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

def create_sample_joint_states(num_states: int = 1) -> List[np.ndarray]:
    joint_states = []
    
    joint1 = np.array([
      -0.10316456313447532,
        1.6569679719419241,
        -1.46040415849622,
        0.0,
        -0.29512739584082315,
        -1.930865117372324,
        0.06600000010803342
    ], dtype=np.float32)
    
    joint2 = np.array([
      -0.11872356248640539,
        1.66713693674247,
        -1.46040415849622,
        0.0,
        -0.28385817484180054,
        -1.9367876682948635,
        0.06600000010803342
    ], dtype=np.float32)
    
    joint3 = np.array([
      -0.13669206887600455,
        1.67255364287783,
        -1.4550079679582415,
        0.0,
        -0.27119455808135773,
        -1.9502419748567668,
        0.06600000010803342
    ], dtype=np.float32)
    
    joint4 = np.array([
      -0.16607040828132943,
        1.6775646391409609,
        -1.4550079679582415,
        0.0,
        -0.2649360510845218,
        -1.9713307577253194,
        0.06600000010803342
    ], dtype=np.float32)
    
    joint5 = np.array([
      -0.20023264660469156,
        1.6826370623073252,
        -1.4550079679582415,
        0.0,
        -0.2590718129709446,
        -1.983222477236864,
        0.06600000010803342
    ], dtype=np.float32)
    
    joint6 = np.array([
      -0.23463441735599533,
        1.6826370623073252,
        -1.4550079679582415,
        0.0,
        -0.25395658381199215,
        -1.9889879338630276,
        0.06600000010803342
    ], dtype=np.float32)
    
    joint7 = np.array([
      -0.2647606645273449,
        1.6826370623073252,
        -1.4499828529018746,
        0.0,
        -0.24269516975692507,
        -2.000894697768732,
        0.06600000010803342
    ], dtype=np.float32)
    
    joint8 = np.array([
      -0.3003712076950401,
        1.6826370623073252,
        -1.4499828529018746,
        0.0,
        -0.23707634903541175,
        -2.000894697768732,
        0.06600000010803342
    ], dtype=np.float32)
    
    joint9 = np.array([
     -0.3325528993194336,
        1.6877621900614073,
        -1.4447530708920395,
        0.0,
        -0.22376535446019763,
        -2.000894697768732,
        0.06600000010803342
    ], dtype=np.float32)
    
    joint10 = np.array([
      -0.38476982230440804,
        1.7096250496426613,
        -1.4447530708920395,
        0.0,
        -0.21812836740703911,
        -1.9954850241477573,
        0.06600000010803342
    ], dtype=np.float32)
    
    joint_states.append(joint1)
    joint_states.append(joint2)
    joint_states.append(joint3)
    joint_states.append(joint4)
    joint_states.append(joint5)
    joint_states.append(joint6)
    joint_states.append(joint7)
    joint_states.append(joint8)
    joint_states.append(joint9)
    joint_states.append(joint10)
    
    
    return joint_states

def main():
    """Main function demonstrating video inference usage."""
    print("Video LeRobot Inference Demo")
    print("=" * 35)
    
    inference_engine = VideoInference("ISdept/piper_arm")
    
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
    
    results = inference_engine.process_video(rgb_video_path, gripper_video_path, depth_video_path, joint_states, 50)
    inference_engine.save_results(results, "temp/inference_results.json")
    print(f"Processed {len(results)} frames")
    

    
if __name__ == "__main__":
    main()
