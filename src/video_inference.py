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
        """
        Initialize the video inference engine.
        
        Args:
            model_path (str): Path to the trained model directory
            dataset_id (str): ID of the dataset used for training
        """
        self.model_path = Path(model_path)
        self.dataset_id = dataset_id
        self.inference_engine = LeRobotInference(model_path, dataset_id)
        self.device = self.inference_engine.device
        
    def load_model(self) -> bool:
        """
        Load the trained model and preprocessors.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        print(f"Loading model from: {self.model_path}")
        return self.inference_engine.load_model()
    
    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.inference_engine.policy is not None
    
    def load_video(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """
        Load a video file for processing.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            cv2.VideoCapture: Video capture object or None if failed
        """
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
        """
        Preprocess a video frame for model input.
        
        Args:
            frame (np.ndarray): Raw video frame
            target_size (tuple): Target size (height, width) for resizing
            
        Returns:
            np.ndarray: Preprocessed frame in channels-first format [C, H, W]
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to expected input size
        frame_resized = cv2.resize(frame_rgb, (target_size[1], target_size[0]))  # width, height
        
        # Normalize to [0, 1] range
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Convert from [H, W, C] to [C, H, W] format (channels first)
        frame_channels_first = np.transpose(frame_normalized, (2, 0, 1))
        
        return frame_channels_first
    
    def process_video(self, rgb_video_path: str, gripper_video_path: str, depth_video_path: str, 
                     joint_states: Optional[List[np.ndarray]] = None, 
                     target_size: Tuple[int, int] = (400, 640)) -> List[Dict[str, Any]]:
        """
        Process three video files (RGB, gripper, depth) and run inference on each frame.
        
        Args:
            rgb_video_path (str): Path to the RGB video file
            gripper_video_path (str): Path to the gripper video file
            depth_video_path (str): Path to the depth video file
            joint_states (list): Optional list of joint states for each frame
            target_size (tuple): Target size (height, width) for frame preprocessing
            
        Returns:
            list: List of inference results for each frame
        """
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
        
        results = []
        frame_count = 0
        
        try:
            print(f"Processing RGB video: {rgb_video_path}")
            print(f"Processing gripper video: {gripper_video_path}")
            print(f"Processing depth video: {depth_video_path}")
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
                
                # Create observation with frames from different cameras
                observation = {
                    "observation.images.rgb": processed_rgb_frame,
                    "observation.images.gripper": processed_gripper_frame,
                    "observation.images.depth": processed_depth_frame
                }
                
                # Add joint state if provided
                if joint_states is not None and frame_count < len(joint_states):
                    observation["observation.state"] = joint_states[frame_count]
                elif joint_states is not None and len(joint_states) > 0:
                    # Use last available joint state
                    observation["observation.state"] = joint_states[-1]
                
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
                    print(f"Processed frame {frame_count}")
                
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
        
        print(f"Finished processing {frame_count} frames")
        return results
    
    def process_single_frame(self, frame_path: str, joint_state: Optional[np.ndarray] = None,
                           target_size: Tuple[int, int] = (400, 640)) -> Dict[str, Any]:
        """
        Process a single image frame and run inference.
        
        Args:
            frame_path (str): Path to the image file
            joint_state (np.ndarray): Optional joint state
            target_size (tuple): Target size (height, width) for frame preprocessing
            
        Returns:
            dict: Inference result
        """
        try:
            # Load image
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Cannot load image: {frame_path}")
                return {"success": False, "error": "Cannot load image"}
            
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame, target_size)
            
            # Create observation
            observation = {
                    "observation.images.rgb": processed_frame,
                    "observation.images.gripper": processed_frame,
                    "observation.images.depth": processed_frame  # Using same frame for both cameras
            }
            
            # Add joint state if provided
            if joint_state is not None:
                observation["observation.state"] = joint_state
            
            # Run inference
            result = self.inference_engine.run_inference(observation)
            return result
            
        except Exception as e:
            print(f"Error processing single frame: {e}")
            return {"success": False, "error": str(e)}
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save inference results to a file.
        
        Args:
            results (list): List of inference results
            output_path (str): Path to save results
        """
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
    
    # Example 1: Process a single image
    print("\nExample 1: Processing a single image")
    # Note: You would need to provide an actual image path
    # result = inference_engine.process_single_frame("path/to/image.jpg")
    # print(f"Single frame result: {result}")
    
    # Example 2: Process a video file (uncomment and provide actual video paths)
    print("\nExample 2: Processing video files")
    rgb_video_path = "input/robot_session_rgb_20251113_080958.mp4"  # Replace with actual RGB video path
    gripper_video_path = "input/robot_session_gripper_20251113_080958.mp4"  # Replace with actual gripper video path
    depth_video_path = "input/depth_video_20251113_080958.mp4"  # Replace with actual depth video path
    joint_states = create_sample_joint_states(100)  # Sample joint states
    results = inference_engine.process_video(rgb_video_path, gripper_video_path, depth_video_path, joint_states)
    print(f"Processed {results} frames")
    
    
    
    
if __name__ == "__main__":
    main()
