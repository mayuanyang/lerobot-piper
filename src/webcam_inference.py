"""
Webcam-based inference for LeRobot policies.
This script demonstrates how to capture frames from a webcam and use them for inference.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time

class WebcamInference:
    """A class to handle webcam-based inference with trained LeRobot policies."""
    
    def __init__(self, model_path: str, dataset_id: str = "ISdept/piper_arm", webcam_id: int = 0):
        """
        Initialize the webcam inference engine.
        
        Args:
            model_path (str): Path to the trained model directory
            dataset_id (str): ID of the dataset used for training
            webcam_id (int): ID of the webcam to use (default: 0)
        """
        self.model_path = Path(model_path)
        self.dataset_id = dataset_id
        self.webcam_id = webcam_id
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        # Determine the appropriate device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.cap = None
        
    def load_model(self) -> bool:
        """
        Load the trained model and preprocessors.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Check if model directory exists
            if not self.model_path.exists():
                print(f"Model directory {self.model_path} not found.")
                return False
                
            # Import LeRobot components
            try:
                from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
                from lerobot.policies.factory import make_pre_post_processors
                from lerobot.configs.types import FeatureType
                from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
                from lerobot.datasets.utils import dataset_to_policy_features
                from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
            except ImportError as e:
                print(f"LeRobot not installed or not in path: {e}")
                print("Please install lerobot: pip install lerobot")
                return False
            
            # Load dataset metadata to get input/output shapes and stats
            print("Loading dataset metadata...")
            dataset_metadata = LeRobotDatasetMetadata(self.dataset_id, force_cache_sync=True, revision="main")
            features = dataset_to_policy_features(dataset_metadata.features)
            output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
            input_features = {key: ft for key, ft in features.items() if key not in output_features}
            
            # Recreate the config used during training
            print("Creating policy configuration...")
            cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
            
            # Initialize the policy
            print("Initializing policy...")
            self.policy = DiffusionPolicy(cfg)
            
            # Load the trained weights
            # First try safetensors format, fallback to pytorch format
            safetensors_path = self.model_path / "model.safetensors"
            pytorch_path = self.model_path / "pytorch_model.bin"
            
            if safetensors_path.exists():
                print(f"Loading model from safetensors format: {safetensors_path}")
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path)
                self.policy.load_state_dict(state_dict)
                self.policy.to(self.device)
            elif pytorch_path.exists():
                print(f"Loading model from pytorch format: {pytorch_path}")
                self.policy.load_state_dict(torch.load(pytorch_path, map_location=self.device))
                self.policy.to(self.device)
            else:
                print(f"Model weights not found at {self.model_path}")
                return False
            self.policy.eval()
            self.policy.to(self.device)
            
            # Load preprocessors
            print("Loading preprocessors...")
            self.preprocessor, self.postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def initialize_webcam(self) -> bool:
        """
        Initialize the webcam.
        
        Returns:
            bool: True if webcam initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.webcam_id)
            if not self.cap.isOpened():
                print(f"Cannot open webcam {self.webcam_id}")
                return False
            
            # Set webcam properties (optional)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Webcam {self.webcam_id} initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing webcam: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the webcam.
        
        Returns:
            np.ndarray: Captured frame or None if failed
        """
        if self.cap is None:
            print("Webcam not initialized")
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                return None
            
            return frame
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a webcam frame for model input.
        
        Args:
            frame (np.ndarray): Raw webcam frame
            
        Returns:
            np.ndarray: Preprocessed frame in channels-first format [C, H, W]
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to expected input size (adjust as needed)
        # Based on the prepare_dataset.py, it seems to expect 640x480
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        
        # Normalize to [0, 1] range
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Convert from [H, W, C] to [C, H, W] format (channels first)
        frame_channels_first = np.transpose(frame_normalized, (2, 0, 1))
        
        return frame_channels_first
    
    def preprocess_observation(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess observation data for model input.
        
        Args:
            observation (dict): Raw observation data
            
        Returns:
            dict: Preprocessed observation as tensors
        """
        processed = {}
        
        # Handle state observation
        if "state" in observation:
            state_tensor = torch.tensor(observation["state"], dtype=torch.float32)
            # Add batch dimension
            state_tensor = state_tensor.unsqueeze(0)
            processed["observation.state"] = state_tensor.to(self.device)
        
        # Handle image observation
        if "image" in observation:
            image_tensor = torch.tensor(observation["image"], dtype=torch.float32)
            # Image is already in [C, H, W] format from preprocess_frame
            # Add batch dimension (B, C, H, W)
            image_tensor = image_tensor.unsqueeze(0)
            processed["observation.images.front_camera"] = image_tensor.to(self.device)
            processed["observation.images.rear_camera"] = image_tensor.to(self.device)
            
        return processed
    
    def predict_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict action given an observation.
        
        Args:
            observation (dict): Current observation
            
        Returns:
            dict: Predicted action and metadata
        """
        if self.policy is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess observation
        batch = self.preprocess_observation(observation)
        
        # Apply preprocessing
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        
        # Get prediction
        with torch.no_grad():
            action = self.policy.select_action(batch)
        
        # Apply postprocessing
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        
        # Convert to numpy for easier handling
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = action
            
        return {
            "action": action_np,
            "timestamp": time.time(),
            "device": str(self.device)
        }
    
    def run_inference(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on an observation.
        
        Args:
            observation (dict): Current observation
            
        Returns:
            dict: Inference results
        """
        try:
            result = self.predict_action(observation)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_webcam_inference(self, joint_state: np.ndarray = None):
        """
        Run continuous inference using webcam frames.
        
        Args:
            joint_state (np.ndarray): Current joint state (if available)
        """
        if not self.initialize_webcam():
            print("Failed to initialize webcam")
            return
        
        print("Starting webcam inference. Press 'q' to quit.")
        
        try:
            while True:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    continue
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                
                # Create observation
                observation = {
                    "image": processed_frame
                }
                
                # Add joint state if provided
                if joint_state is not None:
                    observation["state"] = joint_state
                
                # Run inference
                result = self.run_inference(observation)
                
                if result["success"]:
                    action = result["result"]["action"]
                    print(f"Predicted action: {action}")
                else:
                    print(f"Inference failed: {result['error']}", result)
                
                # Display frame
                cv2.imshow('Webcam Input', frame)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Inference interrupted by user")
        except Exception as e:
            print(f"Error during webcam inference: {e}")
        finally:
            # Clean up
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def create_sample_joint_state():
    """Create a sample joint state for testing."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)

def main():
    """Main function demonstrating webcam usage."""
    print("Webcam LeRobot Inference Demo")
    print("=" * 35)
    
    # Initialize webcam inference engine
    inference_engine = WebcamInference("./model_output")
    
    # Try to load model
    print("Loading model...")
    if not inference_engine.load_model():
        print("Could not load trained model. Using demo mode.")
        print("\nTo use with a real trained model:")
        print("1. Train a model using train.py")
        print("2. Ensure the model is saved to ./model_output")
        print("3. Make sure lerobot is installed")
        
        # Demonstrate with webcam but without model
        print("\n" + "=" * 35)
        print("DEMO MODE - Webcam Only")
        print("=" * 35)
        
        if inference_engine.initialize_webcam():
            print("Webcam initialized successfully")
            print("Displaying webcam feed. Press 'q' to quit.")
            
            try:
                while True:
                    frame = inference_engine.capture_frame()
                    if frame is not None:
                        # Show frame
                        cv2.imshow('Webcam Demo', frame)
                        
                        # Simulate processing
                        if np.random.random() < 0.02:  # Print every ~50 frames
                            print("Processing frame...")
                        
                        # Break on 'q' key press
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            except KeyboardInterrupt:
                print("Demo interrupted by user")
            finally:
                inference_engine.cleanup()
        else:
            print("Failed to initialize webcam")
        
        return
    
    # Run actual webcam inference
    print("\n" + "=" * 35)
    print("RUNNING WEBCAM INFERENCE")
    print("=" * 35)
    
    # Create sample joint state (in a real scenario, this would come from encoders)
    joint_state = create_sample_joint_state()
    print(f"Using joint state: {joint_state}")
    
    # Run webcam inference
    inference_engine.run_webcam_inference(joint_state)

if __name__ == "__main__":
    main()
