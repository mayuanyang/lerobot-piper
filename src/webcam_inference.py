from models.smooth_diffusion.joint_smooth_diffusion import JointSmoothDiffusion
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time


from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

class WebcamInference:
    """A class to handle webcam-based inference with trained LeRobot policies."""
    
    def __init__(self, model_id: str = "ISdept/piper_arm", dataset_id: str = "ISdept/piper_arm", 
                 webcam_rgb_id: int = 0, webcam_depth_id: int = 1, webcam_gripper_id: int = 2):
        """
        Initialize the webcam inference engine.
        
        Args:
            model_id (str): ID or path to the trained model
            dataset_id (str): ID of the dataset used for training
            webcam_rgb_id (int): ID of the RGB webcam (default: 0)
            webcam_depth_id (int): ID of the depth webcam (default: 1)
            webcam_gripper_id (int): ID of the gripper webcam (default: 2)
        """
        self.model_id = model_id
        self.dataset_id = dataset_id
        
        self.rgb_webcam_id = webcam_rgb_id  
        self.depth_webcam_id = webcam_depth_id
        self.gripper_webcam_id = webcam_gripper_id
        
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        
        # Camera capture objects
        self.rgb_cap = None
        self.depth_cap = None
        self.gripper_cap = None
        
        # Determine the appropriate device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        # Use the size specified by the preprocessor if available
        # Default to the size from dataset info (400, 640) based on info.json
        self.target_size = (400, 640)  # (Height, Width)
        
    def load_model(self) -> bool:
        """
        Load the trained model and preprocessors.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
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
            self.policy = JointSmoothDiffusion.from_pretrained(self.model_id)
            
            self.policy.eval()
            self.policy.to(self.device)
            
            # Load preprocessors
            print("Loading preprocessors...")
            self.preprocessor, self.postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def initialize_webcams(self) -> bool:
        """
        Initialize all webcams.
        
        Returns:
            bool: True if all webcams initialized successfully, False otherwise
        """
        try:
            # Initialize RGB camera
            self.rgb_cap = cv2.VideoCapture(self.rgb_webcam_id)
            if not self.rgb_cap.isOpened():
                print(f"Cannot open RGB webcam {self.rgb_webcam_id}")
                return False
            
            # Initialize depth camera
            self.depth_cap = cv2.VideoCapture(self.depth_webcam_id)
            if not self.depth_cap.isOpened():
                print(f"Cannot open depth webcam {self.depth_webcam_id}")
                return False
            
            # Initialize gripper camera
            self.gripper_cap = cv2.VideoCapture(self.gripper_webcam_id)
            if not self.gripper_cap.isOpened():
                print(f"Cannot open gripper webcam {self.gripper_webcam_id}")
                return False
            
            # Set FPS (more reliable than setting resolution)
            for cap in [self.rgb_cap, self.depth_cap, self.gripper_cap]:
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("All webcams initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing webcams: {e}")
            return False
    
    def capture_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Capture frames from all webcams.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Captured frames (RGB, Depth, Gripper) or None if failed
        """
        if self.rgb_cap is None or self.depth_cap is None or self.gripper_cap is None:
            print("Webcams not initialized")
            return None
            
        try:
            # Capture frames from all cameras
            rgb_ret, rgb_frame = self.rgb_cap.read()
            depth_ret, depth_frame = self.depth_cap.read()
            gripper_ret, gripper_frame = self.gripper_cap.read()
            
            if not (rgb_ret and depth_ret and gripper_ret):
                print("Failed to capture frames from one or more cameras")
                return None
            
            # Log actual frame sizes for debugging
            rgb_h, rgb_w = rgb_frame.shape[:2]
            depth_h, depth_w = depth_frame.shape[:2]
            gripper_h, gripper_w = gripper_frame.shape[:2]
            target_h, target_w = self.target_size
            
            if (rgb_h, rgb_w) != (target_h, target_w) or (depth_h, depth_w) != (target_h, target_w) or (gripper_h, gripper_w) != (target_h, target_w):
                print(f"Frame sizes - RGB: {rgb_w}x{rgb_h}, Depth: {depth_w}x{depth_h}, Gripper: {gripper_w}x{gripper_h}")
                print(f"Target size: {target_w}x{target_h} (will resize)")
            
            return rgb_frame, depth_frame, gripper_frame
        except Exception as e:
            print(f"Error capturing frames: {e}")
            return None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a webcam frame for model input.
        
        Args:
            frame (np.ndarray): Raw webcam frame
            
        Returns:
            np.ndarray: Preprocessed frame in channels-first format [C, H, W]
        """
        target_h, target_w = self.target_size
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to expected input size
        frame_resized = cv2.resize(frame_rgb, (target_w, target_h))
        
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
            # Add batch dimension [1, D] where D is the number of joints
            state_tensor = state_tensor.unsqueeze(0)
            processed["observation.state"] = state_tensor.to(self.device)
        
        # Handle image observations
        for key in ["observation.images.rgb", "observation.images.depth", "observation.images.gripper"]:
            if key in observation:
                image_tensor = torch.tensor(observation[key], dtype=torch.float32)
                # Image is already in [C, H, W] format from preprocess_frame
                # Add batch dimension (B, C, H, W) where B=1
                image_tensor = image_tensor.unsqueeze(0)
                processed[key] = image_tensor.to(self.device)
            
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
        if not self.initialize_webcams():
            print("Failed to initialize webcams")
            return
        
        print("Starting webcam inference. Press 'q' to quit.")
        
        try:
            while True:
                # Capture frames from all cameras
                frames = self.capture_frames()
                if frames is None:
                    continue
                
                rgb_frame, depth_frame, gripper_frame = frames
                
                # Preprocess frames
                processed_rgb_frame = self.preprocess_frame(rgb_frame)
                processed_depth_frame = self.preprocess_frame(depth_frame)
                processed_gripper_frame = self.preprocess_frame(gripper_frame)
                
                # Create observation
                observation = {
                    "observation.images.rgb": processed_rgb_frame,
                    "observation.images.depth": processed_depth_frame,
                    "observation.images.gripper": processed_gripper_frame
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
                    print(f"Inference failed: {result['error']}")
                
                # Display frames
                cv2.imshow('RGB Input', rgb_frame)
                cv2.imshow('Depth Input', depth_frame)
                cv2.imshow('Gripper Input', gripper_frame)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Inference interrupted by user")
        except Exception as e:
            print(f"Error during webcam inference: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.rgb_cap is not None:
            self.rgb_cap.release()
        if self.depth_cap is not None:
            self.depth_cap.release()
        if self.gripper_cap is not None:
            self.gripper_cap.release()
        cv2.destroyAllWindows()

def create_sample_joint_state():
    """Create a sample joint state for testing."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)

def main():
    """Main function demonstrating webcam usage."""
    print("Webcam LeRobot Inference Demo")
    print("=" * 35)
    
    # Initialize webcam inference engine
    inference_engine = WebcamInference("ISdept/piper_arm")
    
    # Try to load model
    print("Loading model...")
    if not inference_engine.load_model():
        print("Could not load trained model. Using demo mode.")
        print("\nTo use with a real trained model:")
        print("1. Train a model using train.py")
        print("2. Ensure the model is saved to ./model_output")
        print("3. Make sure lerobot is installed")
        
        # Demonstrate with webcams but without model
        print("\n" + "=" * 35)
        print("DEMO MODE - Webcams Only")
        print("=" * 35)
        
        if inference_engine.initialize_webcams():
            print("Webcams initialized successfully")
            print("Displaying webcam feeds. Press 'q' to quit.")
            
            try:
                while True:
                    frames = inference_engine.capture_frames()
                    if frames is not None:
                        rgb_frame, depth_frame, gripper_frame = frames
                        # Show frames
                        cv2.imshow('RGB Demo', rgb_frame)
                        cv2.imshow('Depth Demo', depth_frame)
                        cv2.imshow('Gripper Demo', gripper_frame)
                        
                        # Simulate processing
                        if np.random.random() < 0.02:  # Print every ~50 frames
                            print("Processing frames...")
                        
                        # Break on 'q' key press
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            except KeyboardInterrupt:
                print("Demo interrupted by user")
            finally:
                inference_engine.cleanup()
        else:
            print("Failed to initialize webcams")
        
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
