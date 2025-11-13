"""
Inference script for LeRobot policies.
This script demonstrates how to load a trained policy and use it for inference.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json

class LeRobotInference:
    """A class to handle inference with trained LeRobot policies."""
    
    def __init__(self, model_path: str, dataset_id: str = "ISdept/piper_arm"):
        """
        Initialize the inference engine.
        
        Args:
            model_path (str): Path to the trained model directory
            dataset_id (str): ID of the dataset used for training
        """
        self.model_path = Path(model_path)
        self.dataset_id = dataset_id
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        # Handle device selection with MPS support
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
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
            # Try to load from local dataset first
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                dataset = LeRobotDataset(self.dataset_id)
                features = dataset.features
                dataset_metadata = dataset.meta
            except:
                # Fallback to loading from HuggingFace
                dataset_metadata = LeRobotDatasetMetadata(self.dataset_id, force_cache_sync=True, revision="main")
                features = dataset_metadata.features
                
            features = dataset_to_policy_features(features)
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
                state_dict = load_file(safetensors_path, device=str(self.device))
                self.policy.load_state_dict(state_dict)
            elif pytorch_path.exists():
                print(f"Loading model from pytorch format: {pytorch_path}")
                self.policy.load_state_dict(torch.load(pytorch_path, map_location=self.device))
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
        if "observation.state" in observation:
            state_tensor = torch.tensor(observation["observation.state"], dtype=torch.float32)
            # Add batch dimension
            state_tensor = state_tensor.unsqueeze(0)
            processed["observation.state"] = state_tensor.to(self.device)
        
        # Handle rgb camera image observation
        if "observation.images.rgb" in observation:
            image_tensor = torch.tensor(observation["observation.images.rgb"], dtype=torch.float32)
            # Add batch dimension (B, C, H, W)
            image_tensor = image_tensor.unsqueeze(0)
            processed["observation.images.rgb"] = image_tensor.to(self.device)
            
        # Handle gripper camera image observation
        if "observation.images.gripper" in observation:
            image_tensor = torch.tensor(observation["observation.images.gripper"], dtype=torch.float32)
            # Add batch dimension (B, C, H, W)
            image_tensor = image_tensor.unsqueeze(0)
            processed["observation.images.gripper"] = image_tensor.to(self.device)
            
        # Handle depth camera image observation
        if "observation.images.depth" in observation:
            image_tensor = torch.tensor(observation["observation.images.depth"], dtype=torch.float32)
            # Add batch dimension (B, C, H, W)
            image_tensor = image_tensor.unsqueeze(0)
            processed["observation.images.depth"] = image_tensor.to(self.device)
            
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
            "timestamp": np.datetime64('now'),
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

def create_sample_observation():
    """Create a sample observation for testing."""
    return {
        "observation.state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),  # 7-DOF joint positions
        "observation.images.rgb": np.random.rand(3, 400, 640).astype(np.float32),  # RGB camera image
        "observation.images.gripper": np.random.rand(3, 400, 640).astype(np.float32),  # Gripper camera image
        "observation.images.depth": np.random.rand(3, 400, 640).astype(np.float32)   # Depth camera image
    }

def main():
    """Main function demonstrating usage."""
    print("LeRobot Inference Demo")
    print("=" * 30)
    
    # Initialize inference engine
    inference_engine = LeRobotInference("src/model_output")
    
    # Try to load model
    print("Loading model...")
    if not inference_engine.load_model():
        print("Could not load trained model. Using dummy mode for demonstration.")
        print("\nTo use with a real trained model:")
        print("1. Train a model using train.py")
        print("2. Ensure the model is saved to ./model_output")
        print("3. Make sure lerobot is installed")
        
        # Demonstrate with dummy data
        print("\n" + "=" * 30)
        print("DEMO MODE - Simulated Inference")
        print("=" * 30)
        
        # Create sample observation
        observation = create_sample_observation()
        print(f"Input observation state: {observation['observation.state']}")
        
        # Simulate prediction
        action = observation["observation.state"] + np.random.normal(0, 0.01, len(observation["observation.state"]))
        print(f"Simulated action: {action}")
        
        return
    
    # Run actual inference
    print("\n" + "=" * 30)
    print("RUNNING INFERENCE")
    print("=" * 30)
    
    # Create sample observation
    observation = create_sample_observation()
    print(f"Input observation state: {observation['observation.state']}")
    
    # Run inference
    result = inference_engine.run_inference(observation)
    
    if result["success"]:
        action = result["result"]["action"]
        print(f"Predicted action: {action}")
        print(f"Device used: {result['result']['device']}")
    else:
        print(f"Inference failed: {result['error']}")

if __name__ == "__main__":
    main()
