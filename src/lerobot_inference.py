"""
Inference script for LeRobot policies.
This script demonstrates how to load a trained policy and use it for inference.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json

# ... (LeRobot imports assumed to be correct) ...
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
# ...

class LeRobotInference:
    """A class to handle inference with trained LeRobot policies."""
    
    def __init__(self, model_path: str, dataset_id: str = "ISdept/piper_arm"):
        self.model_path = Path(model_path)
        self.dataset_id = dataset_id
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
    def load_model(self) -> bool:
        # ... (load_model method remains unchanged) ...
        try:
            # Check if model directory exists
            if not self.model_path.exists():
                print(f"Model directory {self.model_path} not found.")
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
            cfg = DiffusionConfig(input_features=input_features, output_features=output_features, n_obs_steps=4, horizon=16)
            
            # Initialize the policy
            print("Initializing policy...")
            self.policy = DiffusionPolicy(cfg)
            
            # Load the trained weights
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
        Ensures all tensors are in the shape expected by the LeRobot Preprocessor.
        Images are converted from (T, H, W, C) to (T, C, H, W).
        """
        processed = {}
        
        # --- 1. Handle state observation (must be (T, D)) ---
        if "observation.state" in observation:
            try:
                state_np = observation["observation.state"]
                state_tensor = torch.tensor(state_np, dtype=torch.float32)
                # Remove the unsqueeze(0) to match image tensor dimensions
                # Both state and image tensors should have the same number of dimensions
                # before the preprocessor adds batch dimensions
                if len(state_tensor.shape) == 2:  
                    # Keep as (T, D) to match image tensor dimensions (T, C, H, W)
                    pass
                processed["observation.state"] = state_tensor.to(self.device)
            except Exception as e:
                print(f"Error processing 'observation.state': {e}")
        
        # --- 2. Helper for Image Preprocessing (convert to (T, C, H, W)) ---
        def process_image(key):
            if key in observation:
                # 1. Convert NumPy (T, H, W, C) to PyTorch tensor
                image_tensor = torch.tensor(observation[key], dtype=torch.float32) # (T, H, W, C)
                
                if len(image_tensor.shape) == 4:
                    # Transpose from (T, H, W, C) to (T, C, H, W)
                    image_tensor = image_tensor.permute(0, 3, 1, 2)
                    
                    # Do NOT add the batch dimension. 
                    # The preprocessor will handle converting (T, C, H, W) to (1, T, C, H, W)
                    # or flattening to (T, C, H, W) for the image encoder.
                
                processed[key] = image_tensor.to(self.device)

        # Apply helper to all image streams
        process_image("observation.images.rgb")
        process_image("observation.images.gripper")
        process_image("observation.images.depth")
            
        return processed
    
    # ... (predict_action and run_inference methods remain unchanged) ...
    def predict_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.policy is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        batch = self.preprocess_observation(observation)
        
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        
        with torch.no_grad():
            action = self.policy.select_action(batch)
        
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        
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
    # ... (create_sample_observation remains the same) ...
    return {
        "observation.state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),  # 7-DOF joint positions
        "observation.images.rgb": np.random.rand(3, 400, 640).astype(np.float32),  # RGB camera image
        "observation.images.gripper": np.random.rand(3, 400, 640).astype(np.float32),  # Gripper camera image
        "observation.images.depth": np.random.rand(3, 400, 640).astype(np.float32)   # Depth camera image
    }

def main():
    # ... (main function remains the same) ...
    print("LeRobot Inference Demo")
    print("=" * 30)
    
    inference_engine = LeRobotInference("src/model_output")
    
    print("Loading model...")
    if not inference_engine.load_model():
        print("Could not load trained model. Using dummy mode for demonstration.")
        return
    
    print("\n" + "=" * 30)
    print("RUNNING INFERENCE")
    print("=" * 30)
    
    observation = create_sample_observation()
    print(f"Input observation state: {observation['observation.state']}")
    
    result = inference_engine.run_inference(observation)
    
    if result["success"]:
        action = result["result"]["action"]
        print(f"Predicted action: {action}")
        print(f"Device used: {result['result']['device']}")
    else:
        print(f"Inference failed: {result['error']}")

if __name__ == "__main__":
    main()
