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
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
# ...

class LeRobotInference:
    """A class to handle inference with trained LeRobot policies."""
    
    def __init__(self, model_id: str, dataset_id: str = "ISdept/piper_arm"):
        self.model_id = model_id
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
        try:

            # Load dataset metadata to get input/output shapes and stats
            print("Loading dataset metadata...")
            dataset_metadata = LeRobotDatasetMetadata(self.dataset_id, force_cache_sync=True, revision="main")

            features = dataset_metadata.features
            features = dataset_to_policy_features(features)

            dataset_stats = dataset_metadata.stats  # This is what was missing!
                      
            output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
            input_features = {key: ft for key, ft in features.items() if key not in output_features}
            
            # Recreate the config used during training
            print("Creating policy configuration...")
            # Match the configuration from train.py
            cfg = DiffusionConfig(input_features=input_features, output_features=output_features, n_obs_steps=10, horizon=16)
            
            # Initialize the policy
            print("Initializing policy...")
            self.policy = DiffusionPolicy.from_pretrained("ISdept/piper_arm")
            
            
            self.policy.eval()
            self.policy.to(self.device)
            
            # Load preprocessors with proper dataset statistics
            print("Loading preprocessors...")
            if dataset_stats is not None:
                self.preprocessor, self.postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_stats)
                print('The dataset statistics have been loaded successfully for preprocessing.')
            else:
                print("WARNING: Loading preprocessors without dataset statistics. Results may be incorrect.")
                self.preprocessor, self.postprocessor = make_pre_post_processors(cfg, dataset_stats=None)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    
    def preprocess_observation(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
                
        # --- 1. Handle state observation (must be (T, D)) ---
        input_observation = {}
        if "observation.state" in observation:
            state_np = observation["observation.state"]
            state_tensor = torch.tensor(state_np, dtype=torch.float32)
            input_observation["observation.state"] = state_tensor.to(self.device)
            
        
        # --- 2. Helper for Image Preprocessing (convert to (T, C, H, W)) ---
        def process_image(key):
            if key in observation:
                # 1. Convert NumPy (T, H, W, C) to PyTorch tensor
                image_tensor = torch.tensor(observation[key], dtype=torch.float32) # (T, H, W, C)
                
                if len(image_tensor.shape) == 4:
                    # Transpose from (T, H, W, C) to (T, C, H, W)
                    image_tensor = image_tensor.permute(0, 3, 1, 2)
                
                input_observation[key] = image_tensor.to(self.device)

        # Apply helper to all image streams
        process_image("observation.images.rgb")
        process_image("observation.images.gripper")
        process_image("observation.images.depth")
            
        return input_observation
    
    # ... (predict_action and run_inference methods remain unchanged) ...
    def predict_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.policy is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        batch = self.preprocess_observation(observation)
        
        print('observation.state shape:', batch['observation.state'].shape if 'observation.state' in batch else 'No state')
        print('observation.images.rgb shape:', batch['observation.images.rgb'].shape if 'observation.images.rgb' in batch else 'No state')
        print('observation.images.depth shape:', batch['observation.images.depth'].shape if 'observation.images.depth' in batch else 'No state')
        print('observation.images.gripper shape:', batch['observation.images.gripper'].shape if 'observation.images.gripper' in batch else 'No state')
        
        batch = self.preprocessor(batch)
        
        with torch.no_grad():
            action = self.policy.select_action(batch)
            
        print('the raw action output shape:', action.shape)
        
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
      
        result = self.predict_action(observation)
        return {
            "success": True,
            "result": result
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
