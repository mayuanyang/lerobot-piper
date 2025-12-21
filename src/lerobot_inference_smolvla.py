"""
Inference script for LeRobot policies.
This script demonstrates how to load a trained policy and use it for inference.
"""

from models.smooth_diffusion.custom_diffusion_config import CustomDiffusionConfig
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json
import random

# ... (LeRobot imports assumed to be correct) ...
# Import JointSmoothDiffusion instead of DiffusionPolicy
from models.smooth_diffusion.joint_smooth_diffusion import JointSmoothDiffusion
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.datasets.utils import hw_to_dataset_features
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
                        
            
            # Initialize the policy
            print("Initializing policy...")
            self.policy = SmolVLAPolicy.from_pretrained(self.model_id) #JointSmoothDiffusion.from_pretrained(self.model_id)
                        
            
            self.policy.eval()
            self.policy.to(self.device)
            
            # Debug: Print the image features expected by the policy
            print(f"Policy config image features: {self.policy.config.image_features}")
            # print(f"Policy config crop_shape: {self.policy.config.crop_shape}")
            # print(f"Policy config use_group_norm: {self.policy.config.use_group_norm}")
            # print(f"Policy config pretrained_backbone_weights: {self.policy.config.pretrained_backbone_weights}")
            print('The output feature', output_features)
            
            # Load preprocessors with proper dataset statistics
            print("Loading preprocessors...")
            if dataset_stats is not None:
                self.preprocessor, self.postprocessor = make_pre_post_processors(self.policy.config, dataset_stats=dataset_stats)
                print('The dataset statistics have been loaded successfully for preprocessing.')
            else:
                print("WARNING: Loading preprocessors without dataset statistics. Results may be incorrect.")
                self.preprocessor, self.postprocessor = make_pre_post_processors(self.policy.config, dataset_stats=None)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    
    def preprocess_observation(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
                
        # --- 1. Handle state observation ---
        input_observation = {}
        if "observation.state" in observation:
            state_np = observation["observation.state"]
            # Ensure state is in correct format [B, T, D] where B=1, T=10, D=7
            
            # Keep all 7 joints as the dataset contains 7-DOF joint positions
            # Already in correct format [1, T, D]
            state_tensor = torch.tensor(state_np, dtype=torch.float32)
            
            input_observation["observation.state"] = state_tensor.to(self.device)
            
        
        # --- 2. Handle image observations ---
        for key in ["observation.images.front", "observation.images.right", "observation.images.gripper"]:
            if key in observation:
                img_tensor = observation[key]
                input_observation[key] = img_tensor.to(self.device)
                
        # For SmolVLA policy, we need to pass the task as a string in the 'task' key
        # The processor pipeline will handle tokenizing it into tokens and attention masks
        input_observation['task'] = 'pick and place cube into the container'
        return input_observation
    
    def predict_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.policy is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
                
        batch = self.preprocess_observation(observation)
        
        batch = self.preprocessor(batch)
                        
        action = self.policy.select_action(batch)
                
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
        "observation.images.front": np.random.rand(3, 400, 640).astype(np.float32),  # Front camera image
        "observation.images.gripper": np.random.rand(3, 400, 640).astype(np.float32),  # Gripper camera image
        "observation.images.right": np.random.rand(3, 400, 640).astype(np.float32)   # Right camera image
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
