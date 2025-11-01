import torch
from pathlib import Path
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
import numpy as np

def load_model(model_path, dataset_id="ISdept/piper_arm"):
    """
    Load a trained diffusion policy model for inference.
    
    Args:
        model_path (str): Path to the trained model directory
        dataset_id (str): ID of the dataset used for training
    
    Returns:
        tuple: (policy, preprocessor, postprocessor)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset metadata to get input/output shapes and stats
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # Recreate the config used during training
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    
    # Initialize the policy
    policy = DiffusionPolicy(cfg)
    
    # Load the trained weights
    # First try safetensors format, fallback to pytorch format
    safetensors_path = Path(model_path) / "model.safetensors"
    pytorch_path = Path(model_path) / "pytorch_model.bin"
    
    if safetensors_path.exists():
        print(f"Loading model from safetensors format: {safetensors_path}")
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path, device=str(device))
        policy.load_state_dict(state_dict)
    elif pytorch_path.exists():
        print(f"Loading model from pytorch format: {pytorch_path}")
        policy.load_state_dict(torch.load(pytorch_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    policy.eval()
    policy.to(device)
    
    # Load preprocessors
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    
    return policy, preprocessor, postprocessor

def predict_action(policy, preprocessor, postprocessor, observation):
    """
    Predict the next action given an observation.
    
    Args:
        policy: The trained policy model
        preprocessor: Preprocessor for input data
        postprocessor: Postprocessor for output data
        observation (dict): Dictionary containing observation data
        
    Returns:
        dict: Predicted action
    """
    device = next(policy.parameters()).device
    
    # Prepare batch
    batch = {}
    
    # Handle state observation
    if "observation.state" in observation:
        batch["observation.state"] = torch.tensor(observation["observation.state"], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Handle front camera image observation
    if "observation.images.front_camera" in observation:
        batch["observation.images.front_camera"] = torch.tensor(observation["observation.images.front_camera"], dtype=torch.float32).unsqueeze(0).to(device)
        
    # Handle rear camera image observation
    if "observation.images.rear_camera" in observation:
        batch["observation.images.rear_camera"] = torch.tensor(observation["observation.images.rear_camera"], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Apply preprocessing
    batch = preprocessor(batch)
    
    # Get prediction
    with torch.no_grad():
        action = policy.select_action(batch)
    
    # Apply postprocessing
    action = postprocessor(action)
    
    return action

def run_inference(model_path, observation, dataset_id="ISdept/piper_arm"):
    """
    Run inference using a trained model.
    
    Args:
        model_path (str): Path to the trained model directory
        observation (dict): Dictionary containing observation data
        dataset_id (str): ID of the dataset used for training
        
    Returns:
        dict: Predicted action
    """
    # Load model
    policy, preprocessor, postprocessor = load_model(model_path, dataset_id)
    
    # Run prediction
    action = predict_action(policy, preprocessor, postprocessor, observation)
    
    return action

# Example usage
if __name__ == "__main__":
    # Example observation (you would replace this with real sensor data)
    observation = {
        "observation.state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),  # 7-DOF joint positions
        "observation.images.front_camera": np.random.rand(3, 480, 640).astype(np.float32),  # Front camera image
        "observation.images.rear_camera": np.random.rand(3, 480, 640).astype(np.float32)   # Rear camera image
    }
    
    # Path to trained model (update this to your actual model path)
    model_path = "src/model_output"
    
    try:
        # Run inference
        predicted_action = run_inference(model_path, observation)
        print("Predicted action:", predicted_action)
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Make sure you have a trained model at the specified path.")
