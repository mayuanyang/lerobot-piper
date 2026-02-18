#!/usr/bin/env python3
"""
Simple test script to verify heatmap inference functionality.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import torchvision.transforms as transforms

# Import our custom modules
from models.transformer_diffusion.transformer_diffusion_policy import TransformerDiffusionPolicy
from models.transformer_diffusion.processor_transformer_diffusion import make_pre_post_processors
from models.transformer_diffusion.spatial_softmax import save_heatmap_visualization


def test_heatmap_inference(model_path, image_path, output_dir, state=None):
    """Test heatmap inference on a single image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    policy = TransformerDiffusionPolicy.from_pretrained(model_path)
    policy.eval()
    policy.to(device)
    
    # Get model config
    config = policy.config
    
    # Create preprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=None,
        add_grid_overlay=True,
        grid_overlay_cameras=["front"]
    )
    
    # Move preprocessors to device
    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)
    if isinstance(postprocessor, torch.nn.Module):
        postprocessor.to(device)
        
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Load and process image
    print(f"Processing image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Apply transformations
    tensor_image = transform(pil_image)
    
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0).to(device)
    
    # Handle state information
    if state is not None:
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().to(device)
        else:
            state_tensor = state.float().to(device)
        
        # Ensure proper shape (1, 1, state_dim)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
        elif state_tensor.dim() == 2:
            state_tensor = state_tensor.unsqueeze(0)
    else:
        # Use zero state if none provided
        state_tensor = torch.zeros(1, 1, config.state_dim).to(device)
    
    # Create batch dictionary
    batch = {
        "observation.images.front": tensor_image,
        "observation.state": state_tensor
    }
    
    # Preprocess
    batch = preprocessor(batch)
    
    # Enable heatmap saving
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_path / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    
    # Enable heatmap saving in vision encoders
    for encoder in policy.model.image_encoders.values():
        encoder.save_heatmaps = True
        encoder.heatmap_save_dir = str(heatmap_dir)
    
    # Get model predictions with heatmaps
    actions, spatial_outputs = policy.model.get_condition(batch, generate_heatmaps=True)
    
    # Disable heatmap saving
    for encoder in policy.model.image_encoders.values():
        encoder.save_heatmaps = False
    
    # Postprocess actions
    actions = postprocessor(actions)
    
    print(f"Actions shape: {actions.shape}")
    print(f"Spatial outputs keys: {list(spatial_outputs.keys())}")
    
    # Save results
    results_file = output_path / "inference_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Actions shape: {actions.shape}\n")
        f.write(f"Actions (first 10 values): {actions.flatten()[:10].cpu().numpy()}\n")
        f.write(f"Spatial outputs keys: {list(spatial_outputs.keys())}\n")
    
    print(f"Results saved to: {results_file}")
    print(f"Heatmaps saved to: {heatmap_dir}")
    
    return actions, spatial_outputs


def main():
    parser = argparse.ArgumentParser(description="Test heatmap inference on a single image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--state", type=str, help="State values as comma-separated floats (e.g., '0.1,0.2,0.3,...')")
    
    args = parser.parse_args()
    
    # Parse state if provided
    state = None
    if args.state:
        try:
            state_values = [float(x.strip()) for x in args.state.split(',')]
            state = np.array(state_values, dtype=np.float32)
            print(f"Using provided state: {state}")
        except Exception as e:
            print(f"Error parsing state values: {e}")
            print("Using zero state instead.")
    
    test_heatmap_inference(args.model_path, args.image_path, args.output_dir, state)


if __name__ == "__main__":
    main()
