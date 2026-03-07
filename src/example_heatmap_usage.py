#!/usr/bin/env python3
"""
Example script demonstrating heatmap usage with dummy data.
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import cv2

# Import our custom modules
from models.transformer_flow_matching.spatial_softmax import SpatialSoftmax, save_heatmap_visualization


def create_dummy_feature_maps(batch_size=1, channels=32, height=14, width=14):
    """Create dummy feature maps for testing."""
    # Create feature maps with some structured patterns
    feature_maps = torch.randn(batch_size, channels, height, width)
    
    # Add some "interesting" patterns to make heatmaps more visible
    for b in range(batch_size):
        for c in range(channels):
            # Create a gaussian blob at a random location
            center_y = np.random.randint(2, height-2)
            center_x = np.random.randint(2, width-2)
            
            y_coords = torch.arange(height).float()
            x_coords = torch.arange(width).float()
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Create gaussian blob
            sigma = 1.5
            blob = torch.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))
            feature_maps[b, c] = blob + 0.1 * torch.randn_like(blob)
    
    return feature_maps


def test_spatial_softmax_heatmaps():
    """Test spatial softmax with heatmap generation."""
    print("Testing SpatialSoftmax with heatmap generation...")
    
    # Create dummy feature maps
    feature_maps = create_dummy_feature_maps(batch_size=2, channels=5, height=20, width=20)
    print(f"Feature maps shape: {feature_maps.shape}")
    
    # Create spatial softmax with fixed temperature
    spatial_softmax = SpatialSoftmax(temperature=0.1, learn_temperature=False)
    
    # Get coordinates and heatmaps
    coords, heatmaps = spatial_softmax.forward_with_heatmaps(feature_maps)
    print(f"Coordinates shape: {coords.shape}")
    print(f"Heatmaps shape: {heatmaps.shape}")
    print(f"Coordinate range: [{coords.min():.3f}, {coords.max():.3f}]")
    
    # Save some example heatmaps
    output_dir = Path("example_heatmaps")
    output_dir.mkdir(exist_ok=True)
    
    # Create dummy images for overlay
    dummy_images = []
    for b in range(feature_maps.shape[0]):
        # Create a simple gradient image
        h, w = feature_maps.shape[2], feature_maps.shape[3]
        y_coords = np.linspace(0, 1, h)
        x_coords = np.linspace(0, 1, w)
        y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
        dummy_image = np.stack([x_grid, y_grid, 0.5 * np.ones_like(x_grid)], axis=2)
        dummy_images.append(dummy_image)
    
    # Save heatmaps for first batch item
    for c in range(min(3, coords.shape[1])):  # Save first 3 channels
        heatmap = heatmaps[0, c]  # First batch item
        coord = coords[0, c]      # Corresponding coordinate
        
        # Save heatmap visualization
        output_path = output_dir / f"example_heatmap_channel_{c}.png"
        save_heatmap_visualization(
            heatmap, 
            dummy_images[0], 
            output_path,
            point_coords=coord.cpu().numpy()
        )
        print(f"Saved heatmap visualization to: {output_path}")
    
    return coords, heatmaps


def test_vision_encoder_integration():
    """Test vision encoder integration (simplified)."""
    print("\nTesting VisionEncoder integration...")
    
    # This would normally involve loading a full model and processing real images
    # For this example, we'll just show the concept
    
    print("VisionEncoder now includes:")
    print("  - Spatial head with Conv2d + SpatialSoftmax")
    print("  - Point projection to d_model dimensional tokens")
    print("  - Integrated heatmap generation for debugging")
    print("  - generate_debug_heatmaps() method for visualization")
    
    # Show example of how heatmaps are generated in the vision encoder
    print("\nExample heatmap generation flow:")
    print("1. Input image -> ViT backbone -> patch features (B, 768, 14, 14)")
    print("2. patch features -> Conv2d(768->32) -> reduced features (B, 32, 14, 14)")
    print("3. reduced features -> SpatialSoftmax -> coordinates (B, 32, 2)")
    print("4. coordinates -> Linear(2->d_model) -> point tokens (B, 32, d_model)")
    print("5. point tokens concatenated with ViT tokens for final representation")


def main():
    """Main function to run examples."""
    print("=== Heatmap Usage Examples ===\n")
    
    # Test spatial softmax with heatmaps
    coords, heatmaps = test_spatial_softmax_heatmaps()
    
    # Test vision encoder integration
    test_vision_encoder_integration()
    
    print(f"\n=== Results ===")
    print(f"Successfully generated heatmaps for {heatmaps.shape[0]} samples")
    print(f"Each sample has {heatmaps.shape[1]} channels")
    print(f"Example coordinates (first sample, first 3 channels):")
    for i in range(min(3, coords.shape[1])):
        x, y = coords[0, i].cpu().numpy()
        print(f"  Channel {i}: ({x:.3f}, {y:.3f})")
    
    print(f"\nCheck the 'example_heatmaps/' directory for visualization files.")


if __name__ == "__main__":
    main()
