import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax layer that computes spatial softmax over feature maps
    and returns coordinates for use in neural networks.
    
    Supports channel reduction to control the number of output keypoints.
    """
    
    def __init__(self, input_shape=None, num_kp=None, temperature=1.0, learn_temperature=False):
        """
        Args:
            input_shape (tuple, optional): (C, H, W) input feature map shape. Required if num_kp is specified.
            num_kp (int, optional): Number of keypoints in output. If None, output will have the same 
                                  number of channels as input.
            temperature (float): Temperature parameter for softmax.
            learn_temperature (bool): Whether to make temperature a learnable parameter.
        """
        super(SpatialSoftmax, self).__init__()
        
        self.learn_temperature = learn_temperature
        if learn_temperature:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = temperature
            
        # Channel reduction functionality
        if num_kp is not None:
            if input_shape is None:
                raise ValueError("input_shape must be provided when num_kp is specified")
            self._in_c, self._in_h, self._in_w = input_shape
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = None  # Will be determined from input
            
        # Precompute coordinate grid for efficiency
        if input_shape is not None:
            _, H, W = input_shape
            # Create coordinate grids using numpy (similar to lerobot approach)
            pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(-1.0, 1.0, H))
            pos_x = torch.from_numpy(pos_x.reshape(H * W, 1)).float()
            pos_y = torch.from_numpy(pos_y.reshape(H * W, 1)).float()
            # Register as buffer so it's moved to the correct device
            self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))
        else:
            self.pos_grid = None
            
    def forward(self, feature_maps):
        """
        Apply spatial softmax to feature maps.
        
        Args:
            feature_maps: (B, C, H, W) feature maps
            
        Returns:
            coords: (B, K, 2) normalized coordinates in range [-1, 1], where K is the number of keypoints
        """
        coords, _ = self.forward_with_heatmaps(feature_maps)
        return coords
    
    def forward_with_heatmaps(self, feature_maps):
        """
        Apply spatial softmax to feature maps and return both coordinates and heatmaps.
        
        Args:
            feature_maps: (B, C, H, W) feature maps
            
        Returns:
            coords: (B, K, 2) normalized coordinates in range [-1, 1], where K is the number of keypoints
            heatmaps: (B, K, H, W) softmax heatmaps for visualization (K is output channels)
        """
        B, C, H, W = feature_maps.shape
        device = feature_maps.device
        
        # Apply channel reduction if specified
        if self.nets is not None:
            feature_maps = self.nets(feature_maps)
            out_c = self._out_c
        else:
            out_c = C
            
        # Update pos_grid if it wasn't precomputed or shape doesn't match
        if self.pos_grid is None or self.pos_grid.shape[0] != H * W:
            # Create coordinate grids dynamically (your original approach)
            y_coords = torch.linspace(-1, 1, H, device=device)
            x_coords = torch.linspace(-1, 1, W, device=device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Expand grids to match batch and channel dimensions
            y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(B, out_c, -1, -1)  # (B, out_c, H, W)
            x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(B, out_c, -1, -1)  # (B, out_c, H, W)
            
            # Flatten spatial dimensions
            flat_features = feature_maps.view(B, out_c, -1)  # (B, out_c, H*W)
            
            # Apply softmax with temperature
            temp = self.temperature if not self.learn_temperature else self.temperature.clamp(min=1e-6)
            softmax_features = F.softmax(flat_features / temp, dim=-1)  # (B, out_c, H*W)
            
            # Reshape back to spatial dimensions for heatmap
            heatmaps = softmax_features.view(B, out_c, H, W)  # (B, out_c, H, W)
            
            # Compute expected coordinates using element-wise operations (your approach)
            y_coords_expected = (softmax_features * y_grid.reshape(B, out_c, -1)).sum(dim=-1)  # (B, out_c)
            x_coords_expected = (softmax_features * x_grid.reshape(B, out_c, -1)).sum(dim=-1)  # (B, out_c)
            
            # Stack coordinates
            coords = torch.stack([x_coords_expected, y_coords_expected], dim=-1)  # (B, out_c, 2)
        else:
            # Use precomputed grid with matrix multiplication (lerobot approach)
            # Ensure pos_grid is on the correct device
            pos_grid = self.pos_grid.to(device)
            
            # Flatten spatial dimensions
            flat_features = feature_maps.reshape(B * out_c, H * W)  # (B * out_c, H*W)
            
            # Apply softmax with temperature
            temp = self.temperature if not self.learn_temperature else self.temperature.clamp(min=1e-6)
            attention = F.softmax(flat_features / temp, dim=-1)  # (B * out_c, H*W)
            
            # Matrix multiplication with precomputed coordinate grid
            expected_xy = attention @ pos_grid  # (B * out_c, 2)
            
            # Reshape to (B, out_c, 2)
            coords = expected_xy.view(B, out_c, 2)
            
            # Reshape attention back to spatial dimensions for heatmap
            heatmaps = attention.view(B, out_c, H, W)  # (B, out_c, H, W)
        
        return coords, heatmaps


def save_heatmap_visualization(heatmap, image, output_path, point_coords=None):
    """
    Save a visualization of the heatmap overlaid on the image.
    
    Args:
        heatmap: (H, W) heatmap tensor
        image: (H, W, 3) or (H, W) image tensor/array
        output_path: path to save the visualization
        point_coords: optional (x, y) coordinates to mark on the heatmap
    """
    # Convert tensors to numpy if needed
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Normalize heatmap to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Convert to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Handle image formats
    if len(image.shape) == 2:
        # Grayscale image, convert to BGR
        image_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # RGB image, convert to BGR
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # Assume it's already in BGR format
        image_bgr = image.astype(np.uint8) if image.max() > 1 else (image * 255).astype(np.uint8)
    
    # Resize heatmap to match image size if needed
    if heatmap_colormap.shape[:2] != image_bgr.shape[:2]:
        heatmap_colormap = cv2.resize(heatmap_colormap, (image_bgr.shape[1], image_bgr.shape[0]))
    
    # Blend heatmap with image
    blended = cv2.addWeighted(image_bgr, 0.6, heatmap_colormap, 0.4, 0)
    
    # Mark point coordinates if provided
    if point_coords is not None:
        x, y = point_coords
        # Convert normalized coordinates to pixel coordinates
        px = int((x + 1) * image_bgr.shape[1] / 2)
        py = int((y + 1) * image_bgr.shape[0] / 2)
        cv2.circle(blended, (px, py), 5, (0, 255, 0), -1)
        cv2.circle(blended, (px, py), 7, (0, 0, 0), 2)
    
    # Save visualization
    cv2.imwrite(str(output_path), blended)


# Example usage function
def create_spatial_softmax_example():
    """Create an example to demonstrate spatial softmax functionality."""
    # Create dummy feature maps
    B, C, H, W = 1, 5, 20, 20
    feature_maps = torch.randn(B, C, H, W)
    
    # Apply spatial softmax
    spatial_softmax = SpatialSoftmax(temperature=0.1)
    coords = spatial_softmax(feature_maps)
    
    print(f"Feature maps shape: {feature_maps.shape}")
    print(f"Coordinates shape: {coords.shape}")
    print(f"Coordinate range: [{coords.min():.3f}, {coords.max():.3f}]")
    
    return coords


if __name__ == "__main__":
    coords = create_spatial_softmax_example()
