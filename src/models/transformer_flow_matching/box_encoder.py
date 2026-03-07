import torch
from torch import nn
import math
import numpy as np


class BoxEncoder(nn.Module):
    """Box encoder for processing bounding box data with enhanced features."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_bounding_boxes_per_camera = 2
        self.box_token_scale = nn.Parameter(torch.tensor(3.0))
        
        # Category embedding for categorical features
        self.category_embedding = nn.Embedding(3, config.d_model)  # Assuming 3 categories
        
        # Linear projections for different features
        self.geom_proj = nn.Sequential(
            nn.Linear(10, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.d_model)
        )  # Geometric features: [x1, y1, x2, y2, width, height, center_x, center_y, area, aspect_ratio]
        
        self.conf_proj = nn.Linear(1, config.d_model)  # Confidence
        self.pres_proj = nn.Linear(1, config.d_model)   # Presence (if used in the future)
        self.center_proj = nn.Linear(2, config.d_model)  # Center coordinates (center_x, center_y)
        self.missing_box_embedding = nn.Parameter(torch.randn(1, config.d_model))  # Learnable embedding for missing boxes
        
        # Distance token embedding for representing distance between box centers
        self.distance_token_embedding = nn.Linear(1, config.d_model)  # Embedding for distance scalar value
        
        # Camera embedding for distinguishing between different camera views
        self.camera_embedding = nn.Embedding(self.config.num_cameras, config.d_model)  # 3 cameras: gripper, front, right

    def encode_boxes_train(self, box_data, batch, image_width=640.0, image_height=400.0):
        """
        Encode bounding boxes during training.
        
        Args:
            box_data: (B, T_obs, 6, 6) tensor with box data
            batch: The batch dictionary
            image_width: Width of the images
            image_height: Height of the images
            
        Returns:
            bbox_tokens_flat: Encoded box tokens
            distance_tokens_flat: Encoded distance tokens
            coordinates_normalized: Normalized coordinates for visualization
        """
        B, T_obs = box_data.shape[:2]
        
        # Look for any observation.images.xxx in the batch to get dimensions
        for key in batch.keys():
            if key.startswith("observation.images.") and isinstance(batch[key], torch.Tensor):
                # Get the shape of the image tensor
                img_shape = batch[key].shape
                if len(img_shape) >= 4:
                    # Extract height and width (last two dimensions)
                    image_height = float(img_shape[-2])
                    image_width = float(img_shape[-1])
                break  # Use the first image found
        
        # First 4 dimensions are coordinates (x1, y1, x2, y2)
        coordinates = box_data[..., :4]  # (B, T_obs, 6, 4)
        # Additional 2 dimensions are category_id and confidence
        category_id = box_data[..., 4].long()  # (B, T_obs, 6)
        confidence = box_data[..., 5]   # (B, T_obs, 6)
        presence = (coordinates.sum(dim=-1) != 0).float().unsqueeze(-1)  # (B, T_obs, 6, 1)
                    
        # Normalize bounding box coordinates
        normalization_factors = torch.tensor([image_width, image_height, image_width, image_height], 
                                          device=coordinates.device, dtype=coordinates.dtype)
        
        # Normalize the box coordinates
        coordinates_normalized = coordinates / normalization_factors.view(1, 1, 1, 4)
        
        # Reshape to (B, T_obs, 3 cameras, 2 boxes, 4) to process boxes per camera
        B, T_obs, N_boxes, N_coords = coordinates_normalized.shape
        coordinates_normalized = coordinates_normalized.view(B, T_obs, self.config.num_cameras, self.num_bounding_boxes_per_camera, 4)  # (B, T_obs, 3, 2, 4)
        
        # Reshape category_id, confidence
        category_id = category_id.view(B, T_obs, self.config.num_cameras, self.num_bounding_boxes_per_camera)  # (B, T_obs, 3, 2)
        confidence = confidence.view(B, T_obs, self.config.num_cameras, self.num_bounding_boxes_per_camera).unsqueeze(-1)    # (B, T_obs, 3, 2, 1)
        
        # Calculate presence based on whether coordinates are non-zero
        presence = (coordinates_normalized.sum(dim=-1) != 0).float()  # (B, T_obs, 3, 2)
        
        # Sort boxes by presence (real boxes first) and then by category_id to stabilize learning
        # Create composite sorting key: prioritize presence (1 for real boxes, 0 for missing) then category_id
        # We negate presence so that real boxes (1) come before missing boxes (0) after ascending sort
        sorting_key = category_id + (1 - presence) * 1000  # (B, T_obs, 3, 2)
        
        # Create sort indices based on the composite sorting key
        _, sort_indices = torch.sort(sorting_key, dim=-1)  # (B, T_obs, 3, 2)
        
        # Expand sort_indices to match coordinates_normalized shape for gathering
        sort_indices_coords = sort_indices.unsqueeze(-1).expand(-1, -1, -1, -1, coordinates_normalized.size(-1))  # (B, T_obs, 3, 2, 4)
        
        # Sort coordinates_normalized
        coordinates_normalized = torch.gather(coordinates_normalized, dim=-2, index=sort_indices_coords)  # (B, T_obs, 3, 2, 4)
        
        # Expand sort_indices to match other tensors for gathering
        sort_indices_cat = sort_indices  # (B, T_obs, 3, 2)
        sort_indices_conf = sort_indices.unsqueeze(-1)  # (B, T_obs, 3, 2, 1)
        
        # Sort category_id and confidence
        category_id = torch.gather(category_id, dim=-1, index=sort_indices_cat)  # (B, T_obs, 3, 2)
        confidence = torch.gather(confidence, dim=-2, index=sort_indices_conf)  # (B, T_obs, 3, 2, 1)
        
        # Recalculate presence after sorting
        presence = (coordinates_normalized.sum(dim=-1) != 0).float().unsqueeze(-1)  # (B, T_obs, 3, 2, 1)
        
        # Enhance bounding box features with geometric properties
        # Extract coordinates
        x1 = coordinates_normalized[..., 0]  # (B, T_obs, 3, 2)
        y1 = coordinates_normalized[..., 1]  # (B, T_obs, 3, 2)
        x2 = coordinates_normalized[..., 2]  # (B, T_obs, 3, 2)
        y2 = coordinates_normalized[..., 3]  # (B, T_obs, 3, 2)
        
        # Compute derived features
        width = x2 - x1  # (B, T_obs, 3, 2)
        height = y2 - y1  # (B, T_obs, 3, 2)
        center_x = (x1 + x2) * 0.5  # (B, T_obs, 3, 2)
        center_y = (y1 + y2) * 0.5  # (B, T_obs, 3, 2)
        area = width * height  # (B, T_obs, 3, 2)
        aspect_ratio = width / (height + 1e-6)  # Prevent division by zero

        
        # Calculate distances between the two boxes for each camera
        # Reshape to (B, T_obs, 3, 2) to easily access box pairs
        center_x_reshaped = center_x.view(B, T_obs, self.config.num_cameras, self.num_bounding_boxes_per_camera)
        center_y_reshaped = center_y.view(B, T_obs, self.config.num_cameras, self.num_bounding_boxes_per_camera)
        
        # Calculate Euclidean distance between box centers for each camera
        # Distance between box 0 and box 1 for each camera
        dx = center_x_reshaped[:, :, :, 0] - center_x_reshaped[:, :, :, 1]  # (B, T_obs, 3)
        dy = center_y_reshaped[:, :, :, 0] - center_y_reshaped[:, :, :, 1]  # (B, T_obs, 3)
        distances = torch.sqrt(dx**2 + dy**2).unsqueeze(-1)  # (B, T_obs, 3, 1)
        
        # Create distance tokens
        distance_tokens = self.distance_token_embedding(distances)  # (B, T_obs, 3, d_model)
        
        # Flatten distance tokens to (B, T_obs * 3, d_model)
        distance_tokens_flat = distance_tokens.view(B, T_obs * self.config.num_cameras, self.config.d_model)
        
        # Stack geometric features together: [x1, y1, x2, y2, width, height, center_x, center_y, area, aspect_ratio]
        geom_features = torch.stack([x1, y1, x2, y2, width, height, center_x, center_y, area, aspect_ratio], dim=-1)  # (B, T_obs, 3, 2, 10)
        
        # Get category embeddings
        cat_emb = self.category_embedding(category_id)
        
        # Project features and sum them
        geom_proj = self.geom_proj(geom_features)  # (B, T_obs, 3, 2, d_model)
        conf_proj = self.conf_proj(confidence)    # (B, T_obs, 3, 2, d_model)
        
        center = torch.stack([center_x, center_y], dim=-1)
        center_proj = self.center_proj(center)
                    
        # Add camera embedding
        # Camera IDs: 0 for gripper, 1 for front, 2 for right
        camera_ids = torch.arange(self.config.num_cameras, device=box_data.device)  # (3,)
        camera_emb = self.camera_embedding(camera_ids)  # (3, d_model)
        
        # Reshape camera embedding to match bbox_tokens dimensions
        # (3, d_model) -> (1, 1, 3, 1, d_model) to broadcast with (B, T_obs, 3, 2, d_model)
        camera_emb = camera_emb.view(1, 1, self.config.num_cameras, 1, self.config.d_model)
        
        # All tokens
        # Use missing_box_embedding when presence is zero
        missing_mask = (presence == 0)  # (B, T_obs, 3, 2, 1)
        missing_embedding_expanded = self.missing_box_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(2).unsqueeze(2)  # (1, 1, 1, 1, d_model)
        
        # Apply missing box embedding where presence is zero
        bbox_tokens = geom_proj + cat_emb + conf_proj + center_proj + camera_emb
        bbox_tokens = self.box_token_scale * bbox_tokens  # Scale the box tokens to increase their magnitude
        bbox_tokens = torch.where(missing_mask, missing_embedding_expanded, bbox_tokens)
        
        # Flatten to (B, T_obs * 6, d_model)
        bbox_tokens_flat = bbox_tokens.view(B, T_obs * self.config.num_cameras * self.num_bounding_boxes_per_camera, self.config.d_model)
        
        return bbox_tokens_flat, distance_tokens_flat, coordinates_normalized

    def encode_boxes_inference(self, bounding_boxes, cam_index, B_v, T_v, N_v):
        """
        Encode bounding boxes during inference.
        
        Args:
            bounding_boxes: (N_boxes, 4) tensor with box coordinates
            cam_index: Index of the current camera
            B_v, T_v, N_v: Dimensions for reshaping
            
        Returns:
            bbox_tokens_flat: Encoded box tokens
            distance_token: Encoded distance token (if applicable)
        """
        # For inference, we don't have category_id and confidence, so we'll use default values
        N_boxes = bounding_boxes.shape[0]
        device = bounding_boxes.device
        
        category_id = torch.full((N_boxes,), 2, dtype=torch.long, device=device)  # Using category_id = 2 for 'unknown' category
        confidence = torch.ones((N_boxes, 1), device=device)  # (N_boxes, 1)
        
        # Sort boxes by presence (real boxes first) and then by category_id to stabilize learning
        # Calculate presence based on whether coordinates are non-zero
        presence = (bounding_boxes.sum(dim=-1) != 0).float()  # (N_boxes,)
        
        # Create composite sorting key: prioritize presence (1 for real boxes, 0 for missing) then category_id
        # Since all category_ids are 2 in inference, we only sort by presence
        sorting_key = (1 - presence) * 1000  # (N_boxes,)
        
        # Create sort indices based on the composite sorting key
        _, sort_indices = torch.sort(sorting_key, dim=0)  # (N_boxes,)
        
        # Sort bounding_boxes, category_id, and confidence
        bounding_boxes = bounding_boxes[sort_indices]  # (N_boxes, 4)
        category_id = category_id[sort_indices]  # (N_boxes,)
        confidence = confidence[sort_indices]  # (N_boxes, 1)
        presence = presence[sort_indices]  # (N_boxes,)
        
        # Update presence after sorting
        presence = presence.unsqueeze(-1)  # (N_boxes, 1)
                                        
        # Enhance bounding box features with geometric properties for inference
        # Extract coordinates
        x1 = bounding_boxes[:, 0]  # (N_boxes,)
        y1 = bounding_boxes[:, 1]  # (N_boxes,)
        x2 = bounding_boxes[:, 2]  # (N_boxes,)
        y2 = bounding_boxes[:, 3]  # (N_boxes,)
        
        # Compute derived features
        width = x2 - x1  # (N_boxes,)
        height = y2 - y1  # (N_boxes,)
        center_x = (x1 + x2) * 0.5  # (N_boxes,)
        center_y = (y1 + y2) * 0.5  # (N_boxes,)
        area = width * height  # (N_boxes,)
        aspect_ratio = width / (height + 1e-6)  # Prevent division by zero

        # Calculate distance between the two boxes
        distance_token = None
        if N_boxes >= 2:
            dx = center_x[0] - center_x[1]  # Scalar
            dy = center_y[0] - center_y[1]  # Scalar
            distance = torch.sqrt(dx**2 + dy**2).unsqueeze(0)  # (1,)
            
            # Create distance token
            distance_token = self.distance_token_embedding(distance.unsqueeze(0).unsqueeze(0))  # (1, 1, 1, d_model)
            distance_token = distance_token.expand(B_v, T_v, N_v, self.config.d_model)  # (B_v, T_v, N_v, d_model)
        
        # Stack geometric features together: [x1, y1, x2, y2, width, height, center_x, center_y, area, aspect_ratio]
        geom_features = torch.stack([x1, y1, x2, y2, width, height, center_x, center_y, area, aspect_ratio], dim=-1)  # (N_boxes, 10)
        geom_features = geom_features.unsqueeze(0).unsqueeze(0)  # (1, 1, N_boxes, 10)
        
        # Get category embeddings
        cat_emb = self.category_embedding(category_id)
        cat_emb = cat_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, N_boxes, d_cat)
        
        # Project confidence
        confidence = confidence.unsqueeze(0).unsqueeze(0)  # (1, 1, N_boxes, 1)
                                        
        center = torch.stack([center_x, center_y], dim=-1).unsqueeze(0).unsqueeze(0)
        
        # Camera embedding for distinguishing between different camera views
        camera_id = torch.tensor([cam_index], device=device)  # (1,)
        camera_emb = self.camera_embedding(camera_id).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, d_model)
        
        # Repeat the camera embedding for all boxes
        camera_emb_for_boxes = camera_emb.repeat(1, 1, N_boxes, 1)  # (1, 1, N_boxes, d_model)
        
        # Project features and sum them
        geom_proj = self.geom_proj(geom_features) # (1, 1, N_boxes, d_model)
        conf_proj = self.conf_proj(confidence)    # (1, 1, N_boxes, d_model)
        center_proj = self.center_proj(center)  # (1, 1, N_boxes, d_model)
        cam_proj = camera_emb_for_boxes         # (1, 1, N_boxes, d_model)
        
        # Sum all projections to get final bbox tokens                        
        # Use missing_box_embedding when presence is zero
        missing_mask = (presence == 0)  # (1, 1, N_boxes, 1)
        missing_embedding_expanded = self.missing_box_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, 1, d_model)
        
        # Apply missing box embedding where presence is zero
        bbox_tokens = geom_proj + cat_emb + conf_proj + center_proj + cam_proj
        bbox_tokens = self.box_token_scale * bbox_tokens  # Scale the box tokens to increase their magnitude
        bbox_tokens = torch.where(missing_mask, missing_embedding_expanded, bbox_tokens)
        
        # Expand to match vision tokens format
        # Add dimensions to match (B_v, T_v, N_v, N_boxes, d_model)
        bbox_tokens = bbox_tokens.expand(B_v, T_v, N_v, -1, self.config.d_model)  # (B_v, T_v, N_v, N_boxes, d_model)
        
        # Flatten bbox_tokens to (B, T * N_cam * N_boxes, d_model)
        bbox_tokens_flat = bbox_tokens.view(B_v, T_v * N_v * bbox_tokens.shape[-2], self.config.d_model)
        
        return bbox_tokens_flat, distance_token

    def encode_tokens_train(self, box_data, batch):
        """Encode boxes during training into a single token sequence.

        Returns:
            tokens_flat: (B, T_obs * (N_cam + N_cam*N_boxes_per_cam), d_model)
                Token order: [distance tokens per camera] then [box tokens]
            coordinates_normalized: for visualization
        """
        bbox_tokens_flat, distance_tokens_flat, coordinates_normalized = self.encode_boxes_train(
            box_data, batch
        )
        tokens_flat = torch.cat([distance_tokens_flat, bbox_tokens_flat], dim=1)
        return tokens_flat, coordinates_normalized

    def encode_tokens_inference(self, bounding_boxes, cam_index, B_v, T_v, N_v):
        """Encode boxes during inference into a single token sequence for a single camera chunk.

        Returns:
            tokens_flat: (B_v, T_v*N_v*(1 + N_boxes), d_model) if distance token is present,
                         otherwise (B_v, T_v*N_v*(N_boxes), d_model)
                Token order: [distance token] then [box tokens]
        """
        bbox_tokens_flat, distance_token = self.encode_boxes_inference(
            bounding_boxes=bounding_boxes,
            cam_index=cam_index,
            B_v=B_v,
            T_v=T_v,
            N_v=N_v,
        )

        if distance_token is None:
            return bbox_tokens_flat

        distance_token_flat = distance_token.view(B_v, T_v * N_v, self.config.d_model)
        tokens_flat = torch.cat([distance_token_flat, bbox_tokens_flat], dim=1)
        return tokens_flat
