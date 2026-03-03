import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from typing import Dict, Optional, List, Tuple
import math
import numpy as np
import os
from pathlib import Path
import torchvision.models as models
import torchvision.transforms as T


# Import ObjectDetector from separate file
from .object_detector import ObjectDetector, DiffusionSinusoidalPosEmb
      
class PositionalEncoding(nn.Module):
    """Positional encoding for action sequences."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Register a buffer for the positional encoding
        pe = self._generate_positional_encoding(max_len)
        self.register_buffer('pe', pe)

    def _generate_positional_encoding(self, max_len):
        """Generate positional encoding tensor."""
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        
        # If we need more positions than currently available, extend the buffer
        if seq_len > self.pe.size(1):
            # Extend with some buffer room
            new_max_len = max(seq_len + 100, self.max_len * 2)
            print(f"Extending positional encoding from {self.pe.size(1)} to {new_max_len}")
            pe = self._generate_positional_encoding(new_max_len)
            # Ensure the new buffer is on the same device as x
            self.register_buffer('pe', pe.to(x.device))
            
        return x + self.pe[:, :seq_len]

    

class SimpleDiffusionTransformer(nn.Module):
    """Flow matching transformer with separate encoding for vision and state."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_bounding_boxes_per_camera = 2

        # ------------------------------
        # 1. Single shared Object Detector for all cameras (initialize as None)
        # ------------------------------
        self.object_detector = None
        self._object_detector_initialized = False
        
        # Camera names for processing
        self.camera_names = config.cameras_for_vision_state_concat if config.cameras_for_vision_state_concat else [
            f'observation.images.cam_{i}' for i in range(config.num_cameras)
        ]
        self._camera_name_mapping = {}  # Mapping from sanitized names to original names
        for i, cam_name in enumerate(self.camera_names):
            # Sanitize the camera name for use as a module key
            sanitized_name = cam_name.replace('.', '_')
            self._camera_name_mapping[sanitized_name] = cam_name
            
        # ------------------------------
        # 2. Box encoder for processing bounding box data
        # ------------------------------
        # Linear projection for bounding box coordinates to d_model dimension
        self.box_encoder = nn.Sequential(
            nn.Linear(4, config.d_model // 2),  # 4 coordinates per box
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.d_model)
        )
        self.box_positional_encoding = PositionalEncoding(config.d_model)
        
        # Camera embedding for distinguishing between different camera views
        self.camera_embedding = nn.Embedding(3, config.d_model)  # 3 cameras: gripper, front, right
        
        # ------------------------------
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        self.state_positional_encoding = PositionalEncoding(config.d_model)


        # ------------------------------
        # 8. Number of inference steps for flow matching sampling
        # ------------------------------
        # Register as buffer so it's properly handled during device transfers
        self.register_buffer('num_inference_steps', torch.tensor(config.num_inference_steps))

        # ------------------------------
        # 9. Enhanced Action Encoder / Decoder
        # ------------------------------
        # Dedicated action input projection like VLAFlowMatching
        self.action_in_proj = nn.Linear(config.action_dim, config.d_model)
        
        self.action_positional_encoding = PositionalEncoding(config.d_model, config.horizon)

        denoising_layers = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.actions_expert = nn.TransformerDecoder(
            denoising_layers,
            num_layers=config.num_decoder_layers
        )
        self.velocity_prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.Mish(),
            nn.Linear(config.d_model // 2, config.action_dim)
        )
        self.time_embedding = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.d_model),
            nn.Mish(),
            nn.Linear(config.d_model, config.d_model)
        )


    def get_condition(self, batch, generate_heatmaps=False):
        """
        Encode state and bounding box data to get context tokens.
        During training, uses observation.box from dataset.
        During inference, uses Qwen3-VL for object detection.
        Returns:
            context: (B, T_obs * N_tokens, d_model)
            spatial_outputs: for visualization (including bounding boxes if requested)
        """
        B, T_obs = batch["observation.state"].shape[:2]

        # ------------------------------
        # 1. State encoding (compute once for reuse)
        # ------------------------------
        state_tokens = self.state_encoder(batch["observation.state"])  # (B, T, d_model)
        state_tokens = self.state_positional_encoding(state_tokens)
        # Flatten time dimension
        state_tokens_flat = state_tokens.view(B, T_obs, self.config.d_model)
        # Add a singleton token dim to match fusion later
        state_tokens_flat = state_tokens_flat.unsqueeze(2)  # (B, T, 1, d_model)
        state_tokens_flat = state_tokens_flat.view(B, T_obs * 1, self.config.d_model)

        # ------------------------------
        # 2. Bounding box encoding
        # ------------------------------
        spatial_outputs = {}  # for visualization
        all_bbox_tokens = []  # Collect bounding box tokens
        
        # Check if observation.box is available in the batch (from dataset during training)
        if "observation.box" in batch:
            # Use bounding box data from dataset during training
            # observation.box shape: (B, T_obs, 6, 4) - 6 boxes with 4 coordinates each
            box_data = batch["observation.box"]  # (B, T_obs, 6, 4)
            
            # Get image dimensions from observation images
            # Try to get dimensions from any available camera image
            image_width = 640.0  # default
            image_height = 400.0  # default
            
            # Look for any observation.images.xxx in the batch to get dimensions
            for key in batch.keys():
                if key.startswith("observation.images.") and isinstance(batch[key], torch.Tensor):
                    # Get the shape of the image tensor
                    # Expected shape: (B, T, C, H, W) or (B, C, H, W)
                    img_shape = batch[key].shape
                    if len(img_shape) >= 4:
                        # Extract height and width (last two dimensions)
                        image_height = float(img_shape[-2])
                        image_width = float(img_shape[-1])
                    break  # Use the first image found
            
            # Normalize bounding box coordinates during training
            # x coordinates (indices 0 and 2) should be divided by width
            # y coordinates (indices 1 and 3) should be divided by height
            normalization_factors = torch.tensor([image_width, image_height, image_width, image_height], 
                                              device=box_data.device, dtype=box_data.dtype)
            
            # Normalize the box coordinates
            box_data_normalized = box_data / normalization_factors.view(1, 1, 1, 4)
            
            # Reshape to (B, T_obs, 3 cameras, 2 boxes, 4) to process boxes per camera
            B, T_obs, N_boxes, N_coords = box_data_normalized.shape
            box_data_normalized = box_data_normalized.view(B, T_obs, self.config.num_cameras, self.num_bounding_boxes_per_camera, 4)  # (B, T_obs, 3, 2, 4)
            
                        
            # Encode bounding boxes using the box encoder
            bbox_tokens = self.box_encoder(box_data_normalized)  # (B, T_obs, 3, 2, d_model)
            
            # Add camera embedding
            # Camera IDs: 0 for gripper, 1 for front, 2 for right
            camera_ids = torch.arange(3, device=box_data.device)  # (3,)
            camera_emb = self.camera_embedding(camera_ids)  # (3, d_model)
            
            # Reshape camera embedding to match bbox_tokens dimensions
            # (3, d_model) -> (1, 1, 3, 1, d_model) to broadcast with (B, T_obs, 3, 2, d_model)
            camera_emb = camera_emb.view(1, 1, 3, 1, self.config.d_model)
            
            # Add camera embedding to bbox tokens
            bbox_tokens = bbox_tokens + camera_emb  # Broadcasting: (B, T_obs, 3, 2, d_model)
            
            # Flatten to (B, T_obs * 6, d_model) for positional encoding
            bbox_tokens_flat = bbox_tokens.view(B, T_obs * self.config.num_cameras * self.num_bounding_boxes_per_camera, self.config.d_model)
            
            # Apply positional encoding to bounding box tokens
            bbox_tokens_flat = self.box_positional_encoding(bbox_tokens_flat)  # (B, T_obs * 6, d_model)
            
            # Store bounding box tokens
            all_bbox_tokens.append(bbox_tokens_flat)
            
            # Store box data for visualization
            spatial_outputs["observation_box_data"] = box_data_normalized
        else:
            # observation.box is missing usually mean it is in inference, process each camera with the shared object detector
            for frame_idx in range(T_obs):
                for sanitized_cam_key in self._camera_name_mapping.keys():
                    # Get the original camera name from the mapping
                    original_cam_key = self._camera_name_mapping[sanitized_cam_key]
                    batch_key = original_cam_key
                    if batch_key in batch:
                        img = batch[batch_key]
                        if img.dim() == 4:  # (B*T, C, H, W)
                            img = img.view(B, T_obs, 1, *img.shape[-3:])
                        elif img.dim() == 5:  # (B, T, C, H, W)
                            img = img.unsqueeze(2)

                        # Store null values for heatmaps since we're not generating them in this simplified version
                        spatial_outputs[f"{sanitized_cam_key}_heatmap"] = None

                        # Initialize object detector if not already initialized
                        if not self._object_detector_initialized:
                            print("Initializing object detector for inference...")
                            self.object_detector = ObjectDetector(self.config)
                            self._object_detector_initialized = True
                        
                        # Detect objects and get bounding boxes using the shared detector
                        # Extract image for the current frame and camera
                        img_frame_cam = img[:, frame_idx:frame_idx+1, :, :, :]  # (B, 1, 1, C, H, W)
                        B_v, T_v, N_v, C_v, H_v, W_v = img_frame_cam.shape
                        img_reshaped = img_frame_cam.view(B_v * T_v * N_v, C_v, H_v, W_v)  # (B*1*1, C, H, W)
                        bounding_boxes, object_types = self.object_detector.detect_objects_and_get_bounding_boxes(img_reshaped)
                        
                        # Ensure exactly 2 bounding boxes per camera by padding with zeros if needed
                        if bounding_boxes is None or bounding_boxes.numel() == 0:
                            # No detections, create 2 empty boxes with 4 coordinates each for 2D bounding boxes
                            bounding_boxes = torch.zeros((2, 4), device=img_reshaped.device, dtype=torch.float32)
                            object_types = ['unknown', 'unknown']
                        else:
                            # Pad or trim to exactly 2 boxes
                            current_num_boxes = bounding_boxes.shape[0]
                            if current_num_boxes < 2:
                                # Pad with zeros (4 coordinates for 2D bounding boxes)
                                padding = torch.zeros((2 - current_num_boxes, 4), device=bounding_boxes.device, dtype=bounding_boxes.dtype)
                                bounding_boxes = torch.cat([bounding_boxes, padding], dim=0)
                                # Pad object types with 'unknown'
                                object_types.extend(['unknown'] * (2 - current_num_boxes))
                            elif current_num_boxes > 2:
                                # Trim to 2 boxes
                                bounding_boxes = bounding_boxes[:2]
                                object_types = object_types[:2]
                        
                        # Store bounding boxes for visualization
                        spatial_outputs[f"{sanitized_cam_key}_bounding_boxes_frame_{frame_idx}"] = bounding_boxes
                        
                        # Normalize bounding box coordinates during inference to match training
                        bounding_boxes[:, 0::2] /= W_v
                        bounding_boxes[:, 1::2] /= H_v
                        
                        # Reshape bounding box coordinates to match training format
                        # bounding_boxes shape: (N_boxes, 4)
                        # We need to reshape to (1, 1, N_boxes, 4) to match training format (B, T_obs, N_boxes, 4)
                        bbox_data_reshaped = bounding_boxes.unsqueeze(0).unsqueeze(0)  # (1, 1, N_boxes, 4)
                        
                        # Encode bounding boxes using the box encoder (same as training)
                        bbox_tokens = self.box_encoder(bbox_data_reshaped)  # (1, 1, N_boxes, d_model)
                        
                        # Apply positional encoding to bounding box tokens (same as training)
                        bbox_tokens = self.box_positional_encoding(bbox_tokens)  # (1, 1, N_boxes, d_model)
                        
                        # Expand to match vision tokens format
                        # Add dimensions to match (B_v, T_v, N_v, N_boxes, d_model)
                        bbox_tokens = bbox_tokens.expand(B_v, T_v, N_v, -1, self.config.d_model)  # (B_v, T_v, N_v, N_boxes, d_model)
                        
                        # Flatten bbox_tokens to (B, T * N_cam * N_boxes, d_model)
                        bbox_tokens_flat = bbox_tokens.view(B_v, T_v * N_v * bbox_tokens.shape[-2], self.config.d_model)
                        
                        # Collect bounding box tokens, coordinates, frame indices, and object types
                        all_bbox_tokens.append(bbox_tokens_flat)
                        
                        # Collect bounding box coordinates for spatial encoding
                        # Reshape bounding_boxes to match the token structure
                        bounding_boxes_reshaped = bounding_boxes.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N_boxes, 4)
                        bounding_boxes_reshaped = bounding_boxes_reshaped.expand(B_v, T_v, N_v, -1, 4)  # (B_v, T_v, N_v, N_boxes, 4)
                        

            # ------------------------------
            # 3. Apply camera embeddings and positional encoding to bounding box tokens (inference only)
            # ------------------------------
            if all_bbox_tokens:
                # Concatenate all collected tokens
                bbox_tokens_all = torch.cat(all_bbox_tokens, dim=1)  # (B, total_boxes, d_model)
                
                # Reshape to group by camera: (B, T_obs, num_cameras, num_bounding_boxes_per_camera, d_model)
                num_cameras = len(self._camera_name_mapping)
                bbox_tokens_reshaped = bbox_tokens_all.view(B, T_obs, num_cameras, self.num_bounding_boxes_per_camera, self.config.d_model)
                
                # Add camera embedding
                camera_ids = torch.arange(num_cameras, device=bbox_tokens_all.device)  # (num_cameras,)
                camera_emb = self.camera_embedding(camera_ids)  # (num_cameras, d_model)
                
                # Reshape camera embedding to match bbox_tokens dimensions
                # (num_cameras, d_model) -> (1, 1, num_cameras, 1, d_model) to broadcast
                camera_emb = camera_emb.view(1, 1, num_cameras, 1, self.config.d_model)
                
                # Add camera embedding to bbox tokens
                bbox_tokens_with_camera = bbox_tokens_reshaped + camera_emb  # Broadcasting
                
                # Flatten back to (B, T_obs * num_cameras * num_bounding_boxes_per_camera, d_model)
                bbox_tokens_flat = bbox_tokens_with_camera.view(B, T_obs * num_cameras * self.num_bounding_boxes_per_camera, self.config.d_model)
                
                # Apply positional encoding to bounding box tokens
                bbox_tokens_flat = self.box_positional_encoding(bbox_tokens_flat)  # (B, T_obs * num_cameras * num_bounding_boxes_per_camera, d_model)
                
                # Store the processed tokens
                all_bbox_tokens = [bbox_tokens_flat]
            else:
                obs_tokens = torch.empty(B, 0, self.config.d_model, device=batch["observation.state"].device)
        
        # Combine all bounding box tokens
        if all_bbox_tokens:
            bbox_tokens_combined = torch.cat(all_bbox_tokens, dim=1)  # (B, total_boxes, d_model)
        else:
            bbox_tokens_combined = torch.empty(B, 0, self.config.d_model, device=batch["observation.state"].device)
        
        # Combine observation tokens (bounding box tokens) with state tokens
        context = torch.cat([bbox_tokens_combined, state_tokens_flat], dim=1)

        return context, spatial_outputs


    def velocity_field(self, actions, timesteps, obs_context):
        """
        Flow Matching step:
        Predicts the velocity field that transports samples from Gaussian to data distribution.
        """
        B, T_act, _ = actions.shape
        
        # 1. Action & Time Embeddings (enhanced like VLAFlowMatching)
        action_embeddings = self.action_in_proj(actions)  # Use dedicated input projection
        action_embeddings = self.action_positional_encoding(action_embeddings)
        
        # Time embedding: (B, d_model) -> (B, 1, d_model)
        time_emb = self.time_embedding(timesteps.float()).unsqueeze(1)
        
        # 2. Add time to action tokens
        # We expand time across the action horizon
        tgt = action_embeddings + time_emb.expand(-1, T_act, -1)
        
        # 3. Augment Memory with Time
        # We add the time token to the observation context so the 
        # cross-attention mechanism is aware of the flow matching time.
        # extended_memory: (B, 1 + (T_obs * N_tokens), d_model)
        extended_memory = torch.cat([time_emb, obs_context], dim=1)
        
        # 4. Decoder Pass
        # Self-attention ensures T_act is a smooth curve.
        # Cross-attention aligns T_act with your CropConvNet features.
        # Allow actions to attend to all tokens (time, vision, and state)
        velocity_features = self.actions_expert(
            tgt=tgt,
            memory=extended_memory
        )
        #print(f"velocity_features mean abs: {velocity_features.abs().mean():.6f}, max: {velocity_features.abs().max():.6f}")
        
        
        # Residual connection to preserve gradients
        #velocity_features = velocity_features + action_embeddings
        
        # 5. Predict the velocity field
        return self.velocity_prediction_head(velocity_features)

    
    def sample_time(self, bsize, device):
        """Sample time using Beta(1.5, 1.0) distribution for better training dynamics"""
        # Beta(1.5, 1.0) distribution favors earlier timesteps
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001  # Scale to [0.001, 0.999]
        return time

    def compute_loss(self, batch):
        """Flow Matching Training: Learn to predict the velocity field."""
        actions = batch["action"] # (B, Horizon, Action_Dim)
        B, T_act = actions.shape[:2]
        
        # 1. Get observation context
        obs_context, spatial_outputs = self.get_condition(batch) # (B, T_obs, d_model)
                
        # Infer device from model parameters
        device = next(self.parameters()).device
        
        # 2. Sample time using Beta distribution
        timesteps = self.sample_time(B, device)
        
        # 3. Sample Gaussian noise
        noise = torch.randn_like(actions, device=device)
        
        # 4. Construct flow matching targets (straight line coupling)
        # Interpolate between noise and data
        noisy_actions = (1 - timesteps[:, None, None]) * noise + timesteps[:, None, None] * actions
        
        # 5. Predict velocity field
        pred_velocity = self.velocity_field(noisy_actions, timesteps, obs_context)
        #print(f"pred_velocity mean abs: {pred_velocity.abs().mean():.6f}, max: {pred_velocity.abs().max():.6f}")
        
        # 6. Compute flow matching loss
        # Target velocity is the difference between data and noise
        target_velocity = actions - noise
        #print(f"target_velocity mean abs: {target_velocity.abs().mean():.6f}, max: {target_velocity.abs().max():.6f}")
        
        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        
        # 7. Handle padding if present
        if "action_is_pad" in batch:
            # Apply padding mask: True means padded, False means valid
            in_episode_bound = ~batch["action_is_pad"]  # True for valid actions
            loss = loss * in_episode_bound.unsqueeze(-1)
        
        # Return mean loss
        return loss.mean()

    def forward(self, batch):
        """Inference: Solve ODE using learned velocity field."""
        B = batch["observation.state"].shape[0]
        T_act = self.config.horizon
        
        # Get observation context
        obs_context, spatial_outputs = self.get_condition(batch)  # (B, T_obs, d_model)
        
        # Infer device from model parameters
        device = next(self.parameters()).device
                
        # Start from pure Gaussian noise
        samples = torch.randn((B, T_act, self.config.action_dim), device=device)
        
        # Solve ODE using Euler integration
        num_steps = int(self.num_inference_steps.item())
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device)
            velocity = self.velocity_field(samples, t, obs_context)
            samples = samples + dt * velocity
            
        # Return both the actions and spatial outputs for visualization
        return samples, spatial_outputs


# Keep the original class for backward compatibility, but alias to the new simplified version
DiffusionTransformer = SimpleDiffusionTransformer
