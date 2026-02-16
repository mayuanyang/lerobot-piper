import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from typing import Dict, Optional
import math
import numpy as np
import os


      
class PositionalEncoding(nn.Module):
    """Positional encoding for action sequences."""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class DiffusionSinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


import torch
import torch.nn as nn
import torchvision.models as models

class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_cameras = config.num_cameras

        # ------------------------------
        # 1. ViT backbone
        # ------------------------------
        # Extract image size from config (assuming square images)
        image_size = config.input_image_size[0] if isinstance(config.input_image_size, (tuple, list)) else config.input_image_size
        backbone = getattr(models, config.vision_backbone)(
            weights="DEFAULT",
            image_size=image_size
        )
        self.hidden_dim = backbone.hidden_dim  # 768 for vit_b_16
        backbone.heads = nn.Identity()
        self.backbone = backbone

        # ------------------------------
        # 2. Freeze early layers (optional)
        # ------------------------------
        if config.vision_freeze_layers > 0:
            for name, param in self.backbone.named_parameters():
                if "encoder.layers" in name:
                    layer_id = int(name.split("encoder.layers.")[1].split(".")[0])
                    if layer_id < config.vision_freeze_layers:
                        param.requires_grad = False

        # ------------------------------
        # 3. Projection to d_model
        # ------------------------------
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU()
        )

        # ------------------------------
        # 4. Camera embedding
        # ------------------------------
        self.camera_embedding = nn.Embedding(
            self.num_cameras,
            config.d_model
        )

        # ------------------------------
        # 5. num tokens per camera
        # ------------------------------
        # Only using CLS token
        self.num_tokens_per_cam = 1

    # ------------------------------
    # Forward
    # ------------------------------
    def forward(self, x):
        """
        x: (B, T, N_cam, C, H, W)
        return:
            vision_tokens: (B, T, N_cam * num_tokens_per_cam, d_model)
        """

        B, T, N_cam, C, H, W = x.shape

        # Merge batch, time, camera
        x = x.view(B * T * N_cam, C, H, W)

        # Resize images to match the expected input size of the backbone
        if H != self.backbone.image_size or W != self.backbone.image_size:
            import torchvision.transforms.functional as F
            x = F.resize(x, (self.backbone.image_size, self.backbone.image_size))

        # Use VisionTransformer's built-in forward pass which handles CLS token internally
        x = self.backbone(x)  # (B*T*N_cam, hidden_dim)
        
        # Add dimension to match expected format
        x = x.unsqueeze(1)  # (B*T*N_cam, 1, hidden_dim)
        
        # Project to d_model
        x = self.projection(x)  # (B*T*N_cam, 1, d_model)

        # Reshape to separate cameras
        vision_tokens = x.view(B, T, N_cam, 1, self.config.d_model)
        self.num_tokens_per_cam = 1

        # Add camera embedding
        cam_ids = torch.arange(N_cam, device=vision_tokens.device)
        cam_embed = self.camera_embedding(cam_ids)  # (N_cam, d_model)
        vision_tokens = vision_tokens + cam_embed.view(1, 1, N_cam, 1, self.config.d_model)

        # Flatten token dimension for obs transformer
        vision_tokens = vision_tokens.view(
            B, T, N_cam * self.num_tokens_per_cam, self.config.d_model
        )

        return vision_tokens


    

class SimpleDiffusionTransformer(nn.Module):
    """Flow matching transformer with separate encoding for vision and state."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ------------------------------
        # 1. Vision Encoder per camera
        # ------------------------------
        self.image_encoders = nn.ModuleDict({
            cam.replace('.', '_'): VisionEncoder(config)
            for cam in config.image_features.keys()
        })

        # ------------------------------
        # 2. State Encoder (separate)
        # ------------------------------
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model),
            nn.LayerNorm(config.d_model)
        )
        self.state_positional_encoding = PositionalEncoding(config.d_model, 200)

        # ------------------------------
        # 3. Vision Temporal Transformer
        # ------------------------------
        encoder_layer_vision = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.vision_temporal_transformer = nn.TransformerEncoder(
            encoder_layer_vision, num_layers=config.num_encoder_layers
        )
        self.vision_positional_encoding = PositionalEncoding(config.d_model, 200)

                
        # Fusion Transformer Encoder (instead of simple MLP)
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.fusion_projection = nn.TransformerEncoder(
            fusion_encoder_layer,
            num_layers=2  # Lightweight transformer for fusion
        )
        
        self.obs_ln = nn.LayerNorm(config.d_model)


        # ------------------------------
        # 5. Action Encoder / Decoder (same as before)
        # ------------------------------
        self.action_positional_encoding = PositionalEncoding(config.d_model, config.horizon)
        self.action_projection = nn.Linear(config.action_dim, config.d_model)

        denoising_layers = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.denoising_transformer = nn.TransformerDecoder(
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


    def get_condition(self, batch):
        """
        Separately encode vision tokens and state tokens, then fuse.
        Returns:
            context: (B, T_obs * N_tokens, d_model)
            spatial_coords_dict: for visualization
        """
        B, T_obs = batch["observation.state"].shape[:2]
        all_vision_tokens = []

        # ------------------------------
        # 1. Vision encoding per camera
        # ------------------------------
        for cam_key, encoder in self.image_encoders.items():
            batch_key = cam_key.replace('_', '.')
            if batch_key in batch:
                img = batch[batch_key]
                if img.dim() == 4:  # (B*T, C, H, W)
                    img = img.view(B, T_obs, 1, *img.shape[-3:])
                elif img.dim() == 5:  # (B, T, C, H, W)
                    img = img.unsqueeze(2)

                vision_tokens = encoder(img)  # (B, T, N_tokens_per_cam, d_model)
                # Apply temporal transformer per camera to fuse time
                B, T, N, D = vision_tokens.shape
                vision_tokens_flat = vision_tokens.view(B, T * N, D)
                vision_tokens_flat = self.vision_positional_encoding(vision_tokens_flat)
                vision_tokens_encoded = self.vision_temporal_transformer(vision_tokens_flat)
                all_vision_tokens.append(vision_tokens_encoded)

        if all_vision_tokens:
            vision_tokens_all = torch.cat(all_vision_tokens, dim=1)  # concat along token dim
        else:
            vision_tokens_all = torch.empty(B, 0, self.config.d_model, device=batch["observation.state"].device)

        # ------------------------------
        # 2. State encoding
        # ------------------------------
        state_tokens = self.state_encoder(batch["observation.state"])  # (B, T, d_model)
        state_tokens = self.state_positional_encoding(state_tokens)
        # Flatten time dimension
        state_tokens_flat = state_tokens.view(B, T_obs, self.config.d_model)
        # Add a singleton token dim to match fusion later
        state_tokens_flat = state_tokens_flat.unsqueeze(2)  # (B, T, 1, d_model)
        state_tokens_flat = state_tokens_flat.view(B, T_obs * 1, self.config.d_model)

        # ------------------------------
        # 3. Concatenate vision + state tokens
        # ------------------------------
        obs_tokens = torch.cat([vision_tokens_all, state_tokens_flat], dim=1)  # (B, total_tokens, d_model)
        
        obs_tokens = self.obs_ln(obs_tokens)
        
        context = self.fusion_projection(obs_tokens)

        spatial_coords_dict = {}  # for visualization if needed

        return context, spatial_coords_dict


    def velocity_field(self, actions, timesteps, obs_context):
        """
        Flow matching step:
        Predicts the velocity field that transports samples from Gaussian to data distribution.
        """
        B, T_act, _ = actions.shape
        
        # 1. Action & Time Embeddings
        action_embeddings = self.action_projection(actions)
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
        velocity_features = self.denoising_transformer(
            tgt=tgt,
            memory=extended_memory
        )
        
        # Residual connection to preserve gradients
        #velocity_features = velocity_features + action_embeddings
        
        # 5. Predict the velocity field
        return self.velocity_prediction_head(velocity_features)

    
    def compute_loss(self, batch):
        """Flow Matching Training: Learn to predict the velocity field."""
        actions = batch["action"] # (B, Horizon, Action_Dim)
        B, T_act = actions.shape[:2]
        
        # 1. Get observation context
        obs_context, spatial_outputs = self.get_condition(batch) # (B, T_obs, d_model)
                
        # Infer device from model parameters
        device = next(self.parameters()).device
        
        # 2. Sample time uniformly from [0, 1]
        timesteps = torch.rand(B, device=device)
        
        # 3. Sample Gaussian noise
        noise = torch.randn_like(actions, device=device)
        
        # 4. Construct flow matching targets (straight line coupling)
        # Interpolate between noise and data
        noisy_actions = (1 - timesteps[:, None, None]) * noise + timesteps[:, None, None] * actions
        
        # 5. Predict velocity field
        pred_velocity = self.velocity_field(noisy_actions, timesteps, obs_context)
        
        # 6. Compute flow matching loss
        # Target velocity is the difference between data and noise
        target_velocity = actions - noise
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
