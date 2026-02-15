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
        # CLS + all patch tokens
        # For ViT, seq_length includes both patch tokens and the CLS token
        self.num_tokens_per_cam = self.backbone.seq_length

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

        # Patch embedding
        x = self.backbone._process_input(x)
        n = x.shape[0]

        # Add CLS token
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # ViT encoder
        x = self.backbone.encoder(x)  # (B*T*N_cam, num_tokens, hidden_dim)

        # Project
        x = self.projection(x)

        # Reshape to separate cameras
        vision_tokens = x.view(
            B, T, N_cam, self.num_tokens_per_cam, self.config.d_model
        )

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
    """Flow matching transformer for conditional generation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Device will be inferred from the model's device when needed

        # 1. Vision & State Encoding (The "Conditioner")
        self.image_encoders = nn.ModuleDict({
            cam.replace('.', '_'): VisionEncoder(config) 
            for cam in config.image_features.keys()
        })
                
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model),
            nn.LayerNorm(config.d_model)
        )
        
        # Add positional encoding for state sequences
        self.state_positional_encoding = PositionalEncoding(config.d_model, 1200)  # Increased to handle longer sequences
                
        # Calculate the number of observation tokens dynamically
        # Sum up tokens from all image encoders plus one for the state token
        num_obs_tokens = sum(encoder.num_tokens_per_cam for encoder in self.image_encoders.values()) + 1  # +1 for state token
        
        # 2. Observation Transformer (Temporal Fusion)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, 
            nhead=config.nhead, 
            dim_feedforward=config.dim_feedforward, 
            dropout=0.1,  # Added dropout for regularization
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.obs_transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        
        # Positional encoding for observation tokens
        self.obs_positional_encoding = PositionalEncoding(config.d_model, 1200)  # Increased to handle longer sequences

        
        # Positional encoding for action sequences
        self.action_positional_encoding = PositionalEncoding(config.d_model, config.horizon)

        # Action projection to model dimension
        self.action_projection = nn.Linear(config.action_dim, config.d_model)

        # Learnable parameter for number of inference steps
        self.num_inference_steps = nn.Parameter(torch.tensor(100.0))  # Number of integration steps
        
        self.fusion_projection = nn.Sequential(
            nn.Linear(config.d_model * num_obs_tokens, config.d_model),
            nn.Mish(), # Mish helps gradient flow better than ReLU
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Flow matching transformer - predicts velocity field
        # This replaces the complex UNet with a simpler transformer-based approach
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
            num_layers=config.num_decoder_layers  # Configurable number of layers
        )
        
        # Final projection to action space (predicts velocity)
        self.velocity_prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.Mish(),
            nn.Linear(config.d_model // 2, config.action_dim)
        )
        
        # Time embedding for flow matching timesteps
        self.time_embedding = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.d_model),
            nn.Mish(),
            nn.Linear(config.d_model, config.d_model)
        )


    def get_condition(self, batch):
        B, T_obs = batch["observation.state"].shape[:2]
        all_obs_tokens = [] 
        spatial_coords_dict = {} # To store coords for drawing

        for cam_key, encoder in self.image_encoders.items():
            batch_key = cam_key.replace('_', '.')
            if batch_key in batch:
                # Get the image tensor
                img = batch[batch_key]
                
                # The VisionEncoder expects input of shape (B, T, N_cam, C, H, W)
                # We need to reshape the image tensor accordingly
                if img.dim() == 4:  # (B*T, C, H, W)
                    # Reshape to (B, T, 1, C, H, W)
                    img = img.view(B, T_obs, 1, *img.shape[-3:])
                elif img.dim() == 5:  # (B, T, C, H, W)
                    # Add camera dimension to make it (B, T, 1, C, H, W)
                    img = img.unsqueeze(2)
                
                # Pass the image through the VisionEncoder
                vision_tokens = encoder(img)  # (B, T, N_cam * num_tokens_per_cam, d_model)
                
                # Append the vision tokens to all_obs_tokens
                all_obs_tokens.append(vision_tokens)

        # Combine all camera tokens
        if all_obs_tokens:
            cam_tokens_all = torch.cat(all_obs_tokens, dim=2)
        else:
            # If no camera tokens, create empty tensor
            cam_tokens_all = torch.empty(B, T_obs, 0, self.config.d_model, device=batch["observation.state"].device)
        
        # Encode state
        state_tokens = self.state_encoder(batch["observation.state"])  # (B, T, d_model)
        
        # Add positional encoding to state tokens
        state_tokens = self.state_positional_encoding(state_tokens)
        
        # Combine camera tokens and state tokens
        # Add a dimension to state_tokens to match the token dimension
        state_tokens = state_tokens.unsqueeze(2)  # (B, T, 1, d_model)
        
        # Concatenate along the token dimension
        obs_tokens = torch.cat([cam_tokens_all, state_tokens], dim=2)  # (B, T, N_tokens, d_model)
        
        B, T, N, D = obs_tokens.shape
        
        # Flatten time and token dimensions for the Encoder
        obs_tokens_flat = obs_tokens.view(B, T * N, D)
        obs_tokens_flat = self.obs_positional_encoding(obs_tokens_flat)
        
        # Pass through the observation transformer
        context = self.obs_transformer(obs_tokens_flat)

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
        velocity_features = velocity_features + action_embeddings
        
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
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        return loss

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
