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
from pathlib import Path
import torchvision.models as models

# Import our custom spatial softmax
from .spatial_softmax import SpatialSoftmax, save_heatmap_visualization

      
class PositionalEncoding(nn.Module):
    """Positional encoding for action sequences."""
    def __init__(self, d_model, max_len=1000):
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






class VisionEncoder(nn.Module):
    def __init__(self, config, shared_backbone=None, camera_id=None):
        super().__init__()

        self.config = config
        self.num_cameras = config.num_cameras
        self.camera_id = camera_id

        # ------------------------------
        # 1. ViT backbone - shared or individual
        # ------------------------------
        if shared_backbone is not None:
            # Use shared backbone
            self.backbone = shared_backbone
            self.hidden_dim = shared_backbone.hidden_dim
            self.owns_backbone = False
        else:
            # Create individual backbone
            # Extract image size from config (assuming square images)
            image_size = config.input_image_size[0] if isinstance(config.input_image_size, (tuple, list)) else config.input_image_size
            
            # Validate vision backbone
            if not hasattr(models, config.vision_backbone):
                raise ValueError(f"Unsupported vision backbone: {config.vision_backbone}")
                
            backbone = getattr(models, config.vision_backbone)(
                weights="DEFAULT",
                image_size=image_size
            )
            self.hidden_dim = backbone.hidden_dim  # 768 for vit_b_16
            
            # Remove the classification head to get raw features
            # The default head is a Linear layer that maps to 1000 classes
            # We replace it with Identity() to get the raw 768-dim features
            backbone.heads = nn.Identity()
            self.backbone = backbone
            self.owns_backbone = True

        # ------------------------------
        # 2. Improved layer freezing with better parameter handling

        # ------------------------------
        # 3. Add a pooling layer to reduce 14x14 patches to 5x5
        # This reduces 196 tokens to 25 tokens (approximately half)
        # ------------------------------
        self.pool = nn.AdaptiveAvgPool2d((5, 5))

        # ------------------------------
        # 4. Enhanced projection to d_model with better initialization
        # ------------------------------
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU()
        )

        # ------------------------------
        # 6. Camera embedding with Xavier initialization
        # ------------------------------
        # Only create camera embedding if this is not using a shared backbone
        # or if we want to keep per-camera embeddings
        if camera_id is not None:
            self.camera_embedding = nn.Embedding(
                self.num_cameras,
                config.d_model
            )
            # Initialize camera embeddings with Xavier initialization
            nn.init.xavier_uniform_(self.camera_embedding.weight)
        else:
            self.camera_embedding = None

        # ------------------------------
        # 7. Token tracking and configuration
        # ------------------------------
        self.num_tokens_per_cam = None
        
        
        # Flag to enable heatmap saving (for debugging)
        self.save_heatmaps = False
        self.heatmap_save_dir = None
        
        # Initialize all submodules properly
        self._init_weights()

    def _freeze_layers(self, freeze_layers):
        """Improved layer freezing with better parameter handling."""
        if freeze_layers <= 0:
            return
            
        frozen_count = 0
        for name, param in self.backbone.named_parameters():
            # Handle different ViT naming conventions
            if "encoder_layer_" in name:
                # Handle naming convention like "encoder.layers.encoder_layer_0.ln_1.weight"
                layer_part = name.split("encoder_layer_")[1]
                layer_id = int(layer_part.split(".")[0])
                if layer_id < freeze_layers:
                    param.requires_grad = False
                    frozen_count += 1
            elif "encoder.layers" in name:
                # Handle naming convention like "encoder.layers.0.attention.out_proj.weight"
                layer_part = name.split("encoder.layers.")[1]
                layer_id = int(layer_part.split(".")[0])
                if layer_id < freeze_layers:
                    param.requires_grad = False
                    frozen_count += 1
                    
        print(f"Frozen {frozen_count} parameters in VisionEncoder backbone")

    def _init_weights(self):
        """Initialize weights for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Initialize conv layers with Kaiming initialization
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers with Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                # Initialize normalization layers
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        x: (B, T, N_cam, C, H, W)
        return:
            vision_tokens: (B, T, N_cam * num_tokens_per_cam, d_model)
        """
        # Validate input
        self._validate_input(x)
        
        B, T, N_cam, C, H, W = x.shape
        x = x.view(B * T * N_cam, C, H, W)

        # 1. Get ViT features using the standard approach
        # We manually process the input to get both class token and patch tokens separately
        # This is necessary because we want to use both the class token and patch tokens
        # for different parts of our architecture, rather than just the class token
        # which is what the standard forward method returns.
        
        # Process input through convolutional projection (before encoder)
        # This applies the patch embedding convolution (Conv2d 3->768, 16x16 patches)
        # and reshapes to (batch, num_patches, hidden_dim) format
        x_processed = self.backbone._process_input(x)
        n = x_processed.shape[0]
        
        # Add class token
        # The class token is a learnable parameter that serves as a "summary" token
        # It's initialized as a 1x1x768 tensor and gets expanded to match the batch size
        # This allows the transformer to aggregate information from all patches into a single representation
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x_with_cls = torch.cat([batch_class_token, x_processed], dim=1)
        
        # Process through the full encoder to maintain proper gradient flow
        encoded_features = self.backbone.encoder(x_with_cls)
        
        # Extract final tokens
        x = encoded_features
        cls_token = x[:, 0:1, :]      # (Batch, 1, 768)
        patch_tokens = x[:, 1:, :]    # (Batch, 196, 768)
                
        # 5. Spatial Pooling for ViT tokens
        # Reshape to (Batch, 768, 14, 14) to pool spatially
        patch_tokens_pooled = patch_tokens.transpose(1, 2).reshape(n, self.hidden_dim, 14, 14)
        patch_tokens_pooled = self.pool(patch_tokens_pooled)  # (Batch, 768, 5, 5)
        patch_tokens_flattened = patch_tokens_pooled.flatten(2).transpose(1, 2)  # (Batch, 25, 768)

        # 6. Re-combine ViT tokens
        vit_combined = torch.cat([cls_token, patch_tokens_flattened], dim=1)  # (Batch, 26, 768)
        
        # 7. Project ViT tokens
        vit_tokens = self.projection(vit_combined)  # (Batch, 26, d_model)
        
        # Store the number of tokens per camera for later use
        self.num_tokens_per_cam = vit_tokens.shape[1]

        # Reshape to separate cameras
        vision_tokens = vit_tokens.view(B, T, N_cam, self.num_tokens_per_cam, self.config.d_model)

        # Add camera embedding
        cam_ids = torch.arange(N_cam, device=vision_tokens.device)
        cam_embed = self.camera_embedding(cam_ids)  # (N_cam, d_model)
        vision_tokens = vision_tokens + cam_embed.view(1, 1, N_cam, 1, self.config.d_model)

        # Flatten token dimension for obs transformer
        vision_tokens = vision_tokens.view(
            B, T, N_cam * self.num_tokens_per_cam, self.config.d_model
        )

        return vision_tokens

    def _validate_input(self, x):
        """Validate input tensor dimensions and values."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        if x.dim() != 6:
            raise ValueError(f"Expected 6D input (B, T, N_cam, C, H, W), got {x.dim()}D")
            
        if x.shape[-3] != 3:  # Assuming RGB images
            print(f"Warning: Expected 3-channel images, got {x.shape[-3]} channels")
            
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")

    def generate_debug_heatmaps(self, x, camera_name="default", batch_index=0, timestep=0):
        """
        Generate heatmaps for debugging purposes without affecting the normal forward pass.
        
        Args:
            x: Input tensor of shape (B, T, N_cam, C, H, W)
            camera_name: Name of the camera for saving files
            batch_index: Batch index for saving files
            timestep: Timestep for saving files
            
        Returns:
            heatmap_info: Dictionary containing heatmap data and metadata
        """
        with torch.no_grad():
            B, T, N_cam, C, H, W = x.shape
            x = x.view(B * T * N_cam, C, H, W)

            # Get ViT features using the same manual approach as in forward()
            # This gives us access to both class and patch tokens
            
            # Process input through convolutional projection (before encoder)
            # This applies the patch embedding convolution and reshapes to token format
            x_processed = self.backbone._process_input(x)
            n = x_processed.shape[0]
            
            # Add class token
            # The class token is a learnable parameter that serves as a "summary" token
            # It's initialized as a 1x1x768 tensor and gets expanded to match the batch size
            # This allows the transformer to aggregate information from all patches into a single representation
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x_with_cls = torch.cat([batch_class_token, x_processed], dim=1)
            
            # Pass through transformer encoder to get both class and patch tokens
            encoded_features = self.backbone.encoder(x_with_cls) # (Batch, 197, 768)

            # Separate CLS and Patches
            patch_tokens = encoded_features[:, 1:, :]   # (Batch, 196, 768)

            # Reshape patch tokens to feature maps for heatmap generation
            # patch_tokens: (Batch, 196, 768)
            # Reshape to (Batch, 768, 14, 14) for heatmap generation
            patch_features = patch_tokens.transpose(1, 2).reshape(n, self.hidden_dim, 14, 14)
            
            # Create spatial softmax for heatmap generation with proper input shape
            spatial_softmax = SpatialSoftmax(
                input_shape=(self.hidden_dim, 14, 14),
                num_kp=None,  # Use all channels for debugging
                temperature=0.1, 
                learn_temperature=False
            )
            
            # Apply spatial softmax to generate heatmaps and coordinates
            coords, heatmaps = spatial_softmax.forward_with_heatmaps(patch_features)  # coords: (Batch, 768, 2), heatmaps: (Batch, 768, 14, 14)
            
            # Save heatmaps for debugging if requested
            if self.save_heatmaps and self.heatmap_save_dir is not None:
                Path(self.heatmap_save_dir).mkdir(parents=True, exist_ok=True)
                
                # Save heatmaps for debugging
                for i in range(min(4, n)):  # Save heatmaps for first 4 samples
                    # Save one example heatmap per sample
                    heatmap_to_save = heatmaps[i, 0]  # First channel heatmap
                    # Get corresponding input image for overlay
                    input_img = x[i]  # (C, H, W)
                    # Convert to HWC format for visualization
                    input_img_hwc = input_img.permute(1, 2, 0)  # (H, W, C)
                    
                    # Save heatmap visualization
                    heatmap_save_path = Path(self.heatmap_save_dir) / f"{camera_name}_batch{batch_index}_sample{i}_timestep{timestep}_heatmap.png"
                    save_heatmap_visualization(heatmap_to_save, input_img_hwc, heatmap_save_path)
            
            # Store heatmap info for returning
            heatmap_info = {
                'coords': coords,
                'heatmaps': heatmaps,
                'camera_name': camera_name,
                'batch_index': batch_index,
                'timestep': timestep
            }
            
            return heatmap_info


    

class SimpleDiffusionTransformer(nn.Module):
    """Flow matching transformer with separate encoding for vision and state."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ------------------------------
        # 1. Shared ViT backbone for all cameras
        # ------------------------------
        # Extract image size from config (assuming square images)
        image_size = config.input_image_size[0] if isinstance(config.input_image_size, (tuple, list)) else config.input_image_size
        
        # Validate vision backbone
        if not hasattr(models, config.vision_backbone):
            raise ValueError(f"Unsupported vision backbone: {config.vision_backbone}")
            
        self.shared_backbone = getattr(models, config.vision_backbone)(
            weights="DEFAULT",
            image_size=image_size
        )
        self.shared_backbone.hidden_dim = self.shared_backbone.hidden_dim  # 768 for vit_b_16
        
        # Remove the classification head to get raw features
        # The default head is a Linear layer that maps to 1000 classes
        # We replace it with Identity() to get the raw 768-dim features
        self.shared_backbone.heads = nn.Identity()
        
        
        # ------------------------------
        # 2. Vision Encoder per camera (sharing the same backbone)
        # ------------------------------
        self.image_encoders = nn.ModuleDict({
            cam.replace('.', '_'): VisionEncoder(config, shared_backbone=self.shared_backbone, camera_id=i)
            for i, cam in enumerate(config.image_features.keys())
        })
        
        

        # ------------------------------
        # 3. State Encoder (separate)
        # ------------------------------
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model),
            nn.LayerNorm(config.d_model)
        )
        self.state_positional_encoding = PositionalEncoding(config.d_model)

        # ------------------------------
        # 4. Vision Positional Encoding
        # ------------------------------
        self.vision_positional_encoding = PositionalEncoding(config.d_model)
        
        # ------------------------------
        # 5. Cross-Camera Attention
        # ------------------------------
        cross_camera_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.cross_camera_transformer = nn.TransformerEncoder(
            cross_camera_layer, num_layers=config.num_encoder_layers
        )

                        
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
        self.fusion_encoder = nn.TransformerEncoder(
            fusion_encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        self.obs_ln = nn.LayerNorm(config.d_model)

        # ------------------------------
        # 6. Number of inference steps for flow matching sampling
        # ------------------------------
        # Register as buffer so it's properly handled during device transfers
        self.register_buffer('num_inference_steps', torch.tensor(config.num_inference_steps))

        # ------------------------------
        # 7. Action Encoder / Decoder (same as before)
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


    def get_condition(self, batch, generate_heatmaps=False):
        """
        Separately encode vision tokens and state tokens, then fuse.
        Returns:
            context: (B, T_obs * N_tokens, d_model)
            spatial_outputs: for visualization (including heatmaps if requested)
        """
        B, T_obs = batch["observation.state"].shape[:2]
        all_vision_tokens = []

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
        # 2. Vision encoding per camera
        # ------------------------------
        spatial_outputs = {}  # for visualization
        
                
        for cam_key, encoder in self.image_encoders.items():
            batch_key = cam_key.replace('_', '.')
            if batch_key in batch:
                img = batch[batch_key]
                if img.dim() == 4:  # (B*T, C, H, W)
                    img = img.view(B, T_obs, 1, *img.shape[-3:])
                elif img.dim() == 5:  # (B, T, C, H, W)
                    img = img.unsqueeze(2)

                # Generate heatmaps for debugging if requested
                if generate_heatmaps:
                    heatmap_info = encoder.generate_debug_heatmaps(
                        img, 
                        camera_name=cam_key, 
                        batch_index=0, 
                        timestep=0
                    )
                    spatial_outputs[f"{cam_key}_heatmap"] = heatmap_info
                else:
                    spatial_outputs[f"{cam_key}_heatmap"] = None

                vision_tokens = encoder(img)  # (B, T, N_tokens_per_cam, d_model)
                # Apply positional encoding to vision tokens
                B_v, T_v, N_v, D_v = vision_tokens.shape
                vision_tokens_flat = vision_tokens.view(B_v, T_v * N_v, D_v)
                vision_tokens_flat = self.vision_positional_encoding(vision_tokens_flat)
                
                all_vision_tokens.append(vision_tokens_flat)

        if all_vision_tokens:
            vision_tokens_all = torch.cat(all_vision_tokens, dim=1)  # concat along token dim
            
            # Store original vision tokens for residual connection
            vision_tokens_original = vision_tokens_all.clone()
            
            # Apply cross-camera attention to fuse information across cameras
            vision_tokens_fused = self.cross_camera_transformer(vision_tokens_all)
            
            # Add residual connection from original vision tokens
            vision_tokens_fused = vision_tokens_fused + vision_tokens_original
            
            # Skip token reduction since vision_token_linear was removed
            # Concatenate with state tokens directly
            vision_tokens_fused = torch.cat([vision_tokens_fused, state_tokens_flat], dim=1)
        else:
            vision_tokens_fused = torch.empty(B, 0, self.config.d_model, device=batch["observation.state"].device)

        # ------------------------------
        # 3. Final processing
        # ------------------------------
        #obs_tokens = self.obs_ln(vision_tokens_fused)
        context = self.fusion_encoder(vision_tokens_fused)

        return context, spatial_outputs


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
