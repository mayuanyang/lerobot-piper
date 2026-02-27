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

# Import Qwen3-VL-8B-Instruct components
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

      
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
    def __init__(self, config, shared_backbone=None, camera_id=None, camera_name=None, num_kp=None):
        super().__init__()

        self.config = config
        self.num_cameras = config.num_cameras
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.num_kp = num_kp

        # ------------------------------
        # 1. Qwen3-VL-8B-Instruct backbone - shared or individual
        # ------------------------------
        if shared_backbone is not None:
            # Use shared backbone
            self.backbone = shared_backbone
            self.hidden_dim = shared_backbone.config.hidden_size
            self.owns_backbone = False
            # Initialize processor for shared backbone
            self.processor = shared_backbone.processor if hasattr(shared_backbone, 'processor') else None
        else:
            # Create individual Qwen3-VL-8B-Instruct backbone
            print("Loading Qwen3-VL-8B-Instruct model...")
            self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            self.hidden_dim = self.backbone.config.hidden_size  # Should be 4096 for Qwen3-VL-8B
            self.owns_backbone = True
            # Initialize processor
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        # ------------------------------
        # 2. Improved layer freezing with better parameter handling

        # ------------------------------
        # 3. Add a pooling layer to reduce 14x14 patches to 7x7
        # This reduces 196 tokens to 49 tokens (preserves more spatial information)
        # ------------------------------
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        # ------------------------------
        # 4. Enhanced projection to d_model with better initialization
        # ------------------------------
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU()
        )

        # ------------------------------
        # 5. Camera-specific spatial softmax head
        # ------------------------------
        if num_kp is not None and num_kp > 0:
            # We'll initialize the SpatialSoftmax with a placeholder shape
            # The actual shape will be determined dynamically in the forward pass
            self.spatial_head = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, kernel_size=1),  
                SpatialSoftmax(
                    input_shape=(self.hidden_dim // 2, 14, 14),  # Placeholder, will be updated dynamically
                    num_kp=num_kp,  # Camera-specific number of keypoints
                    temperature=0.1, 
                    learn_temperature=True
                ),
                nn.GELU(),
            )
        else:
            self.spatial_head = None

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

        # For Qwen3-VL-8B-Instruct, we need to process the image differently
        # We'll use the vision transformer part of the model directly
        # Get vision features from Qwen3-VL
        vision_outputs = self.backbone.visual(x)
        
        # Extract patch tokens (excluding class token if present)
        if hasattr(vision_outputs, 'last_hidden_state'):
            # For Qwen3-VL, we typically get patch tokens directly
            patch_tokens = vision_outputs.last_hidden_state
        else:
            # Fallback for other formats
            patch_tokens = vision_outputs if isinstance(vision_outputs, torch.Tensor) else vision_outputs[0]
        
        # Handle different output formats
        n = patch_tokens.shape[0]
        
        # If we have a class token, separate it
        if patch_tokens.shape[1] > 1 and hasattr(self.backbone.visual, 'embeddings'):
            # Assume first token is class token
            cls_token = patch_tokens[:, 0:1, :]      # (Batch, 1, hidden_dim)
            patch_tokens = patch_tokens[:, 1:, :]    # (Batch, num_patches, hidden_dim)
        else:
            # No class token or only patch tokens
            cls_token = torch.mean(patch_tokens, dim=1, keepdim=True)  # Create a pseudo class token
            
        # Extract intermediate features for spatial processing if spatial head exists
        # For Qwen3-VL, we'll use the patch tokens directly for spatial processing
        if self.spatial_head is not None and self.num_kp > 0:
            # Dynamically compute patch dimensions
            num_patches = patch_tokens.shape[1]
            h = w = int(math.sqrt(num_patches))
            
            # Process spatial features for spatial softmax
            # Reshape to (Batch, hidden_dim, h, w) for conv operations
            patch_features = patch_tokens.transpose(1, 2).reshape(n, self.hidden_dim, h, w)
            spatial_coords = self.spatial_head(patch_features)
            
            # Store spatial coordinates for later use (e.g., in get_condition)
            self.cached_spatial_coords = spatial_coords
                
        # Spatial Pooling for tokens
        # Dynamically compute patch dimensions
        num_patches = patch_tokens.shape[1]
        h = w = int(math.sqrt(num_patches))
        
        # Reshape to (Batch, hidden_dim, h, w) to pool spatially
        patch_tokens_pooled = patch_tokens.transpose(1, 2).reshape(n, self.hidden_dim, h, w)
        patch_tokens_pooled = self.pool(patch_tokens_pooled)  # (Batch, hidden_dim, 7, 7)
        patch_tokens_flattened = patch_tokens_pooled.flatten(2).transpose(1, 2)  # (Batch, 49, hidden_dim)

        # Re-combine tokens (class token + pooled patch tokens)
        vit_combined = torch.cat([cls_token, patch_tokens_flattened], dim=1)  # (Batch, 50, hidden_dim)
        
        # Project to d_model
        vit_tokens = self.projection(vit_combined)  # (Batch, 50, d_model)
        
        # Store the number of tokens per camera for later use
        self.num_tokens_per_cam = vit_tokens.shape[1]

        # Reshape to separate cameras and time
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
        
        with torch.no_grad():
            B, T, N_cam, C, H, W = x.shape
            x = x.view(B * T * N_cam, C, H, W)

            # For Qwen3-VL-8B-Instruct, we need to process the image differently
            # Get vision features from Qwen3-VL
            vision_outputs = self.backbone.visual(x)
            
            # Extract patch tokens (excluding class token if present)
            if hasattr(vision_outputs, 'last_hidden_state'):
                # For Qwen3-VL, we typically get patch tokens directly
                patch_tokens = vision_outputs.last_hidden_state
            else:
                # Fallback for other formats
                patch_tokens = vision_outputs if isinstance(vision_outputs, torch.Tensor) else vision_outputs[0]
            
            # Handle different output formats
            n = patch_tokens.shape[0]
            
            # If we have a class token, separate it
            if patch_tokens.shape[1] > 1 and hasattr(self.backbone.visual, 'embeddings'):
                # Assume first token is class token
                patch_tokens = patch_tokens[:, 1:, :]    # (Batch, num_patches, hidden_dim)
            
            # Reshape patch tokens to feature maps for heatmap generation
            # Dynamically compute patch dimensions
            num_patches = patch_tokens.shape[1]
            h = w = int(math.sqrt(num_patches))
            
            # Reshape to (Batch, hidden_dim, h, w) for heatmap generation
            patch_features = patch_tokens.transpose(1, 2).reshape(n, self.hidden_dim, h, w)
            
            # Create spatial softmax for heatmap generation with proper input shape
            spatial_softmax = SpatialSoftmax(
                input_shape=(self.hidden_dim, h, w),
                num_kp=None,  # Use all channels for debugging
                temperature=0.1, 
                learn_temperature=False
            )
            
            # Apply spatial softmax to generate heatmaps and coordinates
            coords, heatmaps = spatial_softmax.forward_with_heatmaps(patch_features)  # coords: (Batch, hidden_dim, 2), heatmaps: (Batch, hidden_dim, h, w)
            
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
        # 1. Shared Qwen3-VL-8B-Instruct backbone for all cameras
        # ------------------------------
        print("Loading shared Qwen3-VL-8B-Instruct backbone...")
        self.shared_backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.shared_backbone.hidden_dim = self.shared_backbone.config.hidden_size  # Should be 4096 for Qwen3-VL-8B
        
        # Initialize processor for the shared backbone
        self.shared_backbone.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        
        
        # ------------------------------
        # 2. Vision Encoder per camera (sharing the same backbone)
        # ------------------------------
        self.image_encoders = nn.ModuleDict()
        # Create encoders for each camera
        camera_names = config.cameras_for_vision_state_concat if config.cameras_for_vision_state_concat else [
            f'observation.images.cam_{i}' for i in range(config.num_cameras)
        ]
        for i, cam_name in enumerate(camera_names):
            encoder = VisionEncoder(config, shared_backbone=self.shared_backbone, camera_id=i, camera_name=cam_name)
            if config.vision_freeze_layers > 0:
                encoder._freeze_layers(config.vision_freeze_layers)
            self.image_encoders[cam_name] = encoder
        # ------------------------------
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        self.state_positional_encoding = PositionalEncoding(config.d_model)

        # ------------------------------
        # 4. Vision Positional Encoding
        # -----------------------
        self.vision_positional_encoding = PositionalEncoding(config.d_model)
        # Temporal positional encoding for preserving time structure
        self.temporal_positional_encoding = PositionalEncoding(config.d_model)
        

                        

        # ------------------------------
        # 7. Coordinate projection for spatial features
        # ------------------------------
        self.coord_projection = nn.Linear(2, config.d_model)

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
        Separately encode vision tokens and state tokens, then fuse.
        Returns:
            context: (B, T_obs * N_tokens, d_model)
            spatial_outputs: for visualization (including heatmaps if requested)
        """
        B, T_obs = batch["observation.state"].shape[:2]
        all_vision_tokens = []
        all_coord_tokens = []  # Collect coordinate tokens from all cameras

        # ------------------------------
        # 1. State encoding (compute once for reuse)
        # ------------------------------
        state_tokens = self.state_encoder(batch["observation.state"])  # (B, T, d_model)
        #print(f"state_tokens mean abs: {state_tokens.abs().mean():.6f}, max: {state_tokens.abs().max():.6f}")

        
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
                #print(f"vision_tokens mean abs: {vision_tokens.abs().mean():.6f}, max: {vision_tokens.abs().max():.6f}")
                
                # Apply temporal positional encoding to preserve time structure
                # More efficient approach: compute PE for T timesteps and broadcast to all tokens
                B_v, T_v, N_v, D_v = vision_tokens.shape
                time_pe = self.temporal_positional_encoding(
                    torch.zeros(B_v, T_v, D_v, device=vision_tokens.device)
                )
                time_pe = time_pe.unsqueeze(2)  # (B, T, 1, D)
                vision_tokens = vision_tokens + time_pe

                #print(f"vision_tokens after temporal mean abs: {vision_tokens.abs().mean():.6f}, max: {vision_tokens.abs().max():.6f}")
                
                # Apply positional encoding to vision tokens
                vision_tokens_flat = vision_tokens.view(B_v, T_v * N_v, D_v)
                vision_tokens_flat = self.vision_positional_encoding(vision_tokens_flat)
                
                all_vision_tokens.append(vision_tokens_flat)
                
                # Retrieve cached spatial coordinates if they exist
                if hasattr(encoder, 'cached_spatial_coords') and encoder.cached_spatial_coords is not None:
                    spatial_coords = encoder.cached_spatial_coords
                    spatial_outputs[f"{cam_key}_spatial_coords"] = spatial_coords
                    
                    # Convert spatial coordinates to tokens and add to vision tokens
                    # spatial_coords shape: (B*T*N_cam, num_kp, 2)
                    if spatial_coords is not None and spatial_coords.numel() > 0:
                        # Project coordinates to d_model dimension
                        coord_tokens = self.coord_projection(spatial_coords)  # (B*T*N_cam, num_kp, d_model)
                        
                        # Reshape to match vision tokens format
                        coord_tokens = coord_tokens.view(B, T_v, encoder.num_kp, self.config.d_model)
                        
                        # Apply positional encoding to coordinate tokens
                        coord_tokens_flat = coord_tokens.view(B, T_v * encoder.num_kp, self.config.d_model)
                        
                        # Collect coordinate tokens from all cameras
                        all_coord_tokens.append(coord_tokens_flat)
                        
                else:
                    spatial_outputs[f"{cam_key}_spatial_coords"] = None

        if all_vision_tokens:
            vision_tokens_all = torch.cat(all_vision_tokens, dim=1)  # concat along token dim
        else:
            vision_tokens_all = torch.empty(B, 0, self.config.d_model, device=batch["observation.state"].device)

        # ------------------------------
        # 3. Final processing
        # ------------------------------
        # Combine observation tokens with coordinate tokens from all cameras
        if all_coord_tokens:
            coord_tokens_all = torch.cat(all_coord_tokens, dim=1)
            obs_tokens = torch.cat([vision_tokens_all, coord_tokens_all], dim=1)
        else:
            obs_tokens = vision_tokens_all
        
        # Instead of processing through cross-camera transformer, directly use the concatenated tokens
        # This allows actions to directly query the "pure" vision and state tokens
        context = torch.cat([obs_tokens, state_tokens_flat], dim=1)

        #print(f"context mean abs: {context.abs().mean():.6f}, max: {context.abs().max():.6f}")

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
