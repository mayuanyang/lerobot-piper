import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from typing import Dict, Optional
import math
import numpy as np


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://huggingface.co/papers/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None, temperature=1.0, learnable_temperature=False):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
            temperature (float): temperature for softmax. Lower values lead to sharper peaks.
            learnable_temperature (bool): whether temperature should be learnable.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c
        
        self.temperature = nn.Parameter(torch.tensor(temperature)) if learnable_temperature else None

        # Temperature scaling for sharper peaks
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.clamp(self.temperature, 0.1, 5.0))

        else:
            self.register_buffer('temperature', torch.tensor(temperature))

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # Apply temperature scaling and 2d softmax normalization
        attention = F.softmax(features / self.temperature, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


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


class DiffusionConv1dBlock(nn.Module):
    """Conv1d block with group normalization and Mish activation."""
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    """Residual block with FiLM conditioning."""
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, use_film_scale_modulation=False):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation
        film_layers = []
        if use_film_scale_modulation:
            film_layers.append(nn.Linear(cond_dim, out_channels * 2))
        else:
            film_layers.append(nn.Linear(cond_dim, out_channels))
        film_layers.append(nn.Mish())
        self.film = nn.Sequential(*film_layers)

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # For skip connection
        self.skip_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Activation after residual connection for better gradient flow
        self.residual_activation = nn.Mish()
        
        # Initialize weights for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        # Initialize the final conv layer of conv2 block to small values for stable residual connections
        # This helps prevent gradient explosion while still allowing gradient flow
        # Access the Conv1d layer (index 0) within the block, not the Mish activation (index 2)
        nn.init.normal_(self.conv2.block[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.conv2.block[0].bias)
        
        # Initialize FiLM modulation layers
        for module in self.film:
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize skip connection if it's a convolution
        if isinstance(self.skip_conv, nn.Conv1d):
            # Kaiming initialization for skip connection
            nn.init.kaiming_normal_(self.skip_conv.weight, mode='fan_out', nonlinearity='relu')
            if self.skip_conv.bias is not None:
                nn.init.zeros_(self.skip_conv.bias)

    def forward_with_cond(self, x, cond):
        """Forward pass with conditioning, used for gradient checkpointing."""
        out = self.conv1(x)

        # Handle conditioning with potentially different temporal dimensions
        B, _, T_x = x.shape
        _, _, T_cond = cond.shape
        
        if T_cond == T_x:
            # Same temporal dimensions - use standard conditioning
            cond_proc = cond
        elif T_cond == 1:
            # Single time step conditioning - broadcast to all time steps
            cond_proc = cond.expand(-1, -1, T_x)
        else:
            # Different temporal dimensions - use adaptive pooling to match
            cond_proc = F.adaptive_avg_pool1d(cond, T_x)

        # Move cond to (B*T_x, cond_dim)
        cond_flat = cond_proc.permute(0, 2, 1).reshape(B * T_x, -1)

        film_out = self.film(cond_flat)

        if self.use_film_scale_modulation:
            scale, bias = torch.chunk(film_out, 2, dim=-1)
            scale = scale.view(B, T_x, self.out_channels).permute(0, 2, 1)
            bias  = bias.view(B, T_x, self.out_channels).permute(0, 2, 1)
            # Remove tanh normalization to allow full range of scale and bias
            out = scale * out + bias
        else:
            film_out = film_out.view(B, T_x, self.out_channels).permute(0, 2, 1)
            out = out + film_out

        out = self.conv2(out)
        residual = out + self.skip_conv(x)
        return self.residual_activation(residual)

    def forward(self, x, cond):
        """
        x:    (B, C, T_x)
        cond: (B, cond_dim, T_cond)
        """
        # Note: We no longer require T_x == T_cond, we'll handle mismatched dimensions
        
        out = self.conv1(x)

        # Handle conditioning with potentially different temporal dimensions
        B, _, T_x = x.shape
        _, _, T_cond = cond.shape
        
        if T_cond == T_x:
            # Same temporal dimensions - use standard conditioning
            cond_proc = cond
        elif T_cond == 1:
            # Single time step conditioning - broadcast to all time steps
            cond_proc = cond.expand(-1, -1, T_x)
        else:
            # Different temporal dimensions - use adaptive pooling to match
            cond_proc = F.adaptive_avg_pool1d(cond, T_x)

        # Move cond to (B*T_x, cond_dim)
        cond_flat = cond_proc.permute(0, 2, 1).reshape(B * T_x, -1)

        film_out = self.film(cond_flat)

        if self.use_film_scale_modulation:
            scale, bias = torch.chunk(film_out, 2, dim=-1)
            scale = scale.view(B, T_x, self.out_channels).permute(0, 2, 1)
            bias  = bias.view(B, T_x, self.out_channels).permute(0, 2, 1)
            # Remove tanh normalization to allow full range of scale and bias
            out = scale * out + bias
        else:
            film_out = film_out.view(B, T_x, self.out_channels).permute(0, 2, 1)
            # Remove tanh normalization to allow full range of film outputs
            out = out + film_out

        out = self.conv2(out)
        residual = out + self.skip_conv(x)
        return self.residual_activation(residual)



class DiffusionConditionalUnet1d(nn.Module):
    """
    1D UNet with per-timestep FiLM conditioning.
    Conditioning = diffusion timestep embedding + observation embedding (per time index).
    """

    def __init__(self, config, obs_cond_dim):
        super().__init__()
        self.config = config

        # ---- diffusion timestep encoder ----
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # total conditioning dimension passed to FiLM
        self.cond_dim = config.diffusion_step_embed_dim + obs_cond_dim

        # ---- UNet channel layout ----
        in_out = [(config.action_dim, config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:])
        )

        block_kwargs = dict(
            cond_dim=self.cond_dim,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            use_film_scale_modulation=config.use_film_scale_modulation,
        )

        # ---- Encoder ----
        self.down_modules = nn.ModuleList()
        for i, (cin, cout) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            if not is_last:
                # Non-last layer: Two residual blocks + downsampling
                self.down_modules.append(nn.ModuleList([
                    DiffusionConditionalResidualBlock1d(cin, cout, **block_kwargs),
                    DiffusionConditionalResidualBlock1d(cout, cout, **block_kwargs),
                    nn.Conv1d(cout, cout, 3, stride=2, padding=1),  # Downsampling
                ]))
            else:
                # Last layer: Two residual blocks only (no downsampling)
                self.down_modules.append(nn.ModuleList([
                    DiffusionConditionalResidualBlock1d(cin, cout, **block_kwargs),
                    DiffusionConditionalResidualBlock1d(cout, cout, **block_kwargs),
                    nn.Identity(),  # No downsampling
                ]))

        # ---- Middle ----
        self.mid_modules = nn.ModuleList([
            DiffusionConditionalResidualBlock1d(
                config.down_dims[-1], config.down_dims[-1], **block_kwargs
            ),
            DiffusionConditionalResidualBlock1d(
                config.down_dims[-1], config.down_dims[-1], **block_kwargs
            ),
        ])

        # ---- Decoder ----
        # All decoder blocks concatenate with skip connections
        decoder_pairs = [(cout, cin) for cin, cout in reversed(in_out[1:])]
        self.up_modules = nn.ModuleList()
        for cin, cout in decoder_pairs:
            # All decoder blocks take concatenated input (cin * 2) -> cout
            self.up_modules.append(nn.ModuleList([
                DiffusionConditionalResidualBlock1d(cin * 2, cout, **block_kwargs),
                DiffusionConditionalResidualBlock1d(cout, cout, **block_kwargs),
                nn.ConvTranspose1d(cout, cout, 4, stride=2, padding=1),
            ]))

        # ---- Output ----
        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_dim, 1),
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.enable_gradient_checkpointing = True
        
        # Initialize weights for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        # Initialize the final conv layer of final_conv block to small values
        # This helps prevent gradient explosion while still allowing gradient flow
        # Access the Conv1d layer (index 0) within the DiffusionConv1dBlock, not the Mish activation (index 2)
        final_conv_block = self.final_conv[-1]  # This is the DiffusionConv1dBlock
        if hasattr(final_conv_block, 'block'):
            # Access the Conv1d layer within the block (index 0)
            conv_layer = final_conv_block.block[0]
            if hasattr(conv_layer, 'weight') and hasattr(conv_layer, 'bias'):
                nn.init.normal_(conv_layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(conv_layer.bias)
        
        # Initialize diffusion timestep encoder
        for module in self.diffusion_step_encoder:
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, timestep, obs_cond):
        """
        Args:
            x:        (B, T, action_dim)
            timestep: (B,)
            obs_cond: (B, T, obs_cond_dim)   per-timestep observation tokens
        Returns:
            (B, T, action_dim)
        """
        # ---- reshape to Conv1d ----
        x = x.permute(0, 2, 1)               # (B, action_dim, T)
        obs_cond = obs_cond.permute(0, 2, 1) # (B, obs_cond_dim, T)

        B, _, T = x.shape

        # ---- diffusion timestep embedding ----
        t_emb = self.diffusion_step_encoder(timestep)     # (B, t_dim)
        t_emb = t_emb.unsqueeze(-1).expand(-1, -1, T)     # (B, t_dim, T)

        # ---- full conditioning ----
        cond = torch.cat([t_emb, obs_cond], dim=1)        # (B, cond_dim, T)

        # ------------------------------------------------
        # Encoder
        # ------------------------------------------------
        skips = []

        for res1, res2, down in self.down_modules:
            # The residual blocks now handle different temporal dimensions internally
            x = res1(x, cond)
            x = res2(x, cond)

            skips.append(x)

            x = down(x)

        # ------------------------------------------------
        # Middle
        # ------------------------------------------------
        for mid in self.mid_modules:
            x = mid(x, cond)

        # ------------------------------------------------
        # Decoder
        # ------------------------------------------------
        for res1, res2, up in self.up_modules:
            skip = skips.pop()
            
            x = torch.cat([x, skip], dim=1)

            x = res1(x, cond)
            x = res2(x, cond)
            x = up(x)

        # ---- output ----
        x = self.final_conv(x)
        return x.permute(0, 2, 1)



class VisionEncoder(nn.Module):
    """Improved encoder with Spatial Softmax and Global Features fusion for 7-DOF precision."""
    def __init__(self, config):
        super().__init__()
        # Add resize transform with configurable image size
        self.resize_transform = transforms.Resize(config.input_image_size)
        
        backbone = getattr(models, config.vision_backbone)(weights="DEFAULT" if config.pretrained_backbone_weights else None)
        # Remove fc and avgpool, keep spatial layers
        backbone_children = list(backbone.children())
        self.backbone_features = nn.Sequential(*backbone_children[:-2])
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.backbone_features, 'gradient_checkpointing'):
            self.backbone_features.gradient_checkpointing = True
        
        # Get feature dimensions (e.g., 512 for ResNet18)
        # Update dummy input size to match configured image size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, config.input_image_size[0], config.input_image_size[1])
            backbone_output = self.backbone_features(dummy_input)
            feature_dim = backbone_output.shape[1]
            height = backbone_output.shape[2]
            width = backbone_output.shape[3]
        
        # Use K-point spatial softmax with 2 points
        number_of_keypoints = 16
        # Use moderate temperature and learnable temperature to prevent collapse
        # Higher temperature (closer to 1.0) provides smoother gradients
        self.spatial_softmax = SpatialSoftmax(
            input_shape=(feature_dim, height, width),
            num_kp=number_of_keypoints,
            temperature=1.0,  # Moderate temperature for better gradients
            learnable_temperature=True  # Allow temperature to adapt during training
        )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Global feature extractor - adaptive average pooling to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce to 1x1 spatial features
        
        # Projection for spatial features
        self.spatial_projection = nn.Sequential(
            nn.Linear(number_of_keypoints * 2, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Projection for global features
        self.global_projection = nn.Sequential(
            nn.Linear(feature_dim, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layer to combine spatial and global features
        self.fusion_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # Resize input images to configured size
        x = self.resize_transform(x)
        x = self.backbone_features(x)
        
        # Extract spatial features using spatial softmax
        spatial_coords = self.spatial_softmax(x)
        # Flatten the spatial coordinates for the projection layer
        # From (B, K, 2) to (B, K * 2)
        spatial_coords_flat = spatial_coords.view(spatial_coords.size(0), -1)
        spatial_projected = self.spatial_projection(spatial_coords_flat)
        
        # Extract global features using adaptive pooling
        global_features = self.global_pool(x)
        # Flatten global features (squeeze the spatial dimensions)
        global_features_flat = global_features.view(global_features.size(0), -1)
        global_projected = self.global_projection(global_features_flat)
        
        # Fuse spatial and global features
        fused_features = torch.cat([spatial_projected, global_projected], dim=-1)
        projected = self.fusion_projection(fused_features)
        
        # Return the fused projected features and spatial coordinates (for visualization)
        return projected, spatial_coords
    
    def get_feature_statistics(self, x):
        """Get statistics about the features for debugging purposes."""
        x = self.resize_transform(x)
        x = self.backbone_features(x)
        
        # Compute statistics
        stats = {
            'mean': x.mean().item(),
            'std': x.std().item(),
            'min': x.min().item(),
            'max': x.max().item(),
            'shape': x.shape
        }
        
        # Get spatial softmax output
        spatial_coords = self.spatial_softmax(x)
        stats['spatial_coords_mean'] = spatial_coords.mean().item()
        stats['spatial_coords_std'] = spatial_coords.std().item()
        
        return stats


class DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Vision & State Encoding (The "Conditioner")
        self.image_encoders = nn.ModuleDict({
            cam.replace('.', '_'): VisionEncoder(config) 
            for cam in config.image_features.keys()
        })
                
        self.state_encoder = nn.Linear(config.state_dim, config.d_model)
        
        # Add positional encoding for state sequences
        self.state_positional_encoding = PositionalEncoding(config.d_model, 100)
                
        # Projection layer for concatenated observation features
        # Now we have 1 token per camera (fused spatial + global) plus 1 for state
        num_obs_tokens = len(config.image_features) + 1  # cameras + state
        self.obs_projection = nn.Linear(config.d_model * num_obs_tokens, config.d_model)
        
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
        self.obs_positional_encoding = PositionalEncoding(config.d_model, 100)  # 100 is max length

        # 3. Diffusion Components
        # We denoise the entire action horizon at once: (Horizon, Action_Dim)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=20,  # Reduced from 50 to 20 for less noise
            beta_schedule='scaled_linear',  # Less aggressive noise schedule
            beta_start=0.0001,  # Lower starting noise
            beta_end=0.02,      # Lower ending noise
            clip_sample=True,
            prediction_type='epsilon'
        )

        
        # Positional encoding for action sequences
        self.positional_encoding = PositionalEncoding(config.d_model, config.horizon)

        # Action projection to model dimension
        self.action_projection = nn.Linear(config.action_dim, config.d_model)

        # The UNet Denoiser (Predicts the noise added to actions)
        self.denoiser = DiffusionConditionalUnet1d(config, config.d_model)
        
        # Learnable parameter for number of inference steps
        self.num_inference_steps = nn.Parameter(torch.tensor(20.0))  # Match training steps


    def get_condition(self, batch):
        """Processes images and states into per-timestep observation tokens."""
        B, T_obs = batch["observation.state"].shape[:2]
        tokens = []

        # Store spatial softmax outputs for visualization
        spatial_outputs = {}

        for cam_key, encoder in self.image_encoders.items():
            # Check if the camera data is present in the batch
            batch_key = cam_key.replace('_', '.')
            if batch_key in batch:
                img = batch[batch_key].flatten(0, 1)
                # Get the fused projected features and spatial coordinates
                projected_features, spatial_coords = encoder(img)
                # Add the fused features as a single token
                tokens.append(projected_features.view(B, T_obs, -1))
                # Store spatial coordinates for visualization
                spatial_outputs[cam_key] = (img.view(B, T_obs, *img.shape[1:]), spatial_coords.view(B, T_obs, -1))
            else:
                # If camera data is missing, create zero tokens
                tokens.append(torch.zeros(B, T_obs, self.config.d_model, device=self.device))
                spatial_outputs[cam_key] = None
        
        # Original state encoding (all 7 dimensions including gripper as continuous)
        state_encoded = self.state_encoder(batch["observation.state"])
        # Apply positional encoding to state tokens
        state_with_pos = self.state_positional_encoding(state_encoded)
        tokens.append(state_with_pos)

        
        # Combine tokens
        obs_features = torch.cat(tokens, dim=-1)  # (B, T_obs, d_model * num_modalities)
        
        # Project to d_model
        obs_features = self.obs_projection(obs_features)  # (B, T_obs, d_model)
        
        # Apply positional encoding to observation tokens
        obs_features_pos = self.obs_positional_encoding(obs_features)  # (B, T_obs, d_model)
        
        # Pass through transformer
        context = self.obs_transformer(obs_features_pos)  # (B, T_obs, d_model)

        # Return context and spatial outputs for visualization
        return context, spatial_outputs

    
      
    def compute_loss(self, batch):
        """Diffusion Training: Inject noise and learn to predict it."""
        actions = batch["action"] # (B, Horizon, Action_Dim)
        B, T_act = actions.shape[:2]
        
        # 1. Get observation context
        obs_context, spatial_outputs = self.get_condition(batch) # (B, T_obs, d_model)
        
        # 2. Align observation timesteps with action timesteps
        T_obs = obs_context.shape[1]
        if T_obs != T_act:
            # Use nearest neighbor interpolation to avoid introducing artifacts
            obs_context = F.interpolate(
                obs_context.permute(0, 2, 1),  # (B, d_model, T_obs)
                size=T_act,
                mode="nearest"
            ).permute(0, 2, 1)  # (B, T_act, d_model)

        
        # 3. Sample noise and timesteps
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        
        # 4. Add noise to clean actions (Forward Diffusion)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # 5. Call UNet denoiser with per-timestep conditioning
        pred_noise = self.denoiser(noisy_actions, timesteps, obs_context)
        
        # 6. Compute both original and weighted losses
        # Original MSE loss
        original_loss = F.mse_loss(pred_noise, noise)
        
        # Weighted loss
        # Get joint weights from config and expand to match the shape of noise/pred_noise
        joint_weights = torch.tensor(self.config.joint_weights, device=self.device)
        # Expand weights to match the shape: (B, T_act, action_dim)
        expanded_weights = joint_weights.view(1, 1, -1).expand_as(noise)
        
        # Compute weighted MSE loss
        squared_errors = (pred_noise - noise) ** 2
        weighted_squared_errors = squared_errors * expanded_weights
        weighted_loss = weighted_squared_errors.mean()
        
                
        # Combined loss (original + weighted + spatial regularization)
        # Use a small weight for the spatial regularization loss to avoid overpowering the main losses
        loss = original_loss + weighted_loss

        return loss

    def forward(self, batch):
        """Inference: Start with pure noise and denoise into a smooth trajectory."""
        B = batch["observation.state"].shape[0]
        T_act = self.config.horizon
        
        # Get observation context
        obs_context, spatial_outputs = self.get_condition(batch)  # (B, T_obs, d_model)
        
        # Align observation timesteps with action timesteps
        T_obs = obs_context.shape[1]
        if T_obs != T_act:
            # Use nearest neighbor interpolation to avoid introducing artifacts
            obs_context = F.interpolate(
                obs_context.permute(0, 2, 1),  # (B, d_model, T_obs)
                size=T_act,
                mode="nearest"
            ).permute(0, 2, 1)  # (B, T_act, d_model)
        
        # Start from pure Gaussian noise
        noisy_action = torch.randn((B, T_act, self.config.action_dim), device=self.device)
        
        # Iteratively denoise
        self.noise_scheduler.set_timesteps(int(self.num_inference_steps.item()))
        
        for k in self.noise_scheduler.timesteps:
            # Predict noise using UNet with per-timestep conditioning
            noise_pred = self.denoiser(noisy_action, k.expand(B), obs_context)

            
            # Step back
            noisy_action = self.noise_scheduler.step(noise_pred, k, noisy_action).prev_sample
            
        # Return both the actions and spatial outputs for visualization
        return noisy_action, spatial_outputs
