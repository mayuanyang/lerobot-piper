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
        

        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
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
        
        if isinstance(self.temperature, nn.Parameter):
            temperature = self.temperature.clamp(0.1, 5.0)
        else:
            temperature = self.temperature
        
        # Apply temperature scaling and 2d softmax normalization
        attention = F.softmax(features / temperature, dim=-1)
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


class VisionEncoder(nn.Module):
    """Improved encoder with separate Spatial Softmax and Global Features for 7-DOF precision."""
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
        
        self.spatial_norm = nn.GroupNorm(32, feature_dim) # GroupNorm is safer than BN for small batches
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Global feature extractor - adaptive average pooling to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce to 1x1 spatial features
        
        # Projection for spatial features
        self.spatial_projection = nn.Sequential(
            nn.Linear(number_of_keypoints * 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Projection for global features
        self.global_projection = nn.Sequential(
            nn.Linear(feature_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # Resize input images to configured size
        x = self.resize_transform(x)
        x = self.backbone_features(x)
        
        x = self.spatial_norm(x)
        
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
        
        # Return both projected features separately and spatial coordinates (for visualization)
        return (spatial_projected, global_projected), spatial_coords
    

class SimpleDiffusionTransformer(nn.Module):
    """Simplified diffusion transformer that does denoising directly within the transformer architecture."""
    
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
        # Now we have 2 tokens per camera (spatial + global) plus 1 for state
        num_obs_tokens = len(config.image_features) * 2 + 1  # (cameras * 2) + state
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
        training_steps = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=training_steps,  
            beta_schedule='scaled_linear',  # Less aggressive noise schedule
            beta_start=0.0001,  # Lower starting noise
            beta_end=0.02,      # Lower ending noise
            clip_sample=True,
            prediction_type='epsilon'
        )

        
        # Positional encoding for action sequences
        self.action_positional_encoding = PositionalEncoding(config.d_model, config.horizon)

        # Action projection to model dimension
        self.action_projection = nn.Linear(config.action_dim, config.d_model)

        # Learnable parameter for number of inference steps
        self.num_inference_steps = nn.Parameter(torch.tensor(training_steps * 1.0))  # Match training steps
        
        self.fusion_projection = nn.Sequential(
            nn.Linear(config.d_model * num_obs_tokens, config.d_model),
            nn.Mish(), # Mish helps gradient flow better than ReLU
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Simplified denoising transformer - direct noise prediction
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
        
        # Final projection to action space
        self.noise_prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.Mish(),
            nn.Linear(config.d_model // 2, config.action_dim)
        )
        
        # Time embedding for diffusion timesteps
        self.time_embedding = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.d_model),
            nn.Mish(),
            nn.Linear(config.d_model, config.d_model)
        )


    def get_condition(self, batch):
        B, T_obs = batch["observation.state"].shape[:2]
        cam_tokens_per_timestep = []   # list of (B, T_obs, 2, d_model) for each camera
        spatial_outputs = {}

        for cam_key, encoder in self.image_encoders.items():
            batch_key = cam_key.replace('_', '.')
            if batch_key in batch:
                img = batch[batch_key].flatten(0, 1)      # (B*T_obs, C, H, W)
                (spatial_proj, global_proj), spatial_coords = encoder(img)
                # (B*T_obs, d_model) -> (B, T_obs, d_model)
                spatial_proj = spatial_proj.view(B, T_obs, -1)
                global_proj  = global_proj.view(B, T_obs, -1)
                # 堆成两个 token: dim=2 是 token index（0=spatial,1=global）
                cam_tokens = torch.stack([spatial_proj, global_proj], dim=2)  # (B, T_obs, 2, d_model)
                cam_tokens_per_timestep.append(cam_tokens)

                spatial_outputs[cam_key] = (
                    img.view(B, T_obs, *img.shape[1:]),
                    spatial_coords.view(B, T_obs, -1),
                )
            else:
                zeros = torch.zeros(B, T_obs, 2, self.config.d_model, device=self.device)
                cam_tokens_per_timestep.append(zeros)
                spatial_outputs[cam_key] = None

        # (B, T_obs, 2*#cams, d_model)
        cam_tokens_all = torch.cat(cam_tokens_per_timestep, dim=2)

        # state token: 先编码，再扩一维 token 轴
        state_encoded = self.state_encoder(batch["observation.state"])   # (B, T_obs, d_model)
        state_with_pos = self.state_positional_encoding(state_encoded)   # 也可以再额外加时间 PE
        state_tokens = state_with_pos.unsqueeze(2)                        # (B, T_obs, 1, d_model)

        # 拼成 (B, T_obs, N_tokens, d_model)
        obs_tokens = torch.cat([cam_tokens_all, state_tokens], dim=2)

        # 把 token 维展平到 seq_len 维度，让 Transformer 看到所有 token
        B, T, N_tok, D = obs_tokens.shape
        obs_tokens = obs_tokens.view(B, T * N_tok, D)   # (B, T*N_tokens, d_model)

        # 对这个“长序列”加一次位置编码（可以是 1D PE）
        obs_tokens = self.obs_positional_encoding(obs_tokens)   # (B, T*N_tokens, d_model)

        # 过 Transformer
        context = self.obs_transformer(obs_tokens)  # (B, T*N_tokens, d_model)

        # Instead of summing, we concatenate all token features per timestep
        context_flat = context.view(B, T, N_tok * D) # (B, T, N_tok * D)
        
        # Project down to d_model size. This allows the MLP to decide 
        # how to mix spatial/global/state dynamically per sample.
        context = self.fusion_projection(context_flat) # (B, T, D)

        return context, spatial_outputs


    def denoise_step(self, noisy_actions, timesteps, obs_context):
        """Single denoising step using transformer-based denoiser."""
        B, T_act, _ = noisy_actions.shape
        
        # Project actions to model dimension and add positional encoding
        action_embeddings = self.action_projection(noisy_actions)  # (B, T_act, d_model)
        action_embeddings = self.action_positional_encoding(action_embeddings)  # (B, T_act, d_model)
        
        # Embed timesteps
        time_embeddings = self.time_embedding(timesteps.float())  # (B, d_model)
        time_embeddings = time_embeddings.unsqueeze(1).expand(-1, T_act, -1)  # (B, T_act, d_model)
        
        extended_memory = torch.cat([time_embeddings, obs_context], dim=1)
        
        # Combine action and time embeddings
        action_with_time = action_embeddings + time_embeddings  # (B, T_act, d_model)
        
        # Use transformer decoder for denoising
        # obs_context serves as memory/key/value for cross-attention
        denoised_features = self.denoising_transformer(
            tgt=action_with_time,  # (B, T_act, d_model)
            memory=extended_memory     # (B, T_obs, d_model)
        )  # (B, T_act, d_model)
        
        # Predict noise
        pred_noise = self.noise_prediction_head(denoised_features)  # (B, T_act, action_dim)
        
        return pred_noise

    
    def compute_loss(self, batch):
        """Diffusion Training: Inject noise and learn to predict it."""
        actions = batch["action"] # (B, Horizon, Action_Dim)
        B, T_act = actions.shape[:2]
        
        # 1. Get observation context
        obs_context, spatial_outputs = self.get_condition(batch) # (B, T_obs, d_model)
                
        # 3. Sample noise and timesteps
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        
        # 4. Add noise to clean actions (Forward Diffusion)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # 5. Call simplified transformer denoiser with per-timestep conditioning
        pred_noise = self.denoise_step(noisy_actions, timesteps, obs_context)
        
        # 6. Compute both original and weighted losses
        # Original MSE loss
        loss = F.mse_loss(pred_noise, noise)
        
        return loss

    def forward(self, batch):
        """Inference: Start with pure noise and denoise into a smooth trajectory."""
        B = batch["observation.state"].shape[0]
        T_act = self.config.horizon
        
        # Get observation context
        obs_context, spatial_outputs = self.get_condition(batch)  # (B, T_obs, d_model)
        
                
        # Start from pure Gaussian noise
        noisy_action = torch.randn((B, T_act, self.config.action_dim), device=self.device)
        
        # Iteratively denoise
        self.noise_scheduler.set_timesteps(int(self.num_inference_steps.item()))
        
        for k in self.noise_scheduler.timesteps:
            # Predict noise using simplified transformer denoiser
            noise_pred = self.denoise_step(noisy_action, k.expand(B), obs_context)

            
            # Step back
            noisy_action = self.noise_scheduler.step(noise_pred, k, noisy_action).prev_sample
            
        # Return both the actions and spatial outputs for visualization
        return noisy_action, spatial_outputs


# Keep the original class for backward compatibility, but alias to the new simplified version
DiffusionTransformer = SimpleDiffusionTransformer
