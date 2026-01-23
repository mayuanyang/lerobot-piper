import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from typing import Dict, Optional
import math



class SpatialSoftmax(nn.Module):
    """
    Extracts 2D feature coordinates from feature maps.
    Crucial for pick-and-place to identify 'where' objects are.
    """
    def __init__(self, height, width, num_channels):
        super().__init__()
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, width),
            torch.linspace(-1, 1, height),
            indexing="ij"
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))
        self.num_channels = num_channels

    def forward(self, feature_maps):
        # feature_maps: (B, C, H, W)
        batch_size = feature_maps.size(0)
        probs = F.softmax(feature_maps.view(batch_size, self.num_channels, -1), dim=-1)
        expected_x = torch.sum(probs * self.pos_x, dim=-1, keepdim=True)
        expected_y = torch.sum(probs * self.pos_y, dim=-1, keepdim=True)
        return torch.cat([expected_x, expected_y], dim=-1).view(batch_size, -1)


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

    def forward(self, x, cond):
        # x: (B, C, T)
        # cond: (B, cond_dim)

        # First conv
        out = self.conv1(x)

        # FiLM modulation
        cond_out = self.film(cond)
        if self.use_film_scale_modulation:
            scale, bias = torch.chunk(cond_out, 2, dim=-1)
            scale = scale.unsqueeze(-1)
            bias = bias.unsqueeze(-1)
            out = scale * out + bias
        else:
            cond_out = cond_out.unsqueeze(-1)
            out = out + cond_out

        # Second conv
        out = self.conv2(out)

        # Skip connection
        x = self.skip_conv(x)
        out = out + x
        return out


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning."""
    def __init__(self, config, global_cond_dim):
        super().__init__()
        self.config = config

        # Encoder for the diffusion timestep
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder
        # For the decoder, we just reverse these
        in_out = [(config.action_dim, config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:])
        )

        # Unet encoder
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList([
                    DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                    DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                    # Downsample as long as it is not the last block
                    nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                ])
            )

        # Processing in the middle of the auto-encoder
        self.mid_modules = nn.ModuleList([
            DiffusionConditionalResidualBlock1d(
                config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
            ),
            DiffusionConditionalResidualBlock1d(
                config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
            ),
        ])

        # Unet decoder
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList([
                    # dim_in * 2, because it takes the encoder's skip connection as well
                    DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                    DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                    # Upsample as long as it is not the last block
                    nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                ])
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_dim, 1),
        )

    def forward(self, x, timestep, global_cond=None):
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first
        x = x.permute(0, 2, 1)  # (B, input_dim, T)

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder
        encoder_skip_features = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = x.permute(0, 2, 1)  # (B, T, input_dim)
        return x


class VisionEncoder(nn.Module):
    """Improved encoder with Spatial Softmax for 7-DOF precision."""
    def __init__(self, config):
        super().__init__()
        backbone = getattr(models, config.vision_backbone)(weights="DEFAULT" if config.pretrained_backbone_weights else None)
        # Remove fc and avgpool, keep spatial layers
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Get feature dimensions (e.g., 512 for ResNet18)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 400, 640)
            backbone_output = self.backbone(dummy_input)
            feature_dim = backbone_output.shape[1]
            height = backbone_output.shape[2]
            width = backbone_output.shape[3]
        
        self.spatial_softmax = SpatialSoftmax(height=height, width=width, num_channels=feature_dim)
        self.projection = nn.Linear(feature_dim * 2, config.d_model)

    def forward(self, x):
        x = self.backbone(x)
        x = self.spatial_softmax(x)
        return self.projection(x)


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
        # Original state encoder for all 7 dimensions (including gripper as continuous)
        self.state_encoder = nn.Linear(config.state_dim, config.d_model)
        # Additional embedding for categorical gripper state
        self.gripper_embedding = nn.Embedding(2, config.d_model)  # 0: closed, 1: open
        
        # 2. Observation Transformer (Temporal Fusion)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, 
            nhead=config.nhead, 
            dim_feedforward=config.dim_feedforward, 
            dropout=0.,  # Added dropout for regularization
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
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        # Time embedding for the diffusion step
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, config.d_model)
        )

        # Positional encoding for action sequences
        self.positional_encoding = PositionalEncoding(config.d_model, config.horizon)

        # Action projection to model dimension
        self.action_projection = nn.Linear(config.action_dim, config.d_model)

        # The UNet Denoiser (Predicts the noise added to actions)
        self.denoiser = DiffusionConditionalUnet1d(config, config.d_model)
        

    def get_condition(self, batch):
        """Processes images and states into a single global context vector."""
        B, T_obs = batch["observation.state"].shape[:2]
        tokens = []

        for cam_key, encoder in self.image_encoders.items():
            # Check if the camera data is present in the batch
            batch_key = cam_key.replace('_', '.')
            if batch_key in batch:
                img = batch[batch_key].flatten(0, 1)
                tokens.append(encoder(img).view(B, T_obs, -1))
            else:
                # If camera data is missing, create zero tokens
                tokens.append(torch.zeros(B, T_obs, self.config.d_model, device=self.device))
        
        # Original state encoding (all 7 dimensions including gripper as continuous)
        tokens.append(self.state_encoder(batch["observation.state"]))
        
        # Extract gripper value (7th dimension, index 6) and convert to categorical
        gripper_values = batch["observation.state"][..., 6]  # (B, T_obs)
        # Convert to categorical: 0 if < 0.4 (closed), 1 if >= 0.4 (open)
        gripper_categorical = (gripper_values >= 0.4).long()  # (B, T_obs)
        # Get embedding for categorical gripper state
        gripper_embeddings = self.gripper_embedding(gripper_categorical)  # (B, T_obs, d_model)
        tokens.append(gripper_embeddings)
        
        # Combine tokens
        obs_features = torch.cat(tokens, dim=1)  # (B, seq_len, d_model)
        
        # Apply positional encoding to observation tokens
        obs_features_pos = self.obs_positional_encoding(obs_features)  # (B, seq_len, d_model)
        
        # Pass through transformer
        context = self.obs_transformer(obs_features_pos)
        
        # Global Average Pool across temporal/modality dimension to get "Scene Context"
        scene_context = context.mean(dim=1)

        return scene_context
      
    def compute_loss(self, batch):
        """Diffusion Training: Inject noise and learn to predict it."""
        actions = batch["action"] # (B, Horizon, Action_Dim)
        B = actions.shape[0]
        
        # 1. Get observation context
        obs_cond = self.get_condition(batch) # (B, d_model)
        
        # 2. Sample noise and timesteps
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        
        # 3. Add noise to clean actions (Forward Diffusion)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # 4. Call UNet denoiser
        pred_noise = self.denoiser(noisy_actions, timesteps, global_cond=obs_cond)
        
        # 5. Separate losses with learnable weights
        # Gripper loss (index 6) with learnable weight
        loss = F.mse_loss(pred_noise, noise)

        return loss

    def forward(self, batch):
        """Inference: Start with pure noise and denoise into a smooth trajectory."""
        B = batch["observation.state"].shape[0]
        obs_cond = self.get_condition(batch)
        
        # Start from pure Gaussian noise
        noisy_action = torch.randn((B, self.config.horizon, self.config.action_dim), device=self.device)
        
        # Iteratively denoise
        self.noise_scheduler.set_timesteps(self.config.num_inference_steps)
        
        for k in self.noise_scheduler.timesteps:
            # Predict noise using UNet
            noise_pred = self.denoiser(noisy_action, k.expand(B), global_cond=obs_cond)
            
            # Step back
            noisy_action = self.noise_scheduler.step(noise_pred, k, noisy_action).prev_sample
            
        return noisy_action
