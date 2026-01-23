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

    def forward(self, x, cond):
        """
        x:    (B, C, T)
        cond: (B, cond_dim, T)
        """
        assert x.shape[-1] == cond.shape[-1], "Cond and x must align in time"

        out = self.conv1(x)

        # Move cond to (B*T, cond_dim)
        B, _, T = cond.shape
        cond_flat = cond.permute(0, 2, 1).reshape(B * T, -1)

        film_out = self.film(cond_flat)

        if self.use_film_scale_modulation:
            scale, bias = torch.chunk(film_out, 2, dim=-1)
            scale = scale.view(B, T, self.out_channels).permute(0, 2, 1)
            bias  = bias.view(B, T, self.out_channels).permute(0, 2, 1)
            out = scale * out + bias
        else:
            film_out = film_out.view(B, T, self.out_channels).permute(0, 2, 1)
            out = out + film_out

        out = self.conv2(out)
        return out + self.skip_conv(x)



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
            self.down_modules.append(nn.ModuleList([
                DiffusionConditionalResidualBlock1d(cin, cout, **block_kwargs),
                DiffusionConditionalResidualBlock1d(cout, cout, **block_kwargs),
                nn.Conv1d(cout, cout, 3, stride=2, padding=1) if not is_last else nn.Identity(),
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
        conds = []

        for res1, res2, down in self.down_modules:
            if cond.shape[-1] != x.shape[-1]:
                cond = F.interpolate(cond, size=x.shape[-1], mode="nearest")

            x = res1(x, cond)
            x = res2(x, cond)

            skips.append(x)
            conds.append(cond)

            x = down(x)

        # ------------------------------------------------
        # Middle
        # ------------------------------------------------
        if cond.shape[-1] != x.shape[-1]:
            cond = F.interpolate(cond, size=x.shape[-1], mode="nearest")

        for mid in self.mid_modules:
            x = mid(x, cond)

        # ------------------------------------------------
        # Decoder
        # ------------------------------------------------
        for res1, res2, up in self.up_modules:
            skip = skips.pop()
            cond = conds.pop()

            # Ensure x and skip have the same temporal dimension
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")

            x = torch.cat([x, skip], dim=1)

            x = res1(x, cond)
            x = res2(x, cond)
            x = up(x)

        # ---- output ----
        x = self.final_conv(x)
        return x.permute(0, 2, 1)



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
        # Enhanced state encoder for all 7 dimensions with deeper network
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU()
        )
        # Add positional encoding for state sequences
        self.state_positional_encoding = PositionalEncoding(config.d_model, 100)
        # Additional embedding for categorical gripper state
        self.gripper_embedding = nn.Embedding(2, config.d_model)  # 0: closed, 1: open
        
        # Projection layer for concatenated observation features
        num_obs_tokens = len(config.image_features) + 2  # cameras + state + gripper
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
        
        # Learnable parameter for number of inference steps
        self.num_inference_steps = nn.Parameter(torch.tensor(100.0))


    def get_condition(self, batch):
        """Processes images and states into per-timestep observation tokens."""
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
        obs_features = torch.cat(tokens, dim=-1)  # (B, T_obs, d_model * num_modalities)
        
        # Project to d_model
        obs_features = self.obs_projection(obs_features)  # (B, T_obs, d_model)
        
        # Apply positional encoding to observation tokens
        obs_features_pos = self.obs_positional_encoding(obs_features)  # (B, T_obs, d_model)
        
        # Pass through transformer
        context = self.obs_transformer(obs_features_pos)  # (B, T_obs, d_model)

        return context
      
    def compute_loss(self, batch):
        """Diffusion Training: Inject noise and learn to predict it."""
        actions = batch["action"] # (B, Horizon, Action_Dim)
        B, T_act = actions.shape[:2]
        
        # 1. Get observation context
        obs_context = self.get_condition(batch) # (B, T_obs, d_model)
        
        # 2. Align observation timesteps with action timesteps
        T_obs = obs_context.shape[1]
        if T_obs != T_act:
          obs_context = F.interpolate(
              obs_context.permute(0, 2, 1),  # (B, d_model, T_obs)
              size=T_act,
              mode="linear",
              align_corners=False
          ).permute(0, 2, 1)  # (B, T_act, d_model)

        
        # 3. Sample noise and timesteps
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        
        # 4. Add noise to clean actions (Forward Diffusion)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # 5. Call UNet denoiser with per-timestep conditioning
        pred_noise = self.denoiser(noisy_actions, timesteps, obs_context)
        
        # 6. Compute loss
        loss = F.mse_loss(pred_noise, noise)

        return loss

    def forward(self, batch):
        """Inference: Start with pure noise and denoise into a smooth trajectory."""
        B = batch["observation.state"].shape[0]
        T_act = self.config.horizon
        
        # Get observation context
        obs_context = self.get_condition(batch)  # (B, T_obs, d_model)
        
        # Align observation timesteps with action timesteps
        T_obs = obs_context.shape[1]
        if T_obs != T_act:
          obs_context = F.interpolate(
              obs_context.permute(0, 2, 1),  # (B, d_model, T_obs)
              size=T_act,
              mode="linear",
              align_corners=False
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
            
        return noisy_action
