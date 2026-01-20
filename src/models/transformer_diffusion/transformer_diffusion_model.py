import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from typing import Dict, Optional



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
        self.state_encoder = nn.Linear(config.state_dim, config.d_model)
        
        # 2. Observation Transformer (Temporal Fusion)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.nhead, 
            dim_feedforward=config.dim_feedforward, activation="gelu", 
            batch_first=True, norm_first=True
        )
        self.obs_transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

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

        # The Denoiser MLP (Predicts the noise added to actions)
        # Input: Noisy Action + Time + Obs Context
        input_dim = config.action_dim + config.d_model + config.d_model
        self.denoiser = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, config.action_dim)
        )

    def get_condition(self, batch):
        """Processes images and states into a single global context vector."""
        B, T_obs = batch["observation.state"].shape[:2]
        tokens = []

        for cam_key, encoder in self.image_encoders.items():
            img = batch[cam_key.replace('_', '.')].flatten(0, 1)
            tokens.append(encoder(img).view(B, T_obs, -1))
        
        tokens.append(self.state_encoder(batch["observation.state"]))
        
        # Combine and pass through transformer
        obs_features = torch.cat(tokens, dim=1) 
        context = self.obs_transformer(obs_features)
        
        # Global Average Pool across temporal/modality dimension to get "Scene Context"
        return context.mean(dim=1) 

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
        
        # 4. Predict the noise (Reverse Diffusion)
        time_emb = self.time_mlp(timesteps.unsqueeze(-1).float()) # (B, d_model)
        
        # Reshape to treat horizon as part of the batch for the MLP denoiser
        # (B, Horizon, Dim) -> (B*Horizon, Dim)
        obs_cond_expanded = obs_cond.unsqueeze(1).expand(-1, self.config.horizon, -1)
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, self.config.horizon, -1)
        
        denoiser_input = torch.cat([noisy_actions, obs_cond_expanded, time_emb_expanded], dim=-1)
        pred_noise = self.denoiser(denoiser_input)
        
        # 5. Loss: How well did we predict the added noise?
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
            # Prepare inputs
            time_emb = self.time_mlp(torch.tensor([k], device=self.device).float().unsqueeze(0)).expand(B, -1)
            obs_cond_expanded = obs_cond.unsqueeze(1).expand(-1, self.config.horizon, -1)
            time_emb_expanded = time_emb.unsqueeze(1).expand(-1, self.config.horizon, -1)
            
            denoiser_input = torch.cat([noisy_action, obs_cond_expanded, time_emb_expanded], dim=-1)
            
            # Predict noise
            noise_pred = self.denoiser(denoiser_input)
            
            # Step back
            noisy_action = self.noise_scheduler.step(noise_pred, k, noisy_action).prev_sample
            
        return noisy_action
