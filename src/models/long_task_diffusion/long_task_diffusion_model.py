import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from torch import Tensor
import torch.nn.functional as F
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

class LongTaskDiffusionModel(DiffusionModel):
    """
    Long Task Diffusion Model that extends the base diffusion model with custom loss computation.
    Inherits from DiffusionModel and overrides the compute_loss method.
    """
    
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch[ACTION]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        # 1. Calculate raw element-wise MSE
        loss_steps = F.mse_loss(pred, target, reduction="none") # (B, horizon, action_dim)

        # 2. Apply Padding Mask
        if self.config.do_mask_loss_for_padding:
            in_episode_bound = ~batch["action_is_pad"]
            loss_steps = loss_steps * in_episode_bound.unsqueeze(-1)

        # 3. Temporal Weighting: Weight earlier steps in the horizon more
        # Create a decay curve (e.g., exponential decay), action closer to the start of the horizon get higher weight
        weights = torch.exp(-0.1 * torch.arange(horizon, device=loss_steps.device))
        weights = weights / weights.sum() # Normalize
        loss = (loss_steps.mean(dim=-1) * weights).sum(dim=-1).mean()

        # 4. Velocity/Consistency Loss (The "Shortcut" Killer)
        # Compare the change between adjacent actions in pred vs target, encouraging smooth transitions
        # This helps prevent the model from taking shortcuts by making large jumps in action space
        pred_dist = pred[:, 1:, :] - pred[:, :-1, :]
        target_dist = target[:, 1:, :] - target[:, :-1, :]
        velocity_loss = F.mse_loss(pred_dist, target_dist)

        # Total Loss
        total_loss = loss + (0.5 * velocity_loss) 

        return total_loss
