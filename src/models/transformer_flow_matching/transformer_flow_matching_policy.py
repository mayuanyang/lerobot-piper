import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from .transformer_flow_matching_model import FlowMatchingTransformer
from .transformer_flow_matching_config import TransformerFlowMatchingConfig


class TransformerFlowMatchingPolicy(PreTrainedPolicy):
    """
    Flow matching policy for the Piper 7-DOF robot arm.

    Training: calls model.compute_loss(batch) → scalar loss.
    Inference: calls model.sample_actions(batch) → (B, n_action_steps, action_dim).
    """

    config_class = TransformerFlowMatchingConfig
    name = "transformer_diffusion"

    def __init__(self, config: TransformerFlowMatchingConfig):
        super().__init__(config)
        self.config = config
        self.model = FlowMatchingTransformer(config)

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        pass

    def forward(self, batch: dict) -> tuple:
        """Training forward pass. Returns (loss, loss_dict)."""
        loss = self.model.compute_loss(batch)
        return loss, {}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        """Predict a full action chunk (horizon steps). Returns (B, horizon, action_dim)."""
        self.model.eval()
        return self.model.sample_actions(batch)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """
        Select a single action for real-time control.
        Returns (B, action_dim) — the first action of the sampled chunk.
        """
        self.model.eval()
        actions = self.model.sample_actions(batch)  # (B, n_action_steps, action_dim)
        return actions[:, 0, :]                      # (B, action_dim)
