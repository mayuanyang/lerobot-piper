import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from .wiltechs_moe_model import WiltechsMoETransformer
from .wiltechs_moe_config import WiltechsMoEConfig


class WiltechsMoEPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for the WiltechsMoE mixture-of-experts flow matching model."""

    config_class = WiltechsMoEConfig
    name = "wiltechs_moe"

    def __init__(self, config: WiltechsMoEConfig):
        super().__init__(config)
        self.config = config
        self.model = WiltechsMoETransformer(config)

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        pass

    def forward(self, batch: dict) -> tuple:
        loss = self.model.compute_loss(batch)
        return loss, {}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        self.model.eval()
        return self.model.sample_actions(batch)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        self.model.eval()
        actions = self.model.sample_actions(batch)
        return actions[:, 0, :]