import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from .wilro_model import WilroTransformer
from .wilro_config import WilroConfig


class WilroPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for the WILRO (KV-cache → DiT) flow matching model."""

    config_class = WilroConfig
    name = "wilro"

    def __init__(self, config: WilroConfig):
        super().__init__(config)
        self.config = config
        self.model = WilroTransformer(config)

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