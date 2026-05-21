import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from .interleaved_flow_matching_model import InterleavedFlowMatchingTransformer
from .interleaved_flow_matching_config import InterleavedFlowMatchingConfig


class InterleavedFlowMatchingPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for the SmolVLA-style interleaved flow matching model."""

    config_class = InterleavedFlowMatchingConfig
    name = "interleaved_flow_matching"

    def __init__(self, config: InterleavedFlowMatchingConfig):
        super().__init__(config)
        self.config = config
        self.model = InterleavedFlowMatchingTransformer(config)

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
