from collections import deque

import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from .wilro_moe_model import WilroMoETransformer
from .wilro_moe_config import WilroMoEConfig


class WilroMoEPolicy(PreTrainedPolicy):
    """LeRobot wrapper for WILRO-MoE (SmolVLM2 KV-cache -> expert decoders)."""

    config_class = WilroMoEConfig
    name = "wilro_moe"

    def __init__(self, config: WilroMoEConfig):
        super().__init__(config)
        self.config = config
        self.model = WilroMoETransformer(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        # Chunked execution: commit n_action_steps, re-sample only when drained.
        # Each re-sample draws fresh flow noise, which at small n_action_steps is
        # where most of this family's success rate comes from.
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def forward(self, batch: dict) -> tuple:
        loss = self.model.compute_loss(batch)
        w = float(getattr(self.config, "router_balance_weight", 0.0) or 0.0)
        if w > 0.0:
            bal = self.model.router_balance_loss()
            if bal is not None:
                loss = loss + w * bal
        return loss, {}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        self.model.eval()
        return self.model.sample_actions(batch)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        self.model.eval()
        if len(self._action_queue) == 0:
            actions = self.model.sample_actions(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
