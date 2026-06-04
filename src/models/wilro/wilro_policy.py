from collections import deque

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
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        # Action queue for chunked execution. `lerobot-eval` calls
        # select_action() once per env step; we sample a chunk, commit the
        # first n_action_steps, and only re-sample when the queue drains.
        # Without this, n_action_steps is a no-op and the policy replans every
        # step (independent noise each step → temporally incoherent / jittery
        # actions, and a full VLM+DiT pass per timestep).
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

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
        if len(self._action_queue) == 0:
            # sample_actions already returns (B, n_action_steps, action_dim).
            actions = self.model.sample_actions(batch)[:, : self.config.n_action_steps]
            # Store time-major so each popleft yields one (B, action_dim) step,
            # committed across all parallel envs in lockstep.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()