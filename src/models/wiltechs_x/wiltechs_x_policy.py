from collections import deque

import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from .wiltechs_x_config import WiltechsXConfig
from .wiltechs_x_model import WiltechsXModel


class WiltechsXPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for WiltechsX."""

    config_class = WiltechsXConfig
    name = "wiltechs_x"

    def __init__(self, config: WiltechsXConfig):
        super().__init__(config)
        self.config = config
        self.model = WiltechsXModel(config)
        self._action_queue: deque = deque([], maxlen=1)
        self._queue_batch_size: int | None = None

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        self._action_queue = deque([], maxlen=max(1, int(self.config.n_action_steps)))
        self._queue_batch_size = None

    def forward(self, batch: dict) -> tuple:
        return self.model.compute_loss(batch), {}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        self.model.eval()
        return self.model.sample_actions(batch)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """Serve one action per call, replanning every n_action_steps steps.

        The chunk is executed IN FULL (OFT). At 10 Hz and ~280 steps/episode
        that is 280/n_action_steps prefix computations instead of 280 -- which
        is the stage-B RL budget, not a latency nicety.
        """
        n_steps = max(1, int(self.config.n_action_steps))
        batch_size = batch["observation.state"].shape[0]
        if self._queue_batch_size != batch_size:
            self._action_queue = deque([], maxlen=n_steps)
            self._queue_batch_size = batch_size

        if not self._action_queue:
            self.model.eval()
            chunk = self.model.sample_actions(batch)
            chunk = chunk[:, : min(n_steps, chunk.shape[1])]
            self._action_queue.extend(chunk.transpose(0, 1))

        return self._action_queue.popleft()
