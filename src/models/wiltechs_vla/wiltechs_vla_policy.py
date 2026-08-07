from collections import deque

import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from .wiltechs_vla_model import WiltechsVLATransformer
from .wiltechs_vla_config import WiltechsVLAConfig


class WiltechsVLAPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for the WiltechsVLA interleaved flow matching model."""

    config_class = WiltechsVLAConfig
    name = "wiltechs_vla"

    def __init__(self, config: WiltechsVLAConfig):
        super().__init__(config)
        self.config = config
        self.model = WiltechsVLATransformer(config)
        # Action chunk queue -- see select_action().
        self._action_queue: deque = deque([], maxlen=1)
        self._queue_batch_size: int | None = None

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        # Called once per rollout by lerobot's eval loop. Dropping the queue here
        # is what keeps chunks from leaking across episodes.
        self._action_queue = deque([], maxlen=max(1, int(self.config.n_action_steps)))
        self._queue_batch_size = None

    def forward(self, batch: dict) -> tuple:
        loss = self.model.compute_loss(batch)
        return loss, {}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        self.model.eval()
        return self.model.sample_actions(batch)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """Serve one action per call, replanning only every n_action_steps steps.

        sample_actions() costs a full frozen-VLM forward plus num_inference_steps
        denoising passes over the DiT stack, so calling it per env step and keeping
        only actions[:, 0] discards n_action_steps-1 of every n_action_steps
        predictions. On libero_spatial (280 env steps/episode) that is 280 VLM
        forwards where 280/n_action_steps suffice.

        Setting n_action_steps=1 restores the previous per-step replanning
        behaviour exactly -- the queue then refills on every call.
        """
        n_steps = max(1, int(self.config.n_action_steps))
        batch_size = batch["observation.state"].shape[0]
        if self._queue_batch_size != batch_size:
            # Vec env resized (or first call after __init__ without reset).
            self._action_queue = deque([], maxlen=n_steps)
            self._queue_batch_size = batch_size

        if not self._action_queue:
            self.model.eval()
            chunk = self.model.sample_actions(batch)  # (B, horizon, action_dim)
            chunk = chunk[:, : min(n_steps, chunk.shape[1])]
            # deque over the time axis: each element is (B, action_dim).
            self._action_queue.extend(chunk.transpose(0, 1))

        return self._action_queue.popleft()