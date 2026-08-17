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
        self._state_queue: deque | None = None
        self._state_batch_size: int | None = None

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        self._action_queue = deque([], maxlen=max(1, int(self.config.n_action_steps)))
        self._queue_batch_size = None
        self._state_queue = None
        self._state_batch_size = None

    def _with_state_history(self, batch: dict) -> dict:
        """Stack `observation.state` into the (B, T, D) window the model trains on.

        Training supplies the window through delta_timestamps. A STREAMING
        caller -- lerobot-eval, an RL rollout, a robot loop -- hands over one
        frame per step, and the model's fallback then left-pads that single
        frame: every first difference is zero and the motion-vector path
        (ARCHITECTURE.md 3.5, the LIBERO-Long mechanism) is silently absent at
        inference while it was present in training. Nothing raises; the score
        is just worse, for a reason no log would show.

        Buffering here rather than in each harness means any caller that goes
        through `select_action` gets it. A batch that already carries the
        window passes through untouched, so this is a no-op in training and
        for callers that do their own buffering
        (`eval_wiltechs_x.StateHistory`).
        """
        st = batch.get("observation.state")
        t = max(1, int(self.config.n_obs_steps))
        if st is None or st.dim() != 2 or t == 1:
            return batch
        b = st.shape[0]
        if self._state_queue is None or self._state_batch_size != b:
            # Seed full rather than empty: repeating the first frame is the
            # same left-pad LeRobot applies at an episode boundary, so the
            # window is in-distribution from step 0.
            self._state_queue = deque([st] * t, maxlen=t)
            self._state_batch_size = b
        else:
            self._state_queue.append(st)
        return {**batch,
                "observation.state": torch.stack(list(self._state_queue), dim=1)}

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
        # Push EVERY step, not only on replan: the window has to be current
        # when the next chunk is drawn.
        batch = self._with_state_history(batch)

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
