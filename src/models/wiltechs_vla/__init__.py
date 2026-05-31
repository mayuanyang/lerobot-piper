"""WiltechsVLA — Qwen3-VL-based encoder-decoder flow matching policy.

Architecture: Mixture-of-Transformers (MoT) style. Frozen Qwen3-VL-4B runs
once per inference and exposes KV cache for the last N layers; a trainable
N-layer DiT cross-attends to those caches while running the flow-matching
denoising loop.

Replaces the earlier interleaved (joint attention every layer) design.
"""

from .wiltechs_vla_config import WiltechsVLAConfig
from .wiltechs_vla_model import WiltechsVLATransformer, DiTLayer
from .wiltechs_vla_policy import WiltechsVLAPolicy
from .processor_wiltechs_vla import make_pre_post_processors
from ..interleaved_flow_matching.expert_layer import RMSNorm, SwiGLU

__all__ = [
    "WiltechsVLAConfig",
    "WiltechsVLATransformer",
    "WiltechsVLAPolicy",
    "DiTLayer",
    "make_pre_post_processors",
    "RMSNorm",
    "SwiGLU",
]
