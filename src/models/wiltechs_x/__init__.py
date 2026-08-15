"""WiltechsX — joint-attention (Mixture-of-Transformers) VLA for LIBERO.

Designed as a substrate for RL post-training rather than as a standalone SFT
policy: on standard LIBERO the SFT architecture spread is under 1.5 points
while RL post-training is worth 8-12 (SimpleVLA-RL takes OpenVLA-OFT from 91
to 99, LIBERO-Long 86.5 to 98.5). Rollout throughput and a non-zero per-task
floor therefore outrank SFT average. See ARCHITECTURE.md.

Key departures from wiltechs_vla: the VLM is trainable (LoRA), attention is
joint per layer instead of decoder cross-attention to a captured KV cache, and
knowledge insulation replaces the contrastive hinge.

Before training anything, run:
    python src/models/wiltechs_x/test_components.py   # pure torch, seconds
    python src/models/wiltechs_x/smoke_test.py        # real backbone, minutes
"""

from .wiltechs_x_config import WiltechsXConfig
from .wiltechs_x_model import (
    DiscreteActionHead,
    JointExpertLayer,
    LoRALinear,
    MotionVectorEncoder,
    ProgressHead,
    WiltechsXModel,
    WristTokenizer,
    attach_lora,
    build_joint_attn_mask,
    build_suffix_attn_mask,
)
from .wiltechs_x_policy import WiltechsXPolicy
from .processor_wiltechs_x import make_pre_post_processors

__all__ = [
    "WiltechsXConfig",
    "WiltechsXModel",
    "WiltechsXPolicy",
    "JointExpertLayer",
    "MotionVectorEncoder",
    "ProgressHead",
    "WristTokenizer",
    "DiscreteActionHead",
    "LoRALinear",
    "attach_lora",
    "build_joint_attn_mask",
    "build_suffix_attn_mask",
    "make_pre_post_processors",
]
