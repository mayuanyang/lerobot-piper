"""Component tests for the implemented half of WiltechsX.

Run directly:  python src/models/wiltechs_x/test_components.py

Loads the pure-torch modules WITHOUT importing lerobot, so it runs in an
environment where the policy config's dependencies are unavailable.

The load-bearing assertion here is "prefix output independent of suffix
input". ARCHITECTURE.md section 4 relies on it: if the prefix ever becomes a
function of the noise level, the VLM has to be recomputed at every denoising
step. That failure does not raise -- it just makes rollouts N times slower,
which is the stage-B RL budget -- so it is checked mechanically.
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]          # src/models


def _shell(name, path):
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_shell("models", ROOT)
_shell("models.interleaved_flow_matching", ROOT / "interleaved_flow_matching")
_shell("models.wiltechs_x", ROOT / "wiltechs_x")
_shell("models.wiltechs_vla", ROOT / "wiltechs_vla")
_load("models.interleaved_flow_matching.expert_layer",
      ROOT / "interleaved_flow_matching" / "expert_layer.py")

# wiltechs_x imports the verified Qwen3-VL helpers from wiltechs_vla_model,
# which pulls in lerobot. None of the components tested here call them (RoPE is
# passed as None), so they are stubbed rather than imported.
_vla = types.ModuleType("models.wiltechs_vla.wiltechs_vla_model")
_vla._apply_rope = lambda q, k, cos, sin: (q, k)
_vla._build_mrope_position_ids = lambda *a, **k: None
_vla.preprocess_camera_to_pixels = lambda *a, **k: None
_vla.vlm_pixels_key = lambda c: f"_vlmpix_pv::{c}"
_vla.vlm_grid_key = lambda c: f"_vlmpix_thw::{c}"
sys.modules["models.wiltechs_vla.wiltechs_vla_model"] = _vla

_cfg = types.ModuleType("models.wiltechs_x.wiltechs_x_config")
_cfg.WiltechsXConfig = type("WiltechsXConfig", (), {})
sys.modules["models.wiltechs_x.wiltechs_x_config"] = _cfg

M = _load("models.wiltechs_x.wiltechs_x_model",
          ROOT / "wiltechs_x" / "wiltechs_x_model.py")

_ok = True


def check(label, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


B, Lp, Ls = 2, 6, 4
D_vlm, D_exp, NH, NKV, HD = 64, 32, 4, 2, 16


def test_mask():
    print("build_joint_attn_mask")
    pad = torch.ones(B, Lp, dtype=torch.bool)
    pad[1, 2] = False
    m = M.build_joint_attn_mask(Lp, Ls, pad, True, torch.float32)
    vis = m[0, 0] > -1e30

    check("shape (B,1,L,L)", tuple(m.shape) == (B, 1, Lp + Ls, Lp + Ls))
    check("prefix->prefix full under bidirectional", vis[:Lp, :Lp].all())
    check("prefix->suffix fully masked (noise independence)", not vis[:Lp, Lp:].any())
    check("suffix->prefix full", vis[Lp:, :Lp].all())
    check("suffix->suffix causal",
          torch.equal(vis[Lp:, Lp:], torch.tril(torch.ones(Ls, Ls, dtype=torch.bool))))
    check("padded prefix key masked for every query", not (m[1, 0, :, 2] > -1e30).any())
    check("no all-masked query row", ((m[1, 0] > -1e30).sum(-1) > 0).all())

    mc = M.build_joint_attn_mask(Lp, Ls, pad, False, torch.float32)
    visc = mc[0, 0] > -1e30
    check("prefix->prefix causal when bidirectional=False",
          torch.equal(visc[:Lp, :Lp], torch.tril(torch.ones(Lp, Lp, dtype=torch.bool))))
    return m


class _FakeVLMLayer(nn.Module):
    """Minimal stand-in with the Qwen3-VL decoder-layer attribute names."""

    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(D_vlm)
        self.post_attention_layernorm = nn.LayerNorm(D_vlm)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(D_vlm, NH * HD, bias=False)
        self.self_attn.k_proj = nn.Linear(D_vlm, NKV * HD, bias=False)
        self.self_attn.v_proj = nn.Linear(D_vlm, NKV * HD, bias=False)
        self.self_attn.o_proj = nn.Linear(NH * HD, D_vlm, bias=False)
        self.mlp = nn.Linear(D_vlm, D_vlm)


def test_joint_layer(mask):
    print("\nJointExpertLayer")
    torch.manual_seed(0)
    layer = M.JointExpertLayer(D_exp, D_exp, NH, NKV, HD)
    vlm = _FakeVLMLayer()
    prefix = torch.randn(B, Lp, D_vlm)
    suffix = torch.randn(B, Ls, D_exp)
    t_emb = torch.randn(B, D_exp)

    p_out, s_out = layer(prefix, suffix, vlm, mask, (None, None), t_emb, False)
    check("prefix shape preserved", p_out.shape == prefix.shape)
    check("suffix shape preserved", s_out.shape == suffix.shape)
    check("adaLN-zero: suffix is identity at init", torch.allclose(s_out, suffix))

    p_out2, _ = layer(prefix, torch.randn(B, Ls, D_exp), vlm, mask,
                      (None, None), t_emb, False)
    check("prefix output independent of suffix input",
          torch.allclose(p_out, p_out2, atol=1e-6))

    vlm.zero_grad()
    _, s3 = layer(prefix, suffix, vlm, mask, (None, None), t_emb, True)
    s3.sum().backward()
    g = vlm.self_attn.o_proj.weight.grad
    check("stop_grad_to_prefix cuts the expert->VLM gradient",
          g is None or g.abs().sum() == 0)
    return layer, vlm, prefix, suffix, t_emb


def test_cached_equals_reference(layer, vlm, prefix, suffix, t_emb):
    """The load-bearing equivalence: production runs the cached path, the
    reference joint path defines what it should compute."""
    print("\nforward_cached vs forward (reference)")
    with torch.no_grad():
        pad = torch.ones(B, Lp, dtype=torch.bool)
        pad[1, 2] = False
        joint = M.build_joint_attn_mask(Lp, Ls, pad, True, torch.float32)
        _, s_ref = layer(prefix, suffix, vlm, joint, (None, None), t_emb, False)

        # Cached path: prefix K/V computed once, then the suffix alone.
        _, pk, pv = layer.prefix_qkv(prefix, vlm)
        sub = M.build_suffix_attn_mask(Lp, Ls, pad, torch.float32)
        s_fast = layer.forward_cached(suffix, pk, pv, sub, (None, None), t_emb)

    check("cached suffix output matches the joint reference",
          torch.allclose(s_ref, s_fast, atol=1e-5))
    check("suffix mask rows equal the joint mask's suffix rows",
          torch.equal(sub > -1e30, joint[:, :, Lp:, :] > -1e30))


def test_lora_and_discrete():
    print("\nLoRA / DiscreteActionHead")
    base = nn.Linear(16, 8)
    x = torch.randn(3, 16)
    ref = base(x).clone()
    wrapped = M.LoRALinear(base, rank=4, alpha=8)
    check("LoRA is an exact identity at init (B is zero)",
          torch.allclose(wrapped(x), ref, atol=1e-6))
    check("LoRA freezes the base weight", not base.weight.requires_grad)
    check("LoRA A/B are trainable",
          wrapped.lora_a.requires_grad and wrapped.lora_b.requires_grad)

    mod = nn.Module()
    mod.self_attn = nn.Module()
    mod.self_attn.q_proj = nn.Linear(4, 4)
    mod.self_attn.k_proj = nn.Linear(4, 4)
    mod.self_attn.other = nn.Linear(4, 4)
    n = M.attach_lora(mod, ["q_proj", "k_proj"], 2, 4)
    check("attach_lora wraps only the targeted projections",
          n == 2 and isinstance(mod.self_attn.q_proj, M.LoRALinear)
          and isinstance(mod.self_attn.other, nn.Linear))

    head = M.DiscreteActionHead(D_exp, horizon=4, action_dim=3, n_bins=256)
    a = torch.rand(B, 4, 3) * 6 - 3
    tok = head.tokenize(a)
    check("bin indices in range", bool(((tok >= 0) & (tok < 256)).all()))
    recon = tok.float() / 255.0 * 6 - 3
    check("binning round-trips to within one bin width",
          float((recon - a).abs().max()) < 6 / 255)
    check("logits shape (B,H,A,bins)",
          head(torch.randn(B, D_exp)).shape == (B, 4, 3, 256))


def test_long_horizon():
    print("\nMotionVectorEncoder / ProgressHead")
    mv = M.MotionVectorEncoder(state_dim=8, history_len=8, n_tokens=8, d_out=D_vlm)
    check("motion tokens shape", mv(torch.randn(B, 12, 8)).shape == (B, 8, D_vlm))
    check("short history is left-padded, not an error",
          mv(torch.randn(B, 3, 8)).shape == (B, 8, D_vlm))

    ph = M.ProgressHead(D_exp)
    p = ph(torch.randn(B, D_exp))
    check("progress in [0,1]", bool(((p >= 0) & (p <= 1)).all()))
    check("progress shape", p.shape == (B,))


if __name__ == "__main__":
    mask = test_mask()
    test_cached_equals_reference(*test_joint_layer(mask))
    test_lora_and_discrete()
    test_long_horizon()
    print("\nRESULT:", "ALL PASS" if _ok else "FAILURES ABOVE")
    sys.exit(0 if _ok else 1)
