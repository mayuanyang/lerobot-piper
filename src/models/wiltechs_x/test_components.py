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


def test_motion_history_guard():
    """_check_motion_history is bound to a stub: the real _build_prefix needs a
    Qwen3-VL, but the guard is the part that has to be right. A degraded motion
    window does not raise on its own -- the encoder left-pads and the deltas go
    to zero -- so these branches are the only thing standing between a dead
    feature and a run nobody questions."""
    print("\n_check_motion_history")

    class Stub:
        config = type("C", (), {"motion_history_len": 8,
                                "motion_vector_tokens": 8})()

        def __init__(self, training):
            self.training = training
            self._printed = set()
            self._motion_grace = 0

        _once = M.WiltechsXModel._once
        run = M.WiltechsXModel._check_motion_history

    def run(hist, training=True):
        s = Stub(training)
        try:
            s.run(hist, "observation.state")
            return None, s._printed
        except ValueError as e:
            return str(e), s._printed

    good = torch.randn(B, 8, 8).cumsum(1)                 # a real trajectory
    err, printed = run(good)
    check("healthy (8 frames, moving) passes", err is None and "motion" in printed)
    check("healthy does not emit the DEAD warning", "motion_dead" not in printed)

    err, _ = run(torch.randn(B, 1, 8))
    check("training + 1 frame RAISES", err is not None and "DEAD" in err)

    frozen = torch.randn(B, 1, 8).expand(B, 8, 8).contiguous()
    err, _ = run(frozen)
    check("training + identical frames RAISES", err is not None)

    # At inference the first call is always t=0, where StateHistory.reset has
    # filled the window with one repeated frame. That is a correct left-pad,
    # not a fault, and judging it reported every healthy rollout as dead.
    s = Stub(False)
    frozen1 = torch.randn(B, 1, 8).expand(B, 8, 8).contiguous()
    s.run(frozen1, "observation.state")
    check("inference: episode start is NOT reported dead", not s._printed)

    for _ in range(8):                                    # window fills up
        s.run(frozen1, "observation.state")
    check("inference: still-frozen window IS reported after the grace period",
          "motion_dead" in s._printed)

    s = Stub(False)
    for _ in range(8):
        s.run(good, "observation.state")
    check("inference: a moving window never warns", "motion_dead" not in s._printed)

    s = Stub(True)
    s._printed.add("motion")
    s.run(torch.randn(B, 1, 8), "observation.state")      # would raise if it ran
    check("runs once, then is a no-op on every later step", True)


SPATIAL = [f"pick up the black bowl {r} and place it on the plate" for r in (
    "between the plate and the ramekin", "next to the ramekin",
    "from table center", "on the cookie box", "on the ramekin",
    "next to the cookie box", "on the stove", "next to the plate",
    "on the wooden cabinet", "in the top drawer of the wooden cabinet")]
OBJECT = [f"pick up the {o} and place it in the basket" for o in (
    "alphabet soup", "cream cheese", "salad dressing", "bbq sauce",
    "ketchup", "tomato sauce", "butter", "milk", "orange juice", "chocolate pudding")]
GOAL = ["open the top drawer and put the bowl inside",
        "put the bowl on the stove", "put the wine bottle on top of the cabinet",
        "push the plate to the front of the stove", "turn on the stove"]


def test_hinge_negatives():
    """The hinge swaps sample i's INSTRUCTION while keeping sample i's IMAGE.
    A cross-suite partner names objects that are not in that image, so the two
    predictions can be separated by object presence alone -- no relation
    parsing, which is the whole point of the term. These checks are about the
    negative being drawn from the bucket where every referent IS present."""
    print("\n_suite_map / _hinge_pairs")

    class Stub:
        def __init__(self, thr=0.5):
            self.config = type("C", (), {"contrastive_suite_jaccard": thr})()
            self._printed = set()
            self._suite_tokens = {}
            self._suite_id = {}
        _once = M.WiltechsXModel._once
        _suite_map = M.WiltechsXModel._suite_map
        _hinge_pairs = M.WiltechsXModel._hinge_pairs

    # ---- clustering ----------------------------------------------------
    s = Stub()
    sid = s._suite_map(SPATIAL + OBJECT + GOAL)
    check("LIBERO's suites separate at Jaccard 0.5",
          len({sid[d] for d in SPATIAL}) == 1
          and len({sid[d] for d in OBJECT}) == 1
          and sid[SPATIAL[0]] != sid[OBJECT[0]])
    check("goal tasks do not collapse into spatial",
          all(sid[g] != sid[SPATIAL[0]] for g in GOAL))

    # Clustering must not depend on which tasks a batch happens to sample.
    s2 = Stub()
    s2._suite_map(SPATIAL[:3])
    s2._suite_map(OBJECT + SPATIAL)                    # arrives later
    sid2 = s2._suite_map(GOAL)
    check("grouping is stable when instructions arrive across batches",
          len({sid2[d] for d in SPATIAL}) == 1 and sid2[SPATIAL[0]] != sid2[OBJECT[0]])

    # ---- pairing -------------------------------------------------------
    descs = [SPATIAL[i % 10] for i in range(20)] + [OBJECT[i % 10] for i in range(20)]
    order = list(range(40))
    s = Stub()
    keep, other, hard, dropped = s._hinge_pairs(descs, order, [0.5] * 40)
    check("every sample gets a partner", len(keep) == 40 and dropped == 0)
    check("all negatives are same-suite", hard == 40)
    check("no partner carries the CORRECT instruction",
          all(descs[o] != descs[i] for i, o in zip(keep, other)))
    check("partner is always in the same suite",
          all(sid[descs[o]] == sid[descs[i]] for i, o in zip(keep, other)))

    # The point of a bucket over argmax: the partner varies with the draw, so
    # the model cannot overfit one fixed contrast per task.
    seen = set()
    for d in (0.0, 0.2, 0.4, 0.6, 0.8, 0.99):
        _, o, _, _ = s._hinge_pairs(descs, [0], [d])
        seen.add(o[0])
    check("bucket yields several distinct partners, not one fixed pair",
          len(seen) >= 5)

    # ---- degenerate batches -------------------------------------------
    one = [SPATIAL[0]] * 8
    keep, other, hard, dropped = s._hinge_pairs(one, list(range(8)), [0.5] * 8)
    check("single-instruction batch is dropped, never self-paired",
          keep == [] and dropped == 8)

    # A suite with only one task present must fall back, not drop: a
    # cross-suite negative is weak but still trains language sensitivity.
    lone = [SPATIAL[0]] * 4 + [OBJECT[0]] * 4
    keep, other, hard, dropped = s._hinge_pairs(lone, list(range(8)), [0.5] * 8)
    check("lone-task suite falls back to a cross-suite negative",
          len(keep) == 8 and dropped == 0 and hard == 0)
    check("fallback partner is still a different instruction",
          all(lone[o] != lone[i] for i, o in zip(keep, other)))

    # ---- the off switch ------------------------------------------------
    s0 = Stub(thr=0.0)
    check("suite buckets off -> no clustering", s0._suite_map(SPATIAL) is None)
    keep, other, hard, dropped = s0._hinge_pairs(descs, order, [0.5] * 40)
    check("buckets off still pairs every sample with a different instruction",
          len(keep) == 40 and hard == 0
          and all(descs[o] != descs[i] for i, o in zip(keep, other)))

    # draw=1.0 would index one past the end without the clamp
    keep, other, _, _ = s._hinge_pairs(descs, [0], [1.0])
    check("draw at the open end of [0,1) stays in range", len(other) == 1)


def test_episode_noise():
    """`fixed_episode_noise` is a claim about WHEN the noise is redrawn: once
    per episode, not once per replan. The Euler integration is deterministic
    given x_1, so reusing it is what keeps consecutive chunks on one branch of
    a multimodal action distribution. Getting the lifetime wrong fails
    silently -- redraw too often and nothing changes, too rarely and every
    episode of a run shares one branch."""
    print("\n_chunk_noise lifetime")

    class Stub:
        def __init__(self, fixed):
            self.config = type("C", (), {"horizon": 16, "action_dim": 7,
                                         "n_action_steps": 8,
                                         "fixed_episode_noise": fixed})()
            self._episode_noise = None
            self.model = type("M", (), {
                "sample_noise": staticmethod(lambda shape, device: torch.randn(shape)),
                "parameters": staticmethod(lambda: iter([torch.zeros(1)])),
            })()

    # Bind the REAL method rather than reimplementing it: a test that restates
    # the logic passes even when the shipped logic is wrong. The policy module
    # cannot be imported (it pulls in lerobot), and _chunk_noise is the last
    # method in the file, so slice from its def to EOF.
    src = (ROOT / "wiltechs_x" / "wiltechs_x_policy.py").read_text()
    ns = {"torch": torch}
    exec("class _P:\n" + src[src.index("    def _chunk_noise"):], ns)
    Stub._chunk_noise = ns["_P"]._chunk_noise

    off = Stub(False)
    check("disabled -> model draws its own noise", off._chunk_noise(4) is None)

    s = Stub(True)
    n1 = s._chunk_noise(4)
    check("enabled -> noise has (B, horizon, action_dim)", tuple(n1.shape) == (4, 16, 7))
    n2 = s._chunk_noise(4)
    check("every replan in the episode reuses ONE draw", torch.equal(n1, n2))
    check("and it is the same object, not an equal copy", n1 is n2)

    # reset() is what makes it per-episode. Emulate the two lines that matter.
    s._episode_noise = None
    n3 = s._chunk_noise(4)
    check("reset -> a new episode gets a NEW branch", not torch.equal(n1, n3))

    n4 = s._chunk_noise(8)
    check("batch size change redraws rather than broadcasting",
          tuple(n4.shape) == (8, 16, 7))

    # The eval harness batches envs; two envs in one batch must not be handed
    # the same branch, or 10 "independent" episodes are 1 episode 10 times.
    n5 = s._chunk_noise(8)
    check("envs within a batch get DIFFERENT branches",
          not torch.allclose(n5[0], n5[1]))


def test_attention_mass():
    """The attention diagnostic patches F.scaled_dot_product_attention. Two
    things must hold or it is worse than useless: the patch must not change
    what the model computes, and the recorded probabilities must be the real
    ones. Its segment table must also account for EVERY key -- a share that
    silently sums to less than 1 would understate whatever is missing."""
    print("\nattention_mass diagnostic")

    src = (ROOT.parent / "attention_mass_wiltechs_x.py").read_text()
    ns = {"torch": torch, "F": torch.nn.functional,
          "contextmanager": __import__("contextlib").contextmanager}
    exec(src[src.index("@contextmanager"):src.index("def main(")], ns)
    record_attention, segment_ranges = ns["record_attention"], ns["segment_ranges"]

    # ---- the patch must be transparent -----------------------------------
    torch.manual_seed(0)
    q = torch.randn(2, 4, 25, 16)
    k = torch.randn(2, 4, 100, 16)
    v = torch.randn(2, 4, 100, 16)
    m = torch.zeros(2, 1, 25, 100)
    m[:, :, :, 90:] = torch.finfo(torch.float32).min      # mask a tail
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m)

    store = []
    with record_attention(store, 25):
        got = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m)
    check("patched SDPA returns the unpatched result", torch.allclose(ref, got))
    check("SDPA is restored on exit",
          torch.nn.functional.scaled_dot_product_attention is not None
          and torch.allclose(
              torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m),
              ref))
    check("one call recorded", len(store) == 1)
    check("recorded probs are (B, Lq, Lk), head-averaged",
          tuple(store[0].shape) == (2, 25, 100))
    check("probabilities sum to 1 per query",
          torch.allclose(store[0].sum(-1), torch.ones(2, 25), atol=1e-5))
    check("masked keys get no mass",
          float(store[0][:, :, 90:].abs().max()) < 1e-6)

    # Only the expert's suffix queries should be captured, not the prefix-only
    # VLM layers -- those have L_prefix queries and would swamp the average.
    store2 = []
    with record_attention(store2, 25):
        torch.nn.functional.scaled_dot_product_attention(
            torch.randn(2, 4, 100, 16), k, v)                # L_p queries
        torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m)
    check("only matching query lengths are recorded", len(store2) == 1)

    # ---- the segment table must tile the whole sequence -------------------
    spans = {"lang": (5, 53), "vision": [(55, 119), (121, 185)],
             "wrist": [(185, 441)], "motion": (441, 449), "readout": 451}
    L_p, L_s = 452, 25
    segs = segment_ranges(spans, L_p, L_s, 16)
    covered = torch.zeros(L_p + L_s, dtype=torch.bool)
    dup = False
    for rs in segs.values():
        for s, e in rs:
            if covered[s:e].any():
                dup = True
            covered[s:e] = True
    check("every key belongs to exactly one segment", bool(covered.all()) and not dup)
    check("the chat template / vision brackets are not lost",
          "template" in segs and sum(e - s for s, e in segs["template"]) == 452 - (
              48 + 128 + 256 + 8 + 1))
    check("suffix self-attention is its own segment",
          segs["self(suffix)"] == [(452, 477)])

    # A checkpoint without wrist or motion must not produce phantom segments.
    lean = segment_ranges({"lang": (5, 53), "vision": [(55, 119)], "readout": 120},
                          121, 25, 16)
    check("absent segments are omitted, not zero-filled",
          "wrist" not in lean and "motion" not in lean and "lang" in lean)


def test_paraphrase():
    import re
    """The model scores 60% on an instruction and 0% on a paraphrase of it, so
    it keys on surface form. These checks are about the augmentation being an
    honest fix for that: the original must survive (eval uses it), the MEANING
    must not move (a variant that renames the target trains the wrong task),
    and a sentence the pattern does not understand must be left alone rather
    than mangled."""
    print("\nparaphrase augmentation")

    P = _load("models.wiltechs_x.paraphrase", ROOT / "wiltechs_x" / "paraphrase.py")

    SP = "pick up the black bowl on the stove and place it on the plate"
    v = P.paraphrases(SP)
    check("original is present and first", v[0] == SP)
    check("produces several variants", len(v) >= 6)
    check("all variants are distinct", len(set(v)) == len(v))
    # The relation and the destination are the meaning. If either moves, the
    # policy is being trained on a different task under the same reward.
    check("every variant keeps the relation phrase",
          all("on the stove" in x for x in v))
    check("every variant keeps the destination",
          all(x.rstrip(". ").endswith("the plate") for x in v))
    check("every variant still names the object",
          all("black bowl" in x for x in v))

    # "the bowl that is from table center" is not English; the generator must
    # not attach a qualifier to a source phrase.
    fc = P.paraphrases("pick up the black bowl from table center and place it on the plate")
    check("no qualifier on a non-locative relation",
          not any("that is from" in x or "which is from" in x for x in fc))

    # A multi-word relation containing "and" must not be split at the first one.
    bt = P.paraphrases("pick up the black bowl between the plate and the ramekin "
                       "and place it on the plate")
    check("relation containing 'and' is parsed whole",
          all("between the plate and the ramekin" in x for x in bt))

    # Unknown structure -> leave it alone rather than guess.
    other = P.paraphrases("open the top drawer of the cabinet")
    check("unmatched sentence is returned unchanged, not mangled",
          other == ["open the top drawer of the cabinet"])

    check("limit is respected and keeps the original",
          len(P.paraphrases(SP, limit=3)) == 3
          and P.paraphrases(SP, limit=3)[0] == SP)
    check("deterministic across calls", P.paraphrases(SP) == P.paraphrases(SP))

    # Two different tasks must never generate a shared variant, or a sample of
    # one task would be handed the other's instruction as if it were correct.
    a = set(P.paraphrases(SP))
    b = set(P.paraphrases("pick up the black bowl on the wooden cabinet "
                          "and place it on the plate"))
    check("variant sets of two tasks never collide", not (a & b))

    # A destination reached with "in" must never be rewritten to "on": the
    # reward stays the original task's, so the policy would be trained to do
    # one thing while told another. The first version substituted on/onto
    # unconditionally and would have done this to every libero_object task.
    ob = P.paraphrases("pick up the alphabet soup and place it in the basket")
    check("libero_object form is augmented too (relation is optional)", len(ob) >= 5)
    check("preposition stays inside its equivalence class",
          all(re.search(r"\b(in|into|inside)\s+the basket$", x) for x in ob))
    check("no 'set it into/inside' (not idiomatic)",
          not any("set it into" in x or "set it inside" in x for x in ob))

    cov, under = P.coverage([SP, "turn on the stove"], 8, 5)
    check("coverage names the instructions below the minimum",
          under == ["turn on the stove"] and len(cov[SP]) >= 5)

    tbl = P.build_table([SP, SP, "pick up the black bowl on the ramekin "
                                 "and place it on the plate"])
    check("build_table deduplicates instructions", len(tbl) == 2)

    # A hand-edited table that drops the original would train on phrasings the
    # model is never scored with.
    import json, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({SP: ["grasp the black bowl on the stove and put it on the plate"]}, f)
        bad = f.name
    try:
        P.load_table(bad)
        check("load_table rejects a table missing the original", False)
    except ValueError:
        check("load_table rejects a table missing the original", True)


if __name__ == "__main__":
    mask = test_mask()
    test_cached_equals_reference(*test_joint_layer(mask))
    test_lora_and_discrete()
    test_long_horizon()
    test_motion_history_guard()
    test_hinge_negatives()
    test_episode_noise()
    test_attention_mass()
    test_paraphrase()
    print("\nRESULT:", "ALL PASS" if _ok else "FAILURES ABOVE")
    sys.exit(0 if _ok else 1)
