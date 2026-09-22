"""Build the best-of-N score function from a trained scorer checkpoint.

Two spaces have to be reconciled and neither is negotiable:

  ACTIONS. `sample_actions` returns chunks in the POLICY's normalized space --
  obs2-22k's, i.e. lerobot/libero's mean/std. The scorer was trained on the
  CORPUS's, ISdept/libero-awr-goal's. Those differ by up to 4.9% in std and
  0.053 sigma in mean (measured 2026-09-20 with compare_norm_stats.py). Small,
  but best-of-N is a fine discrimination between candidates that are already
  nearly identical, so a 5% systematic skew is not something to hand-wave.
  Both maps are affine, so the correction is exact and costs one multiply-add.

  IMAGES. The scorer's ResNet trunk expects raw [0, 1] frames and applies its
  own ImageNet normalisation. By the time `select_action` sees the batch, the
  policy's preprocessor has already had it. The eval loop therefore hands over
  the untouched frames separately.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from models.scorer import ActionScorer


def _stats_from(preprocessor, key: str):
    """(mean, std) for `key` out of a lerobot processor pipeline, or None."""
    for step in getattr(preprocessor, "steps", []) or []:
        stats = getattr(step, "stats", None)
        if isinstance(stats, dict) and key in stats:
            s = stats[key]
            get = (lambda k: s.get(k)) if isinstance(s, dict) else (lambda k: getattr(s, k, None))
            m, sd = get("mean"), get("std")
            if m is not None and sd is not None:
                return (torch.as_tensor(np.asarray(m, dtype=np.float32)),
                        torch.as_tensor(np.asarray(sd, dtype=np.float32)))
    return None


def build_selector(scorer_path: str, preprocessor, device, verbose: bool = True):
    """-> (score_fn, info). score_fn(raw_images, batch, candidates) -> (B, K), lower better."""
    ck = torch.load(scorer_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = ActionScorer(
        horizon=a["horizon"], action_dim=ck["a_mean"].numel(),
        state_dim=ck["s_mean"].numel(), n_cams=len(ck["cams"]),
        input_size=a["input_size"], vis_tokens=a["vis_tokens"],
        d_model=a["d_model"], n_layers=a["n_layers"]).to(device).eval()
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad_(False)

    cams = list(ck["cams"])
    sc_am, sc_as = ck["a_mean"].to(device), ck["a_std"].to(device)
    sc_sm, sc_ss = ck["s_mean"].to(device), ck["s_std"].to(device)

    pol_a = _stats_from(preprocessor, "action")
    pol_s = _stats_from(preprocessor, "observation.state")
    if pol_a is None or pol_s is None:
        print("ERROR: could not read action / observation.state normalization out "
              "of the policy's preprocessor, so the scorer's inputs cannot be put "
              "in the space it was trained on. Refusing to score in the wrong "
              "units -- that would look like a working run and produce a "
              "meaningless selection.", file=sys.stderr)
        raise SystemExit(1)

    # policy-normalized -> raw -> scorer-normalized, collapsed to one affine
    a_mul = (pol_a[1].to(device) / sc_as)
    a_add = ((pol_a[0].to(device) - sc_am) / sc_as)
    s_mul = (pol_s[1].to(device) / sc_ss)
    s_add = ((pol_s[0].to(device) - sc_sm) / sc_ss)

    if verbose:
        print(f"[scorer] {Path(scorer_path).name}  step {ck.get('step')}  "
              f"cams {cams}  horizon {a['horizon']}")
        print(f"[scorer] action  scale {a_mul.min():.4f}..{a_mul.max():.4f}   "
              f"offset {a_add.abs().max():.4f} (policy -> scorer units)")
        print(f"[scorer] state   scale {s_mul.min():.4f}..{s_mul.max():.4f}   "
              f"offset {s_add.abs().max():.4f}")
        if float((a_mul - 1).abs().max()) < 1e-6 and float(a_add.abs().max()) < 1e-6:
            print("[scorer] the two normalizations are identical -- the scorer was "
                  "trained on the policy's own stats")

    warned = {"range": False, "cams": False}

    @torch.no_grad()
    def score_fn(raw_images, batch, candidates):
        B, K = candidates.shape[0], candidates.shape[1]
        if raw_images is None:
            raise RuntimeError(
                "best-of-N is on but the eval loop never called "
                "policy.set_selection_images(); the scorer would be reading the "
                "preprocessor's normalized frames instead of raw [0,1] ones.")
        missing = [c for c in cams if c not in raw_images]
        if missing and not warned["cams"]:
            warned["cams"] = True
            print(f"WARNING: scorer wants {cams} but the batch has "
                  f"{sorted(raw_images)}; missing {missing} will be zeros.")
        imgs = torch.stack(
            [raw_images[c].to(device).float() if c in raw_images
             else torch.zeros_like(next(iter(raw_images.values())), device=device)
             for c in cams], dim=1)
        if not warned["range"]:
            warned["range"] = True
            lo, hi = float(imgs.min()), float(imgs.max())
            print(f"[scorer] first image batch {tuple(imgs.shape)} range "
                  f"[{lo:.3f}, {hi:.3f}]")
            if lo < -0.05 or hi > 1.05:
                print("WARNING: those frames are not in [0, 1]. The scorer's "
                      "trunk applies ImageNet normalization on the assumption "
                      "that they are, so every score below is suspect.")

        st = batch["observation.state"]
        st = st[:, -1] if st.dim() == 3 else st          # current frame of the window
        st = st.to(device) * s_mul + s_add
        cand = candidates.to(device) * a_mul + a_add
        return model.score_candidates(imgs, st, cand)

    return score_fn, {"cams": cams, "horizon": a["horizon"], "step": ck.get("step"),
                      "rank_acc_note": "see the training log; below 0.65 do not use"}
