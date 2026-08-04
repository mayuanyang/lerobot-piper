"""WiltechsMoE - Mixture-of-Experts encoder-decoder flow matching policy."""
import math
from contextlib import nullcontext
from typing import Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from .wiltechs_moe_config import WiltechsMoEConfig
from ..wiltechs_vla.task_rewrites import rewrite_instruction
from ..interleaved_flow_matching.expert_layer import RMSNorm, SwiGLU
from ..transformer_flow_matching.robot_visual_encoder import RobotVisualEncoder
from ..wiltechs_vla.wiltechs_vla_model import (
    DiTLayer, LatentQFormer, create_sinusoidal_pos_embedding,
    preprocess_camera_to_pixels, vlm_pixels_key, vlm_grid_key,
    _apply_rope, _build_mrope_position_ids, _modulate, _hard_negative_perm,
)

class MoERouter(nn.Module):
    def __init__(self, hidden_size, num_experts, vlm_hidden_size, temperature=1.0, top_k=0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.top_k = top_k
        # Project pooled VLM hidden state (vlm_hidden_size) -> dit_hidden for router input.
        # The VLM hidden state contains BOTH vision and language information (they are
        # processed together through the VLM's causal attention), so a single
        # pool of the final layer gives the Router access to the full multimodal
        # context: what the scene looks like + what the instruction says.
        self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)
        # Input: [state, vlm_semantic, time, action] each hidden_size
        self.router = nn.Sequential(
            nn.Linear(4 * hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, num_experts))
        # Small random init (NOT zeros). Zero init makes all logits identical at
        # step 0; any tiny perturbation from the data gradient then tips one
        # expert slightly ahead, and the positive-feedback softmax loop
        # collapses to a single expert within ~100 steps (observed: E3=100% by
        # step 200).  std=0.02 gives near-uniform but non-degenerate starting
        # logits so the router can learn to differentiate.
        nn.init.normal_(self.router[-1].weight, std=0.02)
        nn.init.normal_(self.router[-1].bias, std=0.02)
    def forward(self, state_emb, vlm_semantic_emb, time_emb, action_emb):
        """vlm_semantic_emb: pooled VLM hidden state (B, vlm_hidden_size), contains both
        vision and language context from the VLM's final layer."""
        B, H = state_emb.shape[0], state_emb.shape[-1]
        vlm_proj = self.vlm_proj(vlm_semantic_emb)  # (B, hidden_size)
        action_pool = action_emb.mean(dim=1)
        state_flat = state_emb.squeeze(1) if state_emb.dim() == 3 else state_emb
        logits = self.router(torch.cat([state_flat, vlm_proj, time_emb, action_pool], dim=-1)) / max(self.temperature, 1e-6)
        # Diagnostics read the PRE-noise weights: those are what inference
        # actually uses, and the exploration noise below inflates peakedness by
        # a lot at these logit scales (a router with zero input dependence still
        # reports max_w 0.39 / entropy 1.30 once N(0, 0.5) is added, not the
        # 0.25 / 1.386 that "uniform" suggests). Comparing the noisy statistic
        # against the uniform reference therefore reads "differentiating" for a
        # fully collapsed router.
        self._last_clean_weights = F.softmax(logits, dim=-1).detach()
        # Inject noise during training to prevent router collapse. The noise
        # keeps "dead" experts receiving exploration signal early in training,
        # breaking the winner-take-all positive feedback loop. It is a FIXED
        # 0.5, not scaled to the logit magnitude, so it does not wash out as the
        # router grows confident -- it stops mattering only once the learned
        # logit spread is large relative to 0.5.
        if self.training:
            logits = logits + torch.randn_like(logits) * 0.5
        if self.top_k > 0 and self.top_k < self.num_experts:
            _, topk_idx = logits.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
            weights = F.softmax(logits, dim=-1) * mask
            weights = weights / (weights.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        else:
            weights = F.softmax(logits, dim=-1)
        return weights, weights.mean(dim=0)

class ExpertDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, sa_num_heads, sa_num_kv_heads, sa_head_dim,
                 ca_num_heads, ca_num_kv_heads, ca_head_dim, intermediate_size, rms_norm_eps=1e-5, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            DiTLayer(hidden_size=hidden_size, sa_num_heads=sa_num_heads, sa_num_kv_heads=sa_num_kv_heads,
                     sa_head_dim=sa_head_dim, ca_num_heads=ca_num_heads, ca_num_kv_heads=ca_num_kv_heads,
                     ca_head_dim=ca_head_dim, intermediate_size=intermediate_size,
                     rms_norm_eps=rms_norm_eps, dropout=dropout)
            for _ in range(num_layers)])
    def forward(self, x, t_emb, expert_kv_cache, vlm_kv_pad_mask, self_attn_mask):
        for i, layer in enumerate(self.layers):
            vlm_k, vlm_v = expert_kv_cache[i % len(expert_kv_cache)]
            x = layer(x, t_emb=t_emb, vlm_k=vlm_k, vlm_v=vlm_v,
                      vlm_kv_pad_mask=vlm_kv_pad_mask, self_attn_mask=self_attn_mask)
        return x

class WiltechsMoETransformer(nn.Module):
    VLM_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(f"Loading {self.VLM_MODEL_ID} ...")
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(self.VLM_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.processor = AutoProcessor.from_pretrained(self.VLM_MODEL_ID)
        self.vlm_model = vlm.model
        self.visual = self.vlm_model.visual
        self.language_model = self.vlm_model.language_model
        self.num_vlm_layers = len(self.language_model.layers)
        text_cfg = self.language_model.config
        self.hidden_size = int(text_cfg.hidden_size)
        self.num_heads = int(text_cfg.num_attention_heads)
        self.num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(text_cfg, "head_dim", None) or (self.hidden_size // self.num_heads))
        self.intermediate_size = int(self.hidden_size * 2)
        print(f"[wiltechs_moe] VLM hidden_size={self.hidden_size}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, intermediate_size={self.intermediate_size}")
        self.rms_norm_eps = float(getattr(text_cfg, "rms_norm_eps", 1e-5))
        vis_cfg = getattr(vlm.config, "vision_config", None)
        self.spatial_merge_size = int(getattr(vis_cfg, "spatial_merge_size", 2))
        if config.d_model != self.hidden_size: config.d_model = self.hidden_size
        if not hasattr(self.language_model, "rotary_emb"): raise RuntimeError("language_model.rotary_emb not found.")
        for p in self.visual.parameters(): p.requires_grad = False
        for p in self.language_model.parameters(): p.requires_grad = False
        self.visual.eval(); self.language_model.eval(); del vlm
        num_experts = int(config.num_experts)
        expert_depth = int(config.expert_num_layers)
        if config.vlm_capture_layers:
            capture_layers = sorted(config.vlm_capture_layers)
        else:
            total_needed = num_experts * expert_depth
            if total_needed > self.num_vlm_layers: raise ValueError(f"num_experts({num_experts}) * expert_num_layers({expert_depth}) = {total_needed} > VLM layers ({self.num_vlm_layers})")
            capture_layers = list(range(self.num_vlm_layers)) if total_needed == self.num_vlm_layers else np.linspace(0, self.num_vlm_layers - 1, total_needed, dtype=int).tolist()
        self.capture_layers = capture_layers
        print(f"[wiltechs_moe] VLM capture layers ({len(capture_layers)}): {capture_layers}")
        if len(capture_layers) % num_experts != 0: raise ValueError(f"capture_layers count ({len(capture_layers)}) not divisible by num_experts ({num_experts})")
        layers_per_expert = len(capture_layers) // num_experts
        self.expert_kv_blocks = [capture_layers[e * layers_per_expert:(e + 1) * layers_per_expert] for e in range(num_experts)]
        for e, block in enumerate(self.expert_kv_blocks): print(f"  Expert {e}: VLM layers {block}")
        self.dit_hidden = int(getattr(config, "dit_hidden_size", 0)) or self.hidden_size
        if self.dit_hidden % self.head_dim != 0: raise ValueError(f"dit_hidden_size ({self.dit_hidden}) must be divisible by head_dim ({self.head_dim}).")
        ca_nh, ca_nkv, ca_hd = self.num_heads, self.num_kv_heads, self.head_dim
        if self.dit_hidden == self.hidden_size:
            sa_nh, sa_nkv, sa_hd = self.num_heads, self.num_kv_heads, self.head_dim; dit_intermediate = self.intermediate_size
        else:
            sa_hd = self.head_dim; sa_nh = self.dit_hidden // sa_hd
            gqa_ratio = max(1, self.num_heads // max(1, self.num_kv_heads))
            sa_nkv = max(1, sa_nh // gqa_ratio)
            while sa_nh % sa_nkv != 0: sa_nkv -= 1
            # QFormer cross-attn: ca_num_heads must be divisible by VLM ca_nkv
            ca_qformer_nh = (sa_nh // ca_nkv) * ca_nkv
            if ca_qformer_nh == 0: ca_qformer_nh = ca_nkv
            dit_intermediate = self.dit_hidden
        self.experts = nn.ModuleList([ExpertDecoder(hidden_size=self.dit_hidden, num_layers=expert_depth, sa_num_heads=sa_nh, sa_num_kv_heads=sa_nkv, sa_head_dim=sa_hd, ca_num_heads=ca_nh, ca_num_kv_heads=ca_nkv, ca_head_dim=ca_hd, intermediate_size=dit_intermediate, rms_norm_eps=self.rms_norm_eps, dropout=config.dropout) for _ in range(num_experts)])
        self.router = MoERouter(hidden_size=self.dit_hidden, num_experts=num_experts, vlm_hidden_size=self.hidden_size, temperature=float(config.router_temperature), top_k=int(config.router_top_k))
        self.sink_token = nn.Parameter(torch.zeros(1, 1, self.dit_hidden)); nn.init.normal_(self.sink_token, std=0.02)
        self.state_encoder = nn.Sequential(nn.Linear(config.state_dim, self.dit_hidden), RMSNorm(self.dit_hidden, eps=self.rms_norm_eps))
        self.action_in_proj = nn.Linear(config.action_dim, self.dit_hidden)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, config.horizon, self.dit_hidden)); nn.init.normal_(self.action_pos_emb, std=0.02)
        self.final_norm = RMSNorm(self.dit_hidden, eps=self.rms_norm_eps)
        self.action_out_proj = nn.Linear(self.dit_hidden, config.action_dim); nn.init.zeros_(self.action_out_proj.weight); nn.init.zeros_(self.action_out_proj.bias)
        self.time_embedder = nn.Sequential(nn.Linear(self.dit_hidden, self.dit_hidden), nn.SiLU(), nn.Linear(self.dit_hidden, self.dit_hidden))
        if config.use_robot_cnn: self.robot_visual_encoder = RobotVisualEncoder(input_size=config.robot_encoder_input_size, out_tokens=config.robot_encoder_tokens, out_dim=self.dit_hidden)
        else: self.robot_visual_encoder = None
        self.num_latent_tokens = config.num_latent_tokens
        # The LATENT Q-Former is off (num_latent_tokens defaults to 0) -- each
        # expert cross-attends to VLM KV directly. Not to be confused with the
        # THOUGHT Q-Former built below, which is a separate module and is on.
        # The latents plumbing is kept live throughout for the config that
        # re-enables it; "legacy latents" in the sequence comment is this.
        self.latent_qformer = None

        # ------------------------------------------------------------------
        # Thought tokens -- spatial reasoning bottleneck
        # ------------------------------------------------------------------
        # A learned-query Q-Former distills the DEEPEST captured VLM layer's
        # KV cache into K "thought" tokens.  The deepest layer has the most
        # semantic fusion of vision + language (the VLM's causal attention
        # has fully mixed "what the scene looks like" with "what the
        # instruction says to do"), so cross-attending here extracts spatial
        # reasoning: target object location, gripper-to-target relative
        # position, goal placement, etc.
        #
        # These thought tokens are prepended to the expert input sequence
        # (before action tokens) so that in causal self-attention every
        # action token can read the thought.  They are noise-independent
        # (depend only on the observation, not the diffusion timestep), so
        # they are computed ONCE per forward and shared across all N
        # denoising steps -- zero added cost in the denoising loop.
        #
        # This complements (does NOT replace) per-expert cross-attention:
        # experts still cross-attend to their own VLM KV blocks, but the
        # thought tokens provide a compressed, query-focused summary that
        # acts as an explicit reasoning bottleneck -- forcing the model to
        # "think about where things are" before generating actions.
        self.num_thought_tokens = int(getattr(config, "num_thought_tokens", 0))
        if self.num_thought_tokens > 0:
            # Resolve which VLM layer to read KV from for thought generation.
            thought_layer_cfg = int(getattr(config, "thought_vlm_layer_idx", -1))
            if thought_layer_cfg < 0:
                self.thought_vlm_layer = max(self.capture_layers)
            else:
                self.thought_vlm_layer = thought_layer_cfg
            print(f"[wiltechs_moe] Thought tokens: {self.num_thought_tokens} tokens, "
                  f"Q-Former layers={int(getattr(config, 'thought_qformer_layers', 2))}, "
                  f"VLM layer={self.thought_vlm_layer}")
            self.thought_qformer = LatentQFormer(
                dim=self.dit_hidden,
                num_queries=self.num_thought_tokens,
                n_layers=int(getattr(config, "thought_qformer_layers", 2)),
                ca_num_heads=ca_qformer_nh,
                ca_num_kv_heads=ca_nkv,
                ca_head_dim=sa_hd,
                intermediate_size=self.dit_hidden,
                rms_norm_eps=self.rms_norm_eps,
            )
        else:
            self.thought_qformer = None
            self.thought_vlm_layer = -1

        # 128, not 48: the CoT rewrites from task_rewrites.py (enabled by
        # --use_descriptive_objects) run up to ~105 tokens. At 48 they were cut
        # mid-Location, dropping the selector and the entire Action clause, so
        # the model trained on a prompt that ended partway through the
        # description of where the target is. _report_lang_budget prints the
        # real count every run and says TRUNCATED if this is ever too small.
        self._lang_max_len = 128
        self.text_first = bool(getattr(config, "text_first", True))
        self._template_ids_cpu = None; self._template_format_printed = False
        self._lang_len_printed = False; self._vision_grid_printed = set()
        self.gradient_checkpointing = False
    def train(self, mode=True):
        super().train(mode); self.visual.eval(); self.language_model.eval(); return self
    def gradient_checkpointing_enable(self): self.gradient_checkpointing = True; print("[wiltechs_moe] Expert gradient checkpointing ENABLED")
    def gradient_checkpointing_disable(self): self.gradient_checkpointing = False
    def _find_visual_merger(self):
        for owner in (self.visual, self.vlm_model):
            for attr in ("merger", "patch_merger", "visual_merger", "merger_module"):
                candidate = getattr(owner, attr, None)
                if candidate is not None: return candidate
        return None
    def cam_target_size(self, cam_key):
        """Square input side length for this camera's Qwen preprocessing, or 0
        for the processor default. Shared with the DataLoader-worker path so
        both produce identical grids."""
        vs = int(getattr(self.config, "vision_input_size", 0) or 0)
        if vs <= 0: return 0
        hires = list(getattr(self.config, "vision_hires_cameras", None) or [])
        return vs if (not hires or cam_key in hires) else 0

    def _report_vision_grid(self, cam_key, image_grid_thw):
        """One line per camera at startup. The pixel-bound plumbing in
        preprocess_camera_to_pixels varies by transformers version, so this is
        the ONLY trustworthy confirmation that a resolution change took."""
        if cam_key in self._vision_grid_printed: return
        self._vision_grid_printed.add(cam_key)
        g = image_grid_thw[0].tolist() if image_grid_thw.dim() > 1 else image_grid_thw.tolist()
        m = self.spatial_merge_size
        gh, gw = int(g[1]) // m, int(g[2]) // m
        print(f"[wiltechs_moe] vision grid {cam_key}: patch_thw={g} -> {gh}x{gw} merged "
              f"= {gh * gw} tokens/frame (target_size={self.cam_target_size(cam_key) or 'processor default'})")

    def _encode_images(self, batch, B):
        device = batch["observation.state"].device; all_vis = []; grid_thw_list = []
        for cam_key in self.config.cameras_for_vision_state_concat:
            pvk, thwk = vlm_pixels_key(cam_key), vlm_grid_key(cam_key)
            with torch.no_grad():
                if pvk in batch:
                    pv = batch[pvk]; image_grid_thw = batch[thwk]
                    if pv.dim() == 3: pv = pv.reshape(-1, pv.shape[-1])
                    if image_grid_thw.dim() == 1: image_grid_thw = image_grid_thw.unsqueeze(0)
                    pixel_values = pv.to(device=device); image_grid_thw = image_grid_thw.to(device=device)
                elif cam_key in batch:
                    imgs = batch[cam_key]; img = imgs[:, -1] if imgs.dim() == 5 else imgs
                    pixel_values, image_grid_thw = preprocess_camera_to_pixels(
                        self.processor.image_processor, img, target_size=self.cam_target_size(cam_key))
                    pixel_values = pixel_values.to(device=device); image_grid_thw = image_grid_thw.to(device=device)
                else: continue
                self._report_vision_grid(cam_key, image_grid_thw)
                try: vis_tokens = self.visual(pixel_values, grid_thw=image_grid_thw)
                except TypeError: vis_tokens = self.visual(pixel_values, image_thw=image_grid_thw)
                vis_tokens = getattr(vis_tokens, "last_hidden_state", vis_tokens)
                if vis_tokens.shape[-1] != self.hidden_size:
                    merger = self._find_visual_merger()
                    if merger is None: raise RuntimeError("No merger found.")
                    try: vis_tokens = merger(vis_tokens)
                    except TypeError: vis_tokens = merger(vis_tokens, image_grid_thw)
                    vis_tokens = getattr(vis_tokens, "last_hidden_state", vis_tokens)
            if vis_tokens.dim() == 2:
                if vis_tokens.shape[0] % B != 0: raise RuntimeError(f"Cannot unpack vis_tokens shape {tuple(vis_tokens.shape)} with B={B}.")
                vis_tokens = vis_tokens.reshape(B, -1, self.hidden_size)
            all_vis.append(vis_tokens); grid_thw_list.append(image_grid_thw[0].detach())
        if not all_vis: return torch.zeros(B, 0, self.hidden_size, device=device, dtype=torch.bfloat16), []
        return torch.cat(all_vis, dim=1), grid_thw_list
    def _resolve_descs(self, batch, descs_override=None):
        descs = descs_override if descs_override is not None else (batch.get("task_description") or batch.get("task"))
        # rewrite_instruction is idempotent (a rewritten string is not a key in
        # REPHRASINGS), so re-resolving an already-rewritten override is safe.
        if descs and getattr(self.config, "use_descriptive_objects", False): descs = [rewrite_instruction(d) for d in descs]
        return descs
    def _encode_language(self, batch, device):
        descs = self._resolve_descs(batch)
        if not descs or not any(descs): return None
        inputs = self.processor.tokenizer(descs, return_tensors="pt", padding=True, truncation=True, max_length=self._lang_max_len, add_special_tokens=True)
        input_ids = inputs["input_ids"].to(device); lang_mask = inputs["attention_mask"].bool().to(device)
        self._report_lang_budget(descs, input_ids, lang_mask)
        lang_tokens = self.language_model.get_input_embeddings()(input_ids)
        return lang_tokens, lang_mask
    def _report_lang_budget(self, texts, lang_ids, lang_mask):
        """One-time print of the token budget vs. the longest instruction.

        Truncation here is silent and destroys exactly the disambiguating tail
        of the CoT rewrites, so make it visible at startup instead of leaving
        it to be inferred from rollout behaviour."""
        if self._lang_len_printed: return
        self._lang_len_printed = True
        tok = self.processor.tokenizer
        lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        i = int(np.argmax(lens)); full = tok(texts[i], add_special_tokens=False)["input_ids"]
        print(f"[wiltechs_moe] lang budget: max_len={self._lang_max_len}, longest instruction in batch={lens[i]} tokens, kept={int(lang_mask[i].sum().item())}")
        print(f"[wiltechs_moe]   kept: {tok.decode(lang_ids[i][lang_mask[i]])!r}")
        if lens[i] > self._lang_max_len:
            print(f"[wiltechs_moe]   *** TRUNCATED *** dropped tail: {tok.decode(full[self._lang_max_len:])!r}")
    def _get_template_ids(self, device):
        if self._template_ids_cpu is None:
            tok = self.processor.tokenizer
            prefix_ids = tok("<|im_start|>user\n", add_special_tokens=False, return_tensors="pt")["input_ids"][0].long()
            asst_ids = tok("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False, return_tensors="pt")["input_ids"][0].long()
            vs = tok.convert_tokens_to_ids("<|vision_start|>"); ve = tok.convert_tokens_to_ids("<|vision_end|>")
            self._template_ids_cpu = (prefix_ids, torch.tensor([vs], dtype=torch.long), torch.tensor([ve], dtype=torch.long), asst_ids)
        return tuple(t.to(device) for t in self._template_ids_cpu)
    @torch.no_grad()
    def _run_vlm_and_cache_kv(self, batch, descs_override=None, vis_pack=None):
        """Run the frozen VLM and capture per-layer K/V.

        Returns (kv_cache, vlm_kv_pad_mask, vis_mask, lang_span, hidden, vis_pack)
        where lang_span is the (start, end) index range of the per-sample
        instruction tokens inside the VLM sequence, and vis_pack is the
        (vis_tokens, grid_thw_list) ViT output so a second call with a
        different instruction can skip re-encoding the images.
        """
        B = batch["observation.state"].shape[0]; device = batch["observation.state"].device
        if vis_pack is not None: vis_tokens, grid_thw_list = vis_pack
        else: vis_tokens, grid_thw_list = self._encode_images(batch, B)
        vis_pack_out = (vis_tokens, grid_thw_list); L_vis = vis_tokens.shape[1]
        descs = self._resolve_descs(batch, descs_override); have_lang = bool(descs) and any(descs)
        use_template = bool(getattr(self.config, "use_chat_template", False)) and have_lang
        text_first = self.text_first and have_lang
        embed_tokens = self.language_model.get_input_embeddings()
        if text_first:
            # ---- instruction BEFORE images ------------------------------
            # Under the VLM's causal mask this makes every vision patch's K/V
            # at every layer conditioned on the instruction, so the experts'
            # cross-attention reads a language-grounded feature map rather
            # than a language-blind one.
            m = self.spatial_merge_size
            cam_sizes = [int(g[0].item()) * (int(g[1].item()) // m) * (int(g[2].item()) // m) for g in grid_thw_list]
            cam_tokens = list(vis_tokens.split(cam_sizes, dim=1)) if cam_sizes else []
            directive = str(getattr(self.config, "chat_directive", "") or "").strip()
            texts = [(f"{directive} {d}" if directive else str(d)) for d in descs]
            lang = self.processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                            max_length=self._lang_max_len, add_special_tokens=not use_template)
            lang_ids = lang["input_ids"].to(device); lang_mask = lang["attention_mask"].bool().to(device)
            L_lang = lang_ids.shape[1]
            self._report_lang_budget(texts, lang_ids, lang_mask)
            lang_emb = embed_tokens(lang_ids)
            lang_emb = torch.where(lang_mask.unsqueeze(-1), lang_emb, torch.zeros_like(lang_emb))
            parts = []; segments = []; vis_flags = []; head_len = 0; asst_ids = None
            if use_template:
                prefix_ids, vs_id, ve_id, asst_ids = self._get_template_ids(device)
                head_len = prefix_ids.shape[0]
                parts.append(embed_tokens(prefix_ids).unsqueeze(0).expand(B, -1, -1))
                segments.append(("text", head_len)); vis_flags += [False] * head_len
            parts.append(lang_emb); segments.append(("text", L_lang)); vis_flags += [False] * L_lang
            for ct, g in zip(cam_tokens, grid_thw_list):
                if use_template:
                    vs_emb = embed_tokens(vs_id).unsqueeze(0).expand(B, -1, -1)
                    ve_emb = embed_tokens(ve_id).unsqueeze(0).expand(B, -1, -1)
                    parts += [vs_emb, ct, ve_emb]; segments += [("text", 1), ("image", g), ("text", 1)]
                    vis_flags += [False] + [True] * ct.shape[1] + [False]
                else:
                    parts.append(ct); segments.append(("image", g)); vis_flags += [True] * ct.shape[1]
            if use_template:
                parts.append(embed_tokens(asst_ids).unsqueeze(0).expand(B, -1, -1))
                segments.append(("text", asst_ids.shape[0])); vis_flags += [False] * asst_ids.shape[0]
            vlm_seq = torch.cat(parts, dim=1).to(torch.bfloat16); L_vlm = vlm_seq.shape[1]
            lang_span = (head_len, head_len + L_lang)
            vis_mask = torch.tensor(vis_flags, device=device, dtype=torch.bool)
            # Padded instruction positions sit mid-sequence; they are masked out
            # as attention KEYS, so the images never read them. M-RoPE positions
            # are built from the padded length uniformly across the batch, which
            # is a constant offset for the image block -- harmless under RoPE.
            vlm_kv_pad_mask = torch.cat([
                torch.ones(B, head_len, device=device, dtype=torch.bool), lang_mask,
                torch.ones(B, L_vlm - head_len - L_lang, device=device, dtype=torch.bool),
            ], dim=1)
            if not self._template_format_printed:
                self._template_format_printed = True
                print(f"[wiltechs_moe] TEXT-FIRST layout ON (chat_template={use_template}) - "
                      f"L_vlm={L_vlm}, lang span={lang_span}, L_vis={L_vis}")
        elif use_template:
            m = self.spatial_merge_size
            cam_sizes = [int(g[0].item()) * (int(g[1].item()) // m) * (int(g[2].item()) // m) for g in grid_thw_list]
            cam_tokens = list(vis_tokens.split(cam_sizes, dim=1)) if cam_sizes else []
            directive = str(getattr(self.config, "chat_directive", "") or "").strip()
            texts = [(f"{directive} {d}" if directive else str(d)) + "<|im_end|>\n<|im_start|>assistant\n" for d in descs]
            suf = self.processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self._lang_max_len + 24, add_special_tokens=False)
            suffix_ids = suf["input_ids"].to(device); suffix_mask = suf["attention_mask"].bool().to(device)
            suffix_emb = embed_tokens(suffix_ids); suffix_emb = torch.where(suffix_mask.unsqueeze(-1), suffix_emb, torch.zeros_like(suffix_emb))
            prefix_ids, vs_id, ve_id, _ = self._get_template_ids(device)
            prefix_emb = embed_tokens(prefix_ids).unsqueeze(0).expand(B, -1, -1)
            vs_emb = embed_tokens(vs_id).unsqueeze(0).expand(B, -1, -1); ve_emb = embed_tokens(ve_id).unsqueeze(0).expand(B, -1, -1)
            parts = [prefix_emb]; segments = [("text", prefix_ids.shape[0])]; vis_flags = [False] * prefix_ids.shape[0]
            for ct, g in zip(cam_tokens, grid_thw_list):
                parts += [vs_emb, ct, ve_emb]; segments += [("text", 1), ("image", g), ("text", 1)]
                vis_flags += [False] + [True] * ct.shape[1] + [False]
            parts.append(suffix_emb); segments.append(("text", suffix_ids.shape[1])); vis_flags += [False] * suffix_ids.shape[1]
            vlm_seq = torch.cat(parts, dim=1).to(torch.bfloat16); L_vlm = vlm_seq.shape[1]
            text_start = L_vlm - suffix_ids.shape[1]; vis_mask = torch.tensor(vis_flags, device=device, dtype=torch.bool)
            vlm_kv_pad_mask = torch.cat([torch.ones(B, text_start, device=device, dtype=torch.bool), suffix_mask], dim=1)
            lang_span = (text_start, L_vlm)
            if not self._template_format_printed: self._template_format_printed = True; print(f"[wiltechs_moe] chat template ON (text-last) - L_vlm={L_vlm}")
        else:
            lang_result = self._encode_language(batch, device)
            if lang_result is not None:
                lang_tokens, lang_mask = lang_result; lang_tokens = lang_tokens.to(vis_tokens.dtype)
                lang_tokens = torch.where(lang_mask.unsqueeze(-1), lang_tokens, torch.zeros_like(lang_tokens)); L_lang = lang_tokens.shape[1]
            else: lang_tokens, lang_mask, L_lang = None, None, 0
            parts = [vis_tokens]
            if lang_tokens is not None: parts.append(lang_tokens)
            vlm_seq = torch.cat(parts, dim=1).to(torch.bfloat16); L_vlm = vlm_seq.shape[1]; text_start = L_vis
            vis_mask = torch.zeros(L_vlm, device=device, dtype=torch.bool); vis_mask[:L_vis] = True
            segments = [("image", g) for g in grid_thw_list]
            if L_lang > 0: segments.append(("text", L_lang))
            if lang_mask is not None: vlm_kv_pad_mask = torch.cat([torch.ones(B, L_vis, device=device, dtype=torch.bool), lang_mask], dim=1)
            else: vlm_kv_pad_mask = torch.ones(B, L_vlm, device=device, dtype=torch.bool)
            lang_span = (text_start, L_vlm)
        position_ids = _build_mrope_position_ids(segments, B=B, spatial_merge_size=self.spatial_merge_size, device=device)
        cos, sin = self.language_model.rotary_emb(vlm_seq, position_ids)
        causal = torch.triu(torch.full((L_vlm, L_vlm), float("-inf"), device=device, dtype=vlm_seq.dtype), diagonal=1)
        full_mask = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, L_vlm, L_vlm).clone()
        full_mask.masked_fill_((~vlm_kv_pad_mask).unsqueeze(1).unsqueeze(1), float("-inf"))
        capture_set = set(self.capture_layers); hidden = vlm_seq; kv_cache = {}
        for i, layer in enumerate(self.language_model.layers):
            residual = hidden; h_in = layer.input_layernorm(hidden)
            Q = layer.self_attn.q_proj(h_in); K = layer.self_attn.k_proj(h_in); V = layer.self_attn.v_proj(h_in)
            Bn, Ln, _ = Q.shape
            Q = Q.view(Bn, Ln, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(Bn, Ln, self.num_kv_heads, self.head_dim).transpose(1, 2)
            V = V.view(Bn, Ln, self.num_kv_heads, self.head_dim).transpose(1, 2)
            Q, K = _apply_rope(Q, K, cos, sin)
            if i in capture_set: kv_cache[i] = (K.detach(), V.detach())
            if self.num_kv_heads != self.num_heads:
                r = self.num_heads // self.num_kv_heads; K_x = K.repeat_interleave(r, dim=1); V_x = V.repeat_interleave(r, dim=1)
            else: K_x, V_x = K, V
            attn = F.scaled_dot_product_attention(Q, K_x, V_x, attn_mask=full_mask, is_causal=False)
            attn = attn.transpose(1, 2).contiguous().view(Bn, Ln, self.num_heads * self.head_dim)
            attn = layer.self_attn.o_proj(attn); hidden = residual + attn
            residual = hidden; h_in = layer.post_attention_layernorm(hidden); hidden = residual + layer.mlp(h_in)
        return kv_cache, vlm_kv_pad_mask, vis_mask, lang_span, hidden, vis_pack_out
    def _compute_robot_tokens(self, batch):
        if self.robot_visual_encoder is None: return None
        toks_list = []; cnn_cams = getattr(self.config, "robot_cnn_cameras", None) or self.config.cameras_for_vision_state_concat
        for cam_key in cnn_cams:
            if cam_key not in batch: continue
            img = batch[cam_key]
            if img.dim() == 5: img = img[:, -1]
            toks_list.append(self.robot_visual_encoder(img.float()))
        if not toks_list: return None
        toks = torch.cat(toks_list, dim=1)
        vp = float(getattr(self.config, "vision_dropout_prob", 0.0)) if self.training else 0.0
        if vp > 0:
            B, R, _ = toks.shape; keep = torch.rand(B, R, device=toks.device) > vp; toks = toks * keep.unsqueeze(-1).to(toks.dtype)
        return toks
    def _generate_latents(self, kv_cache, vlm_kv_pad_mask):
        if self.num_latent_tokens == 0: return None
        vlm_k, vlm_v = kv_cache[max(self.capture_layers)]; return self.latent_qformer(vlm_k, vlm_v, vlm_kv_pad_mask)

    def _generate_thoughts(self, kv_cache, vlm_kv_pad_mask, record=True):
        """Generate spatial-reasoning "thought" tokens from the deepest captured
        VLM layer's KV cache via a learned-query Q-Former.

        The deepest layer has maximum semantic fusion of vision + language
        (causal attention has fully mixed scene appearance with the
        instruction), so cross-attending here distills spatial reasoning:
        target object location, gripper-to-target relative position, goal
        placement, etc.  Noise-independent -> computed once per forward,
        shared across all N denoising steps (zero added loop cost).
        Returns (B, K, dit_hidden) or None if disabled."""
        if self.num_thought_tokens == 0 or self.thought_qformer is None:
            return None
        vlm_k, vlm_v = kv_cache[self.thought_vlm_layer]
        thoughts = self.thought_qformer(vlm_k, vlm_v, vlm_kv_pad_mask)
        # Magnitude of the thought tokens as they enter the DiT sequence. Read
        # against _last_action_emb_rms: the expert's first RMSNorm renormalises
        # per token, so a tiny ratio does not mean "inert", but a ratio that
        # never moves off its init does.
        if record: self._last_thought_rms = float(thoughts.detach().float().pow(2).mean().sqrt())
        return thoughts

    def _build_expert_input(self, batch, noisy_actions, robot_tokens, latents, thoughts=None):
        B, H, _ = noisy_actions.shape; dtype = noisy_actions.dtype
        sink = self.sink_token.expand(B, -1, -1).to(dtype)
        state = batch["observation.state"].float()
        if state.dim() == 2: state = state.unsqueeze(1)
        state = state.nan_to_num(0.0).clamp(-10.0, 10.0); state_tok = self.state_encoder(state).to(dtype)
        if state_tok.shape[1] > 1: state_tok = state_tok[:, -1:]
        action_emb = (self.action_in_proj(noisy_actions) + self.action_pos_emb[:, :H]).to(dtype)
        self._last_action_emb_rms = float(action_emb.detach().float().pow(2).mean().sqrt())
        # Sequence: [sink, state, robot_cnn, (legacy latents), thoughts, action]
        # Thought tokens go BEFORE action tokens so causal self-attention lets
        # every action token read the spatial reasoning distilled from the VLM.
        parts = [sink, state_tok]
        if robot_tokens is not None: parts.append(robot_tokens.to(dtype))
        if latents is not None: parts.append(latents.to(dtype))
        if thoughts is not None: parts.append(thoughts.to(dtype))
        parts.append(action_emb); seq = torch.cat(parts, dim=1)
        return seq, seq.shape[1] - H, state_tok, action_emb

    def _pool_vlm_semantic(self, vlm_hidden, vlm_kv_pad_mask):
        """Pool the VLM's final hidden state over valid tokens -> (B, hidden_size).

        The VLM hidden state contains BOTH vision and language information (processed
        together through the VLM's causal attention), so mean-pooling the final layer
        gives the Router access to the full multimodal context. Using hidden states
        (not KV cache V) avoids GQA head-count mismatches since hidden_size is always
        the model's hidden_size."""
        mask = vlm_kv_pad_mask.unsqueeze(-1).to(vlm_hidden.dtype)  # (B, L_vlm, 1)
        pooled = (vlm_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)  # (B, hidden_size)
        return pooled

    def _run_moe_dit(self, batch, noisy_actions, timesteps, kv_cache, vlm_kv_pad_mask, robot_tokens, latents, vlm_hidden, thoughts=None, record=True):
        device, dtype = noisy_actions.device, noisy_actions.dtype
        t_emb = self.time_embedder(create_sinusoidal_pos_embedding(timesteps, self.dit_hidden).to(dtype).float()).to(dtype)
        dit_seq, action_start_idx, state_tok, action_emb = self._build_expert_input(batch, noisy_actions, robot_tokens, latents, thoughts)
        L_dit = dit_seq.shape[1]
        causal_mask = torch.triu(torch.full((L_dit, L_dit), float("-inf"), device=device, dtype=dtype), diagonal=1)
        vlm_semantic = self._pool_vlm_semantic(vlm_hidden, vlm_kv_pad_mask).to(dtype)
        weights, usage = self.router(state_tok, vlm_semantic, t_emb, action_emb)
        expert_outputs = []
        for e, expert in enumerate(self.experts):
            expert_kv = [kv_cache[idx] for idx in self.expert_kv_blocks[e]]; x = dit_seq
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(expert, x, t_emb, expert_kv, vlm_kv_pad_mask, causal_mask, use_reentrant=False)
            else:
                x = expert(x, t_emb=t_emb, expert_kv_cache=expert_kv, vlm_kv_pad_mask=vlm_kv_pad_mask, self_attn_mask=causal_mask)
            expert_outputs.append(self.action_out_proj(self.final_norm(x[:, action_start_idx:])))
        stacked = torch.stack(expert_outputs, dim=1)
        v_t = (weights.unsqueeze(-1).unsqueeze(-1) * stacked).sum(dim=1)
        # record=False for the contrastive negative's forward: it runs AFTER
        # this one and would otherwise overwrite every router statistic, so the
        # training log would report routing under PERMUTED instructions while
        # claiming to describe the real forward. (The balance loss is unaffected
        # -- it reads _last_router_usage before the negative branch runs.)
        if record:
            self._last_router_usage = usage
            # usage is the BATCH MEAN, so a perfectly uniform CV^2 is ambiguous:
            # it is produced both by healthy per-sample specialisation that
            # averages out, and by a router that has learned to emit uniform
            # weights for every input -- which zeroes the CV^2 balance penalty
            # for free and turns the MoE into a fixed 4-way average, killing the
            # one module that conditions expert choice on the instruction. These
            # two per-sample statistics separate them: with num_experts=4,
            # uniform-per-sample means max_w -> 0.25 and entropy -> ln(4)=1.386.
            #
            # Taken from the router's PRE-noise weights (see MoERouter.forward),
            # so the reference values above are the right ones to compare with
            # and the numbers describe inference-time routing directly.
            with torch.no_grad():
                w = getattr(self.router, "_last_clean_weights", weights).detach().float()
                self._last_router_max_w = float(w.max(dim=-1).values.mean())
                self._last_router_entropy = float(-(w.clamp_min(1e-9).log() * w).sum(-1).mean())
        return v_t.float()

    def sample_noise(self, shape, device):
        rho = self.config.noise_temporal_correlation; noise = torch.randn(shape, device=device)
        if rho == 0.0 or shape[1] == 1: return noise
        scale = math.sqrt(1.0 - rho * rho)
        for t in range(1, shape[1]): noise[:, t] = rho * noise[:, t - 1] + scale * noise[:, t]
        return noise
    def sample_time(self, B, device): return torch.rand(B, device=device) * 0.998 + 0.001

    def compute_loss(self, batch):
        actions = batch["action"].float().nan_to_num(0.0).clamp(-10.0, 10.0); B = actions.shape[0]; device = actions.device
        kv_cache, vlm_kv_pad_mask, vis_mask, lang_span, vlm_hidden, vis_pack = self._run_vlm_and_cache_kv(batch)
        vkv_p = float(getattr(self.config, "vision_kv_dropout_prob", 0.0)); n_vis = int(vis_mask.sum().item())
        vkv_drop = None
        if self.training and vkv_p > 0.0 and n_vis > 0:
            vis_idx = vis_mask.nonzero(as_tuple=True)[0]; keep = torch.rand(B, n_vis, device=device) > vkv_p
            dead = ~keep.any(dim=1)
            if dead.any(): keep[dead, 0] = True
            vlm_kv_pad_mask = vlm_kv_pad_mask.clone(); vlm_kv_pad_mask[:, vis_idx] &= keep
            # Remembered so the contrastive negative gets the SAME vision KV
            # dropout -- otherwise v_t and v_wrong differ because of dropout,
            # not because of the language, and the hinge measures the wrong thing.
            vkv_drop = (vis_idx, keep)
        robot_tokens = self._compute_robot_tokens(batch)
        latents = self._generate_latents(kv_cache, vlm_kv_pad_mask)
        # Generate spatial-reasoning thought tokens (noise-independent, once per forward)
        thoughts = self._generate_thoughts(kv_cache, vlm_kv_pad_mask)
        noise = self.sample_noise(actions.shape, device); t = self.sample_time(B, device); t_exp = t[:, None, None]
        x_t = t_exp * noise + (1.0 - t_exp) * actions; u_t = noise - actions; x_t_bf16 = x_t.to(torch.bfloat16)
        v_t = self._run_moe_dit(batch, x_t_bf16, t, kv_cache, vlm_kv_pad_mask, robot_tokens, latents, vlm_hidden, thoughts)
        loss = F.mse_loss(v_t, u_t, reduction="none")
        if self.config.action_dim_weights:
            dim_w = torch.tensor(self.config.action_dim_weights, device=loss.device, dtype=loss.dtype); loss = loss * dim_w[None, None, :]
        H = loss.shape[1]; n_exec = self.config.n_action_steps; pos_w = torch.ones(H, device=loss.device, dtype=loss.dtype)
        pos_w[n_exec:] = self.config.future_steps_weight
        if self.config.pos_decay_lambda > 0.0:
            pos = torch.arange(H, device=loss.device, dtype=loss.dtype); pos_w = pos_w * torch.exp(-self.config.pos_decay_lambda * pos)
        loss = loss * pos_w[None, :, None]; loss_dtype = loss.dtype; Bn, Hn, Dn = loss.shape
        is_pad = batch.get("action_is_pad", batch.get("actions_id_pad"))
        valid_t = (~is_pad.bool()).to(loss_dtype) if is_pad is not None else torch.ones(Bn, Hn, device=loss.device, dtype=loss_dtype)
        dim_pad = batch.get("action_dim_pad")
        valid_d = (~dim_pad.bool()).to(loss_dtype) if dim_pad is not None else torch.ones(Bn, Dn, device=loss.device, dtype=loss_dtype)
        valid_cells = valid_t.unsqueeze(-1) * valid_d.unsqueeze(1); loss = loss * valid_cells
        main_loss = loss.sum() / (pos_w[None, :, None] * valid_cells).sum().clamp(min=1e-6)
        balance_w = float(getattr(self.config, "router_balance_weight", 0.0)); balance_loss_val = 0.0
        if balance_w > 0.0 and hasattr(self, "_last_router_usage"):
            usage = self._last_router_usage; cv_sq = (usage.std() / usage.mean().clamp(min=1e-8)).pow(2)
            main_loss = main_loss + balance_w * cv_sq; balance_loss_val = float(cv_sq.detach())
        contrastive_w = float(getattr(self.config, "contrastive_loss_weight", 0.0)); contrastive_v = 0.0
        lang_start, lang_end = lang_span
        L_lang_total = lang_end - lang_start
        descs = self._resolve_descs(batch)
        have_descs = descs is not None and len(descs) == B
        # With text_first the instruction is baked into the vision KV, so the
        # negative has to come from a real re-run of the VLM with permuted
        # instructions -- there is no valid KV slice to swap.
        can_contrast = have_descs if self.text_first else True
        if self.training and contrastive_w > 0.0 and L_lang_total > 0 and B >= 2 and can_contrast:
            if getattr(self.config, "contrastive_hard_negatives", False) and have_descs:
                perm, pair_diff = _hard_negative_perm(descs, device)
            else:
                perm = torch.randperm(B, device=device)
                if (perm == torch.arange(B, device=device)).any(): perm = torch.roll(perm, shifts=1, dims=0)
                if have_descs:
                    perm_cpu = perm.detach().cpu().tolist(); pair_diff = torch.tensor([descs[i] != descs[perm_cpu[i]] for i in range(B)], device=device, dtype=torch.bool)
                else: pair_diff = torch.ones(B, device=device, dtype=torch.bool)
            if pair_diff.any():
                if self.text_first:
                    # Re-run the frozen LM with permuted instructions. The ViT
                    # output is reused via vis_pack, so the extra cost is the 36
                    # LM layers only (no_grad, no activations retained).
                    perm_cpu = perm.detach().cpu().tolist()
                    descs_perm = [descs[i] for i in perm_cpu]
                    # Keep the re-run's OWN hidden state. Two reasons, and the
                    # first one crashes:
                    #  * _hard_negative_perm is explicitly not a bijection, so
                    #    descs_perm is a different multiset from descs and can
                    #    pad to a different length -- L_vlm 406 vs 416, and
                    #    _pool_vlm_semantic multiplies hidden by the mask.
                    #  * Even when the lengths coincide, feeding the CORRECT
                    #    language's hidden into the wrong-language branch gives
                    #    v_t and v_wrong identical router weights, so the
                    #    contrastive gradient never reaches the router at all.
                    shuffled_cache, shuffled_pad_mask, _, _, vlm_hidden_wrong, _ = \
                        self._run_vlm_and_cache_kv(
                            batch, descs_override=descs_perm, vis_pack=vis_pack)
                    if vkv_drop is not None:
                        v_idx, v_keep = vkv_drop
                        shuffled_pad_mask = shuffled_pad_mask.clone(); shuffled_pad_mask[:, v_idx] &= v_keep
                else:
                    shuffled_cache = {}
                    for layer_idx, (K, V) in kv_cache.items():
                        K_shuf, V_shuf = K.clone(), V.clone()
                        K_shuf[:, :, lang_start:, :] = K[perm, :, lang_start:, :]; V_shuf[:, :, lang_start:, :] = V[perm, :, lang_start:, :]
                        shuffled_cache[layer_idx] = (K_shuf, V_shuf)
                    shuffled_pad_mask = vlm_kv_pad_mask.clone(); shuffled_pad_mask[:, lang_start:] = vlm_kv_pad_mask[perm][:, lang_start:]
                    # Same permutation the K/V got, so the router sees the wrong
                    # language here too. Lengths always match on this path.
                    vlm_hidden_wrong = vlm_hidden.clone(); vlm_hidden_wrong[:, lang_start:] = vlm_hidden[perm][:, lang_start:]
                latents_wrong = self._generate_latents(shuffled_cache, shuffled_pad_mask)
                # Also recompute thoughts from wrong-language cache so the thought
                # path is language-forced too.
                thoughts_wrong = self._generate_thoughts(shuffled_cache, shuffled_pad_mask, record=False)
                v_wrong = self._run_moe_dit(batch, x_t_bf16, t, shuffled_cache, shuffled_pad_mask, robot_tokens, latents_wrong, vlm_hidden_wrong, thoughts_wrong, record=False)
                diff_sq = (v_t - v_wrong).pow(2).mean(dim=[1, 2]); margin = float(getattr(self.config, "contrastive_margin", 0.05))
                hinge = F.relu(margin - diff_sq) * pair_diff.float(); loss_contrastive = hinge.sum() / pair_diff.float().sum().clamp(min=1.0)
                contrastive_v = float(loss_contrastive.detach()); main_loss = main_loss + contrastive_w * loss_contrastive
        self._last_loss_components = {"main": float(main_loss.detach() - contrastive_w * contrastive_v - balance_w * balance_loss_val), "contrastive": contrastive_v, "balance": balance_loss_val}
        return main_loss

    def forward(self, batch):
        if self.training: return self.compute_loss(batch), {}
        return self.sample_actions(batch), {}

    def flow_actions_from_noise(self, batch, x_init):
        B = x_init.shape[0]; device = x_init.device
        kv_cache, vlm_kv_pad_mask, _, _, vlm_hidden, _ = self._run_vlm_and_cache_kv(batch)
        robot_tokens = self._compute_robot_tokens(batch)
        latents = self._generate_latents(kv_cache, vlm_kv_pad_mask)
        thoughts = self._generate_thoughts(kv_cache, vlm_kv_pad_mask)
        N = int(getattr(self.config, "num_inference_steps", 5)); x_t = x_init.float(); dt = -1.0 / N; t = torch.ones(B, device=device, dtype=torch.float32)
        for _ in range(N):
            v_t = self._run_moe_dit(batch, x_t.to(torch.bfloat16), t, kv_cache, vlm_kv_pad_mask, robot_tokens, latents, vlm_hidden, thoughts).float()
            x_t = x_t + dt * v_t; t = t + dt
        return x_t

    @torch.no_grad()
    def sample_actions(self, batch):
        B = batch["observation.state"].shape[0]; device = batch["observation.state"].device
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with autocast_ctx:
            kv_cache, vlm_kv_pad_mask, _, _, vlm_hidden, _ = self._run_vlm_and_cache_kv(batch)
            robot_tokens = self._compute_robot_tokens(batch)
            latents = self._generate_latents(kv_cache, vlm_kv_pad_mask)
            thoughts = self._generate_thoughts(kv_cache, vlm_kv_pad_mask)
            N = int(getattr(self.config, "num_inference_steps", 5))
            x_t = self.sample_noise((B, self.config.horizon, self.config.action_dim), device=device)
            dt = -1.0 / N; t = torch.ones(B, device=device, dtype=torch.float32)
            for _ in range(N):
                v_t = self._run_moe_dit(batch, x_t.to(torch.bfloat16), t, kv_cache, vlm_kv_pad_mask, robot_tokens, latents, vlm_hidden, thoughts).float()
                x_t = x_t + dt * v_t; t = t + dt
        return x_t[:, :self.config.n_action_steps]

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}