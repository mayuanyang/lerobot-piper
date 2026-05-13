# Training Log — TransformerFlowMatching

## v1.0 — Step 70000 (2026-05-12)

### Model Configuration
| Parameter | Value |
|---|---|
| Architecture | TransformerFlowMatching |
| VLM backbone | SmolVLM2-500M-Video-Instruct (frozen) |
| VLM layers used | 16 (interleaved 1:1 with decoder) |
| Action expert layers | 16 × TransformerDecoderLayer |
| d_model | 512 |
| nhead | 8 |
| dim_feedforward | 2048 |
| Robot visual encoder | ResNet-18 (layers 1–3), 16 tokens/camera |
| Robot layer projections | 16 × Linear(512, 512) per-layer |
| Cameras | front, gripper, right (all 3) |
| State dim | 7 |
| Action dim | 7 |
| Horizon | 32 |
| n_action_steps | 32 |
| n_obs_steps | 2 |
| Noise | Gaussian (ρ=0.0) |
| Inference steps | 10 (Euler) |
| Batch size | 160 |
| Peak LR | 1e-4 (cosine decay, 1500 warmup steps) |
| action_dim_weights | [1, 1, 1, 0, 1, 1, 1] (joint 4 locked) |
| pos_decay_lambda | 0.0 |
| Dataset | ISdept/piper_arm (episodes ≤ 400) |

### Architecture Changes vs Previous Run
- Grid overlay removed from preprocessor
- Right camera kept (all 3 cameras: front, gripper, right)
- Robot tokens moved from direct concat to per-layer projections (`robot_layer_projs`)
- Per-layer gradient routing: each decoder layer gets its own Linear(512,512) projection of robot tokens
- Switched Heun (20 steps) → Euler (10 steps) for 4× inference speedup
- `pos_decay_lambda` fixed from 0.1 → 0.0 (was suppressing 69% of gradient signal with horizon=32)
- `EpisodeAwareSampler` added: drop_n_first_frames=2, drop_n_last_frames=2
- Per-camera lag compensation: gripper offset +1 frame (delta_timestamps)
- BF16 autocast + cudnn.benchmark + allow_tf32 enabled

---

## Training Stats

### Current Run (New Architecture — from Step 0)

| Step | Epoch | Loss | LR | Grad Norm | State Enc | Robot CNN | Robot Proj | Action Expert |
|---|---|---|---|---|---|---|---|---|
| 18800 | 63 | 0.056–0.061 | 9.75e-05 | 0.60 | 1.27e-4 | 1.07e-4 | 1.7e-5 | 9e-6 |
| 19000 | 64 | 0.061–0.071 | 9.75e-05 | 0.59 | 1.27e-4 | 1.00e-4 | 1.6e-5 | 8e-6 |
| 19200 | 64 | 0.071–0.073 | 9.74e-05 | 0.76 | 1.48e-4 | 1.01e-4 | 1.7e-5 | 1.0e-5 |
| 19400 | 65 | 0.073–0.075 | 9.74e-05 | 0.86 | 1.72e-4 | 1.26e-4 | 1.8e-5 | 1.0e-5 |
| 19600 | 66 | 0.069–0.075 | 9.73e-05 | 0.53 | 1.39e-4 | 9.3e-5  | 1.5e-5 | 8e-6  |
| 19800 | 66 | 0.068–0.069 | 9.72e-05 | 0.65 | 2.30e-4 | 1.01e-4 | 1.6e-5 | 9e-6  |
| 40200 | 138 | 0.036–0.045 | 8.12e-05 | 0.45 | 1.00e-4 | 7.3e-5  | 1.2e-5 | 6e-6  |
| 40400 | 139 | 0.040–0.045 | 8.11e-05 | 0.46 | 1.24e-4 | 7.9e-5  | 1.3e-5 | 6e-6  |
| 40600 | 140 | 0.036–0.040 | 8.10e-05 | 0.39 | 1.60e-4 | 7.6e-5  | 1.3e-5 | 7e-6  |
| 46200 | 159 | 0.036–0.040 | 7.09e-05 | 0.34 | 1.53e-4 | 7.5e-5  | 1.2e-5 | 6e-6  |
| 46400 | 160 | 0.040–0.045 | 7.08e-05 | 0.50 | 1.12e-4 | 7.0e-5  | 1.1e-5 | 6e-6  |
| 46600 | 160 | 0.045–0.047 | 7.07e-05 | 0.41 | 1.49e-4 | 9.5e-5  | 1.5e-5 | 7e-6  |
| 46800 | 161 | 0.047–0.050 | 7.06e-05 | 0.42 | 1.65e-4 | 7.7e-5  | 1.4e-5 | 7e-6  |
| 47000 | 162 | 0.050–0.050 | 7.04e-05 | 0.47 | 9.0e-5  | 8.2e-5  | 1.3e-5 | 7e-6  |
| 60200 | 208 | 0.027–0.045 | 4.45e-05 | 0.40 | 9.0e-5  | 8.6e-5  | 1.3e-5 | 7e-6  |
| 60400 | 209 | 0.045–0.052 | 4.45e-05 | 0.39 | 1.53e-4 | 8.5e-5  | 1.3e-5 | 7e-6  |
| 60600 | 210 | 0.040–0.045 | 4.44e-05 | 0.45 | 1.27e-4 | 9.7e-5  | 1.5e-5 | 7e-6  |
| 70200 | 244 | 0.022–0.033 | 3.00e-05 | 0.45 | 9.9e-5  | 1.03e-4 | 1.5e-5 | 8e-6  |
| 70400 | 244 | 0.016–0.033 | 3.00e-05 | 0.44 | 1.01e-4 | 1.01e-4 | 1.3e-5 | 6e-6  |
| 77600 | 269 | 0.021–0.037 | 2.78e-05 | 0.45 | 1.15e-4 | 9.8e-5  | 1.5e-5 | 8e-6  |
| 77800 | 270 | 0.014–0.037 | 2.78e-05 | 0.39 | 1.02e-4 | 8.9e-5  | 1.3e-5 | 6e-6  |
| 89400 | 311 | 0.013       | 1.36e-05 | 0.32 | 8.3e-5  | 8.4e-5  | 1.2e-5 | 6e-6  |

### Loss Floor Milestones

| Step | Loss Floor | Notes |
|---|---|---|
| ~19k | 0.056 | Fresh start with new architecture |
| ~40k | 0.036 | Steady improvement |
| ~47k | 0.036 | Temporary 7k-step stall |
| ~60k | 0.027 | Broke through stall |
| ~70k | 0.016 | Strong convergence |
| ~78k | 0.014 | Still improving |
| ~89k | **0.013** | New low |

### Previous Run (Old Architecture — archived)

Old architecture: shared robot tokens across all 16 layers (no per-layer projections),
right camera included, grid overlay on front+right, horizon=4, n_action_steps=4.

| Step | Epoch | Loss | LR | Notes |
|---|---|---|---|---|
| 21800 | 73 | 0.044–0.055 | 4.79e-05 | Plateau beginning |
| 22000 | 74 | 0.054–0.055 | 4.78e-05 | — |
| 22200 | 75 | 0.050–0.054 | 4.78e-05 | — |
| 23200 | 77 | 0.042–0.051 | 3.88e-05 | — |
| 23400 | 78 | 0.042–0.050 | 3.88e-05 | — |
| 97600 | 332 | 0.063–0.064 | 1.21e-05 | Hard plateau — pos_decay_lambda=0.1 issue |
| 97800 | 333 | 0.064–0.084 | 1.20e-05 | Abandoned — switched to new architecture |
