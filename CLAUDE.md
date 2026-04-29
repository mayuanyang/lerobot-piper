# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains custom training, inference, and data processing code for the **Piper 7-DOF robot arm** built on top of the [LeRobot](https://github.com/huggingface/lerobot) framework. The primary dataset used is `ISdept/piper_arm` (or similar HuggingFace-hosted LeRobot datasets).

## Common Commands

### Training

```bash
# Train with LongTaskDiffusion policy (ACT/Diffusion-based)
python src/train.py --output_dir ./outputs/my_run --dataset_id ISdept/piper_arm

# Resume from checkpoint
python src/train.py --output_dir ./outputs/my_run --dataset_id ISdept/piper_arm --resume_from_checkpoint ./outputs/my_run/checkpoint-5000

# Train with TransformerFlowMatching policy
python src/train_transformer.py

# Train with SmolVLA policy
python src/train_smolvla.py
```

### Inference

```bash
# TransformerFlowMatching inference
python src/inference_transformer.py

# Video file inference
python src/video_inference.py

# SmolVLA video inference
python src/video_inference_smolvla.py

# Webcam inference
python src/webcam_inference.py
```

### Data Processing

```bash
# Prepare a LeRobot-format dataset from raw episode data
python src/data_processing/prepare_dataset.py

# Check dataset for leakage/contamination
python src/data_processing/dataset_leakage_tester.py --dataset-root ./output

# Add 2D bounding boxes to an existing dataset
python src/add_2d_bounding_boxes_to_lerobot_dataset.py
```

### Installation

```bash
pip install -r requirements_inference.txt
# Note: lerobot==0.4.0 has no numpy constraint — use Colab's pre-installed numpy (2.x)
```

## Architecture

### Model Hierarchy

The repo contains three policy families, all extending `lerobot.policies.pretrained.PreTrainedPolicy`:

1. **`src/models/long_task_diffusion/`** — Extends LeRobot's `DiffusionPolicy` with configurable loss weights and horizon extension for long-horizon tasks.

2. **`src/models/smooth_diffusion/`** — Custom diffusion variant with joint smoothness regularization.

3. **`src/models/transformer_flow_matching/`** — The primary active model. Flow-matching policy with a transformer decoder and SmolVLM vision backbone.
   - `transformer_flow_matching_config.py` — `TransformerFlowMatchingConfig`, registered as `"transformer_flow_matching"`. Key params: `vision_backbone`, `num_cameras`, `state_dim=7`, `action_dim=7`, `horizon=50`, `n_action_steps=8`.
   - `transformer_flow_matching_model.py` — `FlowMatchingTransformer` containing `SmolVLAVisionTokenizer` + transformer decoder + flow matching denoiser.
   - `transformer_flow_matching_policy.py` — `TransformerFlowMatchingPolicy` wrapping the model.
   - `processor_transformer_flow_matching.py` — Custom pre/postprocessors (replaces LeRobot's `make_pre_post_processors`).
   - `box_encoder.py` — Encodes YOLO-detected bounding boxes as tokens for the transformer.
   - `object_detector.py` — YOLOWorld-based zero-shot object detection.
   - `spatial_softmax.py` — Spatial softmax keypoint extractor (used in earlier architecture iterations).
   - `grid_overlay_processor.py` — Overlays a visual grid on images before passing to the vision encoder.

### Vision Backbone: `SmolVLAVisionTokenizer`

Located in `transformer_flow_matching_model.py`. Uses `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`:
- Extracts `vision_model` (SigLIP ViT) — **all layers**, using `last_hidden_state` (not partial layers).
- Uses the pretrained `connector` (pixel-shuffle + MLP resampler) to project vision tokens to VLM text hidden size.
- Projects to `config.d_model` via a trainable linear `proj`.
- Text encoding uses frozen `embed_tokens` from the LM + trainable `text_proj`.
- When `freeze_vision_backbone=True` (default): vision encoder, connector, and text embed_tokens are frozen; only `proj` and `text_proj` train.

### Data Pipeline

Raw robot episodes → `src/data_processing/prepare_dataset.py` → LeRobot-format dataset (parquet + video files).

`src/data_processing/episode_data.py` defines `EpisodeData` and `CameraData` structures for raw episode loading.

### Device Detection Pattern

All training/inference scripts share the same device detection pattern:
```python
if torch.cuda.is_available(): device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"
else: device = "cpu"
```

### Checkpoint Structure

```
outputs/my_run/
├── checkpoint-1000/
│   ├── policy.safetensors
│   ├── preprocessor_config.json
│   ├── postprocessor_config.json
│   └── optimizer_state.pth
```
