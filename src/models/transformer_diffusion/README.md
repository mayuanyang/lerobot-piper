# Transformer Diffusion Policy

This module implements a diffusion-based policy with transformer architecture for long-horizon robotic tasks. The model combines ResNet encoders for visual processing with a transformer-based diffusion approach for action generation.

## Architecture Overview

The TransformerDiffusion model consists of:

1. **ResNet-based Image Encoders**: Process camera observations and convert them to feature embeddings
2. **State Encoder**: Convert observation.state vectors to tokens
3. **Transformer Encoder**: Processes the observation context (images + state)
4. **Diffusion Process**: Generates action sequences through a denoising process
5. **Action Head**: Converts diffusion outputs to actionable robot commands

## Key Features

- **Vision Processing**: Uses ResNet backbones (resnet18, resnet34, resnet50) for robust image feature extraction
- **State Tokenization**: Efficiently converts robot state vectors into transformer-compatible tokens
- **Autoregressive Action Generation**: Generates action sequences step-by-step for long-horizon tasks
- **Flexible Configuration**: Configurable transformer architecture parameters (layers, heads, dimensions)

## Usage

### Training

```bash
python src/train_transformer.py --output_dir ./outputs/transformer_model --dataset_id ISdept/piper_arm
```

### Inference

```bash
python src/inference_transformer.py --model_path ./outputs/transformer_model
```

## Configuration

The model can be configured through the `TransformerDiffusionConfig` class:

```python
config = TransformerDiffusionConfig(
    n_obs_steps=8,          # Number of observation steps
    horizon=16,             # Prediction horizon
    n_action_steps=8,       # Number of action steps to execute
    state_dim=7,            # Robot state dimension
    action_dim=7,           # Action dimension
    d_model=256,            # Transformer model dimension
    nhead=8,                # Number of attention heads
    num_encoder_layers=6,   # Encoder layers
    num_decoder_layers=6,   # Decoder layers
    dim_feedforward=2048,   # Feedforward dimension
    dropout=0.1,            # Dropout rate
    vision_backbone="resnet18"  # Vision backbone
)
```

## Model Components

- `TransformerDiffusionConfig`: Configuration class defining model parameters
- `DiffusionTransformer`: Core neural network implementation
- `TransformerDiffusionPolicy`: High-level policy interface for training and inference
