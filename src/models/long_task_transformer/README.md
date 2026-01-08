# Long Task Transformer Policy

This module implements a transformer-based policy for long-horizon robotic tasks. Unlike the diffusion-based approach, this model uses a transformer architecture with ResNet encoders for processing visual observations.

## Architecture Overview

The LongTaskTransformer model consists of:

1. **ResNet-based Image Encoders**: Process camera observations and convert them to feature embeddings
2. **State Tokenizers**: Convert observation.state vectors to tokens
3. **Transformer Encoder**: Processes the observation context (images + state)
4. **Transformer Decoder**: Autoregressively generates action sequences
5. **Action Head**: Converts decoder outputs to actionable robot commands

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

### Demo

```bash
python src/models/long_task_transformer/demo_usage.py
```

## Configuration

The model can be configured through the `LongTaskTransformerConfig` class:

```python
config = LongTaskTransformerConfig(
    n_obs_steps=8,          # Number of observation steps
    horizon=16,             # Prediction horizon
    n_action_steps=8,       # Number of action steps to execute
    state_dim=7,            # Robot state dimension
    action_dim=7,           # Action dimension
    d_model=128,            # Transformer model dimension
    nhead=4,                # Number of attention heads
    num_encoder_layers=4,   # Encoder layers
    num_decoder_layers=4,   # Decoder layers
    dim_feedforward=512,    # Feedforward dimension
    dropout=0.1,            # Dropout rate
    vision_backbone="resnet18"  # Vision backbone
)
```

## Model Components

- `LongTaskTransformerConfig`: Configuration class defining model parameters
- `LongTaskTransformerModel`: Core neural network implementation
- `LongTaskTransformerPolicy`: High-level policy interface for training and inference
