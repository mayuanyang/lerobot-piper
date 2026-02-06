# Transformer Diffusion Policy

## Simplified Version Available

A simplified version of this model is available that addresses gradient flow issues in the original implementation. See [README_SIMPLE.md](README_SIMPLE.md) for details.

This module implements a diffusion-based policy with transformer architecture for long-horizon robotic tasks. The model combines ResNet encoders with K-point spatial softmax for visual processing and a transformer-based diffusion approach for action generation.

## Visualization

The model includes SpatialSoftmax visualization capabilities to monitor vision encoder outputs during training and inference. Visualizations are saved to `spatial_softmax_visualizations` directory during training, showing:

- Smooth tracking of object positions over time
- Object-relative movement across camera views
- Different camera signals (not identical outputs)
- Multiple timesteps from the same observation window

### Training Visualization Options:

Visualizations are saved every N steps (default 100), showing all timesteps from the first item in each batch. This allows you to see how spatial features evolve over the observation window.

To customize the visualization frequency:
```python
train(output_dir="./outputs", dataset_id="ISdept/piper_arm", visualize_every_n_batches=50)
```

To run standalone visualization:
```bash
python src/visualize_spatial_softmax.py
```

## Architecture Overview

The TransformerDiffusion model consists of:

1. **ResNet-based Image Encoders**: Process camera observations and convert them to feature embeddings
2. **State Encoder**: Convert observation.state vectors to tokens
3. **Transformer Encoder**: Processes the observation context (images + state)
4. **Diffusion Process**: Generates action sequences through a denoising process
5. **Action Head**: Converts diffusion outputs to actionable robot commands

## Key Features

- **Vision Processing**: Uses ResNet backbones (resnet18, resnet34, resnet50) with K-point spatial softmax for robust image feature extraction
- **State Tokenization**: Efficiently converts robot state vectors into transformer-compatible tokens
- **Autoregressive Action Generation**: Generates action sequences step-by-step for long-horizon tasks
- **Flexible Configuration**: Configurable transformer architecture parameters (layers, heads, dimensions)

## K-Point Spatial Softmax

The model uses a K-point spatial softmax mechanism that extracts multiple (x, y) coordinates from feature maps instead of a single point. This allows the model to track multiple objects or regions of interest simultaneously.

Currently configured to use 2 points per camera, but can be easily adjusted to use more points by modifying the `num_points` parameter in the VisionEncoder.

The SpatialSoftmax implementation includes several improvements to prevent the common issue of points collapsing to the center:
- **Temperature scaling**: Uses a low initial temperature (0.1) with learnable parameters to encourage sharp attention peaks
- **Orthogonal initialization**: Initializes the convolutional weights to encourage diverse attention maps
- **Spatial regularization**: Adds a loss term during training to encourage separation between detected points
- **Weight scaling**: Scales initial weights to promote sharper attention distributions

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
    joint_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Weight for each joint in loss computation
    input_image_size=(320, 320),  # Input image size (height, width)
    d_model=256,            # Transformer model dimension
    nhead=8,                # Number of attention heads
    num_encoder_layers=6,   # Encoder layers
    num_decoder_layers=6,   # Decoder layers
    dim_feedforward=2048,   # Feedforward dimension
    dropout=0.1,            # Dropout rate
    vision_backbone="resnet18"  # Vision backbone
)
```

## Weighted Loss

The model supports weighted loss computation for different joints in addition to the original MSE loss. This is useful when certain joints are more important than others in your robotic task. The total loss is computed as the sum of the original MSE loss and the weighted MSE loss. You can adjust the `joint_weights` parameter in the configuration to assign different weights to each joint dimension. Higher weights will make the model focus more on accurately predicting those joint values during training.

## Model Components

- `TransformerDiffusionConfig`: Configuration class defining model parameters
- `DiffusionTransformer`: Core neural network implementation
- `TransformerDiffusionPolicy`: High-level policy interface for training and inference
