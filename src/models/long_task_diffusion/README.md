# Long Task Diffusion Policy

This module implements both a custom diffusion policy and a transformer-based policy designed for long-horizon tasks.

## Components

### Diffusion-based Implementation
1. **LongTaskDiffusionConfig**: Configuration class that extends DiffusionConfig with additional parameters for long task handling
2. **LongTaskDiffusionModel**: Neural network model with customizable loss computation
3. **LongTaskDiffusionPolicy**: Main policy class that inherits from DiffusionPolicy

### Transformer-based Implementation
1. **LongTaskTransformerConfig**: Configuration class for the transformer-based architecture
2. **LongTaskTransformerModel**: Neural network model using ResNet encoders and transformer decoders
3. **LongTaskTransformerPolicy**: Main policy class for the transformer-based approach

## Key Features

### Diffusion-based
- Customizable loss function through the `compute_loss` method placeholder
- Extended horizon support with `horizon_extension_factor`
- Weighted loss combination with `custom_loss_weight`

### Transformer-based
- ResNet-based image encoding for camera observations
- State tokenization for observation.state inputs
- Transformer encoder-decoder architecture for action generation
- Autoregressive action generation for long-horizon tasks

## Usage

### Diffusion-based Policy
To use the diffusion policy, you need to:

1. Implement your custom loss function in the `compute_loss` method of `LongTaskDiffusionPolicy`
2. Configure the model with appropriate parameters in `LongTaskDiffusionConfig`
3. Train the model using the standard lerobot training pipeline

### Transformer-based Policy
To use the transformer policy:

1. Configure the model with appropriate parameters in `LongTaskTransformerConfig`
2. Train the model using the standard lerobot training pipeline

## Customization

### Diffusion-based
The main customization point is the `compute_loss` method in `LongTaskDiffusionPolicy`. Replace the placeholder implementation with your specific loss computation logic for long task scenarios.

### Transformer-based
The transformer policy is designed to be modular and easily customizable:
- Modify the ResNet encoder by changing the `resnet_model` parameter
- Adjust the transformer architecture through parameters like `d_model`, `nhead`, `num_encoder_layers`, etc.
- Customize the action generation by modifying the `ActionHead` class
