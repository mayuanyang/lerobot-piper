# Long Task Diffusion Policy

This module implements a custom diffusion policy designed for long-horizon tasks with a customizable loss function.

## Components

1. **LongTaskDiffusionConfig**: Configuration class that extends DiffusionConfig with additional parameters for long task handling
2. **LongTaskDiffusionModel**: Neural network model with customizable loss computation
3. **LongTaskDiffusionPolicy**: Main policy class that inherits from DiffusionPolicy

## Key Features

- Customizable loss function through the `compute_loss` method placeholder
- Extended horizon support with `horizon_extension_factor`
- Weighted loss combination with `custom_loss_weight`

## Usage

To use this policy, you need to:

1. Implement your custom loss function in the `compute_loss` method of `LongTaskDiffusionPolicy`
2. Configure the model with appropriate parameters in `LongTaskDiffusionConfig`
3. Train the model using the standard lerobot training pipeline

## Customization

The main customization point is the `compute_loss` method in `LongTaskDiffusionPolicy`. Replace the placeholder implementation with your specific loss computation logic for long task scenarios.
