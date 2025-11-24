# Training with Checkpoint Resume Functionality

This document explains how to use the checkpoint resume functionality in the training script.

## Overview

The `train.py` script now supports resuming training from a specific checkpoint. This is useful when:
- Training is interrupted and you want to continue from where you left off
- You want to fine-tune a previously trained model
- You want to experiment with different training configurations starting from a checkpoint

## Usage

### Starting Training from Scratch

To start training from scratch:

```bash
python src/train.py --output_dir ./outputs/my_training_run --dataset_id ISdept/piper_arm
```

### Resuming Training from a Checkpoint

To resume training from a checkpoint:

```bash
python src/train.py --output_dir ./outputs/my_training_run --dataset_id ISdept/piper_arm --resume_from_checkpoint ./outputs/my_training_run/checkpoint-5000
```

### Using the train function directly

You can also call the `train` function directly in Python:

```python
from train import train

# Start training from scratch
train(output_dir='./outputs/my_training_run', dataset_id='ISdept/piper_arm')

# Resume training from a checkpoint
train(output_dir='./outputs/my_training_run', dataset_id='ISdept/piper_arm', resume_from_checkpoint='./outputs/my_training_run/checkpoint-5000')
```

## Checkpoint Structure

When training, checkpoints are saved in the output directory with the following structure:

```
outputs/
└── my_training_run/
    ├── checkpoint-1000/
    │   ├── policy.safetensors
    │   ├── preprocessor_config.json
    │   ├── postprocessor_config.json
    │   └── optimizer_state.pth
    ├── checkpoint-2000/
    │   ├── policy.safetensors
    │   ├── preprocessor_config.json
    │   ├── postprocessor_config.json
    │   └── optimizer_state.pth
    └── ... (final model files)
```

Each checkpoint directory contains:
- `policy.safetensors`: The trained model weights
- `preprocessor_config.json`: Configuration for the preprocessor
- `postprocessor_config.json`: Configuration for the postprocessor
- `optimizer_state.pth`: The optimizer state (important for resuming training)

## How Resume Works

When resuming training:
1. The script loads the policy model from the checkpoint
2. It attempts to load the preprocessor and postprocessor from the checkpoint
3. It creates a new optimizer and loads its state from the checkpoint
4. Training continues from the step number encoded in the checkpoint directory name

## Best Practices

1. **Always use the same output directory**: When resuming, use the same `--output_dir` as the original training run to ensure checkpoints are saved in the correct location.

2. **Verify checkpoint existence**: Make sure the checkpoint directory exists and contains the necessary files before attempting to resume.

3. **Monitor training progress**: When resuming, the step counter will continue from where it left off, so you can track progress accurately.

4. **Keep checkpoints**: Don't delete checkpoint directories if you might want to resume training later.
