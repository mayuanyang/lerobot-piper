#!/usr/bin/env python

"""
Demo script showcasing the usage of the LongTaskTransformer model.
"""

import torch
from src.models.long_task_diffusion.long_task_transformer_config import LongTaskTransformerConfig
from src.models.long_task_diffusion.long_task_transformer_policy import LongTaskTransformerPolicy


def demo_model_creation():
    """Demonstrate creating and using the LongTaskTransformer model."""
    print("=" * 60)
    print("LongTaskTransformer Model Demo")
    print("=" * 60)
    
    # Create configuration
    print("1. Creating model configuration...")
    config = LongTaskTransformerConfig(
        n_obs_steps=8,          # Number of observation steps
        horizon=16,             # Prediction horizon
        n_action_steps=8,       # Number of action steps to execute
        state_dim=7,            # 7-DOF arm state
        action_dim=7,           # 7-DOF arm actions
        d_model=128,            # Transformer model dimension (smaller for demo)
        nhead=4,                # Number of attention heads
        num_encoder_layers=2,   # Number of encoder layers
        num_decoder_layers=2,   # Number of decoder layers
        dim_feedforward=256,    # Feedforward dimension
        dropout=0.1,            # Dropout rate
        resnet_model="resnet18", # ResNet model for image encoding
        pretrained_resnet=True   # Use pretrained ResNet weights
    )
    print(f"   Created config with {config.n_obs_steps} observation steps and {config.horizon} horizon")
    
    # Create policy
    print("\n2. Creating policy...")
    policy = LongTaskTransformerPolicy(config)
    print(f"   Policy created successfully")
    
    # Create mock batch
    print("\n3. Creating mock data...")
    batch_size = 2
    obs_steps = config.n_obs_steps
    horizon = config.horizon
    state_dim = config.state_dim
    action_dim = config.action_dim
    
    batch = {
        "observation.state": torch.randn(batch_size, obs_steps, state_dim),
        "action": torch.randn(batch_size, horizon, action_dim)
    }
    print(f"   Created batch with {batch_size} samples")
    print(f"   Observation shape: {batch['observation.state'].shape}")
    print(f"   Action shape: {batch['action'].shape}")
    
    # Test forward pass
    print("\n4. Testing forward pass (training)...")
    loss, _ = policy.forward(batch)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test action selection
    print("\n5. Testing action selection (inference)...")
    actions = policy.select_action(batch)
    print(f"   Selected actions shape: {actions.shape}")
    print(f"   Expected: ({batch_size}, {config.n_action_steps}, {action_dim})")
    
    # Show model parameters
    print("\n6. Model information...")
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def demo_architecture_details():
    """Show details about the transformer architecture."""
    print("\nArchitecture Details:")
    print("-" * 30)
    print("The LongTaskTransformer model consists of:")
    print("1. ResNet-based image encoders for processing camera observations")
    print("2. State tokenizers for converting observation.state to tokens")
    print("3. Transformer encoder for processing observation context")
    print("4. Transformer decoder for autoregressive action generation")
    print("5. Action head for converting decoder output to actions")
    print("\nThe model processes observations and generates a sequence")
    print("of actions autoregressively for long-horizon tasks.")


if __name__ == "__main__":
    demo_model_creation()
    demo_architecture_details()
