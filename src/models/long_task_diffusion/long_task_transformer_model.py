import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
import torch.nn.functional as F
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from typing import Dict, Optional, Tuple


class ResNetImageEncoder(nn.Module):
    """ResNet-based image encoder for processing camera observations."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize ResNet model
        if config.resnet_model == "resnet18":
            resnet = models.resnet18(pretrained=config.pretrained_resnet)
        elif config.resnet_model == "resnet34":
            resnet = models.resnet34(pretrained=config.pretrained_resnet)
        elif config.resnet_model == "resnet50":
            resnet = models.resnet50(pretrained=config.pretrained_resnet)
        else:
            raise ValueError(f"Unsupported ResNet model: {config.resnet_model}")
        
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add a projection layer to match the transformer dimension
        self.projection = nn.Linear(resnet.fc.in_features, config.d_model)
        
    def forward(self, images: Tensor) -> Tensor:
        """
        Process images through ResNet encoder.
        
        Args:
            images: Tensor of shape (batch_size, channels, height, width)
            
        Returns:
            features: Tensor of shape (batch_size, d_model)
        """
        # Pass through ResNet
        features = self.resnet(images)
        # Flatten spatial dimensions
        features = features.view(features.size(0), -1)
        # Project to transformer dimension
        features = self.projection(features)
        return features


class StateTokenizer(nn.Module):
    """Tokenize observation state into transformer-compatible tokens."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = nn.Linear(config.state_dim, config.state_token_dim)
        self.projection = nn.Linear(config.state_token_dim, config.d_model)
        
    def forward(self, state: Tensor) -> Tensor:
        """
        Tokenize state observations.
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
            
        Returns:
            tokens: Tensor of shape (batch_size, d_model)
        """
        tokens = self.tokenizer(state)
        tokens = self.projection(tokens)
        return tokens


class ActionHead(nn.Module):
    """Generate actions from transformer decoder outputs."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_projection = nn.Linear(config.d_model, config.action_dim)
        
    def forward(self, decoder_output: Tensor) -> Tensor:
        """
        Generate actions from decoder output.
        
        Args:
            decoder_output: Tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            actions: Tensor of shape (batch_size, seq_len, action_dim)
        """
        actions = self.action_projection(decoder_output)
        return actions


class LongTaskTransformerModel(nn.Module):
    """
    Long Task Transformer Model that uses ResNet for image encoding,
    tokenizes state observations, and uses a transformer decoder
    to generate actions autoregressively.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Image encoders (one per camera if configured)
        if config.image_features:
            self.image_encoders = nn.ModuleDict()
            for camera_key in config.image_features.keys():
                self.image_encoders[camera_key] = ResNetImageEncoder(config)
        
        # State tokenizer
        self.state_tokenizer = StateTokenizer(config)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(config.n_obs_steps + config.horizon, config.d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.num_decoder_layers)
        
        # Action head
        self.action_head = ActionHead(config)
        
        # Learnable tokens for action generation
        self.action_tokens = nn.Parameter(torch.randn(config.horizon, config.d_model))
        
    def _prepare_image_features(self, batch: Dict[str, Tensor]) -> Tensor:
        """Extract and encode image features from batch."""
        batch_size = batch[OBS_STATE].shape[0]
        n_obs_steps = batch[OBS_STATE].shape[1]
        
        if not self.config.image_features:
            return None
            
        # Stack images from all cameras and time steps
        image_tensors = []
        for camera_key in self.config.image_features.keys():
            if camera_key in batch:
                # Shape: (batch_size, n_obs_steps, channels, height, width)
                images = batch[camera_key]
                batch_size, n_obs_steps = images.shape[:2]
                # Reshape to process all images at once
                images = images.view(-1, *images.shape[2:])
                # Encode images
                encoded_images = self.image_encoders[camera_key](images)
                # Reshape back
                encoded_images = encoded_images.view(batch_size, n_obs_steps, -1)
                image_tensors.append(encoded_images)
        
        if image_tensors:
            # Concatenate features from all cameras
            image_features = torch.cat(image_tensors, dim=-1)
            return image_features
        return None
    
    def _prepare_state_features(self, batch: Dict[str, Tensor]) -> Tensor:
        """Tokenize state observations."""
        states = batch[OBS_STATE]  # (batch_size, n_obs_steps, state_dim)
        batch_size, n_obs_steps = states.shape[:2]
        
        # Reshape to process all states at once
        states = states.view(-1, states.shape[-1])
        # Tokenize states
        state_tokens = self.state_tokenizer(states)
        # Reshape back
        state_tokens = state_tokens.view(batch_size, n_obs_steps, -1)
        
        return state_tokens
    
    def _prepare_context_tokens(self, batch: Dict[str, Tensor]) -> Tensor:
        """Prepare context tokens from observations."""
        batch_size = batch[OBS_STATE].shape[0]
        
        # Get image features
        image_features = self._prepare_image_features(batch)
        # Get state features
        state_features = self._prepare_state_features(batch)
        
        # Combine features
        if image_features is not None and state_features is not None:
            # Concatenate along feature dimension
            context_tokens = torch.cat([image_features, state_features], dim=-1)
        elif image_features is not None:
            context_tokens = image_features
        elif state_features is not None:
            context_tokens = state_features
        else:
            raise ValueError("No valid input features found")
        
        # Add positional encoding
        n_obs_steps = context_tokens.shape[1]
        context_tokens = context_tokens + self.pos_encoding[:n_obs_steps].unsqueeze(0)
        
        return context_tokens
    
    def forward(self, batch: Dict[str, Tensor], action_queries: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the transformer model.
        
        Args:
            batch: Dictionary containing observation data
            action_queries: Optional action query tokens for generation
            
        Returns:
            predicted_actions: Tensor of shape (batch_size, horizon, action_dim)
        """
        batch_size = batch[OBS_STATE].shape[0]
        
        # Prepare context tokens from observations
        context_tokens = self._prepare_context_tokens(batch)
        
        # Encode context
        memory = self.transformer_encoder(context_tokens)
        
        # Generate action queries if not provided
        if action_queries is None:
            action_queries = self.action_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Add positional encoding to action queries
        action_queries = action_queries + self.pos_encoding[self.config.n_obs_steps:self.config.n_obs_steps + self.config.horizon].unsqueeze(0)
        
        # Decode actions
        decoder_output = self.transformer_decoder(action_queries, memory)
        
        # Generate actions
        predicted_actions = self.action_head(decoder_output)
        
        return predicted_actions
    
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Compute the loss for training.
        
        Args:
            batch: Dictionary containing the input batch data
            
        Returns:
            loss: Scalar tensor with the computed loss
        """
        # Forward pass
        predicted_actions = self.forward(batch)
        
        # Get ground truth actions
        ground_truth_actions = batch[ACTION]
        
        # Compute MSE loss
        loss = F.mse_loss(predicted_actions, ground_truth_actions)
        
        return loss
