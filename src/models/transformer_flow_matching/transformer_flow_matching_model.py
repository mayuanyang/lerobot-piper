import warnings

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.models as models
import math


# Import ObjectDetector from separate file
from .object_detector import ObjectDetector, DiffusionSinusoidalPosEmb
from .box_encoder import BoxEncoder


class ResNet18VisionTokenizer(nn.Module):
    """Shared lightweight vision tokenizer used for all camera streams."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = int(getattr(config, "vision_input_size", 225))
        self.pool_rows = int(getattr(config, "vision_token_rows", 2))
        self.pool_cols = int(getattr(config, "vision_token_cols", 2))
        self.freeze_backbone = bool(getattr(config, "freeze_vision_backbone", True))

        weights = None
        if getattr(config, "use_pretrained_vision_backbone", True):
            weights = models.ResNet18_Weights.IMAGENET1K_V1

        try:
            backbone = models.resnet18(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"Failed to load pretrained ResNet18 weights ({exc}). Falling back to randomly initialized weights.",
                stacklevel=2,
            )
            backbone = models.resnet18(weights=None)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.spatial_pool = nn.AdaptiveAvgPool2d((self.pool_rows, self.pool_cols))
        self.token_projection = nn.Sequential(
            nn.Linear(512, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Apply better initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize vision tokenizer weights with better strategies."""
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                torch.nn.init.constant_(module.bias, 0)
                torch.nn.init.constant_(module.weight, 1.0)

        # Initialize token projection layers
        if hasattr(self, 'token_projection'):
            self.token_projection.apply(_basic_init)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        if images.dtype == torch.uint8 or images.max() > 1.5:
            images = images / 255.0

        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] > 3:
            images = images[:, :3]

        images = F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        images = images.clamp(0.0, 1.0)
        return (images - self.imagenet_mean) / self.imagenet_std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into a small fixed number of spatial tokens.

        Args:
            images: (B, T, C, H, W) or (B, C, H, W)

        Returns:
            tokens: (B, T * pool_rows * pool_cols, d_model)
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if images.dim() != 5:
            raise ValueError(f"Expected images of shape (B, T, C, H, W), got {tuple(images.shape)}")

        bsize, t_obs, channels, height, width = images.shape
        flat_images = images.reshape(bsize * t_obs, channels, height, width)
        flat_images = self._preprocess_images(flat_images)

        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(flat_images)
        else:
            features = self.backbone(flat_images)

        pooled = self.spatial_pool(features)
        pooled = pooled.flatten(2).transpose(1, 2)
        pooled = self.token_projection(pooled)
        num_tokens_per_frame = pooled.shape[1]
        return pooled.reshape(bsize, t_obs * num_tokens_per_frame, self.config.d_model)
      
class PositionalEncoding(nn.Module):
    """Positional encoding for action sequences."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Register a buffer for the positional encoding
        pe = self._generate_positional_encoding(max_len)
        self.register_buffer('pe', pe)

    def _generate_positional_encoding(self, max_len):
        """Generate positional encoding tensor."""
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        
        # If we need more positions than currently available, extend the buffer
        if seq_len > self.pe.size(1):
            # Extend with some buffer room
            new_max_len = max(seq_len + 100, self.max_len * 2)
            print(f"Extending positional encoding from {self.pe.size(1)} to {new_max_len}")
            pe = self._generate_positional_encoding(new_max_len)
            # Ensure the new buffer is on the same device as x
            self.register_buffer('pe', pe.to(x.device))
            
        return x + self.pe[:, :seq_len]


class FlowMatchingTransformer(nn.Module):
    """Flow matching transformer with separate encoding for vision and state."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_bounding_boxes_per_camera = 2
        self.action_chunk_size = 4  # Action chunking size
        
        self.state_scale = nn.Parameter(torch.tensor(1.0))  # Learnable scaling for state tokens
        self.vision_scale = nn.Parameter(torch.tensor(1.0))  # Learnable scaling for vision tokens
        self.box_scale = nn.Parameter(torch.tensor(1.0))     # Learnable scaling for box tokens
        self.time_embedding_scale = nn.Parameter(torch.tensor(0.2))  # Learnable scaling for time embedding

        # ------------------------------
        # 1. Single shared Object Detector for all cameras (initialize as None)
        # ------------------------------
        self.object_detector = None
        self._object_detector_initialized = False
                        
        self.state_token_offsets = nn.Parameter(
            torch.randn(4, config.d_model) * 0.02
        )
        
        # Camera names for processing
        self.camera_names = config.cameras_for_vision_state_concat if config.cameras_for_vision_state_concat else [
            f'observation.images.cam_{i}' for i in range(config.num_cameras)
        ]
        self._camera_name_mapping = {}  # Mapping from sanitized names to original names
        for i, cam_name in enumerate(self.camera_names):
            # Sanitize the camera name for use as a module key
            sanitized_name = cam_name.replace('.', '_')
            self._camera_name_mapping[sanitized_name] = cam_name
            
        # ------------------------------
        # 2. Box encoder for processing bounding box data (training + inference)
        # ------------------------------
        self.use_vision_tokens = bool(getattr(config, "use_vision_tokens", False))
        if self.use_vision_tokens:
            backbone_name = str(getattr(config, "light_weight_vision_backbone", "resnet18")).lower()
            if backbone_name != "resnet18":
                raise ValueError(
                    f"Unsupported lightweight vision backbone '{config.vision_backbone}'. Only 'resnet18' is implemented."
                )
            self.vision_encoder = ResNet18VisionTokenizer(config)
            self.vision_camera_embedding = nn.Embedding(config.num_cameras, config.d_model)
            self.vision_positional_encoding = PositionalEncoding(config.d_model)
        else:
            self.vision_encoder = None
            self.vision_camera_embedding = None
            self.vision_positional_encoding = None

        self.box_encoder = BoxEncoder(config)
        self.box_positional_encoding = PositionalEncoding(config.d_model)

        # ------------------------------
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model // 2),
            nn.Mish(),
            nn.LayerNorm(config.d_model // 2),
            nn.Linear(config.d_model // 2, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.state_positional_encoding = PositionalEncoding(config.d_model)
        
        # State cross attention component
        self.state_cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=0.1,
            batch_first=True
        )
        self.state_cross_attn_norm = nn.LayerNorm(config.d_model)
        self.state_cross_attn_dropout = nn.Dropout(0.1)
        
        # Box-vision self attention component for fusing box and vision tokens
        self.box_vision_self_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=0.1,
            batch_first=True
        )
        self.box_vision_self_attn_norm = nn.LayerNorm(config.d_model)
        self.box_vision_self_attn_dropout = nn.Dropout(0.1)


        # ------------------------------
        # 8. Number of inference steps for flow matching sampling
        # ------------------------------
        # Register as buffer so it's properly handled during device transfers
        self.register_buffer('num_inference_steps', torch.tensor(config.num_inference_steps))

        # ------------------------------
        # 9. Enhanced Action Encoder / Decoder
        # ------------------------------
        # Dedicated action input projection like VLAFlowMatching
        self.action_in_proj = nn.Sequential(
            nn.Linear(config.action_dim, config.d_model // 2),
            nn.Mish(),
            nn.LayerNorm(config.d_model // 2),
            nn.Linear(config.d_model // 2, config.d_model),
            nn.LayerNorm(config.d_model),
        )
                
        self.action_positional_encoding = PositionalEncoding(config.d_model, config.horizon)

        action_layers = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.actions_expert = nn.TransformerDecoder(
            action_layers,
            num_layers=config.num_decoder_layers
        )
        
        self.velocity_prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.action_dim),
        )
        
        self.time_embedding = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.d_model),
            nn.Mish(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )
        
        #self.obs_batch_norm = nn.LayerNorm(config.d_model)

        # Apply better initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with better strategies."""
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                torch.nn.init.constant_(module.bias, 0)
                torch.nn.init.constant_(module.weight, 1.0)

        # Initialize all submodules
        self.apply(_basic_init)
        
        # Initialize specific components with specialized strategies
        
        
        # State token offsets: initialize with smaller variance for stability
        if hasattr(self, 'state_token_offsets'):
            torch.nn.init.normal_(self.state_token_offsets, mean=0.0, std=0.01)
        
        # Initialize embedding layers with proper scaling
        if hasattr(self, 'vision_camera_embedding') and self.vision_camera_embedding is not None:
            torch.nn.init.normal_(self.vision_camera_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize transformer components
        if hasattr(self, 'actions_expert'):
            # Initialize transformer decoder layers
            for decoder_layer in self.actions_expert.layers:
                # Initialize attention layers
                if hasattr(decoder_layer, 'self_attn'):
                    torch.nn.init.xavier_uniform_(decoder_layer.self_attn.in_proj_weight)
                    torch.nn.init.constant_(decoder_layer.self_attn.out_proj.weight, 0)
                if hasattr(decoder_layer, 'multihead_attn'):
                    torch.nn.init.xavier_uniform_(decoder_layer.multihead_attn.in_proj_weight)
                    torch.nn.init.constant_(decoder_layer.multihead_attn.out_proj.weight, 0)
                
                # Initialize FFN layers with scaled init
                if hasattr(decoder_layer, 'linear1'):
                    torch.nn.init.xavier_uniform_(decoder_layer.linear1.weight)
                    torch.nn.init.constant_(decoder_layer.linear1.bias, 0)
                if hasattr(decoder_layer, 'linear2'):
                    torch.nn.init.xavier_uniform_(decoder_layer.linear2.weight)
                    torch.nn.init.constant_(decoder_layer.linear2.bias, 0)

        # Initialize output heads with smaller scale for stability
        if hasattr(self, 'velocity_prediction_head'):
            for module in self.velocity_prediction_head.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0)

    def _reshape_camera_tensor(self, image_tensor: torch.Tensor, batch_size: int, t_obs: int) -> torch.Tensor:
        """Normalize camera tensor shapes to (B, T, C, H, W)."""
        if image_tensor.dim() == 4:
            if image_tensor.shape[0] == batch_size * t_obs and t_obs > 1:
                return image_tensor.reshape(batch_size, t_obs, *image_tensor.shape[-3:])
            if image_tensor.shape[0] == batch_size:
                return image_tensor.unsqueeze(1)
            if t_obs == 1:
                return image_tensor.reshape(batch_size, 1, *image_tensor.shape[-3:])
        elif image_tensor.dim() == 5:
            return image_tensor
        elif image_tensor.dim() == 6 and image_tensor.shape[2] == 1:
            return image_tensor.squeeze(2)

        raise ValueError(
            f"Unsupported image tensor shape {tuple(image_tensor.shape)} for batch_size={batch_size}, t_obs={t_obs}."
        )

    def _encode_vision_tokens(self, batch, batch_size: int, t_obs: int) -> torch.Tensor:
        """Encode lightweight ResNet18 spatial tokens for each available camera."""
        device = batch["observation.state"].device
        if not self.use_vision_tokens or self.vision_encoder is None:
            return torch.empty(batch_size, 0, self.config.d_model, device=device)

        all_vision_tokens = []
        for cam_index, camera_key in enumerate(self.camera_names):
            image_tensor = batch.get(camera_key)
            if not isinstance(image_tensor, torch.Tensor):
                continue

            image_tensor = self._reshape_camera_tensor(image_tensor, batch_size, t_obs)
            camera_tokens = self.vision_encoder(image_tensor)
            camera_tokens = self.vision_positional_encoding(camera_tokens)

            camera_emb = self.vision_camera_embedding(
                torch.tensor(cam_index, device=device, dtype=torch.long)
            ).view(1, 1, -1)
            camera_tokens = camera_tokens + camera_emb
            all_vision_tokens.append(camera_tokens)

        if not all_vision_tokens:
            return torch.empty(batch_size, 0, self.config.d_model, device=device)

        return torch.cat(all_vision_tokens, dim=1)



    def _add_noise_tokens(self, state_tokens: torch.Tensor) -> torch.Tensor:
        """
        Add 3 noise tokens for each input state token with 0-2% noise.
        
        Args:
            state_tokens: (B, T_obs, d_model) - original state tokens
            
        Returns:
            augmented_tokens: (B, T_obs * 4, d_model) - original + 3 noise tokens per original token
        """
        B, T_obs, d_model = state_tokens.shape
        
        # Create noise for 3 additional tokens per original token (0-2% noise)
        noise_scale = torch.rand(B, T_obs, 3, 1, device=state_tokens.device) * 0.02  # (B, T_obs, 3, 1) - 0 to 2% noise
        noise = torch.randn(B, T_obs, 3, d_model, device=state_tokens.device) * noise_scale  # (B, T_obs, 3, d_model)
        
        # Original tokens (no noise) - repeat for 4 tokens per original token
        original_part = state_tokens.unsqueeze(2).expand(-1, -1, 4, -1)[:, :, :1, :]  # (B, T_obs, 1, d_model)
        original_repeated = original_part.expand(-1, -1, 4, -1)  # (B, T_obs, 4, d_model)
        
        # Add noise to the repeated original tokens (first token remains unchanged, next 3 get noise)
        noise_padding = torch.zeros(B, T_obs, 1, d_model, device=state_tokens.device)  # No noise for first token
        noise_with_padding = torch.cat([noise_padding, noise], dim=2)  # (B, T_obs, 4, d_model)
        
        # Combine original tokens with noise
        all_tokens = original_repeated + noise_with_padding  # (B, T_obs, 4, d_model)
        
        # Reshape to (B, T_obs * 4, d_model)
        augmented_tokens = all_tokens.view(B, T_obs * 4, d_model)
        
        return augmented_tokens

    def get_condition(self, batch, generate_heatmaps=False):
        """
        Encode state and bounding box data to get context tokens.
        During training, uses observation.box from dataset.
        During inference, uses Qwen3-VL for object detection.
        Returns:
            context: (B, T_obs * N_tokens, d_model)
            spatial_outputs: for visualization (including bounding boxes if requested)
        """
        B, T_obs = batch["observation.state"].shape[:2]
        spatial_outputs = {}

        # ------------------------------
        # 1. State encoding (compute once for reuse)
        # ------------------------------
        state_tokens = self.state_encoder(batch["observation.state"])  # (B, T, d_model)
        
        state_tokens = self.state_positional_encoding(state_tokens)
        
        # Add noise augmentation: create 3 noise tokens for each original token with 0-2% noise
        state_tokens_flat = self._add_noise_tokens(state_tokens)  # (B, T_obs * 4, d_model)

        # ------------------------------
        # 2. Lightweight vision token encoding
        # ------------------------------
        vision_tokens_flat = self._encode_vision_tokens(batch, B, T_obs)
        
        if vision_tokens_flat.shape[1] > 0:
            spatial_outputs["vision_tokens_shape"] = tuple(vision_tokens_flat.shape)

        # ------------------------------
        # 3. Bounding box encoding
        # ------------------------------
        all_bbox_tokens = []  # Collect bounding box tokens
        
        # Check if observation.box is available in the batch (from dataset during training)
        if "observation.box" in batch:
            box_data = batch["observation.box"]  # (B, T_obs, 6, 6)

            # BoxEncoder returns a single combined token sequence:
            #   [distance tokens] then [box tokens]
            bbox_tokens_flat, coordinates_normalized = self.box_encoder.encode_tokens_train(
                box_data, batch
            )
            spatial_outputs["coordinates_normalized"] = coordinates_normalized

            # Apply positional encoding once over the combined sequence
            bbox_tokens_flat = self.box_positional_encoding(bbox_tokens_flat)

            all_bbox_tokens.append(bbox_tokens_flat)

        else:
            # observation.box is missing usually mean it is in inference, process each camera with the shared object detector
            for frame_idx in range(T_obs):
                for cam_index, sanitized_cam_key in enumerate(
                    sorted(self._camera_name_mapping.keys())
                ):
                    # Get the original camera name from the mapping
                    original_cam_key = self._camera_name_mapping[sanitized_cam_key]
                    batch_key = original_cam_key
                    if batch_key in batch:
                        img = self._reshape_camera_tensor(batch[batch_key], B, T_obs).unsqueeze(2)

                        # Initialize object detector if not already initialized
                        if not self._object_detector_initialized:
                            print("Initializing object detector for inference...")
                            self.object_detector = ObjectDetector(self.config)
                            self._object_detector_initialized = True
                        
                        # Detect objects and get bounding boxes using the shared detector
                        # Extract image for the current frame and camera
                        img_frame_cam = img[:, frame_idx:frame_idx+1, :, :, :]  # (B, 1, 1, C, H, W)
                        B_v, T_v, N_v, C_v, H_v, W_v = img_frame_cam.shape
                        img_reshaped = img_frame_cam.view(B_v * T_v * N_v, C_v, H_v, W_v)  # (B*1*1, C, H, W)
                        bounding_boxes, object_types = self.object_detector.detect_objects_and_get_bounding_boxes(img_reshaped)
                        
                        # Ensure exactly 2 bounding boxes per camera by padding with zeros if needed
                        if bounding_boxes is None or bounding_boxes.numel() == 0:
                            # No detections, create 2 empty boxes with 4 coordinates each for 2D bounding boxes
                            bounding_boxes = torch.zeros((2, 4), device=img_reshaped.device, dtype=torch.float32)
                            object_types = ['unknown', 'unknown']
                        else:
                            # Pad or trim to exactly 2 boxes
                            current_num_boxes = bounding_boxes.shape[0]
                            if current_num_boxes < 2:
                                # Pad with zeros (4 coordinates for 2D bounding boxes)
                                padding = torch.zeros((2 - current_num_boxes, 4), device=bounding_boxes.device, dtype=bounding_boxes.dtype)
                                bounding_boxes = torch.cat([bounding_boxes, padding], dim=0)
                                # Pad object types with 'unknown'
                                object_types.extend(['unknown'] * (2 - current_num_boxes))
                            elif current_num_boxes > 2:
                                # Trim to 2 boxes
                                bounding_boxes = bounding_boxes[:2]
                                object_types = object_types[:2]
                        
                        
                        # Normalize bounding box coordinates during inference to match training
                        bounding_boxes[:, 0::2] /= W_v
                        bounding_boxes[:, 1::2] /= H_v
                        
                        bbox_tokens_flat = self.box_encoder.encode_tokens_inference(
                            bounding_boxes=bounding_boxes,
                            cam_index=cam_index,
                            B_v=B_v,
                            T_v=T_v,
                            N_v=N_v,
                        )

                        all_bbox_tokens.append(bbox_tokens_flat)
                        

            # ------------------------------
            # 3. Apply positional encoding to bounding box tokens (inference only)
            # ------------------------------
            if all_bbox_tokens:
                # Concatenate all collected tokens
                bbox_tokens_all = torch.cat(all_bbox_tokens, dim=1)  # (B, total_boxes, d_model)
                
                # Apply positional encoding to bounding box tokens
                bbox_tokens_flat = self.box_positional_encoding(bbox_tokens_all)  # (B, total_boxes, d_model)
                
                # Store the processed tokens
                all_bbox_tokens = [bbox_tokens_flat]
            else:
                obs_tokens = torch.empty(B, 0, self.config.d_model, device=batch["observation.state"].device)
        
        # Combine all bounding box tokens
        if all_bbox_tokens:
            bbox_tokens_combined = torch.cat(all_bbox_tokens, dim=1)  # (B, total_boxes, d_model)
        else:
            bbox_tokens_combined = torch.empty(B, 0, self.config.d_model, device=batch["observation.state"].device)
        
        vision_tokens_flat = vision_tokens_flat * self.vision_scale
        bbox_tokens_combined = bbox_tokens_combined * self.box_scale
        state_tokens_flat = state_tokens_flat * self.state_scale

        
        # print(f"vision_tokens_flat norm: {vision_tokens_flat.norm():.6f}, max: {vision_tokens_flat.abs().max():.6f}")
        # print(f"bbox_tokens_combined norm: {bbox_tokens_combined.norm():.6f}, max: {bbox_tokens_combined.abs().max():.6f}")
        # print(f"state_tokens_flat norm: {state_tokens_flat.norm():.6f}, max: {state_tokens_flat.abs().max():.6f}")
        
        # Apply self-attention between box and vision tokens to create fused tokens
      
        # Concatenate vision and box tokens for self-attention
        box_vision_tokens = torch.cat([bbox_tokens_combined, vision_tokens_flat], dim=1)  # (B, vision_len + box_len, d_model)
        
        # Apply self-attention between box and vision tokens
        box_vision_attn_out, _ = self.box_vision_self_attn(
            query=box_vision_tokens,
            key=box_vision_tokens,
            value=box_vision_tokens
        )
        box_vision_tokens = self.box_vision_self_attn_norm(
            box_vision_tokens + self.box_vision_self_attn_dropout(box_vision_attn_out)
        )
        

        
        # Combine observation tokens (fused vision + fused boxes + state)
        context_parts = [tokens for tokens in [state_tokens_flat, box_vision_tokens, vision_tokens_flat  ,bbox_tokens_combined] if tokens.shape[1] > 0]
        
        
        context = torch.cat(context_parts, dim=1)
        
        return context, spatial_outputs


    def velocity_field(self, noisy_actions, timesteps, obs_context):
        """
        Flow Matching step:
        Predicts the velocity field that transports samples from Gaussian to data distribution.
        """
        B, T_act, _ = noisy_actions.shape
        
        # 1. Action & Time Embeddings (enhanced like VLAFlowMatching)
        action_embeddings = self.action_in_proj(noisy_actions)  # Use dedicated input projection
        action_embeddings = self.action_positional_encoding(action_embeddings)
        
        # Time embedding: (B, d_model) -> (B, 1, d_model)
        time_emb = self.time_embedding_scale * self.time_embedding(timesteps.float()).unsqueeze(1)
        
        #print(f"time_emb mean abs: {time_emb.norm():.6f}, max: {time_emb.abs().max():.6f}")
        
        
        # 2. Add time to action tokens
        # We expand time across the action horizon so that each action token is aware of the flow matching time, which can help the model learn time-dependent velocity fields.
        tgt = action_embeddings + time_emb.expand(-1, T_act, -1)
        
        # 3. The Memory - Concatenate time embedding with observation context
        # Expand time embedding to match the sequence length of obs_context
        time_emb_expanded = time_emb.expand(-1, obs_context.shape[1], -1)  # (B, seq_len, d_model)
        extended_memory = torch.cat([obs_context, time_emb_expanded], dim=1)  # Concatenate along sequence dimension
                
        # print(f"action_embeddings mean abs: {torch.norm(action_embeddings, dim=2).mean()}, max: {action_embeddings.abs().max():.6f}")
        # print(f"time_emb mean abs: {torch.norm(time_emb, dim=2).mean()}, max: {time_emb.abs().max():.6f}")
        # print(f"extended_memory mean abs: {torch.norm(extended_memory, dim=2).mean()}, max: {extended_memory.abs().max():.6f}")
        # print(f"tgt mean abs: {torch.norm(tgt, dim=2).mean()}, max: {tgt.abs().max():.6f}")

        # 4. Decoder Pass
        # Self-attention ensures T_act is a smooth curve.
        # Cross-attention aligns T_act with your CropConvNet features.
        # Allow actions to attend to all tokens (time, vision, boxes, state, and time embedding)
        velocity_features = self.actions_expert(
            tgt=tgt,
            memory=extended_memory
        )
        

        #print(f"velocity_features mean abs: {velocity_features.norm():.6f}, max: {velocity_features.abs().max():.6f}")
        
        
        # Residual connection to preserve gradients
        #velocity_features = velocity_features + action_embeddings
        
        # 5. Predict the velocity field
        return self.velocity_prediction_head(velocity_features)

    
    def sample_time(self, bsize, device):
        # Sample time from a Beta distribution to encourage learning across the entire flow matching trajectory, with more emphasis on mid-range times.
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def compute_loss(self, batch):
        """Flow Matching Training: Learn to predict the velocity field with improved loss computation."""
        
        actions = batch["action"]     # [B, T, D]
        
        B, T_act = actions.shape[:2]
        
        # 1. Get observation context
        obs_context, _ = self.get_condition(batch)  # (B, T_obs, d_model)
                
        # Infer device from model parameters
        device = next(self.parameters()).device
        
        # 2. Sample time using Beta distribution
        timesteps = self.sample_time(B, device)
        
        # 3. Sample Gaussian noise
        noise = torch.randn_like(actions, device=device)

        # 4. Construct flow matching targets (straight line coupling)
        # Interpolate between noise and data
        noisy_actions = (1 - timesteps[:, None, None]) * noise + timesteps[:, None, None] * actions
        
        # 5. Predict velocity field
        pred_velocity = self.velocity_field(noisy_actions, timesteps, obs_context)
        
        # 6. Compute flow matching loss
        # Target velocity is the difference between data and noise
        target_velocity = actions - noise
        
        # Use a weighting scheme that down weight mid-range times to encourage learning at the endpoints of the flow, which can help with stability and convergence.
        weight = (timesteps[:, None, None] ** 2 + (1 - timesteps[:, None, None]) ** 2)

        loss_steps = weight * F.mse_loss(
            pred_velocity,
            target_velocity,
            reduction="none"
        )

        
        # 7. Handle padding if present
        if "action_is_pad" in batch:
            # Apply padding mask: True means padded, False means valid
            in_episode_bound = ~batch["action_is_pad"]  # True for valid actions
            loss_steps = loss_steps * in_episode_bound.unsqueeze(-1)
        
        loss = loss_steps.mean()
        return loss

    def forward(self, batch):
        """Inference: Solve ODE using learned velocity field."""
        B = batch["observation.state"].shape[0]
        T_act = self.config.horizon
        
        # Get observation context
        obs_context, spatial_outputs = self.get_condition(batch)  # (B, T_obs, d_model)
        
        # Infer device from model parameters
        device = next(self.parameters()).device
                
        # Start from pure Gaussian noise
        samples = torch.randn((B, T_act, self.config.action_dim), device=device)
        
        # This is RK2 / midpoint solver.
        num_steps = int(self.num_inference_steps.item())
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device)

            v1 = self.velocity_field(samples, t, obs_context)

            midpoint = samples + 0.5 * dt * v1
            t_mid = torch.full((B,), (i + 0.5) / num_steps, device=device)

            v2 = self.velocity_field(midpoint, t_mid, obs_context)

            samples = samples + dt * v2
            
            
        # Return both the actions and spatial outputs for visualization
        return samples, spatial_outputs

