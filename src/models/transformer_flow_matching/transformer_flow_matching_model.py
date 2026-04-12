import warnings
import contextlib

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.models as models
import math
from transformers import AutoModel, AutoProcessor


# Import ObjectDetector from separate file
from .object_detector import ObjectDetector, DiffusionSinusoidalPosEmb
from .box_encoder import BoxEncoder


class SmolVLAVisionTokenizer(nn.Module):
    """Vision tokenizer using SmolVLM backbone, mirroring the original SmolVLA approach:
    - Uses the pretrained SigLIP ViT vision encoder (configurable layers via vision_num_layers, last_hidden_state)
    - Uses the pretrained SmolVLM connector (pixel-shuffle + MLP resampler)
    - Keeps all patch tokens (no spatial pooling/collapsing)
    - Scales embeddings by sqrt(hidden_dim) before downstream use
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = int(getattr(config, "vision_input_size", 224))
        self.freeze_backbone = bool(getattr(config, "freeze_vision_backbone", True))
        self.model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

        # Initialize SmolVLM vision encoder
        print(f"Loading SmolVLM vision backbone: {self.model_id}")
        self.vision_model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # Extract SigLIP ViT vision encoder
        if hasattr(self.vision_model, 'vision_model'):
            self.vision_encoder = self.vision_model.vision_model
        else:
            self.vision_encoder = self.vision_model

        # Optionally truncate to first N transformer layers
        vision_num_layers = getattr(config, "vision_num_layers", None)
        if vision_num_layers is not None:
            self.vision_encoder.encoder.layers = self.vision_encoder.encoder.layers[:vision_num_layers]
            print(f"Truncated SigLIP ViT to first {vision_num_layers} layers")

        # Use the pretrained SmolVLM connector (pixel-shuffle + MLP resampler)
        # This projects from vision hidden size → VLM text hidden size
        if hasattr(self.vision_model, 'connector'):
            self.connector = self.vision_model.connector
            connector_out_dim = self.vision_model.config.text_config.hidden_size
        else:
            raise ValueError("SmolVLM model does not have a 'connector' attribute. Check model structure.")

        # Projection from VLM text dim → config.d_model (only if dims differ)
        if connector_out_dim != config.d_model:
            self.proj = nn.Linear(connector_out_dim, config.d_model)
        else:
            self.proj = nn.Identity()

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

        # ── Text encoding (task description) ──────────────────────────────────
        # We reuse the LM's embed_tokens layer (frozen) and add a small trainable
        # projection to map from LM hidden size → d_model.
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        lm_hidden_size = self.vision_model.config.text_config.hidden_size
        # embed_tokens: (vocab_size, lm_hidden_size)
        self.text_embed_tokens = self.vision_model.text_model.embed_tokens

        # Free the full VLM — only the extracted submodules above are needed.
        # This releases the LLM transformer layers (~1.8 GB fp32) from GPU memory.
        del self.vision_model

        # Trainable projection: lm_hidden_size → d_model
        self.text_proj = nn.Linear(lm_hidden_size, config.d_model)
        torch.nn.init.xavier_uniform_(self.text_proj.weight)
        torch.nn.init.constant_(self.text_proj.bias, 0)
        # ──────────────────────────────────────────────────────────────────────

        if self.freeze_backbone:
            self._freeze_vision_encoder()

        # Only initialize the projection layer (connector is pretrained)
        if isinstance(self.proj, nn.Linear):
            torch.nn.init.xavier_uniform_(self.proj.weight)
            torch.nn.init.constant_(self.proj.bias, 0)

    def _freeze_vision_encoder(self):
        """Freeze vision encoder, connector, and text embed_tokens (all are pretrained)."""
        self.vision_encoder.eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.connector.eval()
        for param in self.connector.parameters():
            param.requires_grad = False
        # Freeze the LM embed_tokens; only text_proj remains trainable
        for param in self.text_embed_tokens.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.vision_encoder.eval()
            self.connector.eval()
            self.text_embed_tokens.eval()
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
            mode="bicubic",
            align_corners=False,
        )
        images = images.clamp(0.0, 1.0)
        return (images - self.imagenet_mean) / self.imagenet_std

    def encode_text(self, descriptions: list[str]) -> torch.Tensor:
        """Encode task description strings → (B, L, d_model) tokens.

        Uses the pre-trained LM embed_tokens layer (frozen) followed by a small
        trainable linear projection.  All tokens (up to max_length=64) are kept
        so the action decoder can attend to the full instruction.

        Args:
            descriptions: list of B strings, one per sample.

        Returns:
            (B, L, d_model) float tensor on the same device as the vision encoder.
        """
        device = next(self.vision_encoder.parameters()).device

        encoded = self.processor.tokenizer(
            descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(device)

        ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with ctx:
            # (B, L, lm_hidden_size)
            text_embeds = self.text_embed_tokens(encoded["input_ids"]).float()

        # (B, L, d_model)
        return self.text_proj(text_embeds)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into patch tokens using SmolVLM backbone (all layers, all tokens).

        Mirrors SmolVLA's embed_image:
          1. SigLIP ViT → last_hidden_state (all patch tokens)
          2. Pretrained connector (pixel-shuffle + MLP resampler)
          3. sqrt(hidden_dim) scaling
          4. Linear projection → config.d_model

        Args:
            images: (B, T, C, H, W) or (B, C, H, W)

        Returns:
            tokens: (B, T * num_patch_tokens, d_model)
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if images.dim() != 5:
            raise ValueError(f"Expected images of shape (B, T, C, H, W), got {tuple(images.shape)}")

        bsize, t_obs, channels, height, width = images.shape
        flat_images = images.reshape(bsize * t_obs, channels, height, width)
        flat_images = self._preprocess_images(flat_images)

        vision_dtype = next(self.vision_encoder.parameters()).dtype

        # Step 1: Run SigLIP ViT — all layers, keep all patch tokens
        if self.freeze_backbone:
            with torch.no_grad():
                patch_tokens = self.vision_encoder(
                    pixel_values=flat_images.to(dtype=vision_dtype)
                ).last_hidden_state  # (B*T, num_patches, vision_hidden_size)
        else:
            patch_tokens = self.vision_encoder(
                pixel_values=flat_images.to(dtype=vision_dtype)
            ).last_hidden_state  # (B*T, num_patches, vision_hidden_size)

        # Step 2: Pretrained connector (pixel-shuffle + MLP resampler)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.connector(patch_tokens)  # (B*T, num_tokens, connector_out_dim)
        else:
            features = self.connector(patch_tokens)  # (B*T, num_tokens, connector_out_dim)

        # Step 3: Cast back to float32 for the trainable projection layers
        features = features.float()

        # Step 4: Scale by sqrt(hidden_dim), matching SmolVLA's embed_prefix normalization
        features = features * math.sqrt(features.shape[-1])

        # Step 5: Project to d_model
        tokens = self.proj(features)  # (B*T, num_tokens, d_model)

        num_tokens_per_frame = tokens.shape[1]
        return tokens.reshape(bsize, t_obs * num_tokens_per_frame, self.config.d_model)
      
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

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=2048):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feedforward (VERY important for capacity)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # Cross-attention
        attn_out, _ = self.cross_attn(query=query, key=key, value=value)
        x = self.norm1(query + self.dropout1(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x
      
class FlowMatchingTransformer(nn.Module):
    """Flow matching transformer with separate encoding for vision and state."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_scale = nn.Parameter(torch.tensor(1.0))  # Learnable scaling for state tokens
        self.vision_scale = nn.Parameter(torch.tensor(1.0))  # Learnable scaling for vision tokens
        self.box_scale = nn.Parameter(torch.tensor(1.0))     # Learnable scaling for box tokens
        # modality IDs: 0=state, 1=vision, 2=box, 3=text
        self.modality_embedding = nn.Embedding(4, config.d_model)
        self.text_scale = nn.Parameter(torch.tensor(1.0))

        # ------------------------------
        # 1. Single shared Object Detector for all cameras
        # ------------------------------
        self.object_detector = ObjectDetector(self.config)
                        
                
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
        # Make vision tokens mandatory
        self.use_vision_tokens = True
        self.vision_encoder = SmolVLAVisionTokenizer(config)
        self.vision_camera_embedding = nn.Embedding(config.num_cameras, config.d_model)
        self.vision_positional_encoding = PositionalEncoding(config.d_model)

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
        
                
        self.box_to_vision_cross_attn = nn.ModuleList([
            CrossAttentionBlock(
                d_model=config.d_model,
                nhead=config.nhead,
                dropout=0.1,
                dim_feedforward=config.dim_feedforward
            )
            for _ in range(2)  # e.g. 2–4
        ])
        
        self.state_to_vision_cross_attn = nn.ModuleList([
            CrossAttentionBlock(
                d_model=config.d_model,
                nhead=config.nhead,
                dropout=0.1,
                dim_feedforward=config.dim_feedforward
            )
            for _ in range(2)  # e.g. 2–4
        ])
        
        self.state_to_box_cross_attn = nn.ModuleList([
            CrossAttentionBlock(
                d_model=config.d_model,
                nhead=config.nhead,
                dropout=0.1,
                dim_feedforward=config.dim_feedforward
            )
            for _ in range(2)  # e.g. 2–4
        ])

        # Cross-attention where text is the query and each modality is key/value.
        # This produces context-grounded text representations (text reads from the scene).
        self.text_to_state_cross_attn = nn.ModuleList([
            CrossAttentionBlock(
                d_model=config.d_model,
                nhead=config.nhead,
                dropout=0.1,
                dim_feedforward=config.dim_feedforward
            )
            for _ in range(2)
        ])

        self.text_to_vision_cross_attn = nn.ModuleList([
            CrossAttentionBlock(
                d_model=config.d_model,
                nhead=config.nhead,
                dropout=0.1,
                dim_feedforward=config.dim_feedforward
            )
            for _ in range(2)
        ])

        self.text_to_box_cross_attn = nn.ModuleList([
            CrossAttentionBlock(
                d_model=config.d_model,
                nhead=config.nhead,
                dropout=0.1,
                dim_feedforward=config.dim_feedforward
            )
            for _ in range(2)
        ])

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

        # MLP that fuses action embeddings with time embedding (mirrors SmolVLA's action_time_mlp).
        # Concat [action, time] along last dim → project back to d_model with a nonlinearity,
        # allowing the model to gate action features based on the current flow timestep.
        self.action_time_mlp = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
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
        state_tokens_flat = state_tokens  # (B, T_obs, d_model)

        # ------------------------------
        # 2. Task description text tokens (if provided)
        # ------------------------------
        text_tokens_flat = None
        if "task_description" in batch and batch["task_description"]:
            descriptions = batch["task_description"]  # list[str] of length B
            text_tokens_flat = self.vision_encoder.encode_text(descriptions)  # (B, L, d_model)

        # ------------------------------
        # 2b. Lightweight vision token encoding
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

            # Skip box encoding if all coordinates are zero (no detections in entire batch)
            box_coords = box_data[..., :4]  # (B, T_obs, 6, 4)
            if box_coords.abs().sum() > 0:
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

                        # Detect objects and get bounding boxes using the shared detector
                        # Extract image for the current frame and camera
                        img_frame_cam = img[:, frame_idx:frame_idx+1, :, :, :]  # (B, 1, 1, C, H, W)
                        B_v, T_v, N_v, C_v, H_v, W_v = img_frame_cam.shape
                        img_reshaped = img_frame_cam.view(B_v * T_v * N_v, C_v, H_v, W_v)  # (B*1*1, C, H, W)
                        # YOLOWorld returns (N, 4) absolute pixel coords [x1, y1, x2, y2]
                        bounding_boxes, object_types = self.object_detector.detect_objects_and_get_bounding_boxes(img_reshaped)
                        
                        # Ensure exactly 2 bounding boxes per camera by padding/trimming
                        if bounding_boxes is None or bounding_boxes.numel() == 0:
                            bounding_boxes = torch.zeros((2, 4), device=img_reshaped.device, dtype=torch.float32)
                            object_types = ['unknown', 'unknown']
                        else:
                            current_num_boxes = bounding_boxes.shape[0]
                            if current_num_boxes < 2:
                                padding = torch.zeros((2 - current_num_boxes, 4), device=bounding_boxes.device, dtype=bounding_boxes.dtype)
                                bounding_boxes = torch.cat([bounding_boxes, padding], dim=0)
                                object_types = list(object_types) + ['unknown'] * (2 - current_num_boxes)
                            elif current_num_boxes > 2:
                                bounding_boxes = bounding_boxes[:2]
                                object_types = list(object_types)[:2]

                        # Normalise absolute pixel coords to [0, 1] to match the training
                        # format expected by encode_boxes_inference (which computes features
                        # like distance_to_right = 1 - x2 and center_x - 0.5).
                        norm_factors = torch.tensor(
                            [W_v, H_v, W_v, H_v],
                            device=bounding_boxes.device, dtype=bounding_boxes.dtype
                        )
                        bounding_boxes = bounding_boxes / norm_factors.clamp(min=1.0)
                        
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
        
        # Apply modality embeddings to each token type
        dev = batch["observation.state"].device
        if state_tokens_flat.shape[1] > 0:
            state_modality_emb = self.modality_embedding(torch.tensor(0, device=dev)).unsqueeze(0).unsqueeze(0)
            state_tokens_flat = state_tokens_flat + state_modality_emb.expand_as(state_tokens_flat)

        if vision_tokens_flat.shape[1] > 0:
            vision_modality_emb = self.modality_embedding(torch.tensor(1, device=dev)).unsqueeze(0).unsqueeze(0)
            vision_tokens_flat = vision_tokens_flat + vision_modality_emb.expand_as(vision_tokens_flat)

        if bbox_tokens_combined.shape[1] > 0:
            box_modality_emb = self.modality_embedding(torch.tensor(2, device=dev)).unsqueeze(0).unsqueeze(0)
            bbox_tokens_combined = bbox_tokens_combined + box_modality_emb.expand_as(bbox_tokens_combined)

        if text_tokens_flat is not None and text_tokens_flat.shape[1] > 0:
            text_modality_emb = self.modality_embedding(torch.tensor(3, device=dev)).unsqueeze(0).unsqueeze(0)
            text_tokens_flat = text_tokens_flat + text_modality_emb.expand_as(text_tokens_flat)

        # Apply scaling after modality embeddings
        vision_tokens_flat = vision_tokens_flat * self.vision_scale
        bbox_tokens_combined = bbox_tokens_combined * self.box_scale
        state_tokens_flat = state_tokens_flat * self.state_scale
        if text_tokens_flat is not None:
            text_tokens_flat = text_tokens_flat * self.text_scale

        
        # print(f"vision_tokens_flat norm: {vision_tokens_flat.norm():.6f}, max: {vision_tokens_flat.abs().max():.6f}")
        # print(f"bbox_tokens_combined norm: {bbox_tokens_combined.norm():.6f}, max: {bbox_tokens_combined.abs().max():.6f}")
        # print(f"state_tokens_flat norm: {state_tokens_flat.norm():.6f}, max: {state_tokens_flat.abs().max():.6f}")
        
        # Cross-attention fusion — only run when the key/value sequence is non-empty.
        # PyTorch MultiheadAttention with 0-length keys produces NaN.

        # boxes attending to vision (skip if no boxes or no vision)
        has_boxes = bbox_tokens_combined.shape[1] > 0
        has_vision = vision_tokens_flat.shape[1] > 0

        if has_boxes and has_vision:
            box_vision_tokens = bbox_tokens_combined
            for layer in self.box_to_vision_cross_attn:
                box_vision_tokens = layer(
                    query=box_vision_tokens,
                    key=vision_tokens_flat,
                    value=vision_tokens_flat,
                )
        else:
            box_vision_tokens = bbox_tokens_combined  # empty or unchanged

        # state attending to vision (skip if no vision)
        if has_vision:
            state_vision_tokens = state_tokens_flat
            for layer in self.state_to_vision_cross_attn:
                state_vision_tokens = layer(
                    query=state_vision_tokens,
                    key=vision_tokens_flat,
                    value=vision_tokens_flat,
                )
        else:
            state_vision_tokens = state_tokens_flat  # unchanged

        # state attending to boxes (skip if no boxes)
        if has_boxes:
            state_box_tokens = state_tokens_flat
            for layer in self.state_to_box_cross_attn:
                state_box_tokens = layer(
                    query=state_box_tokens,
                    key=bbox_tokens_combined,
                    value=bbox_tokens_combined,
                )
        else:
            state_box_tokens = state_tokens_flat  # unchanged

        # text attending to each modality — text is query, modality is key/value.
        # Produces scene-grounded text representations that capture what the task
        # description means *given the current robot state and visual observations*.
        has_text = text_tokens_flat is not None and text_tokens_flat.shape[1] > 0

        if has_text:
            # text reads from state: grounds instruction with joint positions/velocity
            text_state_tokens = text_tokens_flat
            for layer in self.text_to_state_cross_attn:
                text_state_tokens = layer(
                    query=text_state_tokens,
                    key=state_tokens_flat,
                    value=state_tokens_flat,
                )

            # text reads from vision: grounds instruction with visual observations
            if has_vision:
                text_vision_tokens = text_tokens_flat
                for layer in self.text_to_vision_cross_attn:
                    text_vision_tokens = layer(
                        query=text_vision_tokens,
                        key=vision_tokens_flat,
                        value=vision_tokens_flat,
                    )
            else:
                text_vision_tokens = None

            # text reads from boxes: grounds instruction with detected object positions
            if has_boxes:
                text_box_tokens = text_tokens_flat
                for layer in self.text_to_box_cross_attn:
                    text_box_tokens = layer(
                        query=text_box_tokens,
                        key=bbox_tokens_combined,
                        value=bbox_tokens_combined,
                    )
            else:
                text_box_tokens = None
        else:
            text_state_tokens = None
            text_vision_tokens = None
            text_box_tokens = None

        # Combine observation tokens:
        # Include RAW tokens alongside cross-attended versions so each encoder has
        # a direct gradient path (cross-attention queries alone vanish when dominated
        # by vision value vectors).
        context_parts = [tokens for tokens in [
            state_tokens_flat,      # direct gradient to state encoder
            bbox_tokens_combined,   # direct gradient to box encoder (empty → filtered)
            state_box_tokens,       # state enriched with object positions (or unchanged)
            state_vision_tokens,    # state enriched with visual context (or unchanged)
            box_vision_tokens,      # objects enriched with visual context (empty → filtered)
            text_tokens_flat,       # task description tokens (None → filtered)
            text_state_tokens,      # text grounded by robot state (None → filtered)
            text_vision_tokens,     # text grounded by visual observations (None → filtered)
            text_box_tokens,        # text grounded by object positions (None → filtered)
        ] if tokens is not None and tokens.shape[1] > 0]
        
        context = torch.cat(context_parts, dim=1)  # (B, total_seq_len, d_model)
        return context, spatial_outputs


    def velocity_field(self, noisy_actions, timesteps, obs_context):
        """
        Flow Matching step:
        Predicts the velocity field that transports samples from Gaussian to data distribution.
        """
        B, T_act, _ = noisy_actions.shape
        
        # 1. Action embeddings + positional encoding
        action_embeddings = self.action_in_proj(noisy_actions)
        action_embeddings = self.action_positional_encoding(action_embeddings)

        # 2. Time embedding: (B, d_model), then broadcast to (B, T_act, d_model)
        time_emb = self.time_embedding(timesteps.float())                        # (B, d_model)
        time_emb_seq = time_emb.unsqueeze(1).expand(-1, T_act, -1)              # (B, T_act, d_model)

        # 3. MLP fusion: nonlinearly gate action features by the flow timestep.
        #    Concat along feature dim → project back to d_model (mirrors SmolVLA).
        tgt = self.action_time_mlp(
            torch.cat([action_embeddings, time_emb_seq], dim=-1)
        )                                                                        # (B, T_act, d_model)

        # 4. Decoder pass — memory is the observation context only
        velocity_features = self.actions_expert(
            tgt=tgt,
            memory=obs_context,
        )
        

        #print(f"velocity_features mean abs: {velocity_features.norm():.6f}, max: {velocity_features.abs().max():.6f}")
        
        
        # Residual: direct gradient shortcut from output head → action_in_proj
        velocity_features = velocity_features + action_embeddings
        
        # 5. Predict the velocity field
        return self.velocity_prediction_head(velocity_features)

    
    def log_scales(self) -> dict:
        """Return current learnable scale values for monitoring during training."""
        return {
            "state_scale": self.state_scale.item(),
            "vision_scale": self.vision_scale.item(),
            "box_scale": self.box_scale.item(),
            "text_scale": self.text_scale.item(),
            "box_token_scale": self.box_encoder.box_token_scale.item(),
        }

    def sample_time(self, bsize, device):
        # Uniform sampling over (0, 1) — standard CFM formulation.
        return torch.rand(bsize, device=device, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)

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

        loss_steps = F.mse_loss(
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

