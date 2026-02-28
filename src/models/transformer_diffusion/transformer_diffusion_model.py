import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from typing import Dict, Optional, List, Tuple
import math
import numpy as np
import os
from pathlib import Path
import torchvision.models as models
import torchvision.transforms as T
import re
import json
import random
import cv2


# Import Qwen3-VL-8B-Instruct components
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

      
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


class DiffusionSinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ObjectDetector:
    """Simplified object detector using Qwen3-VL for pure bounding box detection."""
    
    def __init__(self, config, system_prompt=None, user_prompt=None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Qwen3-VL model and processor
        print("Loading Qwen3-VL-4B-Instruct model for object detection...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        
        # Default system prompt - defines the format
        self.system_prompt = system_prompt or "You are a precise object detector for robotic manipulation. For each object, provide its 3D bounding box in JSON format with 'bbox_3d' field containing [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw] and a 'label' field indicating the object type (e.g., 'object_to_grasp', 'container', 'obstacle')."
        
        # Default user prompt - defines what the actual user requirement is
        self.user_prompt = user_prompt or "Locate objects in the picture that are relevant for robotic manipulation."
        
        # Create embeddings for different object types
        # We'll use a small embedding dimension for object type (e.g., 32) and combine it with the coordinate projection
        self.object_type_embedding_dim = 32
        self.object_type_embedding = nn.Embedding(10, self.object_type_embedding_dim)  # Support up to 10 object types
        # Update coordinate projection to account for the additional embedding dimensions
        # Now using 9 parameters for 3D bounding boxes instead of 4
        self.coord_projection = nn.Linear(9 + self.object_type_embedding_dim, self.config.d_model)
        
        # Object type to index mapping
        self.object_type_to_idx = {
            'object_to_grasp': 0,
            'container': 1,
            'obstacle': 2,
            'robot_arm': 3,
            'workspace': 4,
            'unknown': 5
        }

    def detect_objects_and_get_bounding_boxes(self, image_tensor, user_prompt=None):
        """
        Use Qwen3-VL to detect objects in the image and extract bounding boxes.
        
        Args:
            image_tensor: (B, C, H, W) tensor of images
            user_prompt: Optional override for the user prompt
            
        Returns:
            bounding_boxes: (B, N, 9) tensor of 3D bounding boxes [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
            object_types: List of lists of object type strings
        """
        # Use provided user prompt or fall back to instance default
        actual_user_prompt = user_prompt or self.user_prompt
        
        B, C, H, W = image_tensor.shape
        bounding_boxes_list = []
        object_types_list = []
        
        # Process each image in the batch
        for i in range(B):
            # Convert tensor to PIL Image for Qwen3-VL processing
            img = image_tensor[i].detach().cpu()
            # Convert from [-1, 1] to [0, 1] if needed
            if img.min() < 0:
                img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            # Convert to PIL Image
            pil_transform = T.ToPILImage()
            pil_image = pil_transform(img)
            
            # Prepare messages for Qwen3-VL with system and user roles
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_image,
                        },
                        {"type": "text", "text": actual_user_prompt},
                    ],
                }
            ]
            
            # Apply chat template
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Parse bounding boxes and object types from output text
            bounding_boxes, object_types = self._parse_bounding_boxes_from_text(output_text[0], W, H)
            bounding_boxes_list.append(bounding_boxes)
            object_types_list.append(object_types)
        
        # Pad bounding boxes and object types to same number across batch
        if bounding_boxes_list:
            # Find maximum number of boxes
            max_boxes = max([boxes.shape[0] if boxes is not None else 0 for boxes in bounding_boxes_list])
            
            # Pad all boxes and object types to max_boxes
            padded_boxes = []
            padded_object_types = []
            
            for boxes, obj_types in zip(bounding_boxes_list, object_types_list):
                if boxes is not None and boxes.numel() > 0:
                    # Pad boxes
                    if boxes.shape[0] < max_boxes:
                        padding = torch.zeros((max_boxes - boxes.shape[0], 9), device=boxes.device)
                        boxes = torch.cat([boxes, padding], dim=0)
                    padded_boxes.append(boxes)
                    
                    # Pad object types
                    if len(obj_types) < max_boxes:
                        obj_types.extend(['unknown'] * (max_boxes - len(obj_types)))
                    padded_object_types.append(obj_types)
                else:
                    # Create empty tensors
                    empty_boxes = torch.zeros((max_boxes, 9), device=image_tensor.device)
                    empty_object_types = ['unknown'] * max_boxes
                    
                    padded_boxes.append(empty_boxes)
                    padded_object_types.append(empty_object_types)
            
            bounding_boxes_batch = torch.stack(padded_boxes, dim=0)  # (B, max_boxes, 9)
            object_types_batch = padded_object_types  # List of lists
        else:
            bounding_boxes_batch = torch.zeros((B, 0, 9), device=image_tensor.device)
            object_types_batch = [['unknown'] * 0 for _ in range(B)]  # Empty list for each batch item
        
        return bounding_boxes_batch, object_types_batch

    def _parse_bounding_boxes_from_text(self, text, img_width, img_height):
        """
        Parse 3D bounding boxes and object types from Qwen3-VL JSON text output.
        
        Args:
            text: String output from Qwen3-VL
            img_width: Width of the input image
            img_height: Height of the input image
            
        Returns:
            tuple: (boxes, object_types) where boxes is a torch.Tensor of shape (N, 4) 
                   with selected coordinates from 3D bounding boxes, and object_types is a 
                   list of strings representing object types, or (None, []) if none found
        """
        try:
            # Use the provided function to parse 3D bounding boxes
            bounding_boxes_data = self.parse_bbox_3d_from_text(text)
            print(f"Successfully parsed {len(bounding_boxes_data)} bounding box entries.")

            # Extract bounding boxes and object types
            boxes = []
            object_types = []
            for item in bounding_boxes_data:
                if 'bbox_3d' in item:
                    # Extract 3D bounding box parameters
                    bbox_3d = item['bbox_3d']
                    x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw = bbox_3d
                    
                    # Use all 9 parameters for the 3D bounding box representation
                    boxes.append([x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw])
                    
                    # Extract object type/label
                    label = item.get('label', 'unknown')
                    object_types.append(label)
            
            if boxes:
                return torch.tensor(boxes, dtype=torch.float32), object_types
            else:
                return None, []
        except Exception as e:
            print(f"Error parsing bounding boxes: {e}")
            return None, []

    def _parse_json(self, json_output):
        """
        Parse JSON from Qwen3-VL output, removing markdown fencing.
        
        Args:
            json_output: String output from Qwen3-VL
            
        Returns:
            str: Cleaned JSON string
        """
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        
        return json_output

    def parse_bbox_3d_from_text(self, text: str) -> list:
        """
        Parse 3D bounding box information from assistant response.
        
        Args:
            text: Assistant response text containing JSON with bbox_3d information
            
        Returns:
            List of dictionaries containing bbox_3d data
        """
        try:
            # Find JSON content
            if "```json" in text:
                start_idx = text.find("```json")
                end_idx = text.find("```", start_idx + 7)
                if end_idx != -1:
                    json_str = text[start_idx + 7:end_idx].strip()
                else:
                    json_str = text[start_idx + 7:].strip()
            else:
                # Find first [ and last ]
                start_idx = text.find('[')
                end_idx = text.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    json_str = text[start_idx:end_idx + 1]
                else:
                    return []
            
            # Parse JSON
            bbox_data = json.loads(json_str)
            
            # Normalize to list format
            if isinstance(bbox_data, list):
                return bbox_data
            elif isinstance(bbox_data, dict):
                return [bbox_data]
            else:
                return []
                
        except (json.JSONDecodeError, IndexError, KeyError):
            return []

    def convert_3dbbox(self, point, cam_params):
        """Convert 3D bounding box to 2D image coordinates"""
        x, y, z, x_size, y_size, z_size, pitch, yaw, roll = point
        hx, hy, hz = x_size / 2, y_size / 2, z_size / 2
        local_corners = [
            [ hx,  hy,  hz],
            [ hx,  hy, -hz],
            [ hx, -hy,  hz],
            [ hx, -hy, -hz],
            [-hx,  hy,  hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [-hx, -hy, -hz]
        ]

        def rotate_xyz(_point, _pitch, _yaw, _roll):
            x0, y0, z0 = _point
            x1 = x0
            y1 = y0 * math.cos(_pitch) - z0 * math.sin(_pitch)
            z1 = y0 * math.sin(_pitch) + z0 * math.cos(_pitch)

            x2 = x1 * math.cos(_yaw) + z1 * math.sin(_yaw)
            y2 = y1
            z2 = -x1 * math.sin(_yaw) + z1 * math.cos(_yaw)

            x3 = x2 * math.cos(_roll) - y2 * math.sin(_roll)
            y3 = x2 * math.sin(_roll) + y2 * math.cos(_roll)
            z3 = z2

            return [x3, y3, z3]
        
        img_corners = []
        for corner in local_corners:
            rotated = self.rotate_xyz(corner, np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll))
            X, Y, Z = rotated[0] + x, rotated[1] + y, rotated[2] + z
            if Z > 0:
                x_2d = cam_params['fx'] * (X / Z) + cam_params['cx']
                y_2d = cam_params['fy'] * (Y / Z) + cam_params['cy']
                img_corners.append([x_2d, y_2d])

        return img_corners

    def rotate_xyz(self, point, pitch, yaw, roll):
        """Rotate a 3D point by the given angles"""
        x0, y0, z0 = point
        x1 = x0
        y1 = y0 * math.cos(pitch) - z0 * math.sin(pitch)
        z1 = y0 * math.sin(pitch) + z0 * math.cos(pitch)

        x2 = x1 * math.cos(yaw) + z1 * math.sin(yaw)
        y2 = y1
        z2 = -x1 * math.sin(yaw) + z1 * math.cos(yaw)

        x3 = x2 * math.cos(roll) - y2 * math.sin(roll)
        y3 = x2 * math.sin(roll) + y2 * math.cos(roll)
        z3 = z2

        return [x3, y3, z3]

    def draw_3dbboxes(self, image_path, cam_params, bbox_3d_list, color=None):
        """Draw multiple 3D bounding boxes on the same image and return matplotlib figure"""
        # Read image
        annotated_image = cv2.imread(image_path)
        if annotated_image is None:
            print(f"Error reading image: {image_path}")
            return None

        edges = [
            [0,1], [2,3], [4,5], [6,7],
            [0,2], [1,3], [4,6], [5,7],
            [0,4], [1,5], [2,6], [3,7]
        ]
        
        # Draw 3D box for each bbox
        for bbox_data in bbox_3d_list:
            # Extract bbox_3d from the dictionary
            if isinstance(bbox_data, dict) and 'bbox_3d' in bbox_data:
                bbox_3d = bbox_data['bbox_3d']
            else:
                bbox_3d = bbox_data
            
            # Convert angles multiplied by 180 to degrees
            bbox_3d = list(bbox_3d)  # Convert to list for modification
            bbox_3d[-3:] = [_x * 180 for _x in bbox_3d[-3:]]
            bbox_2d = self.convert_3dbbox(bbox_3d, cam_params)

            if len(bbox_2d) >= 8:
                # Generate random color for each box
                box_color = [random.randint(0, 255) for _ in range(3)]
                for start, end in edges:
                    try:
                        pt1 = tuple([int(_pt) for _pt in bbox_2d[start]])
                        pt2 = tuple([int(_pt) for _pt in bbox_2d[end]])
                        cv2.line(annotated_image, pt1, pt2, box_color, 2)
                    except:
                        continue

        # Convert BGR to RGB for matplotlib
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(annotated_image_rgb)
        ax.axis('off')
        
        return fig


    

class SimpleDiffusionTransformer(nn.Module):
    """Flow matching transformer with separate encoding for vision and state."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ------------------------------
        # 1. Single shared Object Detector for all cameras
        # ------------------------------
        self.object_detector = ObjectDetector(config)
        
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
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        self.state_positional_encoding = PositionalEncoding(config.d_model)

        # ------------------------------
        # 4. Vision Positional Encoding
        # -----------------------
        self.vision_positional_encoding = PositionalEncoding(config.d_model)
        # Temporal positional encoding for preserving time structure
        self.temporal_positional_encoding = PositionalEncoding(config.d_model)
        

        # ------------------------------
        # 8. Number of inference steps for flow matching sampling
        # ------------------------------
        # Register as buffer so it's properly handled during device transfers
        self.register_buffer('num_inference_steps', torch.tensor(config.num_inference_steps))

        # ------------------------------
        # 9. Enhanced Action Encoder / Decoder
        # ------------------------------
        # Dedicated action input projection like VLAFlowMatching
        self.action_in_proj = nn.Linear(config.action_dim, config.d_model)
        
        self.action_positional_encoding = PositionalEncoding(config.d_model, config.horizon)

        denoising_layers = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.actions_expert = nn.TransformerDecoder(
            denoising_layers,
            num_layers=config.num_decoder_layers
        )
        self.velocity_prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.Mish(),
            nn.Linear(config.d_model // 2, config.action_dim)
        )
        self.time_embedding = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.d_model),
            nn.Mish(),
            nn.Linear(config.d_model, config.d_model)
        )


    def get_condition(self, batch, generate_heatmaps=False):
        """
        Encode state and use Qwen3-VL for object detection to get bounding box tokens.
        Returns:
            context: (B, T_obs * N_tokens, d_model)
            spatial_outputs: for visualization (including bounding boxes if requested)
        """
        B, T_obs = batch["observation.state"].shape[:2]

        # ------------------------------
        # 1. State encoding (compute once for reuse)
        # ------------------------------
        state_tokens = self.state_encoder(batch["observation.state"])  # (B, T, d_model)
        state_tokens = self.state_positional_encoding(state_tokens)
        # Flatten time dimension
        state_tokens_flat = state_tokens.view(B, T_obs, self.config.d_model)
        # Add a singleton token dim to match fusion later
        state_tokens_flat = state_tokens_flat.unsqueeze(2)  # (B, T, 1, d_model)
        state_tokens_flat = state_tokens_flat.view(B, T_obs * 1, self.config.d_model)

        # ------------------------------
        # 2. Object detection per camera using Qwen3-VL
        # ------------------------------
        spatial_outputs = {}  # for visualization
        all_bbox_tokens = []  # Collect bounding box tokens from all cameras
        all_bbox_coords = []   # Collect bounding box coordinates for spatial encoding
        all_frame_indices = [] # Collect frame indices for temporal encoding
        all_object_types = []  # Collect object types for embedding
        
        # Process each camera with the shared object detector
        for frame_idx in range(T_obs):
            for sanitized_cam_key in self._camera_name_mapping.keys():
                # Get the original camera name from the mapping
                original_cam_key = self._camera_name_mapping[sanitized_cam_key]
                batch_key = original_cam_key
                if batch_key in batch:
                    img = batch[batch_key]
                    if img.dim() == 4:  # (B*T, C, H, W)
                        img = img.view(B, T_obs, 1, *img.shape[-3:])
                    elif img.dim() == 5:  # (B, T, C, H, W)
                        img = img.unsqueeze(2)

                    # Store null values for heatmaps since we're not generating them in this simplified version
                    spatial_outputs[f"{sanitized_cam_key}_heatmap"] = None

                    # Detect objects and get bounding boxes using the shared detector
                    # Extract image for the current frame and camera
                    img_frame_cam = img[:, frame_idx:frame_idx+1, :, :, :]  # (B, 1, 1, C, H, W)
                    B_v, T_v, N_v, C_v, H_v, W_v = img_frame_cam.shape
                    img_reshaped = img_frame_cam.view(B_v * T_v * N_v, C_v, H_v, W_v)  # (B*1*1, C, H, W)
                    bounding_boxes, object_types = self.object_detector.detect_objects_and_get_bounding_boxes(img_reshaped)
                    
                    # Store bounding boxes for visualization
                    spatial_outputs[f"{sanitized_cam_key}_bounding_boxes_frame_{frame_idx}"] = bounding_boxes
                    
                    # Generate tokens from bounding box coordinates and object types
                    if bounding_boxes is not None and bounding_boxes.numel() > 0:
                        # Get object type embeddings
                        object_type_indices = torch.tensor([
                            self.object_detector.object_type_to_idx.get(obj_type, self.object_detector.object_type_to_idx['unknown']) 
                            for obj_type in object_types
                        ], device=bounding_boxes.device)
                        object_embeddings = self.object_detector.object_type_embedding(object_type_indices)  # (N, object_type_embedding_dim)
                        
                        # Concatenate bounding box coordinates with object type embeddings
                        combined_features = torch.cat([bounding_boxes, object_embeddings], dim=1)  # (N, 9 + object_type_embedding_dim)
                        
                        # Project to d_model dimension
                        bbox_tokens = self.object_detector.coord_projection(combined_features)  # (N, d_model)
                        
                        # Reshape bbox_tokens to match vision tokens format
                        # Add dimensions to match (B_v, T_v, N_v, N_boxes, d_model)
                        bbox_tokens = bbox_tokens.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N_boxes, d_model)
                        bbox_tokens = bbox_tokens.expand(B_v, T_v, N_v, -1, self.config.d_model)  # (B_v, T_v, N_v, N_boxes, d_model)
                        
                        # Flatten bbox_tokens to (B, T * N_cam * N_boxes, d_model)
                        bbox_tokens_flat = bbox_tokens.view(B_v, T_v * N_v * bbox_tokens.shape[-2], self.config.d_model)
                        
                        # Collect bounding box tokens, coordinates, frame indices, and object types
                        if bbox_tokens_flat.shape[1] > 0:  # Only collect if there are bounding boxes
                            all_bbox_tokens.append(bbox_tokens_flat)
                            
                            # Collect bounding box coordinates for spatial encoding
                            # Reshape bounding_boxes to match the token structure
                            bounding_boxes_reshaped = bounding_boxes.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N_boxes, 9)
                            bounding_boxes_reshaped = bounding_boxes_reshaped.expand(B_v, T_v, N_v, -1, 9)  # (B_v, T_v, N_v, N_boxes, 9)
                            bounding_boxes_flat = bounding_boxes_reshaped.view(B_v, T_v * N_v * bounding_boxes_reshaped.shape[-2], 9)  # (B, T*N_cam*N_boxes, 9)
                            all_bbox_coords.append(bounding_boxes_flat)
                            
                            # Collect frame indices for temporal encoding
                            frame_indices = torch.full((B_v, bbox_tokens_flat.shape[1]), frame_idx, device=bbox_tokens_flat.device)  # (B, T*N_cam*N_boxes)
                            all_frame_indices.append(frame_indices)
                            
                            # Collect object types
                            all_object_types.extend(object_types)

        # ------------------------------
        # 3. Apply combined positional encoding to bounding box tokens
        # ------------------------------
        if all_bbox_tokens and all_bbox_coords and all_frame_indices:
            # Concatenate all collected tokens, coordinates, and frame indices
            bbox_tokens_all = torch.cat(all_bbox_tokens, dim=1)  # (B, total_boxes, d_model)
            bbox_coords_all = torch.cat(all_bbox_coords, dim=1)   # (B, total_boxes, 9)
            frame_indices_all = torch.cat(all_frame_indices, dim=1)  # (B, total_boxes)
            
            # Apply spatial positional encoding based on bounding box coordinates
            # We'll use the x_center and y_center coordinates (first two values) for spatial encoding
            # Normalize coordinates to [0, 1] for positional encoding
            spatial_coords = bbox_coords_all[:, :, :2]  # (B, total_boxes, 2) - x_center, y_center
            
            # Create spatial positional encoding
            # For simplicity, we'll create a learnable embedding based on the coordinates
            # In a more advanced implementation, we could use a sinusoidal encoding
            B, N, _ = spatial_coords.shape
            spatial_pe = torch.zeros(B, N, self.config.d_model, device=spatial_coords.device)
            # Simple linear transformation of coordinates to positional encoding
            # This is a basic approach - a more sophisticated method would use a dedicated spatial encoding layer
            for i in range(B):
                for j in range(N):
                    # Use coordinates to modulate the positional encoding
                    # This is a simplified approach - in practice, you might want a more complex spatial encoding
                    spatial_pe[i, j, 0::2] = torch.sin(spatial_coords[i, j, 0] * 100)  # x coordinate
                    spatial_pe[i, j, 1::2] = torch.cos(spatial_coords[i, j, 1] * 100)  # y coordinate
            
            # Apply temporal positional encoding based on frame indices
            # We'll reshape frame_indices to use with the existing temporal_positional_encoding
            max_frame_idx = frame_indices_all.max().item() if frame_indices_all.numel() > 0 else 0
            # Ensure we have enough temporal positions
            if max_frame_idx >= self.temporal_positional_encoding.pe.shape[1]:
                # Extend temporal positional encoding if needed
                new_max_len = max(max_frame_idx + 10, self.temporal_positional_encoding.max_len * 2)
                print(f"Extending temporal positional encoding from {self.temporal_positional_encoding.pe.shape[1]} to {new_max_len}")
                pe_temporal = self.temporal_positional_encoding._generate_positional_encoding(new_max_len)
                self.temporal_positional_encoding.register_buffer('pe', pe_temporal.to(self.temporal_positional_encoding.pe.device))
            
            # Apply temporal positional encoding
            temporal_pe = self.temporal_positional_encoding.pe[0, frame_indices_all]  # (B, total_boxes, d_model)
            
            # Combine spatial and temporal positional encodings with the bounding box tokens
            # Add both positional encodings to the bbox tokens
            obs_tokens = bbox_tokens_all + spatial_pe + temporal_pe
        else:
            obs_tokens = torch.empty(B, 0, self.config.d_model, device=batch["observation.state"].device)
        
        # Combine observation tokens (only bounding box tokens) with state tokens
        context = torch.cat([obs_tokens, state_tokens_flat], dim=1)

        return context, spatial_outputs


    def velocity_field(self, actions, timesteps, obs_context):
        """
        Flow Matching step:
        Predicts the velocity field that transports samples from Gaussian to data distribution.
        """
        B, T_act, _ = actions.shape
        
        # 1. Action & Time Embeddings (enhanced like VLAFlowMatching)
        action_embeddings = self.action_in_proj(actions)  # Use dedicated input projection
        action_embeddings = self.action_positional_encoding(action_embeddings)
        
        # Time embedding: (B, d_model) -> (B, 1, d_model)
        time_emb = self.time_embedding(timesteps.float()).unsqueeze(1)
        
        # 2. Add time to action tokens
        # We expand time across the action horizon
        tgt = action_embeddings + time_emb.expand(-1, T_act, -1)
        
        # 3. Augment Memory with Time
        # We add the time token to the observation context so the 
        # cross-attention mechanism is aware of the flow matching time.
        # extended_memory: (B, 1 + (T_obs * N_tokens), d_model)
        extended_memory = torch.cat([time_emb, obs_context], dim=1)
        
        # 4. Decoder Pass
        # Self-attention ensures T_act is a smooth curve.
        # Cross-attention aligns T_act with your CropConvNet features.
        # Allow actions to attend to all tokens (time, vision, and state)
        velocity_features = self.actions_expert(
            tgt=tgt,
            memory=extended_memory
        )
        #print(f"velocity_features mean abs: {velocity_features.abs().mean():.6f}, max: {velocity_features.abs().max():.6f}")
        
        
        # Residual connection to preserve gradients
        #velocity_features = velocity_features + action_embeddings
        
        # 5. Predict the velocity field
        return self.velocity_prediction_head(velocity_features)

    
    def sample_time(self, bsize, device):
        """Sample time using Beta(1.5, 1.0) distribution for better training dynamics"""
        # Beta(1.5, 1.0) distribution favors earlier timesteps
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001  # Scale to [0.001, 0.999]
        return time

    def compute_loss(self, batch):
        """Flow Matching Training: Learn to predict the velocity field."""
        actions = batch["action"] # (B, Horizon, Action_Dim)
        B, T_act = actions.shape[:2]
        
        # 1. Get observation context
        obs_context, spatial_outputs = self.get_condition(batch) # (B, T_obs, d_model)
                
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
        #print(f"pred_velocity mean abs: {pred_velocity.abs().mean():.6f}, max: {pred_velocity.abs().max():.6f}")
        
        # 6. Compute flow matching loss
        # Target velocity is the difference between data and noise
        target_velocity = actions - noise
        #print(f"target_velocity mean abs: {target_velocity.abs().mean():.6f}, max: {target_velocity.abs().max():.6f}")
        
        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        
        # 7. Handle padding if present
        if "action_is_pad" in batch:
            # Apply padding mask: True means padded, False means valid
            in_episode_bound = ~batch["action_is_pad"]  # True for valid actions
            loss = loss * in_episode_bound.unsqueeze(-1)
        
        # Return mean loss
        return loss.mean()

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
        
        # Solve ODE using Euler integration
        num_steps = int(self.num_inference_steps.item())
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device)
            velocity = self.velocity_field(samples, t, obs_context)
            samples = samples + dt * velocity
            
        # Return both the actions and spatial outputs for visualization
        return samples, spatial_outputs


# Keep the original class for backward compatibility, but alias to the new simplified version
DiffusionTransformer = SimpleDiffusionTransformer
