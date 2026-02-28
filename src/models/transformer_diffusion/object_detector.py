import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
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
            attn_implementation="sdpa"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        
        # Default system prompt - defines the format
        self.system_prompt = system_prompt or "You are a precise object detector for robotic manipulation. For each object, provide its 3D bounding box in JSON format with 'bbox_3d' field containing [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw] and a 'label' field indicating the object type (e.g., 'object_to_grasp', 'container', 'obstacle')."
        
        # Default user prompt - defines what the actual user requirement is
        self.user_prompt = user_prompt or "Locate objects in the picture that are relevant for robotic manipulation."
        
        # Create embeddings for different object types
        # We'll use a small embedding dimension for object type (e.g., 32) and combine it with the coordinate projection
        # Now using 9 parameters for 3D bounding boxes instead of 4
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
                    "content": [{"type": "text", "text": self.system_prompt}]
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