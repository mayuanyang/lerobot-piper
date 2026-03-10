#!/usr/bin/env python3
"""
Standalone script to precompute bounding boxes for a video using Qwen3-VL object detection.
This version avoids importing problematic modules with NumPy compatibility issues.

Supports both 3D and 2D object detection:
- 3D mode: Detects 3D bounding boxes with [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
- 2D mode: Detects 2D bounding boxes with [x1, y1, x2, y2] coordinates

Usage:
    python precompute_video_bounding_boxes_standalone.py \\
        --video_path /path/to/video.mp4 \\
        --user_prompt "Locate objects in the picture that are relevant for robotic manipulation." \\
        --output_path /path/to/output.json \\
        --mode 3d

    For 2D detection:
    python precompute_video_bounding_boxes_standalone.py \\
        --video_path /path/to/video.mp4 \\
        --user_prompt "locate every instance that belongs to the following categories: 'plate/dish, scallop, wine bottle, tv, bowl, spoon, air conditioner, coconut drink, cup, chopsticks, person'. Report bbox coordinates in JSON format." \\
        --output_path /path/to/output.json \\
        --mode 2d
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import traceback
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import sys
import os

# Conditional import of transformers to handle cases where it might not be available
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not available. Object detection will not work.")


class StandaloneObjectDetector:
    """Standalone object detector using Qwen3-VL for bounding box detection."""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {self.device}")
        
        if not TRANSFORMERS_AVAILABLE:
            print("WARNING: Transformers library not available. Object detection will not work.")
            self.model = None
            self.processor = None
            return
        
        # Load Qwen3-VL model and processor
        print("Loading Qwen3-VL-4B-Instruct model for object detection...")
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-4B-Instruct",
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="sdpa"
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        except Exception as e:
            print(f"Error loading Qwen3-VL model: {e}")
            self.model = None
            self.processor = None
            return
        
        # Default system prompt - defines the format
        self.system_prompt = "You are a precise object detector for robotic manipulation. For each object, provide its 3D bounding box in JSON format with 'bbox_3d' field containing [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw] and a 'label' field indicating the object type (e.g., 'object_to_grasp', 'container', 'obstacle')."
        
        # Default user prompt - defines what the actual user requirement is
        self.user_prompt = "Locate objects in the picture that are relevant for robotic manipulation."
        
        # System prompt for 2D bounding box detection
        self.system_prompt_2d = "You are a precise 2D object detector. For each object, provide its 2D bounding box in JSON format with 'bbox_2d' field containing [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. All coordinates should be normalized to [0, 1000]. Also provide a 'label' field indicating the object category."
        
        # Default user prompt for 2D detection
        self.user_prompt_2d = 'locate every instance that belongs to the following categories: "plate/dish, scallop, wine bottle, tv, bowl, spoon, air conditioner, coconut drink, cup, chopsticks, person". Report bbox coordinates in JSON format.'
        
        print("Object detector initialized successfully!")

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
        if self.model is None or self.processor is None:
            # Return empty results if model is not available
            B = image_tensor.shape[0]
            return torch.zeros((B, 0, 9)), [[] for _ in range(B)]
        
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=16384)
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

    def detect_objects_and_get_2d_bounding_boxes(self, image_tensor, user_prompt=None):
        """
        Use Qwen3-VL to detect objects in the image and extract 2D bounding boxes.
        
        Args:
            image_tensor: (B, C, H, W) tensor of images
            user_prompt: Optional override for the user prompt
            
        Returns:
            bounding_boxes: List of lists of dictionaries with 'bbox_2d' and 'label' fields
        """
        if self.model is None or self.processor is None:
            # Return empty results if model is not available
            B = image_tensor.shape[0]
            return [[] for _ in range(B)]
        
        # Use provided user prompt or fall back to instance default
        actual_user_prompt = user_prompt or self.user_prompt_2d
        
        B, C, H, W = image_tensor.shape
        bounding_boxes_list = []
        
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
            
            # Prepare messages for Qwen3-VL with system and user roles for 2D detection
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt_2d}]
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            
            # Parse 2D bounding boxes from output text
            bounding_boxes = self._parse_2d_bounding_boxes_from_text(output_text[0])
            bounding_boxes_list.append(bounding_boxes)
        
        return bounding_boxes_list

    def _parse_2d_bounding_boxes_from_text(self, text):
        """
        Parse 2D bounding boxes and object types from Qwen3-VL JSON text output.
        
        Args:
            text: String output from Qwen3-VL
            
        Returns:
            List of dictionaries with 'bbox_2d' and 'label' fields
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
                # Validate each item in the list has required fields
                validated_data = []
                for item in bbox_data:
                    if isinstance(item, dict) and 'bbox_2d' in item and 'label' in item:
                        # Ensure bbox_2d has 4 coordinates
                        if len(item['bbox_2d']) == 4:
                            validated_data.append(item)
                return validated_data
            elif isinstance(bbox_data, dict):
                # Check if single dict has required fields
                if 'bbox_2d' in bbox_data and 'label' in bbox_data:
                    # Ensure bbox_2d has 4 coordinates
                    if len(bbox_data['bbox_2d']) == 4:
                        return [bbox_data]
                return []
            else:
                return []
                
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
            print(f"Error parsing 2D bounding boxes: {e}, the text was: {text}")
            print(f"Stack trace:\n{traceback.format_exc()}")
            return []

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


class VideoBoundingBoxPrecomputer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the bounding box precomputer."""
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize the object detector
        self.object_detector = StandaloneObjectDetector(device)
        
        # Image transformations
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        
        print("Video bounding box precomputer initialized successfully!")
    
    def process_video(self, video_path, user_prompt, output_path, max_frames=None):
        """
        Process video and precompute bounding boxes for each frame.
        
        Args:
            video_path: Path to input video file
            user_prompt: User prompt to guide object detection
            output_path: Path to save the bounding box results (JSON format)
            max_frames: Maximum number of frames to process (None for all)
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Store bounding box results
        bounding_box_results = {
            "video_path": str(video_path),
            "user_prompt": user_prompt,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "processed_frames": 0,
            "frames": []
        }
        
        # Process frames
        frame_count = 0
        processed_count = 0
        
        with tqdm(total=max_frames or total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if max_frames and processed_count >= max_frames:
                    break
                
                try:
                    # Process frame
                    frame_result = self.process_single_frame(frame, user_prompt, frame_count)
                    
                    # Store results
                    bounding_box_results["frames"].append({
                        "frame_index": frame_count,
                        "bounding_boxes": frame_result["bounding_boxes"],
                        "object_types": frame_result["object_types"]
                    })
                    
                    processed_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"Objects detected": len(frame_result["bounding_boxes"])})
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    print(f"Stack trace:\n{traceback.format_exc()}")
                    # Still record the frame but with empty results
                    bounding_box_results["frames"].append({
                        "frame_index": frame_count,
                        "bounding_boxes": [],
                        "object_types": []
                    })
                    continue
                
                frame_count += 1
        
        # Cleanup
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Ignore errors when running in headless environments
            pass
        
        # Update metadata
        bounding_box_results["processed_frames"] = processed_count
        
        # Save results to JSON
        with open(output_path, 'w') as f:
            json.dump(bounding_box_results, f, indent=2)
        
        print(f"Processed {processed_count} frames")
        print(f"Bounding box results saved to: {output_path}")
        
        return bounding_box_results
    
    def process_single_frame(self, frame, user_prompt, frame_index=0):
        """
        Process a single frame and extract bounding boxes.
        
        Args:
            frame: numpy array (H, W, C) in BGR format
            user_prompt: User prompt to guide object detection
            frame_index: Frame index (for logging purposes)
            
        Returns:
            dict with bounding boxes and object types
        """
        with torch.no_grad():
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Apply transformations
            tensor_image = self.transform(pil_image)
            
            # Add batch dimension and move to device
            tensor_image = tensor_image.unsqueeze(0).to(self.device)
            
            # Detect objects and get bounding boxes
            bounding_boxes_batch, object_types_batch = self.object_detector.detect_objects_and_get_bounding_boxes(
                tensor_image, user_prompt
            )
            
            # Extract results (batch size is 1)
            if bounding_boxes_batch is not None and bounding_boxes_batch.numel() > 0:
                bounding_boxes = bounding_boxes_batch[0].cpu().numpy()
                object_types = object_types_batch[0] if object_types_batch else []
            else:
                bounding_boxes = np.array([])
                object_types = []
            
            # Handle case where no bounding boxes were detected
            if bounding_boxes.size == 0:
                return {
                    "bounding_boxes": [],
                    "object_types": []
                }
            
            # Filter out empty bounding boxes (all zeros)
            valid_indices = np.any(bounding_boxes != 0, axis=1)
            valid_bounding_boxes = bounding_boxes[valid_indices].tolist()
            valid_object_types = [obj_type for i, obj_type in enumerate(object_types) if valid_indices[i]] if len(object_types) > 0 else []
            
            return {
                "bounding_boxes": valid_bounding_boxes,
                "object_types": valid_object_types
            }

    def process_single_frame_2d(self, frame, user_prompt, frame_index=0):
        """
        Process a single frame and extract 2D bounding boxes.
        
        Args:
            frame: numpy array (H, W, C) in BGR format
            user_prompt: User prompt to guide object detection
            frame_index: Frame index (for logging purposes)
            
        Returns:
            dict with 2D bounding boxes
        """
        with torch.no_grad():
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Apply transformations
            tensor_image = self.transform(pil_image)
            
            # Add batch dimension and move to device
            tensor_image = tensor_image.unsqueeze(0).to(self.device)
            
            # Detect objects and get 2D bounding boxes
            bounding_boxes_list = self.object_detector.detect_objects_and_get_2d_bounding_boxes(
                tensor_image, user_prompt
            )
            
            # Extract results (batch size is 1)
            bounding_boxes = bounding_boxes_list[0] if bounding_boxes_list else []
            
            return {
                "bounding_boxes_2d": bounding_boxes
            }

    def process_video_2d(self, video_path, user_prompt, output_path, max_frames=None):
        """
        Process video and precompute 2D bounding boxes for each frame.
        
        Args:
            video_path: Path to input video file
            user_prompt: User prompt to guide object detection
            output_path: Path to save the bounding box results (JSON format)
            max_frames: Maximum number of frames to process (None for all)
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Store bounding box results
        bounding_box_results = {
            "video_path": str(video_path),
            "user_prompt": user_prompt,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "processed_frames": 0,
            "frames": []
        }
        
        # Process frames
        frame_count = 0
        processed_count = 0
        
        with tqdm(total=max_frames or total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if max_frames and processed_count >= max_frames:
                    break
                
                try:
                    # Process frame with 2D detection
                    frame_result = self.process_single_frame_2d(frame, user_prompt, frame_count)
                    
                    # Store results
                    bounding_box_results["frames"].append({
                        "frame_index": frame_count,
                        "bounding_boxes_2d": frame_result["bounding_boxes_2d"]
                    })
                    
                    processed_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"Objects detected": len(frame_result["bounding_boxes_2d"])})
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    print(f"Stack trace:\n{traceback.format_exc()}")
                    # Still record the frame but with empty results
                    bounding_box_results["frames"].append({
                        "frame_index": frame_count,
                        "bounding_boxes_2d": []
                    })
                    continue
                
                frame_count += 1
        
        # Cleanup
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Ignore errors when running in headless environments
            pass
        
        # Update metadata
        bounding_box_results["processed_frames"] = processed_count
        
        # Save results to JSON
        with open(output_path, 'w') as f:
            json.dump(bounding_box_results, f, indent=2)
        
        print(f"Processed {processed_count} frames")
        print(f"2D bounding box results saved to: {output_path}")
        
        return bounding_box_results


def main():
    parser = argparse.ArgumentParser(description="Precompute bounding boxes for video using Qwen3-VL")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--user_prompt", type=str, required=True, help="User prompt to guide object detection")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save bounding box results (JSON)")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--mode", type=str, choices=["3d", "2d"], default="3d", help="Detection mode: 3d or 2d")
    
    args = parser.parse_args()
    
    # Initialize precomputer
    precomputer = VideoBoundingBoxPrecomputer()
    
    # Process video based on mode
    if args.mode == "2d":
        # For 2D detection
        results = precomputer.process_video_2d(
            video_path=args.video_path,
            user_prompt=args.user_prompt,
            output_path=args.output_path,
            max_frames=args.max_frames
        )
    else:
        # Default 3D detection
        results = precomputer.process_video(
            video_path=args.video_path,
            user_prompt=args.user_prompt,
            output_path=args.output_path,
            max_frames=args.max_frames
        )
    
    print("Bounding box precomputation completed successfully!")


if __name__ == "__main__":
    main()
