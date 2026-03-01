#!/usr/bin/env python3
"""
Script to add 2D bounding box information to an existing LeRobot dataset using the LeRobotDataset class.
This script loads a dataset using LeRobotDataset, processes each frame to detect objects using Qwen3-VL,
and adds the bounding box information to the dataset.
"""

import json
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import traceback
import sys
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

# Conditional import of transformers to handle cases where it might not be available
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not available. Object detection will not work.")


class LeRobot2DBoundingBoxAdder:
    """Class to add 2D bounding box information to LeRobot datasets using LeRobotDataset class."""
    
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
        
        # System prompt for 2D bounding box detection
        self.system_prompt_2d = "You are a precise 2D object detector. For each object, provide its 2D bounding box in JSON format with 'bbox_2d' field containing [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. All coordinates should be normalized to [0, 1000]. Also provide a 'label' field indicating the object category."
        
        # Default user prompt for 2D detection
        self.user_prompt_2d = 'locate every instance that belongs to the following categories: "plate/dish, scallop, wine bottle, tv, bowl, spoon, air conditioner, coconut drink, cup, chopsticks, person". Report bbox coordinates in JSON format.'
        
        # Image transformations (only convert to tensor, no resizing)
        self.transform = T.ToTensor()
        
        print("LeRobot 2D bounding box adder initialized successfully!")

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
            print(f"Error parsing 2D bounding boxes: {e}")
            return []

    def process_lerobot_dataset(self, repo_id, user_prompt=None, output_path=None, max_samples=None, revision="main", push_to_hub=False, hf_token=None):
        """
        Load and process a LeRobot dataset, adding 2D bounding box information.
        
        Args:
            repo_id: Repository ID for the LeRobot dataset
            user_prompt: Optional custom prompt for object detection
            output_path: Output path for the modified dataset
            max_samples: Maximum number of samples to process (for testing)
            revision: Revision of the dataset to load
            push_to_hub: Whether to push updates to HuggingFace Hub
            hf_token: HuggingFace token for authentication
        """
        print(f"Loading LeRobot dataset {repo_id}...")
        
        # Load dataset using LeRobotDataset
        dataset = LeRobotDataset(repo_id, force_cache_sync=True, revision=revision)
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Limit samples if max_samples is specified
        if max_samples:
            # For LeRobotDataset, we need to slice differently
            indices = list(range(min(max_samples, len(dataset))))
            # We'll process only up to max_samples in the loop below
            print(f"Limited to {len(indices)} samples for processing")
        else:
            indices = list(range(len(dataset)))
        
        # Instead of trying to access dataset.info directly, let's work with the dataset structure
        # We'll identify camera keys by looking at a sample
        sample = dataset[0]
        camera_keys = [key for key in sample.keys() if key.startswith("observation.images.")]
        print(f"Found camera keys: {camera_keys}")
        
        # Process samples to add bounding box information
        bounding_boxes_data = []
        
        # Initialize tqdm with custom format
        pbar = tqdm(indices, desc="Processing samples", postfix={"boxes": 0})
        
        # Process each sample (limited by max_samples if specified)
        for idx in pbar:
            try:
                # Get sample
                sample = dataset[idx]
                
                # Check if current observation.box values are all zeros
                needs_detection = False
                if 'observation.box' in sample:
                    current_boxes = sample['observation.box']
                    # Handle different possible formats
                    if isinstance(current_boxes, torch.Tensor):
                        # Handle PyTorch tensor format
                        current_boxes_array = current_boxes.detach().cpu().numpy()
                        # Handle different possible shapes
                        if current_boxes_array.size == 0:
                            # Empty tensor, treat as all zeros
                            needs_detection = True
                        elif current_boxes_array.ndim >= 2:
                            # Multi-dimensional tensor, check if all values are zeros
                            if np.all(current_boxes_array == 0):
                                needs_detection = True
                            else:
                                print(f"Sample {idx} already has non-zero boxes, skipping detection...")
                        else:
                            # Unexpected format, perform detection
                            needs_detection = True
                    else:
                        needs_detection = True
                else:
                    needs_detection = True
                
                # Extract bounding boxes for each camera/image in the sample
                sample_bounding_boxes = {}
                sample_boxes_count = 0
                
                if needs_detection:
                    # Look for image observations in the sample
                    for key, value in sample.items():
                        if key.startswith("observation.images."):
                            # Extract camera name
                            camera_name = key.replace("observation.images.", "")
                            
                            try:
                                # Convert image to tensor
                                if isinstance(value, torch.Tensor):
                                    # Convert tensor to PIL Image
                                    # Assuming tensor is in CHW format
                                    if value.dim() == 3:
                                        # Convert from tensor to PIL Image
                                        # Convert from [-1, 1] to [0, 1] if needed
                                        img_tensor = value.detach().cpu()
                                        if img_tensor.min() < 0:
                                            img_tensor = (img_tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
                                        
                                        # Convert to PIL Image (CHW to HWC)
                                        img_array = img_tensor.permute(1, 2, 0).numpy()
                                        img_array = (img_array * 255).astype(np.uint8)
                                        pil_image = Image.fromarray(img_array)
                                        
                                        # Convert to tensor without resizing
                                        tensor_image = T.ToTensor()(pil_image)
                                        tensor_image = tensor_image.unsqueeze(0).to(self.device)
                                        
                                        # Detect objects and get 2D bounding boxes
                                        bounding_boxes_list = self.detect_objects_and_get_2d_bounding_boxes(tensor_image, user_prompt)
                                        bounding_boxes = bounding_boxes_list[0] if bounding_boxes_list else []
                                        
                                        # Store bounding boxes for this camera
                                        sample_bounding_boxes[camera_name] = bounding_boxes
                                        sample_boxes_count += len(bounding_boxes)
                                    else:
                                        print(f"Unexpected tensor dimensions for {key}: {value.dim()}")
                                        sample_bounding_boxes[camera_name] = []
                                else:
                                    print(f"Unexpected image format for {key}: {type(value)}")
                                    sample_bounding_boxes[camera_name] = []
                            except Exception as e:
                                print(f"Error processing image {key} in sample {idx}: {e}")
                                sample_bounding_boxes[camera_name] = []
                else:
                    # No detection needed, store empty bounding boxes
                    sample_bounding_boxes = {}
                
                bounding_boxes_data.append(sample_bounding_boxes)
                pbar.set_postfix({"boxes": sample_boxes_count})
                
                # Update parquet files every 10 samples
                if (idx + 1) % 10 == 0 or (idx + 1) == len(indices):
                    print(f"Updating dataset with {idx + 1} samples processed...")
                    self._update_dataset_with_bounding_boxes_partial(repo_id, bounding_boxes_data, indices[:idx+1], revision)
                    
                    # Push to HuggingFace Hub if requested
                    if push_to_hub:
                        self._push_to_huggingface(repo_id, hf_token)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                bounding_boxes_data.append({})
        
        # Add bounding_boxes data to dataset
        # This is a simplified approach - in practice, you might want to save this to a separate file
        # or modify the dataset structure to include this information
        print(f"Processed {len(bounding_boxes_data)} samples with bounding box data")
        
        # Save bounding box data to a JSON file
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save bounding box data
            bounding_boxes_file = output_path / "bounding_boxes.json"
            with open(bounding_boxes_file, 'w') as f:
                json.dump(bounding_boxes_data, f, indent=2)
            print(f"Bounding box data saved to {bounding_boxes_file}")
            
            # Save a mapping of indices to bounding boxes for easy lookup
            index_mapping = {}
            for i, bbox_data in enumerate(bounding_boxes_data):
                index_mapping[i] = bbox_data
            
            mapping_file = output_path / "index_to_bounding_boxes.json"
            with open(mapping_file, 'w') as f:
                json.dump(index_mapping, f, indent=2)
            print(f"Index mapping saved to {mapping_file}")
            
            # Update the dataset with bounding box information
            self._update_dataset_with_bounding_boxes(repo_id, bounding_boxes_data, revision)
            
            # Push to HuggingFace Hub if requested
            if push_to_hub:
                self._push_to_huggingface(repo_id, hf_token)
        
        return bounding_boxes_data

    def _update_dataset_with_bounding_boxes_partial(self, repo_id, bounding_boxes_data, processed_indices, revision="main"):
        """
        Update the LeRobot dataset with bounding box information for processed samples only.
        
        Args:
            repo_id: Repository ID for the LeRobot dataset
            bounding_boxes_data: List of bounding box data for each sample
            processed_indices: List of indices that have been processed
            revision: Revision of the dataset to update
        """
        print(f"Partially updating dataset {repo_id} with bounding box information for {len(processed_indices)} samples...")
        
        # Construct dataset directory path based on user-provided information
        try:
            # Use the standard HuggingFace cache directory for lerobot datasets
            dataset_cache_root = Path(f"/root/.cache/huggingface/lerobot/{repo_id}")
            
            dataset_dir = dataset_cache_root
            print(f"Dataset directory: {dataset_dir}")
            
            # Check if the directory exists
            if not dataset_dir.exists():
                print(f"Dataset directory does not exist: {dataset_dir}")
                return
        except Exception as e:
            print(f"Error constructing dataset directory path: {e}")
            return
        
        # Update info.json to include the new observation.box feature (only once)
        info_json_path = dataset_dir / "meta" / "info.json"
        if info_json_path.exists():
            try:
                with open(info_json_path, 'r') as f:
                    info_data = json.load(f)
                
                # Add observation.box feature if not already present
                if "observation.box" not in info_data["features"]:
                    info_data["features"]["observation.box"] = {
                        "shape": [3, 4, 2],  # 3 boxes, 4 points each, 2 coordinates each
                        "dtype": "float32"
                    }
                    
                    with open(info_json_path, 'w') as f:
                        json.dump(info_data, f, indent=2)
                    print("Updated info.json with observation.box feature")
            except Exception as e:
                print(f"Error updating info.json: {e}")
        else:
            print(f"info.json not found at {info_json_path}")
        
        # Update parquet files with bounding box data
        data_dir = dataset_dir / "data"
        if data_dir.exists():
            for chunk_dir in data_dir.glob("chunk-*"):
                for parquet_file in chunk_dir.glob("*.parquet"):
                    try:
                        # Read existing parquet file
                        df = pd.read_parquet(parquet_file)
                        print(f"Loaded parquet file with {len(df)} rows")
                        
                        # Update bounding box column for processed indices only
                        # Convert bounding_boxes_data to a format suitable for the dataframe
                        for i, idx in enumerate(processed_indices):
                            if idx < len(df):
                                # Check if current observation.box values are all zeros
                                current_boxes = df.loc[idx, 'observation.box']
                                # Convert to numpy array for easier comparison
                                if isinstance(current_boxes, list):
                                    current_boxes_array = np.array(current_boxes)
                                    # Check if all values are zeros
                                    if np.all(current_boxes_array == 0):
                                        print(f"Sample {idx} has all-zero boxes, performing detection...")
                                        
                                        # Combine bounding boxes from all cameras for this sample
                                        all_boxes = []
                                        if i < len(bounding_boxes_data):
                                            for camera_boxes in bounding_boxes_data[i].values():
                                                for box_data in camera_boxes:
                                                    # Convert bbox_2d to a flat list
                                                    if 'bbox_2d' in box_data:
                                                        all_boxes.extend(box_data['bbox_2d'])
                                        
                                        # Reshape to list of 4-element lists (representing boxes with 4 points each)
                                        reshaped_boxes = []
                                        if all_boxes:
                                            # Reshape to list of 4-element lists
                                            reshaped_boxes = [all_boxes[i:i+4] for i in range(0, len(all_boxes), 4)]
                                        
                                        # Pad with zero boxes to ensure consistent length (exactly 3 boxes)
                                        while len(reshaped_boxes) < 3:
                                            reshaped_boxes.append([0.0, 0.0, 0.0, 0.0])
                                        
                                        # Truncate to exactly 3 boxes if more were detected
                                        if len(reshaped_boxes) > 3:
                                            reshaped_boxes = reshaped_boxes[:3]
                                            
                                        # Convert to the required format: [3, 4, 2]
                                        # Each box should have 4 points, each point with 2 coordinates
                                        final_boxes = []
                                        for box in reshaped_boxes:
                                            # Each box should have 4 points with 2 coordinates each
                                            # For simplicity, we'll convert the 4 coordinates [x1, y1, x2, y2] 
                                            # to 4 points [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
                                            if len(box) == 4:
                                                x1, y1, x2, y2 = box
                                                box_points = [
                                                    [x1, y1],  # Top-left
                                                    [x2, y1],  # Top-right
                                                    [x2, y2],  # Bottom-right
                                                    [x1, y2]   # Bottom-left
                                                ]
                                                final_boxes.append(box_points)
                                            else:
                                                # If box doesn't have 4 coordinates, pad with zeros
                                                box_points = [[0.0, 0.0] for _ in range(4)]
                                                final_boxes.append(box_points)
                                        
                                        # Pad with zero boxes if needed to reach exactly 3 boxes
                                        while len(final_boxes) < 3:
                                            zero_box = [[0.0, 0.0] for _ in range(4)]
                                            final_boxes.append(zero_box)
                                        
                                        # Use loc instead of at to avoid the ndarray error
                                        df.loc[idx, 'observation.box'] = final_boxes
                                    else:
                                        print(f"Sample {idx} already has non-zero boxes, skipping detection...")
                                else:
                                    print(f"Sample {idx} has unexpected box format, skipping...")
                        
                        # Save updated dataframe back to parquet
                        df.to_parquet(parquet_file, index=False)
                        print(f"Updated parquet file: {parquet_file}")
                    except Exception as e:
                        print(f"Error updating parquet file {parquet_file}: {e}")
        else:
            print(f"Data directory not found at {data_dir}")
        
        print("Partial dataset update completed!")

    def _push_to_huggingface(self, repo_id, hf_token=None):
        """
        Push the updated dataset to HuggingFace Hub.
        
        Args:
            repo_id: Repository ID for the LeRobot dataset
            hf_token: HuggingFace token for authentication
        """
        print(f"Pushing dataset {repo_id} to HuggingFace Hub...")
        
        try:
            # Construct dataset directory path
            dataset_cache_root = Path(f"/root/.cache/huggingface/lerobot/{repo_id}")
            # Convert repo_id to a safe directory name (replace / with ___)
            
            dataset_dir = dataset_cache_root
            print(f"Dataset directory: {dataset_dir}")
            
            # Check if the directory exists
            if not dataset_dir.exists():
                print(f"Dataset directory does not exist: {dataset_dir}")
                # Try alternative paths
                alt_paths = [
                    Path.home() / ".cache/huggingface/lerobot",
                    Path("/tmp/lerobot")
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        dataset_dir = alt_path
                        print(f"Found dataset directory at alternative location: {dataset_dir}")
                        break
                else:
                    print("Could not find dataset directory at any expected location")
                    return
            
            # Initialize HuggingFace API
            api = HfApi(token=hf_token)
            
            # Upload all files in the dataset directory
            api.upload_folder(
                folder_path=str(dataset_dir),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Update dataset with 2D bounding box information"
            )
            
            print(f"Dataset successfully pushed to HuggingFace Hub: {repo_id}")
        except Exception as e:
            print(f"Error pushing to HuggingFace Hub: {e}")

    def _update_dataset_with_bounding_boxes(self, repo_id, bounding_boxes_data, revision="main"):
        """
        Update the LeRobot dataset with bounding box information.
        
        Args:
            repo_id: Repository ID for the LeRobot dataset
            bounding_boxes_data: List of bounding box data for each sample
            revision: Revision of the dataset to update
        """
        print(f"Updating dataset {repo_id} with bounding box information...")
        
        # Construct dataset directory path
        try:
            # Use the standard HuggingFace cache directory for lerobot datasets
            dataset_cache_root = Path("/root/.cache/huggingface/lerobot")
            # Convert repo_id to a safe directory name (replace / with ___)
            safe_repo_id = repo_id.replace("/", "___")
            dataset_dir = dataset_cache_root / safe_repo_id
            print(f"Dataset directory: {dataset_dir}")
            
            # Check if the directory exists
            if not dataset_dir.exists():
                print(f"Dataset directory does not exist: {dataset_dir}")
                # Try alternative paths
                alt_paths = [
                    Path.home() / ".cache/huggingface/lerobot" / safe_repo_id,
                    Path("/tmp/lerobot") / safe_repo_id
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        dataset_dir = alt_path
                        print(f"Found dataset directory at alternative location: {dataset_dir}")
                        break
                else:
                    print("Could not find dataset directory at any expected location")
                    return
        except Exception as e:
            print(f"Error constructing dataset directory path: {e}")
            return
        
        # Update info.json to include the new observation.box feature
        info_json_path = dataset_dir / "meta" / "info.json"
        if info_json_path.exists():
            try:
                with open(info_json_path, 'r') as f:
                    info_data = json.load(f)
                
                # Add observation.box feature with correct shape
                info_data["features"]["observation.box"] = {
                    "shape": [3, 4, 2],  # 3 boxes, 4 points each, 2 coordinates each
                    "dtype": "float32"
                }
                
                with open(info_json_path, 'w') as f:
                    json.dump(info_data, f, indent=2)
                print("Updated info.json with observation.box feature")
            except Exception as e:
                print(f"Error updating info.json: {e}")
        else:
            print(f"info.json not found at {info_json_path}")
        
        # Update parquet files with bounding box data
        data_dir = dataset_dir / "data"
        if data_dir.exists():
            for chunk_dir in data_dir.glob("chunk-*"):
                for parquet_file in chunk_dir.glob("*.parquet"):
                    try:
                        # Read existing parquet file
                        df = pd.read_parquet(parquet_file)
                        print(f"Loaded parquet file with {len(df)} rows")
                        
                        # Update bounding box column for all samples
                        for idx in range(len(df)):
                            if idx < len(bounding_boxes_data):
                                # Check if current observation.box values are all zeros
                                current_boxes = df.loc[idx, 'observation.box']
                                # Convert to numpy array for easier comparison
                                if isinstance(current_boxes, list):
                                    current_boxes_array = np.array(current_boxes)
                                    # Check if all values are zeros
                                    if np.all(current_boxes_array == 0):
                                        print(f"Sample {idx} has all-zero boxes, performing detection...")
                                        
                                        # Combine bounding boxes from all cameras for this sample
                                        all_boxes = []
                                        for camera_boxes in bounding_boxes_data[idx].values():
                                            for box_data in camera_boxes:
                                                # Convert bbox_2d to a flat list
                                                if 'bbox_2d' in box_data:
                                                    all_boxes.extend(box_data['bbox_2d'])
                                        
                                        # Reshape to list of 4-element lists (representing boxes with 4 points each)
                                        reshaped_boxes = []
                                        if all_boxes:
                                            # Reshape to list of 4-element lists
                                            reshaped_boxes = [all_boxes[i:i+4] for i in range(0, len(all_boxes), 4)]
                                        
                                        # Pad with zero boxes to ensure consistent length (exactly 3 boxes)
                                        while len(reshaped_boxes) < 3:
                                            reshaped_boxes.append([0.0, 0.0, 0.0, 0.0])
                                        
                                        # Truncate to exactly 3 boxes if more were detected
                                        if len(reshaped_boxes) > 3:
                                            reshaped_boxes = reshaped_boxes[:3]
                                            
                                        # Convert to the required format: [3, 4, 2]
                                        # Each box should have 4 points, each point with 2 coordinates
                                        final_boxes = []
                                        for box in reshaped_boxes:
                                            # Each box should have 4 points with 2 coordinates each
                                            # For simplicity, we'll convert the 4 coordinates [x1, y1, x2, y2] 
                                            # to 4 points [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
                                            if len(box) == 4:
                                                x1, y1, x2, y2 = box
                                                box_points = [
                                                    [x1, y1],  # Top-left
                                                    [x2, y1],  # Top-right
                                                    [x2, y2],  # Bottom-right
                                                    [x1, y2]   # Bottom-left
                                                ]
                                                final_boxes.append(box_points)
                                            else:
                                                # If box doesn't have 4 coordinates, pad with zeros
                                                box_points = [[0.0, 0.0] for _ in range(4)]
                                                final_boxes.append(box_points)
                                        
                                        # Pad with zero boxes if needed to reach exactly 3 boxes
                                        while len(final_boxes) < 3:
                                            zero_box = [[0.0, 0.0] for _ in range(4)]
                                            final_boxes.append(zero_box)
                                        
                                        # Use loc instead of at to avoid the ndarray error
                                        df.loc[idx, 'observation.box'] = final_boxes
                                    else:
                                        print(f"Sample {idx} already has non-zero boxes, skipping detection...")
                                else:
                                    print(f"Sample {idx} has unexpected box format, skipping...")
                            else:
                                # Handle case where we don't have bounding box data for this sample
                                # Just leave the existing boxes as they are
                                pass
                        
                        # Save updated dataframe back to parquet
                        df.to_parquet(parquet_file, index=False)
                        print(f"Updated parquet file: {parquet_file}")
                    except Exception as e:
                        print(f"Error updating parquet file {parquet_file}: {e}")
        else:
            print(f"Data directory not found at {data_dir}")
        
        print("Dataset update completed!")


def main():
    parser = argparse.ArgumentParser(description="Add 2D bounding box information to LeRobot dataset")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID for the LeRobot dataset (e.g., ISDept/piper_arm)")
    parser.add_argument("--user_prompt", type=str, 
                       default='locate every instance that belongs to the following categories: "plate/dish, scallop, wine bottle, tv, bowl, spoon, air conditioner, coconut drink, cup, chopsticks, person". Report bbox coordinates in JSON format.',
                       help="User prompt to guide object detection")
    parser.add_argument("--output_path", type=str, help="Output path for the bounding box data")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process (for testing)")
    parser.add_argument("--revision", type=str, default="main", help="Revision of the dataset to load")
    parser.add_argument("--push_to_hub", action="store_true", help="Push updates to HuggingFace Hub")
    parser.add_argument("--hf_token", type=str, help="HuggingFace token for authentication")
    
    args = parser.parse_args()
    
    # Initialize the adder
    adder = LeRobot2DBoundingBoxAdder()
    
    # Process the dataset
    try:
        bounding_boxes_data = adder.process_lerobot_dataset(
            repo_id=args.repo_id,
            user_prompt=args.user_prompt,
            output_path=args.output_path,
            max_samples=args.max_samples,
            revision=args.revision,
            push_to_hub=args.push_to_hub,
            hf_token=args.hf_token
        )
        print("2D bounding box information added to dataset successfully!")
        print(f"Processed {len(bounding_boxes_data)} samples with bounding box data")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()