#!/usr/bin/env python3
"""
Script to add 2D bounding box information to an existing LeRobot dataset using the LeRobotDataset class.
This script loads a dataset using LeRobotDataset, processes each frame to detect objects using YOLOWorld
(zero-shot open-vocabulary detection), and adds the bounding box information to the dataset.

Install dependency: pip install ultralytics
"""

import json
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import traceback
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi


class LeRobot2DBoundingBoxAdder:
    """Add 2D bounding box information to LeRobot datasets using YOLOWorld zero-shot detection.

    YOLOWorld uses CLIP-based text–image matching so you can detect arbitrary object
    classes by name without any fine-tuning.  Typical throughput: ~25 ms/image on GPU.

    Args:
        detection_classes: List of class names to detect, e.g. ["cube", "container"].
        detection_conf:    Confidence threshold in [0, 1].  Lower = more recalls.
        device:            "cuda" or "cpu".
    """

    def __init__(
        self,
        detection_classes: list[str] | None = None,
        detection_conf: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.detection_classes = detection_classes or ["cube", "container"]
        self.detection_conf = detection_conf
        print(f"Using device: {self.device}")
        print(f"Detection classes: {self.detection_classes}  conf≥{self.detection_conf}")

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics is required: pip install ultralytics")

        print("Loading YOLOWorld model (yolov8x-worldv2.pt)…")
        self.model = YOLO("yolov8x-worldv2.pt")
        self.model.set_classes(self.detection_classes)
        print("YOLOWorld model ready.")

        # Label → category_id mapping (matches the training pipeline)
        self.label_to_id: dict[str, int] = {
            label: idx for idx, label in enumerate(self.detection_classes)
        }

    # ------------------------------------------------------------------
    # Core detection API (returns same dict format as the old Qwen code)
    # ------------------------------------------------------------------

    def detect_objects_and_get_2d_bounding_boxes(
        self, image_tensor: torch.Tensor, user_prompt: str | None = None
    ) -> list[list[dict]]:
        """Detect objects in a batch of images using YOLOWorld.

        Args:
            image_tensor: (B, C, H, W) float tensor in [0, 1] or uint8 [0, 255].
            user_prompt:  Ignored (kept for API compatibility).

        Returns:
            List of length B.  Each element is a list of dicts::

                {
                    "bbox_2d":     [x1, y1, x2, y2],   # absolute pixel coords
                    "label":       str,
                    "confidence":  float,
                    "category_id": int,
                }
        """
        B, C, H, W = image_tensor.shape

        # Convert to list of HWC uint8 numpy arrays
        img_cpu = image_tensor.detach().cpu()
        if img_cpu.max() <= 1.0 + 1e-3:
            img_cpu = (img_cpu * 255).clamp(0, 255)
        img_cpu = img_cpu.byte()

        imgs_np = [img_cpu[i].permute(1, 2, 0).numpy() for i in range(B)]

        try:
            results = self.model(imgs_np, conf=self.detection_conf, verbose=False)
        except Exception as e:
            print(f"YOLOWorld inference error: {e}")
            return [[] for _ in range(B)]

        bounding_boxes_list: list[list[dict]] = []
        for result in results:
            dets: list[dict] = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu()          # absolute pixel coords
                confs = boxes.conf.cpu()
                cls_ids = boxes.cls.cpu().long()
                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j].tolist()
                    conf = float(confs[j].item())
                    cls_id = int(cls_ids[j].item())
                    label = (
                        self.detection_classes[cls_id]
                        if cls_id < len(self.detection_classes)
                        else "unknown"
                    )
                    dets.append({
                        "bbox_2d":     [x1, y1, x2, y2],
                        "label":       label,
                        "confidence":  conf,
                        "category_id": self.label_to_id.get(label, -1),
                    })
            bounding_boxes_list.append(dets)

        return bounding_boxes_list

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
                
                # Check if current observation.box values are all zeros or null
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
                        elif np.isnan(current_boxes_array).any():
                            # Tensor contains NaN values, trigger detection
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
                    elif current_boxes is None:
                        # Null value, trigger detection
                        needs_detection = True
                    else:
                        # Other formats, check for null-like values
                        try:
                            # Convert to numpy array for consistent checking
                            current_boxes_array = np.array(current_boxes)
                            if current_boxes_array.size == 0 or np.isnan(current_boxes_array).any():
                                # Empty array or contains NaN values, trigger detection
                                needs_detection = True
                            elif np.all(current_boxes_array == 0):
                                # All zeros, trigger detection
                                needs_detection = True
                            else:
                                print(f"Sample {idx} already has non-zero boxes, skipping detection...")
                        except:
                            # If conversion fails, trigger detection
                            needs_detection = True
                else:
                    needs_detection = True
                
                # Extract bounding boxes for each camera/image in the sample
                sample_bounding_boxes = {}
                sample_boxes_count = 0
                
                # Add episode_index and frame_index to the sample_bounding_boxes
                if 'episode_index' in sample:
                    sample_bounding_boxes['episode_index'] = sample['episode_index'].item() if isinstance(sample['episode_index'], torch.Tensor) else sample['episode_index']
                if 'frame_index' in sample:
                    sample_bounding_boxes['frame_index'] = sample['frame_index'].item() if isinstance(sample['frame_index'], torch.Tensor) else sample['frame_index']
                
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
                                        
                                        # Convert detected bounding boxes to required format immediately
                                        if bounding_boxes:
                                            # Initialize with default empty boxes for this camera
                                            converted_boxes = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(2)]
                                            
                                            # Process up to 2 boxes per camera
                                            for box_idx, box_data in enumerate(bounding_boxes[:2]):
                                                if 'bbox_2d' in box_data and len(box_data['bbox_2d']) == 4:
                                                    # Get category_id and confidence, with defaults if not present
                                                    category_id = box_data.get('category_id', 0.0)  # Default to 0 (cube)
                                                    confidence = box_data.get('confidence', 1.0)    # Default to 1.0
                                                    
                                                    # Store the box coordinates with category_id and confidence
                                                    converted_boxes[box_idx] = box_data['bbox_2d'] + [float(category_id), float(confidence)]
                                            
                                            # Store converted bounding boxes for this camera
                                            sample_bounding_boxes[camera_name] = converted_boxes
                                            sample_boxes_count += len(bounding_boxes)
                                            
                                            # Also store additional information (category_id, confidence)
                                            # Create extended data structure for this camera
                                            extended_camera_data = []
                                            for box_data in bounding_boxes[:2]:
                                                if 'bbox_2d' in box_data and len(box_data['bbox_2d']) == 4:
                                                                                                        
                                                    # Create extended data with all required information
                                                    extended_data = {
                                                        'bbox_2d': box_data['bbox_2d'],
                                                        'category_id': box_data.get('category_id', -1),
                                                        'confidence': box_data.get('confidence', 1.0),
                                                        'label': box_data.get('label', 'unknown')
                                                    }
                                                    extended_camera_data.append(extended_data)
                                            
                                            # Store extended data for this camera
                                            if extended_camera_data:
                                                sample_bounding_boxes[f"{camera_name}_extended"] = extended_camera_data
                                        else:
                                            # Store empty boxes for this camera
                                            sample_bounding_boxes[camera_name] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(2)]
                                            sample_boxes_count += 0
                                    else:
                                        print(f"Unexpected tensor dimensions for {key}: {value.dim()}")
                                        sample_bounding_boxes[camera_name] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(2)]
                                else:
                                    print(f"Unexpected image format for {key}: {type(value)}")
                                    sample_bounding_boxes[camera_name] = []
                            except Exception as e:
                                print(f"Error processing image {key} in sample {idx}: {e}")
                                sample_bounding_boxes[camera_name] = []
                else:
                    # No detection needed, store empty bounding boxes
                    sample_bounding_boxes = {}
                
                # Store the bounding boxes data directly in the list for this sample
                bounding_boxes_data.append(sample_bounding_boxes)
                
                pbar.set_postfix({"boxes": sample_boxes_count})
                
                # Update parquet files every 10 samples
                if (idx + 1) % 500 == 0 or (idx + 1) == len(indices):
                    self._update_dataset_with_bounding_boxes_partial(repo_id, bounding_boxes_data, indices[:idx+1], revision)
                    
                    # Clear bounding_boxes_data after updating parquet files
                    bounding_boxes_data.clear()
                    
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
            
            # Extract and save extended information separately
            extended_data = []
            for bbox_data in bounding_boxes_data:
                sample_extended = {}
                for key, value in bbox_data.items():
                    if key.endswith('_extended'):
                        # This is extended data, store it
                        camera_name = key.replace('_extended', '')
                        sample_extended[camera_name] = value
                if sample_extended:
                    # Add episode and frame info if available
                    if 'episode_index' in bbox_data:
                        sample_extended['episode_index'] = bbox_data['episode_index']
                    if 'frame_index' in bbox_data:
                        sample_extended['frame_index'] = bbox_data['frame_index']
                    extended_data.append(sample_extended)
            
            # Save extended information to a separate file
            if extended_data:
                extended_file = output_path / "bounding_boxes_extended.json"
                with open(extended_file, 'w') as f:
                    json.dump(extended_data, f, indent=2)
                print(f"Extended bounding box data saved to {extended_file}")
            
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
        print(f"Partially updating dataset {repo_id} with bounding box information for {len(bounding_boxes_data)} samples...")
        
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
        
        
        # Update parquet files with bounding box data
        data_dir = dataset_dir / "data"
        if data_dir.exists():
            for chunk_dir in data_dir.glob("chunk-*"):
                for parquet_file in chunk_dir.glob("*.parquet"):
                    try:
                        # Read existing parquet file
                        df = pd.read_parquet(parquet_file)
                        print(f"Loaded parquet file with {len(df)} rows")
                        
                        # Loop through bounding_boxes_data and find matching frames in the parquet file
                        for bbox_data in bounding_boxes_data:
                            # Extract episode_index and frame_index from bounding_boxes_data
                            if 'episode_index' in bbox_data and 'frame_index' in bbox_data:
                                episode_index = bbox_data['episode_index']
                                frame_index = bbox_data['frame_index']
                                
                                # Use where condition to locate the specific frame
                                mask = (df['episode_index'] == episode_index) & (df['frame_index'] == frame_index)
                                matching_rows = df[mask]
                                print(f"Mask length: {len(mask)}")
                                print(f"Matching rows count: {len(matching_rows)}")
                                
                                if len(matching_rows) > 0:
                                    # Remove episode_index and frame_index from bbox_data before updating
                                    bbox_data_copy = bbox_data.copy()
                                    bbox_data_copy.pop('episode_index', None)
                                    bbox_data_copy.pop('frame_index', None)
                                    
                                    # Convert camera-based bbox_data to required [6,4] format
                                    final_boxes = self._convert_detected_bounding_boxes_to_required_format(bbox_data_copy)
                                    
                                    # Update the observation.box column for the matching row(s)
                                    # Print debug information about the existing data structure
                                    existing_box = df.loc[mask, 'observation.box'].iloc[0]
                                    print(f"Existing box type: {type(existing_box)}")
                                    print(f"Existing box shape: {existing_box.shape if hasattr(existing_box, 'shape') else 'N/A'}")
                                    print(f"Existing box value: {existing_box}")
                                    
                                    # Print debug information about the new data structure
                                    new_boxes = np.array(final_boxes)
                                    print(f"New box type: {type(new_boxes)}")
                                    print(f"New box shape: {new_boxes.shape}")
                                    print(f"New box value: {new_boxes}")
                                    
                                    converted_boxes = [
                                        [float(x) for x in new_boxes[i]]
                                        for i in range(6)
                                    ]
                                    
                                                                        
                                    # Check if we're trying to assign to multiple rows
                                    if len(matching_rows) > 1:
                                        print(f"Warning: Found {len(matching_rows)} matching rows. Attempting to update all of them.")
                                        # We need to assign a list of values, one for each matching row
                                        # Create a list of converted_boxes, one for each matching row
                                        values_to_assign = [converted_boxes] * len(matching_rows)
                                        print(f"Values to assign length: {len(values_to_assign)}")
                                        df.loc[mask, 'observation.box'] = values_to_assign
                                    else:
                                        # Assign the converted boxes
                                        row_index = df.index[mask][0]
                                        df.at[row_index, 'observation.box'] = converted_boxes
                                    print(f"Updated frame with episode_index={episode_index}, frame_index={frame_index}")
                                else:
                                    print(f"No matching frame found for episode_index={episode_index}, frame_index={frame_index}")
                            else:
                                print(f"Missing episode_index or frame_index in bounding_boxes_data")
                        
                        # Save updated dataframe back to parquet
                        df.to_parquet(parquet_file, index=False)
                        print(f"Updated parquet file: {parquet_file}")
                    except Exception as e:
                        print(f"Error updating parquet file {parquet_file}: {e}")
                        print(traceback.format_exc())
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

    def _convert_detected_bounding_boxes_to_required_format(self, bbox_data):
        """
        Convert detected bounding box data to the required format: [6, 6]
        Where:
        - 6 elements total (2 per camera for 3 cameras)
        - Each element is an array of 6 floats [x1, y1, x2, y2, category_id, confidence]
        - Each camera must have exactly 2 boxes, pad with [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] if needed
        
        Args:
            bbox_data: Dictionary with camera names as keys and lists of bounding boxes as values
                       Each bounding box is a list of 6 floats [x1, y1, x2, y2, category_id, confidence]
                       OR
                       A dictionary with the format:
                       [{'episode_index': 0, 'frame_index': 14, 'gripper': [[37, 600, 162, 788, 0, 0.9], [376, 500, 539, 838, 1, 0.8]], 
                         'front': [[273, 405, 350, 525, 0, 0.95], [486, 405, 616, 575, 1, 0.85]], 
                         'right': [[524, 412, 642, 638, 0, 0.92], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}, ...]
            
        Returns:
            List of lists representing the bounding boxes in the required format
        """
        # Initialize with default empty boxes - 6 boxes with 6 elements each (total 36 elements)
        final_boxes = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(6)]
        
        # Handle the case where bbox_data is in the user-provided format
        # (a dictionary with episode_index, frame_index, and camera keys)
        if isinstance(bbox_data, dict) and ('episode_index' in bbox_data or 'frame_index' in bbox_data):
            # Remove episode_index and frame_index if they exist
            bbox_data_copy = bbox_data.copy()
            bbox_data_copy.pop('episode_index', None)
            bbox_data_copy.pop('frame_index', None)
            bbox_data = bbox_data_copy
        
        # Define the order of cameras
        camera_order = ['gripper', 'front', 'right']
        
        # Process bounding boxes for each camera
        for cam_idx, camera_name in enumerate(camera_order):
            # Ensure each camera has exactly 2 boxes
            camera_boxes = []
            if camera_name in bbox_data:
                camera_boxes = bbox_data[camera_name]
            
            # Process up to 2 boxes per camera, pad with zeros if needed
            for box_idx in range(2):  # Always process 2 boxes per camera
                if box_idx < len(camera_boxes):
                    box_coords = camera_boxes[box_idx]
                    if len(box_coords) == 6:
                        # Store the box coordinates directly
                        final_boxes[cam_idx * 2 + box_idx] = box_coords
                    elif len(box_coords) == 4:
                        # If we only have 4 coordinates, pad with default values for category_id and confidence
                        final_boxes[cam_idx * 2 + box_idx] = box_coords + [0.0, 1.0]  # Default category_id=0, confidence=1.0
                # If there's no box at this index, it remains [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (already initialized)
        
        print('The final_boxes is: ', final_boxes)
        return final_boxes

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
        except Exception as e:
            print(f"Error constructing dataset directory path: {e}")
            return
        
                
        # Update parquet files with bounding box data
        data_dir = dataset_dir / "data"
        if data_dir.exists():
            for chunk_dir in data_dir.glob("chunk-*"):
                for parquet_file in chunk_dir.glob("*.parquet"):
                    try:
                        # Read existing parquet file
                        df = pd.read_parquet(parquet_file)
                        print(f"Loaded parquet file with {len(df)} rows")
                        
                        # Loop through bounding_boxes_data and find matching frames in the parquet file
                        for bbox_data in bounding_boxes_data:
                            # Extract episode_index and frame_index from bounding_boxes_data
                            if 'episode_index' in bbox_data and 'frame_index' in bbox_data:
                                episode_index = bbox_data['episode_index']
                                frame_index = bbox_data['frame_index']
                                
                                # Use where condition to locate the specific frame
                                mask = (df['episode_index'] == episode_index) & (df['frame_index'] == frame_index)
                                matching_rows = df[mask]
                                print(f"Mask length: {len(mask)}")
                                print(f"Matching rows count: {len(matching_rows)}")
                                
                                if len(matching_rows) > 0:
                                    # Remove episode_index and frame_index from bbox_data before updating
                                    bbox_data_copy = bbox_data.copy()
                                    bbox_data_copy.pop('episode_index', None)
                                    bbox_data_copy.pop('frame_index', None)
                                    
                                    # Convert camera-based bbox_data to required [6,4] format
                                    final_boxes = self._convert_detected_bounding_boxes_to_required_format(bbox_data_copy)
                                    
                                    # Update the observation.box column for the matching row(s)
                                    # Print debug information about the existing data structure
                                    existing_box = df.loc[mask, 'observation.box'].iloc[0]
                                    print(f"Existing box type: {type(existing_box)}")
                                    print(f"Existing box shape: {existing_box.shape if hasattr(existing_box, 'shape') else 'N/A'}")
                                    print(f"Existing box value: {existing_box}")
                                    
                                    # Print debug information about the new data structure
                                    new_boxes = np.array(final_boxes)
                                    print(f"New box type: {type(new_boxes)}")
                                    print(f"New box shape: {new_boxes.shape}")
                                    print(f"New box value: {new_boxes}")
                                    
                                    converted_boxes = [
                                        [float(x) for x in new_boxes[i]]
                                        for i in range(6)
                                    ]
                                    
                                    
                                    # Check if we're trying to assign to multiple rows
                                    if len(matching_rows) > 1:
                                        print(f"Warning: Found {len(matching_rows)} matching rows. Attempting to update all of them.")
                                        # We need to assign a list of values, one for each matching row
                                        # Create a list of converted_boxes, one for each matching row
                                        values_to_assign = [converted_boxes] * len(matching_rows)
                                        print(f"Values to assign length: {len(values_to_assign)}")
                                        df.loc[mask, 'observation.box'] = values_to_assign
                                    else:
                                        # Assign the converted boxes
                                        row_index = df.index[mask][0]
                                        df.at[row_index, 'observation.box'] = converted_boxes
                                    print(f"Updated frame with episode_index={episode_index}, frame_index={frame_index}")
                                else:
                                    print(f"No matching frame found for episode_index={episode_index}, frame_index={frame_index}")
                            else:
                                print(f"Missing episode_index or frame_index in bounding_boxes_data")
                        
                        # Save updated dataframe back to parquet
                        df.to_parquet(parquet_file, index=False)
                        print(f"Updated parquet file: {parquet_file}")
                    except Exception as e:
                        print(f"Error updating parquet file {parquet_file}: {e}")
        else:
            print(f"Data directory not found at {data_dir}")
        
        print("Dataset update completed!")


def main():
    parser = argparse.ArgumentParser(description="Add 2D bounding box information to LeRobot dataset (using YOLOWorld)")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID for the LeRobot dataset (e.g., ISdept/piper_arm)")
    parser.add_argument(
        "--detection_classes", type=str, nargs="+", default=["cube", "container"],
        help="Object class names to detect, e.g. --detection_classes cube container tray",
    )
    parser.add_argument(
        "--detection_conf", type=float, default=0.1,
        help="YOLOWorld confidence threshold (default: 0.1)",
    )
    parser.add_argument("--output_path", type=str, help="Output path for the bounding box data")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process (for testing)")
    parser.add_argument("--revision", type=str, default="main", help="Revision of the dataset to load")
    parser.add_argument("--push_to_hub", action="store_true", help="Push updates to HuggingFace Hub")
    parser.add_argument("--hf_token", type=str, help="HuggingFace token for authentication")

    args = parser.parse_args()

    # Initialize the adder with YOLOWorld
    adder = LeRobot2DBoundingBoxAdder(
        detection_classes=args.detection_classes,
        detection_conf=args.detection_conf,
    )

    # Process the dataset
    try:
        bounding_boxes_data = adder.process_lerobot_dataset(
            repo_id=args.repo_id,
            output_path=args.output_path,
            max_samples=args.max_samples,
            revision=args.revision,
            push_to_hub=args.push_to_hub,
            hf_token=args.hf_token,
        )
        print("2D bounding box information added to dataset successfully!")
        print(f"Processed {len(bounding_boxes_data)} samples with bounding box data")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()