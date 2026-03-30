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
        """Load and process a LeRobot dataset, adding 2D bounding box information via YOLOWorld."""
        print(f"Loading LeRobot dataset {repo_id}...")
        dataset = LeRobotDataset(repo_id, force_cache_sync=True, revision=revision)
        print(f"Dataset loaded with {len(dataset)} samples")

        indices = list(range(min(max_samples, len(dataset)) if max_samples else len(dataset)))
        if max_samples:
            print(f"Limited to {len(indices)} samples")

        sample0 = dataset[0]
        camera_keys = [k for k in sample0.keys() if k.startswith("observation.images.")]
        print(f"Camera keys: {camera_keys}")

        bounding_boxes_data = []
        pbar = tqdm(indices, desc="Detecting boxes", postfix={"boxes": 0})

        for idx in pbar:
            try:
                sample = dataset[idx]
                sample_bounding_boxes = {}
                sample_boxes_count = 0

                # Always record episode/frame indices for parquet lookup
                if 'episode_index' in sample:
                    sample_bounding_boxes['episode_index'] = sample['episode_index'].item() if isinstance(sample['episode_index'], torch.Tensor) else sample['episode_index']
                if 'frame_index' in sample:
                    sample_bounding_boxes['frame_index'] = sample['frame_index'].item() if isinstance(sample['frame_index'], torch.Tensor) else sample['frame_index']

                # Always run detection on every sample (ignore existing box data)
                for key, value in sample.items():
                    if not key.startswith("observation.images."):
                        continue
                    camera_name = key.replace("observation.images.", "")
                    try:
                        if isinstance(value, torch.Tensor) and value.dim() == 3:
                            # Build a (1, C, H, W) uint8 tensor for YOLOWorld
                            img_t = value.detach().cpu().float()
                            if img_t.min() < 0:
                                img_t = (img_t + 1.0) / 2.0
                            img_t = img_t.clamp(0, 1)
                            tensor_image = img_t.unsqueeze(0).to(self.device)

                            detections = self.detect_objects_and_get_2d_bounding_boxes(tensor_image)[0]

                            # Build 2-box slot (pad with zeros if fewer than 2 detected)
                            converted_boxes = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(2)]
                            for box_idx, box_data in enumerate(detections[:2]):
                                coords = box_data.get('bbox_2d', [0.0]*4)
                                if len(coords) == 4:
                                    converted_boxes[box_idx] = coords + [
                                        float(box_data.get('category_id', 0)),
                                        float(box_data.get('confidence', 1.0)),
                                    ]

                            sample_bounding_boxes[camera_name] = converted_boxes
                            sample_boxes_count += len(detections)
                        else:
                            sample_bounding_boxes[camera_name] = [[0.0]*6 for _ in range(2)]
                    except Exception as e:
                        print(f"Error on sample {idx} camera {camera_name}: {e}")
                        sample_bounding_boxes[camera_name] = [[0.0]*6 for _ in range(2)]
                
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
        """Update parquet files with bounding box detections for the processed batch."""
        dataset_dir = Path(f"/root/.cache/huggingface/lerobot/{repo_id}")
        if not dataset_dir.exists():
            print(f"Dataset directory not found: {dataset_dir}")
            return

        data_dir = dataset_dir / "data"
        if not data_dir.exists():
            print(f"Data directory not found: {data_dir}")
            return

        updated = 0
        for chunk_dir in data_dir.glob("chunk-*"):
            for parquet_file in chunk_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(parquet_file)
                    changed = False
                    for bbox_data in bounding_boxes_data:
                        if 'episode_index' not in bbox_data or 'frame_index' not in bbox_data:
                            continue
                        episode_index = bbox_data['episode_index']
                        frame_index = bbox_data['frame_index']
                        mask = (df['episode_index'] == episode_index) & (df['frame_index'] == frame_index)
                        if not mask.any():
                            continue

                        bbox_data_copy = {k: v for k, v in bbox_data.items() if k not in ('episode_index', 'frame_index')}
                        final_boxes = self._convert_detected_bounding_boxes_to_required_format(bbox_data_copy)
                        new_boxes = np.array(final_boxes)
                        converted_boxes = [[float(x) for x in new_boxes[i]] for i in range(6)]

                        if mask.sum() > 1:
                            df.loc[mask, 'observation.box'] = [converted_boxes] * mask.sum()
                        else:
                            df.at[df.index[mask][0], 'observation.box'] = converted_boxes
                        updated += 1
                        changed = True

                    if changed:
                        df.to_parquet(parquet_file, index=False)
                except Exception as e:
                    print(f"Error updating {parquet_file}: {e}")
                    print(traceback.format_exc())

        print(f"Partial update complete: {updated} frames written.")

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
        
        return final_boxes

    def _update_dataset_with_bounding_boxes(self, repo_id, bounding_boxes_data, revision="main"):
        """Update all parquet files with the full bounding_boxes_data list."""
        self._update_dataset_with_bounding_boxes_partial(repo_id, bounding_boxes_data, [], revision)


def main():
    parser = argparse.ArgumentParser(description="Add 2D bounding box information to LeRobot dataset (using YOLOWorld)")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID for the LeRobot dataset (e.g., ISdept/piper_arm)")
    parser.add_argument(
        "--detection_classes", type=str, nargs="+", default=["blue cube", "pink box"],
        help="Object class names to detect, e.g. --detection_classes cube container tray",
    )
    parser.add_argument(
        "--detection_conf", type=float, default=0.05,
        help="YOLOWorld confidence threshold (default: 0.05)",
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