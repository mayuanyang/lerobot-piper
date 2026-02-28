# Adding 2D Bounding Boxes to LeRobot Datasets

This document explains how to add 2D bounding box information to existing LeRobot datasets using the Qwen3-VL model for object detection. Two approaches are provided:

1. Using the Hugging Face datasets library directly
2. Using the LeRobotDataset class (recommended for LeRobot datasets)

## Overview

The script `add_2d_bounding_boxes_to_hf_dataset.py` downloads a LeRobot dataset from Hugging Face, processes each frame to detect objects using the Qwen3-VL model, and adds the 2D bounding box information to the dataset.

## Prerequisites

1. Python 3.8 or higher
2. Required Python packages:
   ```bash
   pip install torch torchvision pillow transformers datasets huggingface_hub tqdm opencv-python numpy pandas pyarrow
   ```

3. Access to the Qwen3-VL-4B-Instruct model (automatically downloaded when running the script)

## Usage

### Basic Usage

To add 2D bounding boxes to a Hugging Face dataset:

```bash
python add_2d_bounding_boxes_to_hf_dataset.py \
    --repo_id ISDept/piper_arm \
    --output_path ./piper_arm_with_bounding_boxes
```

### Advanced Usage

#### Specify a Custom Prompt

You can customize the object detection prompt:

```bash
python add_2d_bounding_boxes_to_hf_dataset.py \
    --repo_id ISDept/piper_arm \
    --user_prompt 'locate every instance that belongs to the following categories: "cube, container, table, robot arm". Report bbox coordinates in JSON format.' \
    --output_path ./piper_arm_with_bounding_boxes
```

#### Process Only a Subset of Samples

For testing purposes, you can limit the number of samples processed:

```bash
python add_2d_bounding_boxes_to_hf_dataset.py \
    --repo_id ISDept/piper_arm \
    --max_samples 100 \
    --output_path ./piper_arm_with_bounding_boxes
```

#### Specify Dataset Split

Process a specific dataset split (default is "train"):

```bash
python add_2d_bounding_boxes_to_hf_dataset.py \
    --repo_id ISDept/piper_arm \
    --split test \
    --output_path ./piper_arm_test_with_bounding_boxes
```

## Output Format

The script adds a new column `observation.bounding_boxes` to the dataset. This column contains a dictionary for each sample with bounding box information for each camera:

```json
{
  "observation.bounding_boxes": {
    "camera1": [
      {
        "bbox_2d": [100, 150, 300, 400],
        "label": "cup"
      },
      {
        "bbox_2d": [400, 200, 600, 500],
        "label": "bowl"
      }
    ],
    "camera2": [
      {
        "bbox_2d": [200, 100, 500, 300],
        "label": "person"
      }
    ]
  }
}
```

Each bounding box is represented as:
- `bbox_2d`: Array of 4 integers [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner
- `label`: String indicating the object category
- Coordinates are normalized to the range [0, 1000]

## How It Works

1. The script downloads the specified dataset from Hugging Face
2. For each sample in the dataset:
   - For each camera image in the sample:
     - The image is converted to a tensor without resizing (preserving original dimensions)
     - The Qwen3-VL model is used to detect objects in the image
     - The detected objects and their bounding boxes are extracted
   - The bounding box information is stored in the `observation.bounding_boxes` column
3. The updated dataset is saved to the specified output path

## Performance Considerations

- Object detection using Qwen3-VL is computationally intensive
- Processing time depends on the number of samples and cameras per sample
- Using a GPU significantly speeds up processing
- For large datasets, consider processing in batches using the `--max_samples` parameter

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors, try:

1. Using a smaller batch size (currently hard-coded to 1)
2. Using CPU processing by setting `CUDA_VISIBLE_DEVICES=""` before running the script
3. Processing fewer samples at a time

### Model Loading Issues

If the Qwen3-VL model fails to load:

1. Ensure you have sufficient disk space (the model is several GB in size)
2. Check your internet connection (the model is downloaded automatically)
3. Try manually downloading the model using:
   ```python
   from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
   model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
   processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
   ```

## Updating Existing Datasets

If you have already added bounding boxes to a dataset and want to update them with a different prompt or model, you can run the script again with the same output path. The script will overwrite the existing `observation.bounding_boxes` column.

## Using LeRobotDataset Class (Recommended for LeRobot datasets)

For LeRobot datasets, we recommend using the `add_2d_bounding_boxes_to_lerobot_dataset.py` script which is specifically designed to work with the LeRobotDataset class.

### Basic Usage

```bash
python add_2d_bounding_boxes_to_lerobot_dataset.py \
    --repo_id ISDept/piper_arm \
    --output_path ./piper_arm_bounding_boxes
```

### Advanced Usage

#### Specify a Custom Prompt

```bash
python add_2d_bounding_boxes_to_lerobot_dataset.py \
    --repo_id ISDept/piper_arm \
    --user_prompt 'locate every instance that belongs to the following categories: "cube, container, table, robot arm". Report bbox coordinates in JSON format.' \
    --output_path ./piper_arm_bounding_boxes
```

#### Process Only a Subset of Samples

```bash
python add_2d_bounding_boxes_to_lerobot_dataset.py \
    --repo_id ISDept/piper_arm \
    --max_samples 100 \
    --output_path ./piper_arm_bounding_boxes
```

#### Specify Dataset Revision

```bash
python add_2d_bounding_boxes_to_lerobot_dataset.py \
    --repo_id ISDept/piper_arm \
    --revision v1.0 \
    --output_path ./piper_arm_bounding_boxes
```

### How It Works

1. The script loads the dataset using the LeRobotDataset class
2. For each sample in the dataset:
   - For each camera image in the sample:
     - The image is converted to a tensor without resizing (preserving original dimensions)
     - The Qwen3-VL model is used to detect objects in the image
     - The detected objects and their bounding boxes are extracted
   - The bounding box information is stored in a JSON file for easy integration
3. The bounding box data is saved to the specified output path

### Output Format

The script saves the bounding box data in JSON format:

1. `bounding_boxes.json` - Array of bounding box data for each sample
2. `dataset_info.json` - Copy of the dataset information
3. `index_to_bounding_boxes.json` - Mapping of sample indices to bounding box data for easy lookup

Each bounding box entry follows the same format as described above.
