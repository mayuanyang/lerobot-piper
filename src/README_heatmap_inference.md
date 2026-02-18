# Heatmap Inference and Visualization

This document explains how to use the heatmap inference scripts to analyze the transformer diffusion model's spatial attention mechanisms.

## Overview

The transformer diffusion model includes spatial softmax layers that extract keypoints from visual features. These keypoints can be visualized as heatmaps to understand where the model is focusing its attention.

## Available Scripts

### 1. Video Inference with Heatmaps (`video_inference_with_heatmaps.py`)

Processes video frames and generates heatmap visualizations for each frame.

#### Usage:
```bash
python src/video_inference_with_heatmaps.py \
    --model_path /path/to/trained/model \
    --video_path /path/to/input/video.mp4 \
    --output_dir /path/to/output/directory
```

#### Features:
- Processes each video frame through the model
- Generates heatmap visualizations for each frame
- Creates overlay videos showing heatmaps on original frames
- Includes performance benchmarking option

#### Benchmark Mode:
```bash
python src/video_inference_with_heatmaps.py \
    --model_path /path/to/trained/model \
    --video_path /path/to/input/video.mp4 \
    --benchmark \
    --benchmark_frames 100
```

### 2. Simple Image Test (`test_heatmap_inference.py`)

Tests heatmap inference on a single image.

#### Usage:
```bash
python src/test_heatmap_inference.py \
    --model_path /path/to/trained/model \
    --image_path /path/to/input/image.jpg \
    --output_dir /path/to/output/directory
```

#### With State Information:
```bash
python src/test_heatmap_inference.py \
    --model_path /path/to/trained/model \
    --image_path /path/to/input/image.jpg \
    --output_dir /path/to/output/directory \
    --state "0.1,0.2,0.3,0.4,0.5,0.6,0.7"
```

### 3. Video Inference with State Data (`video_inference_with_states.py`)

Processes video frames with state information provided for each frame.

#### Usage:
```bash
python src/video_inference_with_states.py \
    --model_path /path/to/trained/model \
    --video_path /path/to/input/video.mp4 \
    --state_file /path/to/state/data.csv \
    --output_dir /path/to/output/directory
```

#### Supported State File Formats:
- CSV: Comma-separated values, one state per line
- JSON: Array of state arrays
- NPZ: NumPy compressed file with 'states' key

#### Create Sample State File:
```bash
python src/video_inference_with_states.py \
    --create_sample \
    --sample_file sample_states.csv \
    --sample_frames 100
```

## Output Directories

When running the inference scripts, the following directories will be created:

- `heatmaps/`: Individual heatmap visualizations for each processed frame/image
- `heatmap_overlay.mp4`: Video with heatmaps overlaid on original frames (video inference only)
- `inference_results.txt`: Text file with inference results and model outputs

## Understanding Heatmaps

The heatmaps show where the model's spatial softmax layers are focusing attention:

- **Hot spots (red/yellow)**: Areas of high attention
- **Cool spots (blue/dark)**: Areas of low attention
- Each heatmap corresponds to one of the 32 spatial keypoints extracted by the model

## Performance Considerations

- Heatmap generation adds minimal overhead during inference
- The spatial softmax layers are integrated into the model architecture
- Visualization is only generated when explicitly requested (`generate_heatmaps=True`)

## Example Usage

### Process a demonstration video:
```bash
python src/video_inference_with_heatmaps.py \
    --model_path ./outputs/train/checkpoint-50000 \
    --video_path ./data/demo_video.mp4 \
    --output_dir ./results/heatmap_analysis \
    --max_frames 100
```

### Benchmark inference speed:
```bash
python src/video_inference_with_heatmaps.py \
    --model_path ./outputs/train/checkpoint-50000 \
    --video_path ./data/demo_video.mp4 \
    --benchmark \
    --benchmark_frames 200
```

### Test on a single image:
```bash
python src/test_heatmap_inference.py \
    --model_path ./outputs/train/checkpoint-50000 \
    --image_path ./data/test_image.jpg \
    --output_dir ./results/single_image_test
```

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size or use CPU inference
2. **Missing model files**: Ensure the model path contains all required files
3. **Video codec issues**: Try converting video to MP4 format

### Debugging Tips:

- Check that the model was trained with spatial softmax layers enabled
- Verify that the input image/video dimensions match training data
- Ensure sufficient disk space for output files
