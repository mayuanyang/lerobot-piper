# Action Visualization Tools

This repository includes tools to visualize the comparison between ground truth actions and predicted actions from your trained model.

## Installation

The visualization tools require additional dependencies:

```bash
pip install -r requirements_visualization.txt
```

## NumPy Compatibility Issue

**Important**: There is a known compatibility issue with NumPy 2.x. The visualization tools require NumPy 1.x versions. If you encounter errors like:

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
```

You can resolve this by downgrading NumPy:

```bash
pip install "numpy>=1.21.0,<2.0.0"
```

The `requirements_visualization.txt` file already specifies compatible versions, but if you have an existing environment with NumPy 2.x, you may need to downgrade manually.

## Tools Overview

### 1. Interactive GUI Visualizer (`src/action_visualizer.py`)

A tkinter-based graphical interface that allows you to:
- Load your trained model
- Load the dataset
- Navigate through episodes and frames
- Visualize ground truth vs predicted actions in real-time

To run the GUI:
```bash
python src/action_visualizer.py
```

Features:
- Model loading dialog
- Dataset loading
- Episode/frame navigation
- Bar chart comparison of joint actions
- Real-time visualization updates

### 2. Batch Visualizer (`src/batch_action_visualizer.py`)

A command-line tool that generates plots for multiple frames and saves them to disk.

To run with default settings:
```bash
python src/batch_action_visualizer.py
```

To customize the visualization:
```bash
python src/batch_action_visualizer.py \
  --model-path src/model_output \
  --dataset-id ISdept/piper_arm \
  --num-frames 20 \
  --output-dir my_action_plots
```

The batch visualizer will generate:
- Individual bar charts comparing ground truth vs predicted actions for each frame
- An overall MSE (Mean Squared Error) plot showing model performance across frames
- Summary statistics

### 3. Simple Text-based Visualizer (`src/simple_action_visualizer.py`)

A lightweight command-line tool that prints action comparisons to the console without requiring matplotlib.

To run:
```bash
python src/simple_action_visualizer.py --num-frames 5
```

### 4. Local Dataset Visualizer (`src/local_action_visualizer.py`)

A version that works with local datasets and handles dataset format issues.

To run:
```bash
python src/local_action_visualizer.py --num-frames 5
```

## Understanding the Visualizations

### Bar Charts
Each bar chart shows:
- Blue bars: Ground truth actions (actual joint positions from the dataset)
- Red bars: Predicted actions (model output)
- X-axis: Joint indices (0-6 for the 7-DOF arm)
- Y-axis: Action values

### MSE Plot
The Mean Squared Error plot shows how the model's predictions deviate from the ground truth across frames. Lower values indicate better performance.

## Usage Examples

### Using the Interactive GUI

1. Run the GUI:
   ```bash
   python src/action_visualizer.py
   ```

2. Click "Load Model" and select your trained model directory (default: `src/model_output`)

3. Click "Load Dataset" to load the training dataset

4. Use the spinboxes to select an episode and frame

5. Click "Update" to generate the visualization

### Using the Batch Visualizer

Generate visualizations for the first 10 frames:
```bash
python src/batch_action_visualizer.py --num-frames 10
```

Generate visualizations with a custom model path:
```bash
python src/batch_action_visualizer.py --model-path /path/to/your/model
```

### Using the Simple Text-based Visualizer

Compare the first 5 frames:
```bash
python src/simple_action_visualizer.py --num-frames 5
```

### Using the Local Dataset Visualizer

Compare the first 5 frames with local dataset handling:
```bash
python src/local_action_visualizer.py --num-frames 5
```

## Output

The batch visualizer saves plots to the specified output directory (default: `action_plots`):
- `action_comparison_frame_*.png`: Individual frame comparisons
- `mse_over_frames.png`: Overall performance plot

The simple text-based and local dataset visualizers print results directly to the console.

## Troubleshooting

### Common Issues

1. **"Module not found" errors**: Make sure you've installed all dependencies:
   ```bash
   pip install -r requirements_inference.txt
   pip install -r requirements_visualization.txt
   ```

2. **NumPy compatibility errors**: Downgrade to NumPy 1.x:
   ```bash
   pip install "numpy>=1.21.0,<2.0.0"
   ```

3. **Model loading errors**: Ensure your model path contains:
   - `config.json`
   - `model.safetensors` or `pytorch_model.bin`
   - `policy_preprocessor.json`
   - `policy_postprocessor.json`

4. **Dataset loading errors**: If you encounter dataset format issues:
   - Try using `src/local_action_visualizer.py` which handles local datasets
   - Check that your dataset files are properly formatted
   - Ensure camera names in episodes.jsonl match what the code expects

5. **Hugging Face dataset issues**: If you get errors about dataset loading:
   - The local visualizer tools use dummy data when the dataset can't be loaded
   - For real data visualization, ensure your dataset is properly formatted

### Getting Help

If you encounter issues, check the console output for error messages. Common problems include:
- Incorrect file paths
- Missing dependencies
- Incompatible model/dataset versions
- NumPy version incompatibilities
- Dataset format mismatches

## Development Notes

The visualization tools are designed to be modular and extensible:

1. **action_visualizer.py**: Full-featured GUI with matplotlib integration
2. **batch_action_visualizer.py**: Batch processing with file output
3. **simple_action_visualizer.py**: Console-based output for quick checks
4. **local_action_visualizer.py**: Handles local dataset issues and format mismatches

You can extend these tools by:
- Adding new visualization types (line plots, heatmaps, etc.)
- Supporting additional data formats
- Adding export options (CSV, JSON, etc.)
- Implementing statistical analysis features
