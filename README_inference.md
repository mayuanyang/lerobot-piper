# LeRobot Inference Code

This repository contains inference code for the LeRobot framework, specifically designed for the Piper robot arm project.

## Files

1. `inference.py` - Full LeRobot-based inference implementation
2. `simple_inference.py` - Simplified inference implementation with dummy mode
3. `lerobot_inference.py` - Object-oriented inference implementation with demo mode
4. `webcam_inference.py` - Webcam-based inference implementation for real-time robot control

## Usage

### Webcam Inference

To run webcam-based inference:

```bash
python webcam_inference.py
```

This script will:
1. Initialize the webcam (default: camera 0)
2. Capture frames in real-time
3. Process frames for model input
4. Run inference if a trained model is available
5. Display the webcam feed

### Webcam Requirements

- A webcam connected to your computer
- OpenCV installed: `pip install opencv-python`

### Customization

To customize the webcam inference:
1. Change the webcam ID in the `WebcamInference` constructor
2. Adjust frame preprocessing parameters (resolution, normalization)
3. Modify the joint state input based on your robot's sensors

## Usage

### With a trained model

1. First, train a model using `train.py`:
   ```bash
   python train.py
   ```

2. Run inference:
   ```bash
   python lerobot_inference.py
   ```

### Without a trained model (demo mode)

The inference scripts include a demo mode that simulates inference without requiring a trained model.

## Requirements

- Python 3.7+
- PyTorch
- LeRobot framework
- NumPy < 2.0 (due to compatibility issues)
- OpenCV (for webcam inference)

### Installation

To install the required packages, run:

```bash
pip install -r requirements_inference.txt
```

Or install individually:

```bash
pip install torch
pip install "numpy<2.0.0"
pip install opencv-python
pip install lerobot
```

## Troubleshooting

### NumPy Compatibility Issues

If you encounter NumPy compatibility issues like:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Try downgrading NumPy:
```bash
pip install "numpy<2"
```

### LeRobot Installation

If LeRobot is not installed, install it with:
```bash
pip install lerobot
```

## Implementation Details

The inference code follows the LeRobot framework patterns:

1. **Model Loading**: Loads trained policy weights and configuration
2. **Preprocessing**: Normalizes input observations using dataset statistics
3. **Inference**: Runs the policy to predict actions
4. **Postprocessing**: Denormalizes output actions to real-world values

## Customization

To customize for your specific robot:

1. Modify the observation format in `create_sample_observation()`
2. Adjust the action processing based on your robot's DOF (degrees of freedom)
3. Update the dataset ID if using a different dataset

## Example Output

When running in demo mode:
```
LeRobot Inference Demo
==============================
Loading model...
Could not load trained model. Using dummy mode for demonstration.

==============================
DEMO MODE - Simulated Inference
==============================
Input observation: [0.1 0.2 0.3 0.4 0.5 0.6]
Simulated action: [0.10942388 0.1982365  0.30545381 0.39105978 0.52178071 0.58944586]
