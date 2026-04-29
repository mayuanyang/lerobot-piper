#!/bin/bash
# Colab setup script for lerobot-piper training
# Run once at the top of your Colab notebook:
#   !bash lerobot-piper/colab_setup.sh
# or paste the pip commands directly into a cell.

set -e

# 1. Pin numpy before anything else — LeRobot requires <2.0.0
pip install -q "numpy<2.0.0"

# 2. LeRobot (install from source for latest API)
pip install -q "git+https://github.com/huggingface/lerobot.git"

# 3. HuggingFace stack
pip install -q \
  "transformers>=4.45.0" \
  "safetensors>=0.4.0" \
  "huggingface_hub>=0.24.0" \
  "datasets>=2.19.0"

# 4. LoRA — torchao must be >=0.16.0 or peft raises ImportError
pip install -q "peft>=0.9.0" "torchao>=0.16.0"

# 5. Vision / data
pip install -q "opencv-python>=4.5.0" "Pillow>=9.0.0" "pyarrow>=14.0.0"

# 6. Optional: YOLO-based object detector
# pip install -q "ultralytics>=8.0.0"

echo "Setup complete."
