#!/bin/bash
# Colab setup script for lerobot-piper training
# Run once at the top of your notebook:
#   !bash lerobot-piper/colab_setup.sh

set -e

# 1. Pin numpy before everything else — LeRobot requires <2.0.0
pip install -q "numpy<2.0.0"

# 2. LeRobot 0.4.0 with SmolVLA support
#    This pulls in: transformers>=4.53.0, huggingface_hub>=0.34.2,<0.36.0,
#                   datasets>=4.0.0,<4.2.0, safetensors>=0.4.3
pip install -q "lerobot[smolvla]==0.4.0"

# 3. Re-pin HuggingFace packages to lerobot 0.4.0 constraints
#    (Colab's pre-installed versions may be incompatible)
pip install -q \
  "huggingface_hub[cli,hf-transfer]>=0.34.2,<0.36.0" \
  "transformers>=4.53.0,<5.0.0" \
  "datasets>=4.0.0,<4.2.0" \
  "safetensors>=0.4.3,<1.0.0"

# 4. LoRA — torchao must be installed first at >=0.16.0, or peft raises ImportError
pip install -q "torchao>=0.16.0"
pip install -q "peft>=0.13.0,<1.0.0"

# 5. Vision / data
pip install -q "opencv-python>=4.5.0" "Pillow>=9.0.0" "pyarrow>=14.0.0"

# 6. Optional: YOLO-based object detector
# pip install -q "ultralytics>=8.0.0"

echo "Setup complete."
