#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot environment setup for WilR VLA training on a fresh server.
#
#   bash setup_env.sh            # creates conda env `wilro` and installs all deps
#
# Designed for a headless multi-GPU box (SSH). Reproduces the Colab deps but in
# a clean conda env, with the install ORDER that avoids the torch/torchaudio and
# mujoco/robosuite version drift that broke imports before. Run once; then use
# run_rl_8gpu.sh to train.
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${ENV_NAME:-wilro}"
PY_VER="${PY_VER:-3.10}"          # 3.10 is the sweet spot for robosuite/LIBERO/lerobot

echo "=== [1/9] conda env: $ENV_NAME (python $PY_VER) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
# Idempotent: reuse the env if it already exists so this script is safe to re-run
# after a partial/network failure (pip below skips already-installed packages).
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "    env '$ENV_NAME' already exists — reusing it (safe to re-run)."
else
    conda create -y -n "$ENV_NAME" python="$PY_VER"
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# Survive flaky networks: longer timeout + more retries on every pip call below.
export PIP_DEFAULT_TIMEOUT=100
export PIP_RETRIES=10

# --- [2/9] System GL/EGL libs for headless MuJoCo (EGL offscreen rendering) ---
# Needs sudo. On an NVIDIA box libEGL is usually already provided by the driver;
# uncomment if env construction fails with an EGL/GL error.
# sudo apt-get update && sudo apt-get install -y \
#     libegl1 libgl1 libgles2 libosmesa6 libglfw3 xvfb

echo "=== [3/9] lerobot (pulls a CUDA-enabled torch) ==="
pip install lerobot==0.4.0

echo "=== [4/9] model / data / misc deps ==="
pip install transformers decord opencv-python num2words \
    "qwen-vl-utils[decord]" ultralytics pyvirtualdisplay bitsandbytes
pip install hydra-core==1.2.0

echo "=== [5/9] simulator stack — PINNED, installed AFTER lerobot so nothing upgrades them ==="
# mujoco MUST stay 3.1.6: newer mujoco changed mj_fullM's signature and breaks
# robosuite 1.4.1's controller init (TypeError in mj_fullM).
pip install bddl "robosuite==1.4.1" "mujoco==3.1.6"

echo "=== [6/9] remove torchaudio (ABI-mismatched, breaks 'import transformers', unused) ==="
pip uninstall -y torchaudio || true

echo "=== [7/9] LIBERO (no-deps so it can't drag versions) ==="
[ -d LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
# editable_mode=compat: LIBERO's `libero` is a NAMESPACE package (no top-level
# __init__.py); modern setuptools' PEP 660 import-hook editable install fails to
# expose it ("pip show" lists it but `import libero` errors). compat mode writes
# a plain .pth that puts the dir on sys.path, which works for namespace packages.
pip install -e ./LIBERO --no-deps --config-settings editable_mode=compat
# LIBERO runtime deps that --no-deps skips (small, pure-Python, no version pins):
pip install easydict thop

echo "=== [8/9] clone the training repo + apply the groot import patch ==="
[ -d lerobot-piper ] || git clone https://github.com/mayuanyang/lerobot-piper.git
python - <<'PY'
import lerobot, os
init = os.path.join(os.path.dirname(lerobot.__file__), "policies", "__init__.py")
src = open(init).read()
needle = "from .groot.configuration_groot import GrootConfig as GrootConfig"
if needle in src and "# groot-patch" not in src:
    src = src.replace(needle,
        "try:\n"
        "    from .groot.configuration_groot import GrootConfig as GrootConfig\n"
        "except Exception as _e:  # groot-patch: this lerobot build's groot dataclass is broken\n"
        "    GrootConfig = None\n"
        "    print('[groot-patch] skipped groot import:', _e)")
    open(init, "w").write(src); print("[groot-patch] patched", init)
else:
    print("[groot-patch] already patched or line not found")
PY

echo "=== [9/9] verify the import chain + GPUs ==="
python - <<'PY'
import torch, mujoco, transformers, robosuite, lerobot
from lerobot.envs.libero import LiberoEnv  # the import that kept breaking
print("torch     :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| GPUs:", torch.cuda.device_count())
print("mujoco    :", mujoco.__version__, "(must be 3.1.6)")
print("robosuite :", robosuite.__version__, "(must be 1.4.1)")
print("OK: all imports clean")
PY

cat <<EOF

===========================================================================
Done. Next steps:
  conda activate $ENV_NAME
  huggingface-cli login          # or: export HF_TOKEN=...; huggingface-cli login --token \$HF_TOKEN
  bash lerobot-piper/run_rl_8gpu.sh

If 'cuda: False' above, your driver doesn't match the bundled torch CUDA — install a
matching build, e.g.:  pip install --force-reinstall torch torchvision --index-url \\
  https://download.pytorch.org/whl/cu121   (then re-run step 6 to drop torchaudio).
===========================================================================
EOF
