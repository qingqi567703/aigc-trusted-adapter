#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"
"$BASE_PYTHON" scripts/verify_bundle.py
"$BASE_PYTHON" -c 'import sys,torch; assert sys.version_info[:2] == (3,10), sys.version; assert torch.__version__.split("+")[0] == "2.2.2", torch.__version__; print("Base Torch", torch.__version__, "CUDA", torch.version.cuda)'
"$BASE_PYTHON" -m venv --without-pip --system-site-packages env
env/bin/python -m pip install --no-index --no-deps --find-links offline/wheelhouse -r requirements-linux-cp310.txt
HF_HOME="$PWD/work/hf_cache" env/bin/python -c 'import torch,diffusers,transformers,accelerate,cv2; print(torch.__version__,diffusers.__version__,transformers.__version__,accelerate.__version__); assert torch.cuda.is_available()'
echo "Offline environment imports passed."
