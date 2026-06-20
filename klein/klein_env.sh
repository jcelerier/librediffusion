#!/usr/bin/env bash
# Source this to get the klein export env: flux venv (torch/trt11/modelopt) + git diffusers 0.39 side-loaded.
# Set these to your local paths before sourcing (placeholders shown).
export VENV=${VENV:-./.flux-venv}                 # flux venv (torch/trt11/modelopt)
export SIDE=${SIDE:-./klein-pydeps}               # side-loaded git diffusers 0.39
export PY=$VENV/bin/python
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
TRTLIBS=$($PY -c "import tensorrt_libs,os;print(os.path.dirname(tensorrt_libs.__file__))" 2>/dev/null)
NVLIBS=$($PY -c "import nvidia,os,glob;print(':'.join(glob.glob(os.path.dirname(nvidia.__file__)+'/*/lib')))" 2>/dev/null)
export LD_LIBRARY_PATH=$TRTLIBS:$NVLIBS:$LD_LIBRARY_PATH
export PATH=$VENV/bin:$PATH
export PYTHONPATH=$SIDE
