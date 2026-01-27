#!/bin/env zsh

set -euo pipefail

export CUDA_HOME=/usr/local/cuda
# export CUDA_HOME=/usr/local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}



