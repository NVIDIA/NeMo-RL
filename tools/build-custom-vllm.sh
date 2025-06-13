#!/bin/bash

set -eoux pipefail

# Default values
DEFAULT_GIT_URL="https://github.com/vegaluisjose/vllm.git"
DEFAULT_BRANCH="ssm-fp32-exp"
DEFAULT_VLLM_COMMIT=a3319f4f04fbea7defe883e516df727711e516cd # use full commit hash from the main branch

# Parse command line arguments
GIT_URL=${1:-$DEFAULT_GIT_URL}
BRANCH=${2:-$DEFAULT_BRANCH}
export VLLM_COMMIT=${3:-$DEFAULT_VLLM_COMMIT}
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${DEFAULT_VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

# Create a temporary directory for the build
#BUILD_DIR=$(mktemp -d)
#trap 'rm -rf "$BUILD_DIR"' EXIT
BUILD_DIR=/tmp/vllm

echo "Building vLLM from:"
echo "  Vllm Git URL: $GIT_URL"
echo "  Vllm Branch: $BRANCH"
echo "  VLLM Wheel Commit: $VLLM_COMMIT"
echo "  VLLM Precompiled Wheel Location: $VLLM_PRECOMPILED_WHEEL_LOCATION"

# Clone the repository
echo "Cloning repository..."
git clone "$GIT_URL" "$BUILD_DIR"
cd "$BUILD_DIR"
git checkout "$BRANCH"

# Create a new Python environment using uv
echo "Creating Python environment..."
uv venv
source .venv/bin/activate

# Install dependencies
#echo "Installing dependencies..."
#uv pip install --upgrade pip
#uv pip install numpy
#uv pip install torch==2.7.0 --torch-backend=auto
#uv pip install setuptools setuptools_scm

# Install vLLM using precompiled wheel
echo "Installing vLLM with precompiled wheel..."
uv pip install -e .

echo "Build completed successfully!"
echo "The built vLLM is available in: $BUILD_DIR"
echo "You can now update your pyproject.toml to use this local version."