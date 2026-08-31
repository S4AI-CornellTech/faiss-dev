#!/usr/bin/env bash

set -e

echo "======================================"
echo "🔧 Step 1: Configuring CMake build..."
echo "======================================"

cmake -B build \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_C_API=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_ENABLE_CUVS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release

echo "======================================"
echo "🚀 Step 2: Building core FAISS library..."
echo "======================================"

cmake --build build --target faiss -j$(nproc)

echo "======================================"
echo "🐍 Step 3: Building Python (SWIG) bindings..."
echo "======================================"

cmake --build build --target swigfaiss -j$(nproc)

echo "======================================"
echo "📦 Step 4: Installing Python bindings..."
echo "======================================"

(cd build/faiss/python && python setup.py install)

echo "======================================"
echo "📂 Step 5: Installing FAISS libraries..."
echo "======================================"

cmake --install build --prefix "$CONDA_PREFIX"

echo "======================================"
echo "✅ Build and installation complete!"
echo "======================================"