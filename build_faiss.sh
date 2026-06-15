╰▶🙃 cat build_faiss.sh 
#!/usr/bin/env bash
set -e  # stop on first error

echo "======================================"
echo "🔧 Step 1: Configuring CMake build..."
echo "======================================"
cmake -B build \
  -DFAISS_ENABLE_GPU=ON \
  -DBUILD_TESTING=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
#  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release

echo "======================================"
echo "🚀 Step 2: Building core FAISS library..."
echo "======================================"
make -C build -j$(nproc) faiss

echo "======================================"
echo "🐍 Step 3: Building Python (SWIG) bindings..."
echo "======================================"
make -C build -j$(nproc) swigfaiss

echo "======================================"
echo "📦 Step 4: Installing Python bindings..."
echo "======================================"
(cd build/faiss/python && python setup.py install)

echo "======================================"
echo "📂 Step 5: Installing FAISS libraries into environment..."
echo "======================================"
cmake --install build --prefix "$CONDA_PREFIX"

echo "======================================"
echo "✅ Build and installation complete!"
echo "======================================"
