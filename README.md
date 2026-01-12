# TensileSME
TensileSME is a vertical-slice compiler that lowers high-level Python tensor operations directly to Armv9 Streaming SVE assembly. It bridges the gap between abstract linear algebra (NumPy) and bare-metal hardware acceleration. Unlike heavy frameworks (PyTorch, TensorFlow) which rely on large, pre-compiled runtime libraries, TensileSME uses MLIR to JIT-compile custom kernels specifically optimized for the Arm SME architecture.

## Core features
- Zero-Overhead Abstraction: Python decorators (@sme.jit) compile directly to machine code.
- Hardware-Aware Tiling: Automatically breaks large matrices into hardware-friendly tiles (ZA-Tile alignment).
- Kernel Fusion: Fuses Matmul + Bias + ReLU into single loop nests to minimize memory traffic.
- No Bloat: Designed for high-performance edge inference where every kilobyte of memory matters.

## Example
```python
import tensile_sme as sme
import numpy as np

# 1. Setup Data (Host Side)
# Standard NumPy arrays. No custom data types required.
A = np.random.rand(128, 128).astype(np.float32)
B = np.random.rand(128, 128).astype(np.float32)
Bias = np.random.rand(128).astype(np.float32)

# 2. Define the Kernel
# The JIT compiler lowers this pure Python function to an MLIR Module,
# applies tiling and bufferization passes, and emits an Arm object file.
@sme.jit
def fused_linear_layer(a, b, bias):
    # TensileSME detects the pattern and fuses these into a
    # single hardware kernel loop.
    dense = sme.matmul(a, b)
    biased = dense + bias  # Implicit broadcasting
    return sme.relu(biased)

# 3. Execute
# Triggers compilation and executes on the SME hardware/emulator.
C = fused_linear_layer(A, B, Bias)

print(f"Inference Complete. Result shape: {C.shape}")
```

## Usage
```bash
./tensor-sme-opt model.mlir <--examine-mlir>    # NOTE: Not final usage
```

## Build

### Prerequisites

- LLVM/IR built from source (requires C++ 20)
- Python 3.10 with MLIR Python bindings
- CMake 3.20+
### Build driver
```bash
mkdir build_driver
cd build_driver
cmake ..
make tensor-sme-opt -j8        # Adjust based on number of CPU cores available
```