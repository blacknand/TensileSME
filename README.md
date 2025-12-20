# TensileSME
Compiler for lowering high-level Python tensor operations directly to bare-metal Arm SME assembly. TensileSME aims to bridge the gap between abstract linear alebra with Python/NumPy and hardware-accelerated loop nests with Armv9 Streaming SVE without relying on heavy runtimes like PyTorch or TensorFlow.

> NOTE: not working example
## Basic matrix multiplication example
```python
import numpy as np
import tensor_sme as sme

# Define standard data
A = np.random.rand(128, 128).astype(np.float32)
B = np.random.rand(128, 128).astype(np.float32)

@ame.jit
def matmul_kernel(a, b):
    return s.matmul(a, b)

# Generate SME code, execute it and return a NumPy Array
C = kernel(A, B)
```

## Bias add example
```python
import numpy as np
import tensor_sme as sme

weight = np.random.rand(128, 128).astype(np.float32)
bias = np.random.rand(128, 128).astype(np.float32)

@sme.jit
def dense_layer(input_data):
    partial = sme.matmu(input_data, weights)
    final = sme.bias_add(partial, bias)
    return final

input_data = np.random.rand(128, 128).astype(np.float32)
result = dense_layer(input_data)
```

## Architecture
The compiler operates in a strict multi-phase pipeline:
1. **Frontend**: Generates MLIR using Python bindings. Hanldes shape inference and type verification programatically.
2. **Driver**: A executable that ingests the generated IR, manages the MLIR context, dialect registry, and pass manager. Verifices the semantic correctness.
3. **Middle-end**: Converts abstract math into hardware friendly loops. Performs one-shot bufferization to map immutable Tensors to mutable MemRefs.
4. **Backend**: Lowers loops to `vector` and `arm_sme` intrinsics. Manages ZA tile state and streaming mode. 

## Usage
```bash
./tensor-sme-opt model.mlir <--examine-mlir>
```

## Build
### Prerequisites
- LLVM/IR built from source (requires C++ 20)
- Python 3.10 with numpy and MLIR Python bindings
- CMake 3.20+
### Build driver
```bash
mkdir build_driver
cd build_driver
cmake ..
make tensor-sme-opt -j8        # Adjust based on number of CPU cores available
```