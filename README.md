# TensileSME
JIT/AOT compiler for matrix multiplication and linear layers in Python targeting AArch64 Arm SME built on MLIR
> NOTE: TensileSME is not a Python compiler. It only compiles matrix and linear layer operations in Python.

## Why use this?
Dont. Use CUDA or something else.

## Example
```python
> NOTE: not working example
import numpy as np
import tensor_sme as sme

# Define standard data
A = np.random.rand(128, 128).astype(np.float32)
B = np.random.rand(128, 128).astype(np.float32)

@ame.jit
def matmul_kernel(a, b):
    return s.matmul(a, b)

@ame.jit
def linear_layer_kernel(a, b):
    return s.

# Generate SME code, execute it and return a NumPy Array
C = kernel(A, B)
```

## Usage
```bash
./tensor-sme-opt model.mlir <--examine-mlir>
```

## Build
- TensileSME uses the LLVM and MLIR. You will need to clone and build LLVM and MLIR on your local machine, in the root of your TensileSME clone. **Build instructions are not complete. Follow the MLIR build instructions.**
```bash
mkdir build_driver
cd build_driver
cmake ..
make tensor-sme-opt -j8        # Adjust based on number of CPU cores available
```