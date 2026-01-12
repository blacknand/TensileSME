# TensileSME

TensileSME is a high-performance, vertical-slice graph compiler that lowers Python tensor programs directly to bare-metal **Armv9 Streaming SVE** assembly. It bypasses standard runtime intermediaries (like the PyTorch Dispatcher) to perform **hardware-aware compilation** specifically for the Arm Scalable Matrix Extension (SME).

Unlike general-purpose linear algebra libraries, TensileSME specializes in **Linalg Fusion**. It detects producer-consumer chains (e.g., `MatMul -> BiasAdd -> ReLU`) and fuses them into a single, hardware-optimized loop nest. This maximizes arithmetic intensity by keeping data resident in the CPU's registers and L1 cache, minimizing expensive round-trips to main memory.

## Core Features
- **Graph-Level Optimization:** Automatically identifies and fuses element-wise operations (Add, ReLU, Sigmoid) into the preceding matrix multiplication kernel using Polyhedral compilation principles.
- **Hardware-Aware Tiling:** Implements a custom "Tile-and-Fuse" pass that decomposes large tensors into blocks aligned with the specific **ZA Tile** dimensions of the target hardware.
- **Scalable Vectorization:** Generates assembly that adapts at runtime to the physical Vector Length (VL) of the processor (128-bit to 2048-bit), utilizing LLVM's `vscale` semantics.
- **Zero-Copy JIT:** A lightweight C++ driver compiles the MLIR intermediate representation to machine code in memory, executing kernels with zero runtime overhead or bloat.

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
# The JIT compiler lowers this pure Python function to an MLIR Module.
# It identifies that 'dense', 'biased', and the return value form a 
# single producer-consumer chain.
@sme.jit
def fused_linear_layer(a, b, bias):
    # TensileSME fuses these three operations into a single
    # hardware loop nest, keeping the accumulator in the ZA tile.
    dense = sme.matmul(a, b)
    biased = dense + bias  # Implicit broadcasting
    return sme.relu(biased)

# 3. Execute
# Triggers compilation and executes on the SME hardware/emulator.
# Data is passed via zero-copy pointers.
C = fused_linear_layer(A, B, Bias)

print(f"Inference Complete. Result shape: {C.shape}")