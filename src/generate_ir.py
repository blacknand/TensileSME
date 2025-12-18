# Frontend for Python matmul operations

import sys
print(f"Python executable: {sys.executable}")
print(f"Path: {sys.path}\n\n")
import numpy as np
from mlir.ir import Context, Module, InsertionPoint, Location, RankedTensorType, F32Type, FunctionType
from mlir.dialects import func, linalg, arith, tensor

"""
The following is an example of what TensorSME generates,
build_matmul_ir then uses the MLIR API to generate the MLIR
to create the matmul_py subroutine.
"""
def matmul_py(matrix1, matrix2):
    mat1 = np.random.rand(128, 128).astype(np.float32)
    mat2 = np.random.rand(128, 128).astype(np.float32)
    res = 0
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            res += (mat1[i][j] * mat2[i][j])
    return res

def save_mlir(module, filename):
    with open(filename, "w") as f:
        f.write(str(module))

def build_matmul_ir():
    with Context() as _, Location.unknown():
        module = Module.create() 
        f32 = F32Type.get()
        tensor_type = RankedTensorType.get([128, 128], f32)
        func_type = FunctionType.get(inputs=[tensor_type, tensor_type], results=[tensor_type])
        func_op = func.FuncOp(name="matmul", type=func_type)
        entry_block = func_op.add_entry_block()
        with InsertionPoint(entry_block):
            arg0 = entry_block.arguments[0]
            arg1 = entry_block.arguments[1]
            zero_op = arith.ConstantOp(f32, 0.0)
            zero_val = zero_op.results[0]
            empty_op = tensor.EmptyOp([128, 128], f32)
            empty_val = empty_op.results[0]
            filled_val = linalg.fill(zero_val, outs=[empty_val])
            matmul_res = linalg.matmul(arg0, arg1, outs=[filled_val])
            func.ReturnOp([matmul_res])
        module.body.append(func_op)

        try:
            module.operation.verify()
            print("matmul verified successfully.")
            save_mlir(module, "module.mlir")
        except Exception as e:
            print("matmul verification unsuccessful.")
            print(e)
            print(module)

def build_bias_add():
    with Context() as _, Location.unkown():
        module = Module.create()
        func_type = None
        func_op = func.FuncOp(name="biasadd", type=func_type)
        entry_block = func_op.add_entry_block()
        with InsertionPoint(entry_block):

            func.ReturnOp([])
        module.body.append(func_op)

        try:
            module.operation.verify()
            print("biasadd verified successfully.")
        except Exception as e:
            print("biasadd verification unsuccessful.")
            print(e)
            print(module)

if __name__ == "__main__":
    build_matmul_ir()