import sys
print(f"Python executable: {sys.executable}")
print(f"Path: {sys.path}\n\n")
import numpy as np
from colorama import Fore
from mlir.ir import Context, Module, InsertionPoint, Location, RankedTensorType, F32Type, FunctionType, AffineMap, AffineDimExpr
from mlir.dialects import func, linalg, arith, tensor

def build_matmul_ir(module):
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

def build_bias_add(module):
    f32 = F32Type.get()
    matrix_type = RankedTensorType.get([128, 128], f32)
    vector_type = RankedTensorType.get([128], f32)
    d0 = AffineDimExpr.get(0)
    d1 = AffineDimExpr.get(1)
    map_matrix = AffineMap.get(2, 0, [d0, d1])
    map_vector = AffineMap.get(2, 0, [d1])
    map_output = AffineMap.get(2, 0, [d0, d1])
    maps = [map_matrix, map_vector, map_output]
    iterators = ["parallel", "parallel"]
    func_type = FunctionType.get(inputs=[matrix_type, vector_type], results=[matrix_type])
    func_op = func.FuncOp(name="bias_add", type=func_type)
    entry_block = func_op.add_entry_block()
    with InsertionPoint(entry_block):
        matrix_arg0 = entry_block.arguments[0]
        vector_arg1 = entry_block.arguments[1]
        init_tensor = tensor.EmptyOp([128, 128], f32)
        generic_op = linalg.GenericOp(
            result_tensors=[matrix_type],
            inputs=[matrix_arg0, vector_arg1],
            outputs=[init_tensor],
            indexing_maps=maps,
            iterator_types=iterators
        )
        generic_block = generic_op.regions[0].blocks.append(f32, f32, f32)
        with InsertionPoint(generic_block):
            m_val = generic_block.arguments[0]
            v_val = generic_block.arguments[1]
            m_v_add_op = arith.AddFOp(m_val, v_val) 
            linalg.YieldOp([m_v_add_op.result])
        func.ReturnOp([generic_op.results[0]])
    module.body.append(func_op)

if __name__ == "__main__":
    with Context() as ctx, Location.unknown():
        module = Module.create()
        build_matmul_ir(module)
        build_bias_add(module)
        if module.operation.verify():
            print(Fore.GREEN + "module successfully verified.")
            with open("module.mlir", "w") as f:
                 f.write(str(module))
        else:
            print(Fore.RED + "verification failed.")
            print(module)