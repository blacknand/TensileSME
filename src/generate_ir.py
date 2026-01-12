import sys
print(f"Python executable: {sys.executable}")
print(f"Path: {sys.path}\n\n")
from colorama import Fore
from mlir.ir import Context, Module, InsertionPoint, Location, RankedTensorType, F32Type, FunctionType, AffineMap, AffineDimExpr, UnitAttr
from mlir.dialects import func, linalg, arith, tensor

def build_matmul_ir(module):
    """
    Creates the 'matmul' function to multiplying 2 128 * 128 matricies.
    """
    f32 = F32Type.get()
    tensor_type = RankedTensorType.get([128, 128], f32)
    func_type = FunctionType.get(inputs=[tensor_type, tensor_type], results=[tensor_type])
    func_op = func.FuncOp(name="matmul", type=func_type)
    entry_block = func_op.add_entry_block()
    with InsertionPoint(entry_block):
        arg0 = entry_block.arguments[0]
        arg1 = entry_block.arguments[1]
        # prealloc empty tensor and zero out
        zero_op = arith.ConstantOp(f32, 0.0)
        zero_val = zero_op.results[0]
        empty_op = tensor.EmptyOp([128, 128], f32)
        empty_val = empty_op.results[0]
        filled_val = linalg.fill(zero_val, outs=[empty_val])
        # Fill result of matrix multiplication into filled_val tensor
        matmul_res = linalg.matmul(arg0, arg1, outs=[filled_val])
        func.ReturnOp([matmul_res])
    module.body.append(func_op)

def build_bias_add(module):
    """
    Creates the 'bias_add' function which adds a 1D vector (the bias)
    to every row of a 2D matrix.
    """
    f32 = F32Type.get()
    matrix_type = RankedTensorType.get([128, 128], f32)
    vector_type = RankedTensorType.get([128], f32)
    # Loop iterators representing i and j, respectively
    d0 = AffineDimExpr.get(0)
    d1 = AffineDimExpr.get(1)
    map_matrix = AffineMap.get(2, 0, [d0, d1]) # At loop (i, j), access the matrix at [i][j]: (d0, d1) -> (d0, d1)
    map_vector = AffineMap.get(2, 0, [d1])     # At loop (i, j), access the vector at [j]: (d0, d1) -> (d1)
    map_output = AffineMap.get(2, 0, [d0, d1]) # Write result to [i][j]: (d0, d1) -> (d0, d1)
    maps = [map_matrix, map_vector, map_output]
    # Use two parallel iterators here because neither iterator
    # needs to access data that is being used by the other iterator,
    # in other words, we have two independent loops
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
            # The addressing defined above will be used on the inputs and outputs
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

def build_linear_layer_ir(module):
    """
    Creates main function to perform the matrix multiplication
    and add the bias to the result.
    """
    f32 = F32Type.get()
    matrix_type = RankedTensorType.get([128, 128], f32)
    vector_type = RankedTensorType.get([128], f32)
    func_type = FunctionType.get(
        inputs=[matrix_type, matrix_type, vector_type],
        results=[matrix_type]
    )
    func_op = func.FuncOp(name="main", type=func_type)
    func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()        # Generate C compatiable wrapper
    entry_block = func_op.add_entry_block()
    with InsertionPoint(entry_block):
        A = entry_block.arguments[0]
        B = entry_block.arguments[1]
        Bias = entry_block.arguments[2]
        zero = arith.ConstantOp(f32, 0.0).results[0]
        empty_mat = tensor.EmptyOp([128, 128], f32)
        filled_mat = linalg.fill(zero, outs=[empty_mat])

        matmul_res = linalg.matmul(A, B, outs=[filled_mat])
        
        # Take matmul_res and add the Bias to it
        # Maps:
        #       - Input 1 (matrix): (d0, d1) -> (d0, d1)
        #       - Input 2 (Bias):   (d0, d1) -> (d1)
        #       - Output:           (d0, d1) -> (d0, d1)
        d0 = AffineDimExpr.get(0)
        d1 = AffineDimExpr.get(1)
        map_matrix = AffineMap.get(2, 0, [d0, d1])
        map_vector = AffineMap.get(2, 0, [d1])
        init_bias_out = tensor.EmptyOp([128, 128], f32).results[0]
        generic_op = linalg.GenericOp(
            result_tensors=[matrix_type],
            inputs=[matmul_res, Bias],
            outputs=[init_bias_out],
            indexing_maps=[map_matrix, map_vector, map_matrix],
            iterator_types=["parallel", "parallel"]
        )

        generic_block = generic_op.regions[0].blocks.append(f32, f32, f32)
        with InsertionPoint(generic_block):
            in_mat_val = generic_block.arguments[0]
            in_vec_val = generic_block.arguments[1]
            add_op = arith.AddFOp(in_mat_val, in_vec_val)
            zero_op = arith.ConstantOp(f32, 0.0)
            relu_res = arith.MaximumFOp(add_op.result, zero_op.result)
            linalg.YieldOp([relu_res.result])
        func.ReturnOp([generic_op.results[0]])
    module.body.append(func_op)

if __name__ == "__main__":
    with Context() as ctx, Location.unknown():
        module = Module.create()
        build_linear_layer_ir(module)

        if module.operation.verify():
            print(Fore.GREEN + "module successfully verified.")
            with open("module.mlir", "w") as f:
                 f.write(str(module))
        else:
            print(Fore.RED + "verification failed.")
            print(module)