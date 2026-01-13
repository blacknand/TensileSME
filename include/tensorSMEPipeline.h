#ifndef TENSOR_SME_PIPELINE_H
#define TENSOR_SME_PIPELINE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

class TensorSMEPipeline {
public:
    void static buildTensorSMEPipeline(mlir::OpPassManager &pm);
};

#endif //TENSOR_SME_PIPELINE_H 