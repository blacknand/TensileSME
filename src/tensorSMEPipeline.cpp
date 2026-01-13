#include "tensorSMEPipeline.h"

void TensorSMEPipeline::buildTensorSMEPipeline(mlir::OpPassManager &pm)
{
    // Nest pass manager that operates on func.func operates
    // nested directly under top-level module
    mlir::OpPassManager &funcModule = pm.nest<mlir::func::FuncOp>(); 
    // Add canonicalizer pass to func.func module
    // so that every func.func module will be canonicalized seprately
    funcModule.addPass(mlir::createCanonicalizerPass());
    funcModule.addPass(mlir::createCSEPass());

} 