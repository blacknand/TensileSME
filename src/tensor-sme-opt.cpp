// Main driver
#include <memory>
#include <string>
#include <system_error>
#include <utility>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "tensorSMEPipeline.h"

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"));

int main(int argc, char** argv) {
    // Core LLVM setup
    llvm::InitLLVM y(argc, argv);
    cl::ParseCommandLineOptions(argc, argv, "TensorSME compiler\n");

    // Dialect Registry
    mlir::DialectRegistry registry;
    // mlir::registerAllPasses(registry);
    registry.insert<mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect,
                    mlir::tensor::TensorDialect,
                    mlir::affine::AffineDialect,
                    mlir::vector::VectorDialect,
                    mlir::scf::SCFDialect,
                    mlir::arm_sme::ArmSMEDialect>();

    // Create context
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    // Parse input file
    llvm::SourceMgr sourceMgr;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    // Parse MLIR file into a ModuleOp
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error parsing input file.\n";
        return -1;
    }

    // Build optimisation pipeline
    auto pm = mlir::PassManager::on<mlir::ModuleOp>(&context);
    pm.enableVerifier(true); 
    TensorSMEPipeline::buildTensorSMEPipeline(pm);

    // Print pipeline
    llvm::errs() << "Pipeline: ";
    pm.printAsTextualPipeline(llvm::errs());
    llvm::errs() << "\n";

    // Execute pipeline
    if (failed(pm.run(*module))) {
        llvm::errs() << "Pass pipeline failed\n";
        return -1;
    }

    llvm::outs() << "Pass pipeline succeeded\n";

    if (failed(mlir::verify(*module))) {
        module->emitError("Module verification failed");
        return -1;
    }

    module->print(llvm::outs());
    return 0;
}