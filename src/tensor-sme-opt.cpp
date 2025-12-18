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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"));

int main(int argc, char** argv) {
    // python3 generate_ir.py > model.mlir
    // ./tensor-opt-sme model.mlir
    llvm::InitLLVM y(argc, argv);
    cl::ParseCommandLineOptions(argc, argv, "TensorSME compiler\n");
    // mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect>();
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    llvm::SourceMgr sourceMgr;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error parsing input file.\n";
        return -1;
    }

    if (failed(mlir::verify(*module))) {
        module->emitError("Module verification failed");
        return -1;
    }

    module->print(llvm::outs());
    return 0;
}