# # Root CMake
cmake_minimum_required(VERSION 3.20.0)
project(TensileSME)

# TODO: Dynamically find LLVM clone
set(LLVM_BUILD_DIR "/Users/nathanblackburn/Programming/TensileSME/llvm-project/build")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${LLVM_BUILD_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_BUILD_DIR}/lib/cmake/llvm")
list(APPEND CMAKE_MODULE_PATH "${LLVM_BUILD_DIR}/lib/cmake/mlir")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

include(MLIRConfig)
include(AddMLIR)
include(AddLLVM)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(TOOL_NAME tensor-sme-opt)
# add_executable(${TOOL_NAME} src/tensor-sme-opt.cpp)
add_llvm_executable(${TOOL_NAME} src/tensor-sme-opt.cpp)

target_link_libraries(${TOOL_NAME} PRIVATE
    # Dialects used in C++
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRArithDialect 
    MLIRAffineDialect
    MLIRTensorDialect
    MLIRVectorDialect
    MLIRSCFDialect
    MLIRArmSMEDialect
    
    # Core Infrastructure
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRPass
    MLIROptLib
    
    # Execution Engine / Runner Utils (Useful for later)
    # MLIRExecutionEngine
    LLVMSupport
)

message(STATUS "TensorSME configued successfuly.")
message(STATUS "LLVM build at: ${LLVM_BUILD_DIR}")