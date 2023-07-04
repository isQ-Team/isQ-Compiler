#include <cstdio>
#include <cassert>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <memory>

#include "isq/Dialect.h"
#include <isq/IR.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"


namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input isq file>"),cl::init("-"),cl::value_desc("filename"));


#define STR_(x) #x
#define STR(x) STR_(x)
static void PrintVersion(mlir::raw_ostream &OS) {
  OS << '\n';
  OS << "isQ IR Optimizer " << STR(ISQ_BUILD_SEMVER) << '\n';
  OS << "Git revision: "<<STR(ISQ_BUILD_REV)<< ((STR(ISQ_BUILD_FROZEN)[0])=='1'?"":" (dirty)") << "\n";
  OS << "Build type: "<<STR(ISQ_OPT_BUILD_TYPE)<<"\n";
  OS << "Website: https://arclight-quantum.github.io/isQ-Compiler/\n";
}


int isq_mlir_opt_main(int argc, char **argv) {
    llvm::cl::AddExtraVersionPrinter(PrintVersion);
    mlir::DialectRegistry registry;
    isq::ir::ISQToolsInitialize(registry);
    return mlir::asMainReturnCode(mlir::MlirOptMain(
        argc, argv, "MLIR modular optimizer driver for ISQ dialect\n", registry,
        true));
}


int main(int argc, char **argv) { 
    return isq_mlir_opt_main(argc, argv); 
}