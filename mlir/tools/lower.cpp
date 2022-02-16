#include <cstdio>
#include <cassert>
#include <memory>
#include <isq/IR.h>


#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include <mlir/InitAllDialects.h>
#include "mlir/Support/MlirOptMain.h"
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input qir file>"),cl::init("-"),cl::value_desc("filename"));
static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));


enum Action { None, DumpMLIR, DumpMLIRLLVM, DumpLLVMIR };

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR LLVM Dialect dump")));


int isq_mlir_codegen_main(int argc, char **argv) {
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "isQ MLIR Dialect Codegen\n");
    mlir::DialectRegistry registry;
    isq::ir::ISQToolsInitialize(registry);
    mlir::MLIRContext context(registry);
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        return -1;
    }
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningModuleRef module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    auto module_op = module.get();

    std::string errorMessage;
    auto output = mlir::openOutputFile(outputFilename, &errorMessage);
    if (!output) {
        llvm::errs() << errorMessage << "\n";
        return 1;
    }

    if (emitAction == Action::DumpMLIR){
        module_op->print(output->os());
        output->keep();
        return 0;
    }
    
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);
    pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    pm.addPass(mlir::isqLower::createLowerToLLVMPass());
    
    if (mlir::failed(pm.run(module_op))){
        llvm::errs() << "lower to mlir-llvm error\n";
        return -1;
    }
    
    if (emitAction == Action::DumpMLIRLLVM){
        module_op->print(output->os());
        output->keep();
        return 0;
    }
    

    mlir::registerLLVMDialectTranslation(*module_op->getContext());

    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module_op, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    /// Optionally run an optimization pipeline over the llvm module.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
        llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }

    if (emitAction == Action::DumpLLVMIR){
        output->os() << *llvmModule << "\n";
        output->keep();
        return 0;
    }
    
    return 0;
}

int main(int argc, char **argv) { 
    return isq_mlir_codegen_main(argc, argv); 
}