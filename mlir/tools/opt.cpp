#include <cstdio>
#include <cassert>
#include <memory>

#include "isq/Dialect.h"
#include <isq/IR.h>

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

namespace{
    enum BackendType {None, OpenQASM3, QCIS, EQASM};
}

static cl::opt<enum BackendType> emitBackend(
    "target", cl::desc("Choose the backend for code generation"),
    cl::values(clEnumValN(None, "none", "output the MLIR as-is")),
    cl::values(clEnumValN(OpenQASM3, "openqasm3", "generate (logic) OpenQASM3 program")),
    cl::values(clEnumValN(QCIS, "qcis", "generate qcis")),
    cl::values(clEnumValN(EQASM, "eqasm", "generate eqasm"))
);

static cl::opt<bool> printAst(
    "printast", cl::desc("print mlir ast."));

int isq_mlir_codegen_main(int argc, char **argv) {
    
    mlir::DialectRegistry registry;
    isq::ir::ISQToolsInitialize(registry);
    mlir::MLIRContext context(registry);

    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");
    cl::ParseCommandLineOptions(argc, argv, "isQ MLIR Dialect Codegen\n");
    
    mlir::PassManager pm(&context, mlir::OpPassManager::Nesting::Implicit);
    pm.enableVerifier(true);
    applyPassManagerCLOptions(pm);
    auto res = passPipeline.addToPipeline(pm, [&](const llvm::Twine &msg) {
        emitError(mlir::UnknownLoc::get(pm.getContext())) << msg;
        return mlir::failure();
    });
    if (mlir::failed(res)){
        llvm::errs() << "init pass error\n";
        return -1;
    }

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        return -1;
    }
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error: can't load file " << inputFilename << "\n";
        return 3;
    }

    auto module_op = module.get();
    if (mlir::failed(pm.run(module_op))){
        llvm::errs() << "Error: Lower\n";
        return -1;
    }

    if(emitBackend==None){
        module->print(llvm::outs());
    }else if(emitBackend==OpenQASM3){
        if(failed(isq::ir::generateOpenQASM3Logic(context, module_op, llvm::outs()))){
            llvm::errs() << "Error: Generate OpenQASM3 failed.\n";
            return -2;
        }
    }else if (emitBackend==QCIS){
        if(failed(isq::ir::generateQCIS(context, module_op, llvm::outs(), printAst))){
            llvm::errs() << "Error: Generate QCIS failed.\n";
            return -2;
        }
    }else if (emitBackend==EQASM){
        if(failed(isq::ir::generateEQASM(context, module_op, llvm::outs(), printAst))){
            llvm::errs() << "Error: Generate EQASM failed.\n";
            return -2;
        }
    }else{
        llvm::errs() << "Bad backend.\n";
        return -1;
    }
    
    return 0;
}

int main(int argc, char **argv) { 
    return isq_mlir_codegen_main(argc, argv); 
}