#include <cstdio>
#include <cassert>
#include <memory>
#include <mlir/IR/Types.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/Parser.h>
#include <mlir/Parser/AsmParserState.h>
#include <mlir/IR/DialectImplementation.h>
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/InitAllDialects.h>
#include "isq/Backends.h"
#include "isq/Dialect.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "llvm/Support/CommandLine.h"
#include <mlir/InitAllPasses.h>
#include <mlir/IR/BuiltinTypes.h>
#include <algorithm>
#include <isq/IR.h>
#include <llvm/Support/CommandLine.h>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input qir file>"),cl::init("-"),cl::value_desc("filename"));
namespace{
    enum BackendType {None, OpenQASM3Logic, QCIS};
}

static cl::opt<enum BackendType> emitBackend(
    "target", cl::desc("Choose the backend for code generation"),
    cl::values(clEnumValN(None, "none", "output the MLIR as-is")),
    cl::values(clEnumValN(OpenQASM3Logic, "openqasm3-logic", "generate (logic) OpenQASM3 program")),
    cl::values(clEnumValN(QCIS, "qcis", "generate qcis"))
);

static cl::opt<bool> printAst(
    "printast", cl::desc("print mlir ast."));



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
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    if(emitBackend==None){
        module->print(llvm::outs());
    }else if(emitBackend==OpenQASM3Logic){
        auto module_op = module.get();
        if(failed(isq::ir::generateOpenQASM3Logic(context, module_op, llvm::outs()))){
            llvm::errs() << "Generate OpenQASM3 failed.\n";
            return -2;
        }
    }else if (emitBackend==QCIS){
        auto module_op = module.get();
        if(failed(isq::ir::generateQCIS(context, module_op, llvm::outs(), printAst))){
            llvm::errs() << "Generate QCIS failed.\n";
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