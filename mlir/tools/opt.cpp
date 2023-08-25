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
#include <nlohmann/json.hpp>


#define STR_(x) #x
#define STR(x) STR_(x)
static void PrintVersion(mlir::raw_ostream &OS) {
  OS << '\n';
  OS << "isQ IR Codegen " << STR(ISQ_BUILD_SEMVER) << '\n';
  OS << "Git revision: "<<STR(ISQ_BUILD_REV)<< ((STR(ISQ_BUILD_FROZEN)[0])=='1'?"":" (dirty)") << "\n";
  OS << "Build type: "<<STR(ISQ_OPT_BUILD_TYPE)<<"\n";
  OS << "Website: https://arclight-quantum.github.io/isQ-Compiler/\n";
}

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(
    cl::Positional, 
    cl::desc("<input file>"),
    cl::init("-"),
    cl::value_desc("filename")
);
static cl::opt<bool> formatOutput(
    "format-out",
    cl::desc("format output/error through json"),
    cl::init(false)
);
      
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


struct qLoc{
    std::string source_file;
    int line;
    int col;
};


nlohmann::json gen_err_info(qLoc loc, std::string tag, std::string msg){
    nlohmann::json err_info = {
        {"pos",{
            {"filename", loc.source_file},
            {"line", loc.line},
            {"column", loc.col}
        }},
        {"tag", tag},
        {"msg", msg}
    };
    return err_info;
}

int isq_mlir_codegen_main(int argc, char **argv) {
    llvm::cl::AddExtraVersionPrinter(PrintVersion);
    mlir::DialectRegistry registry;
    isq::ir::ISQToolsInitialize(registry);
    mlir::MLIRContext context(registry);

    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");
    cl::ParseCommandLineOptions(argc, argv, "isQ MLIR Dialect Codegen\n");
    
    mlir::DiagnosticEngine &engine = context.getDiagEngine();

    // Handle the reported diagnostic.
    // Return success to signal that the diagnostic has either been fully
    // processed, or failure if the diagnostic should be propagated to the
    // previous handlers.
    nlohmann::json err;
    err["Left"] = nlohmann::json::array();

    engine.registerHandler([&](mlir::Diagnostic &diag) -> mlir::LogicalResult {
        //std::cout << "Dumping Module after error.\n";
        if (diag.getSeverity() == mlir::DiagnosticSeverity::Error){

            mlir::FileLineColLoc flc = diag.getLocation().dyn_cast<mlir::FileLineColLoc>();
            qLoc loc = qLoc(flc.getFilename().strref().str(), flc.getLine(), flc.getColumn());
            
            nlohmann::json err_diag = gen_err_info(loc, "OptimizationError", diag.str());

            err["Left"].insert(err["Left"].end(), err_diag);
        }
        //std::cout << err.dump() << std::endl;
        bool should_propagate_diagnostic = true;
        return mlir::success(should_propagate_diagnostic);
    });

    mlir::PassManager pm(&context, mlir::OpPassManager::Nesting::Implicit);
    pm.enableVerifier(true);
    applyPassManagerCLOptions(pm);
    auto res = passPipeline.addToPipeline(pm, [&](const llvm::Twine &msg) {
        emitError(mlir::UnknownLoc::get(pm.getContext())) << msg;
        return mlir::failure();
    });

    if (mlir::failed(res)){
        llvm::outs() << err.dump();
        return 0;
    }

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code EC = fileOrErr.getError()) {
        nlohmann::json ec_err = gen_err_info(qLoc(inputFilename, 0, 0), "FileNotFound", EC.message());
        err["Left"].insert(err["Left"].end(), ec_err);
        llvm::outs() << err.dump();
        return 0;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::outs() << err.dump();
        //llvm::errs() << "Error: can't load file " << inputFilename << "\n";
        return 0;
    }

    auto module_op = module.get();
    if (mlir::failed(pm.run(module_op))){
        llvm::outs() << err.dump();
        return 0;
    }


    std::string s;
    llvm::raw_string_ostream os(s);

    if (emitBackend==None){
        module->print(os);
    }else if(emitBackend==OpenQASM3){
        if(failed(isq::ir::generateOpenQASM3Logic(context, module_op, os))){
            llvm::outs() << err.dump();
            return 0;
        }
    }else if (emitBackend==QCIS){
        if(failed(isq::ir::generateQCIS(context, module_op, os, printAst))){
            llvm::outs() << err.dump();
            return 0;
        }
    }else if (emitBackend==EQASM){
        if(failed(isq::ir::generateEQASM(context, module_op, os, printAst))){
            llvm::outs() << err.dump();
            return 0;
        }
    }else{
        nlohmann::json backend_err = gen_err_info(qLoc("", 0, 0), "BackendError", "Bad backend");
        err["Left"].insert(err["Left"].end(), backend_err);
        llvm::outs() << err.dump();
        return 0;
    }

    if (formatOutput){
        nlohmann::json out_json = {
            {"Right", s}
        };
        llvm::outs() << out_json.dump();
    }else{
        llvm::outs() << s;
    }
    return 0;
}

int main(int argc, char **argv) { 
    return isq_mlir_codegen_main(argc, argv); 
}