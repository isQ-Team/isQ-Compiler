#include <cstdio>
#include <cassert>
#include <mlir/IR/Types.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/Parser.h>
#include <mlir/Parser/AsmParserState.h>
#include <mlir/IR/DialectImplementation.h>
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/InitAllDialects.h>
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include <mlir/InitAllPasses.h>
#include <mlir/IR/BuiltinTypes.h>
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include <algorithm>

#include <isq/IR.h>

int isq_mlir_lsp_server_main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    isq::ir::ISQToolsInitialize(registry);
    return failed(mlir::MlirLspServerMain(argc, argv, registry));
}

int main(int argc, char **argv) { return isq_mlir_lsp_server_main(argc, argv); }