#include <mlir/InitAllPasses.h>
#include <mlir/InitAllDialects.h>
#include <isq/IR.h>
#include <isq/passes/Passes.h>
namespace isq {
namespace ir {
void ISQToolsInitialize(mlir::DialectRegistry &registry) {
    mlir::registerAllPasses();
    passes::registerDecorateFolding();
    passes::registerQSD();
    passes::registerExpandDecomposition();
    passes::registerLowerToQIRRep();
    passes::registerQIR2LLVM();
    passes::registerPureGateDetect();
    mlir::registerAllDialects(registry);
    registry.insert<isq::ir::ISQDialect>();
}
} // namespace ir
} // namespace isq