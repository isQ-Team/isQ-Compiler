#include "isq/contrib/Affine.h"
#include "isq/dialects/Extra.h"
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
    passes::registerRecognizeFamousGates();
    passes::registerSQRot2U3();
    passes::registerDecomposeCtrlU3();
    passes::registerRemoveTrivialSQGates();
    passes::registerTargetQCISSet();
    passes::registerRemoveGPhase();
    passes::registerEliminateNegCtrl();
    passes::registerISQCanonicalizer();
    passes::registerOracleDecompose();
    passes::registerAffineSWP();
    isq::contrib::mlir::registerAffineScalarReplacementPass();
    mlir::registerAllDialects(registry);
    registry.insert<isq::ir::ISQDialect>();
    registry.insert<isq::extra::ISQExtraDialect>();
}
} // namespace ir
} // namespace isq
