#include "isq/contrib/Affine.h"
#include "isq/dialects/Extra.h"
#include <mlir/InitAllPasses.h>
#include <mlir/InitAllDialects.h>
#include <isq/IR.h>
#include <isq/passes/Passes.h>
#include "logic/IR.h"
namespace isq {
namespace ir {
void ISQToolsInitialize(mlir::DialectRegistry &registry) {
    mlir::registerAllPasses();
    passes::registerLogicToISQ();
    passes::registerStatePreparation();
    passes::registerDecorateFolding();
    passes::registerQSD();
    passes::registerExpandDecomposition();
    passes::registerLowerToQIRRep();
    passes::registerLowerSwitchOp();
    passes::registerQIR2LLVM();
    passes::registerPureGateDetect();
    passes::registerRecognizeFamousGates();
    passes::registerSQRot2U3();
    passes::registerDecomposeCtrlU3();
    passes::registerRemoveTrivialSQGates();
    passes::registerTargetQCISSet();
    passes::registerRemoveGPhase();
    passes::registerRemoveReset();
    passes::registerEliminateNegCtrl();
    passes::registerISQCanonicalizer();
    passes::registerOracleDecompose();
    passes::registerAffineSWP();
    passes::registerGlobalThreadLocal();
    passes::registerReuseQubit();
    passes::registerRedundant();
    isq::contrib::mlir::registerAffineScalarReplacementPass();
    mlir::registerAllDialects(registry);
    registry.insert<isq::ir::ISQDialect>();
    registry.insert<isq::extra::ISQExtraDialect>();
    registry.insert<logic::ir::LogicDialect>();
}
} // namespace ir
} // namespace isq
