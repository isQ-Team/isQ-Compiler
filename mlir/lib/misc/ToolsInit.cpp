#include <mlir/InitAllPasses.h>
#include <mlir/InitAllDialects.h>
#include <isq/IR.h>
#include <isq/passes/Passes.h>
namespace isq {
namespace ir {
void ISQToolsInitialize(mlir::DialectRegistry &registry) {
    mlir::registerAllPasses();
    passes::registerDecorateFolding();
    mlir::registerAllDialects(registry);
    registry.insert<isq::ir::ISQDialect>();
}
} // namespace ir
} // namespace isq