#include <mlir/InitAllPasses.h>
#include <mlir/InitAllDialects.h>
#include <isq/IR.h>
namespace isq {
namespace ir {
void ISQToolsInitialize(mlir::DialectRegistry &registry) {
    mlir::registerAllPasses();
    mlir::registerAllDialects(registry);
    registry.insert<isq::ir::ISQDialect>();
}
} // namespace ir
} // namespace isq