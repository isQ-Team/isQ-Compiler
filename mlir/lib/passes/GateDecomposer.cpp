#include "isq/QTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include <Eigen/Dense>
#include <isq/IR.h>
#include <string>
/*

class GateDecomposePattern{
    mlir::OpBuilder builder;
    mlir::ModuleOp me;
    GateDecomposePattern(mlir::ModuleOp mod): me(mod), builder(mod){
        // nope.
    }
    static std::string decomposedName(mlir::StringRef decompositionName, mlir::StringRef decomposedGate){
        std::string name = "__isq_decomposition_";
        name += decompositionName.str();
        name += "_";
        name += decomposedGate.str();
        return name;
    }
    mlir::OpBuilder createDecompositionFunction(mlir::StringRef decompositionName, mlir::StringRef decomposedGate, mlir::ModuleOp mod, size_t size){
        builder.setInsertionPointToEnd(mod->getBlock());
        auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(), decomposedName(decompositionName, decomposedGate));
        auto memref = mlir::MemRefType::getChecked(builder.getUnknownLoc(), size, isq::ir::QStateType::get(mod->getContext()));
        func.setType(mlir::FunctionType::get(mod.getContext(), memref, {}));
        builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
        return builder;
    }
    virtual mlir::LogicalResult checkDecompose(){
        return mlir::failure();
    }
    virtual void decompose(mlir::OpBuilder& ){

    }
};
*/