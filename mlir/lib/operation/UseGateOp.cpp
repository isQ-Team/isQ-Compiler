#include <isq/Operations.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Casting.h>
#include <mlir/Support/LLVM.h>
namespace isq {
namespace ir {
::mlir::LogicalResult
UseGateOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
    auto symbol_def = symbolTable.lookupNearestSymbolFrom(*this, this->name());
    if (auto gatedef = llvm::dyn_cast_or_null<DefgateOp>(symbol_def)) {
        
        if (this->getResult().getType() != gatedef.getTypeWhenUsed()) {
            this->emitOpError()
                << "type mismatch, expected " << gatedef.getTypeWhenUsed();
            return mlir::failure();
        }
        /*
        if(gatedef.parameters()){
            auto params = *gatedef.parameters();
            ::mlir::SmallVector<::mlir::Type> curr_params_list;
            for(auto param : this->args().getTypes()){
                curr_params_list.push_back(param);
            }
            ::mlir::SmallVector<::mlir::Type> expected_params_list;
            for(auto attr: params){
                auto param_type = attr.cast<::mlir::TypeAttr>().getValue();
                expected_params_list.push_back(param_type);
            }
            
            if (curr_params_list!=expected_params_list) {
                this->emitOpError()
                << "type mismatch, expected " << params;
                return mlir::failure();
            }
        }*/
        
        return mlir::success();
    }
    this->emitOpError() << "symbol `" << this->name()
                        << "` not found or has wrong type";
    return mlir::failure();
}


/*
mlir::LogicalResult verify(GateOp op) {
if (op.gate_type() != op.getResult().getType()) {
    op.emitOpError("gate dimension or trait mismatch.");
    return mlir::failure();
}
return mlir::success();
}
*/
} // namespace ir
} // namespace isq