#include "isq/Operations.h"
#include "isq/QStructs.h"
#include "isq/passes/Passes.h"
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
namespace isq{
namespace ir{
namespace passes{
const char* ISQ_FAMOUS_GATE = "isq_famous_gate";
struct RecognizeFamousGatePass : public mlir::PassWrapper<RecognizeFamousGatePass, mlir::OperationPass<DefgateOp>>{
    void markAs(DefgateOp op, const char* tag){
        op->setAttr(ISQ_FAMOUS_GATE, mlir::StringAttr::get(op->getContext(), tag));
    }
    void runOnOperation() override{
        auto op = this->getOperation();
        if(!op.definition()) return;
        for(auto attr: *op.definition()){
            auto def = attr.cast<GateDefinition>();
            if(def.type().strref()=="qir"){
                auto flat_symbol = def.value().cast<mlir::FlatSymbolRefAttr>();
#define FAMOUS(a,b) if(flat_symbol.getAttr().strref()==#a){ return markAs(op, #b); }
                FAMOUS(__quantum__qis__cnot, CNOT);
                FAMOUS(__quantum__qis__h__body, H);
                FAMOUS(__quantum__qis__s__body, S);
                FAMOUS(__quantum__qis__t__body, T);
                FAMOUS(__quantum__qis__x__body, X);
                FAMOUS(__quantum__qis__y__body, Y);
                FAMOUS(__quantum__qis__z__body, Z);
                FAMOUS(__quantum__qis__rx__body, Rx);
                FAMOUS(__quantum__qis__ry__body, Ry);
                FAMOUS(__quantum__qis__rz__body, Rz);
                FAMOUS(__quantum__qis__gphase, GPhase);
                FAMOUS(__quantum__qis__u3, U3);
            }
        }
    }
    mlir::StringRef getArgument() const final{
        return "isq-recognize-famous-gates";
    }
    mlir::StringRef getDescription() const final{
        return "Recognize famous quantum gates from their QIR description.";
    }
};

void registerRecognizeFamousGates(){
    mlir::PassRegistration<RecognizeFamousGatePass>();
}
}

}
}
