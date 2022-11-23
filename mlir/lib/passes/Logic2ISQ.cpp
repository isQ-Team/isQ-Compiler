#include <string>
#include <unordered_map>

#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "mockturtle/algorithms/simulation.hpp"
#include "mockturtle/networks/xag.hpp"

#include "logic/IR.h"
#include "isq/IR.h"

namespace isq {
namespace ir {
namespace passes {
namespace {

template<typename Ty>
std::ostream& operator<<(std::ostream& os, const std::vector<Ty>& v) {
    os << '[';
    for (int i=0; i<v.size(); i++) {
        os << v[i] << ", ";
    }
    os << ']';
    return os;
}

class RuleReplaceLogicFunc : public mlir::OpRewritePattern<logic::ir::FuncOp> {
public:
    RuleReplaceLogicFunc(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::FuncOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::FuncOp op, mlir::PatternRewriter &rewriter) const override {

        // Build the XAG.
        mockturtle::xag_network xag; // the xag to be built
        auto hash = [](const std::pair<std::string, int> &p){
            return std::hash<std::string>()(p.first) * 31 + std::hash<int>()(p.second);
        };
        std::unordered_map<std::pair<std::string, int>, mockturtle::xag_network::signal, decltype(hash)> symbol_table(8, hash);

        // A helper function that get the SSA identifier linked to a value
        mlir::AsmState state(op);
        auto value2str = [&](mlir::Value value) -> std::string {
            std::string res;
            llvm::raw_string_ostream output(res);
            value.printAsOperand(output, state);
            return res;
        };

        // Create input signals based on the function inputs.
        unsigned int input_num = op.getNumArguments();
        for (int i=0; i<input_num; i++) {
            mlir::Value arg = op.getArgument(i);
            std::string str = value2str(arg);
            int width = getBitWidth(arg);
            for (int j=0; j<width; j++) {
                symbol_table[{str, j}] = xag.create_pi();
            }
        }

        // Binary operator processing template
        auto vec_binary = [&](mlir::Value lhs, mlir::Value rhs, mlir::Value res,
            mockturtle::xag_network::signal(mockturtle::xag_network::*create)(mockturtle::xag_network::signal, mockturtle::xag_network::signal)) {
            std::string lname = value2str(lhs);
            std::string rname = value2str(rhs);
            std::string res_name = value2str(res);
            int width = getBitWidth(res);
            for (int j=0; j<width; j++) {
                symbol_table[{res_name, j}] = (xag.*create)(symbol_table[{lname, j}], symbol_table[{rname, j}]);
            }
        };

        // Process each statement in the funciton body.
        for (mlir::Operation &it : op.getRegion().getOps()) {
            if (logic::ir::AndvOp binop = llvm::dyn_cast<logic::ir::AndvOp>(it)) {
                vec_binary(binop.lhs(), binop.rhs(), binop.result(), &mockturtle::xag_network::create_and);
            }
            else if (logic::ir::OrvOp binop = llvm::dyn_cast<logic::ir::OrvOp>(it)) {
                vec_binary(binop.lhs(), binop.rhs(), binop.result(), &mockturtle::xag_network::create_or_no_const);
            }
            else if (logic::ir::XorvOp binop = llvm::dyn_cast<logic::ir::XorvOp>(it)) {
                vec_binary(binop.lhs(), binop.rhs(), binop.result(), &mockturtle::xag_network::create_xor);
            }
            else if (logic::ir::ReturnOp ret = llvm::dyn_cast<logic::ir::ReturnOp>(it)) {
                mlir::Value oprand = ret.getOperand(0);
                std::string str = value2str(oprand);
                int width = getBitWidth(oprand);
                for (int i=0; i<width; i++) {
                    mockturtle::xag_network::signal sig = symbol_table[{str, i}];
                    xag.create_po(sig);
                }
            } else {
                std::string str;
                llvm::raw_string_ostream output(str);
                it.getName().print(output);
                std::cout << str << std::endl;
                assert(0 && "unrecognized operation");
            }
        }
        //mockturtle::xag_network::signal res = xag.create_and(pi, pi2);
        std::vector<bool> results = mockturtle::simulate<bool>(xag, mockturtle::default_simulator<bool>(std::vector<bool>(6, false)));
        std::cout << results << std::endl;
        results = mockturtle::simulate<bool>(xag, mockturtle::default_simulator<bool>(std::vector<bool>{true, true, true, false, false, false}));
        std::cout << results << std::endl;
        results = mockturtle::simulate<bool>(xag, mockturtle::default_simulator<bool>(std::vector<bool>(6, true)));
        std::cout << results << std::endl;
        //mockturtle::write_dot(xag, std::cout);


        // Convert XAG to quantum circuit
        // To be added...


        // Below is a demonstration of creating a quantum circuit with isQ dialect.
        // The circuit is
        // in0[0] ---X--*--X---
        // in0[1] ------|------
        //    .   ------|------
        //    .   ------|------
        //    .   ------|------
        // out[0] ------X------
        // out[1] -------------
        //    .   -------------
        //    .   -------------
        //    .   -------------
        // i.e., X(in0[0]); CNOT(in0[0], out[0]); X(in0[0]);

        mlir::MLIRContext *ctx = op.getContext();
        mlir::Location loc = op.getLoc(); // The location of the oracle function in the source code.

        // Construct function signature.
        mlir::SmallVector<::mlir::Type> argtypes;
        mlir::SmallVector<::mlir::Type> returntypes;
        isq::ir::QStateType qstate = isq::ir::QStateType::get(ctx);
        mlir::MemRefType memref_3_qstate = mlir::MemRefType::get(mlir::ArrayRef<int64_t>{3}, qstate);
        for(int i=0; i<=input_num; i++){
            argtypes.push_back(memref_3_qstate);
        }
        mlir::FunctionType functype = mlir::FunctionType::get(ctx, argtypes, returntypes);

        // Create a FuncOp that represent the quantum circuit.
        mlir::FuncOp funcop = rewriter.create<mlir::FuncOp>(loc, op.sym_name(), functype);
        mlir::Block *entry_block = funcop.addEntryBlock(); // Arguments are automatically created based on the function signature.
        mlir::OpBuilder builder(entry_block, entry_block->begin());
        mlir::BlockArgument in0 = entry_block->getArgument(0);
        mlir::BlockArgument out = entry_block->getArgument(input_num);

        // Fetch the original qstates.
        mlir::arith::ConstantIndexOp const0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::memref::LoadOp in00 = builder.create<mlir::memref::LoadOp>(loc, qstate, in0, mlir::ValueRange{const0});
        mlir::memref::LoadOp out0 = builder.create<mlir::memref::LoadOp>(loc, qstate, out, mlir::ValueRange{const0});

        // Load the quantum gates. The last argument is the parameters of the gate, e.g., `theta` for Rz(theta, q);
        mlir::Value x_gate = builder.create<isq::ir::UseGateOp>(loc, isq::ir::GateType::get(ctx, 1, GateTrait::General),
            mlir::FlatSymbolRefAttr::get(ctx, "X"), mlir::ValueRange{}).getResult();
        mlir::Value cnot_gate = builder.create<isq::ir::UseGateOp>(loc, isq::ir::GateType::get(ctx, 2, GateTrait::General),
            mlir::FlatSymbolRefAttr::get(ctx, "CNOT"), mlir::ValueRange{}).getResult();

        // Apply gates to qstates. The last argument is the qstates to be applied on.
        isq::ir::ApplyGateOp applied_x = builder.create<isq::ir::ApplyGateOp>(loc, mlir::ArrayRef<mlir::Type>{qstate},
            x_gate, mlir::ArrayRef<mlir::Value>({in00}));
        isq::ir::ApplyGateOp applied_cnot = builder.create<isq::ir::ApplyGateOp>(loc, mlir::ArrayRef<mlir::Type>{qstate, qstate},
            cnot_gate, mlir::ArrayRef<mlir::Value>({applied_x.getResult(0), out0}));
        isq::ir::ApplyGateOp applied_x2 = builder.create<isq::ir::ApplyGateOp>(loc, mlir::ArrayRef<mlir::Type>{qstate},
            x_gate, mlir::ArrayRef<mlir::Value>({applied_cnot.getResult(0)}));

        // Store qstates back to registers (i.e., the Memref<!isq.qstate> struct).
        builder.create<mlir::memref::StoreOp>(loc, applied_x2.getResult(0), in0, mlir::ValueRange{const0});
        builder.create<mlir::memref::StoreOp>(loc, applied_cnot.getResult(1), out, mlir::ValueRange{const0});
        builder.create<mlir::ReturnOp>(loc); // dummy terminator

        rewriter.eraseOp(op); // Remove original logic.func op
        return mlir::success();
    }
private:
    int getBitWidth(mlir::Value val) const {
        auto mem_type = val.getType().dyn_cast<mlir::MemRefType>();
        assert(mem_type && "Returned value is not of MemRefType");
        return mem_type.getDimSize(0);
    }
};

class RuleReplaceLogicCall : public mlir::OpRewritePattern<logic::ir::CallOp> {
public:
    RuleReplaceLogicCall(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::CallOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::CallOp op, mlir::PatternRewriter &rewriter) const override {
        auto ctx = op.getContext();
        rewriter.create<mlir::CallOp>(mlir::UnknownLoc::get(ctx), op.callee(), (mlir::TypeRange){}, op.operands());

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

class RuleReplaceLogicReturn : public mlir::OpRewritePattern<logic::ir::ReturnOp> {
public:
    RuleReplaceLogicReturn(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::ReturnOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::ReturnOp op, mlir::PatternRewriter &rewriter) const override {
        auto ctx = op.getContext();
        rewriter.create<mlir::ReturnOp>(mlir::UnknownLoc::get(ctx));

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

class RuleReplaceLogicApply : public mlir::OpRewritePattern<logic::ir::ApplyGateOp> {
public:
    RuleReplaceLogicApply(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::ApplyGateOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::ApplyGateOp op, mlir::PatternRewriter &rewriter) const override {
        auto ctx = op.getContext();
        auto qst = QStateType::get(rewriter.getContext());
        mlir::SmallVector<mlir::Type> types;
        types.push_back(qst);
        rewriter.replaceOpWithNewOp<isq::ir::ApplyGateOp, mlir::ArrayRef<mlir::Type>, ::mlir::Value, ::mlir::ValueRange>(op, types, op.gate(), op.args());
        return mlir::success();
    }
};

class RuleReplaceLogicUse : public mlir::OpRewritePattern<logic::ir::UseGateOp> {
public:
    RuleReplaceLogicUse(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::UseGateOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::UseGateOp op, mlir::PatternRewriter &rewriter) const override {
        auto ctx = op.getContext();
        //auto gate_type = ;
        //rewriter.create<isq::ir::UseGateOp>(mlir::UnknownLoc::get(ctx), op.result(), op.name(), op.parameters());
        rewriter.replaceOpWithNewOp<isq::ir::UseGateOp, isq::ir::GateType, ::mlir::SymbolRefAttr, ::mlir::ValueRange>(op, isq::ir::GateType::get(ctx, 1, GateTrait::General), op.name(), op.parameters());
        return mlir::success();
    }
};

struct LogicToISQPass : public mlir::PassWrapper<LogicToISQPass, mlir::OperationPass<mlir::ModuleOp>> {
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();

        mlir::RewritePatternSet rps(ctx);
        rps.add<RuleReplaceLogicReturn>(ctx);
        rps.add<RuleReplaceLogicFunc>(ctx);
        rps.add<RuleReplaceLogicApply>(ctx);
        rps.add<RuleReplaceLogicUse>(ctx);
        mlir::FrozenRewritePatternSet frps(std::move(rps));
        (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);

        mlir::RewritePatternSet rps2(ctx);
        rps2.add<RuleReplaceLogicCall>(ctx);
        mlir::FrozenRewritePatternSet frps2(std::move(rps2));
        (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps2);
    }
    mlir::StringRef getArgument() const final {
        return "logic-lower-to-isq";
    }
    mlir::StringRef getDescription() const final {
        return  "Generate iSQ gate based on logic oracle specification.";
    }
};

}

void registerLogicToISQ() {
    mlir::PassRegistration<LogicToISQPass>();
}

}
}
}