#include <llvm/ADT/Hashing.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "mockturtle/algorithms/simulation.hpp"
#include "mockturtle/networks/xag.hpp"
#include "isq/passes/LHRSOracle.hpp"
#include "caterpillar/synthesis/strategies/bennett_mapping_strategy.hpp"
#include "caterpillar/synthesis/strategies/eager_mapping_strategy.hpp"
#include "isq/passes/GreedyPebbling.hpp"
#include "mockturtle/algorithms/lut_mapping.hpp"
#include "mockturtle/algorithms/collapse_mapped.hpp"
#include "mockturtle/networks/klut.hpp"
#include "mockturtle/views/mapping_view.hpp"
#undef X // ghack.hpp fix
#include "logic/IR.h"
#include "isq/IR.h"

namespace isq {
namespace ir {
namespace passes {
namespace logic_to_isq{

// debug helper function
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
            return std::to_string((size_t)llvm::hash_value(value.getImpl()));
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
        auto binary = [&]<typename T>(mlir::Value lhs, mlir::Value rhs, mlir::Value res,
            T create) {
            std::string lname = value2str(lhs);
            std::string rname = value2str(rhs);
            std::string res_name = value2str(res);
            symbol_table[{res_name, -1}] = (xag.*create)(symbol_table.at({lname, -1}), symbol_table.at({rname, -1}));
        };

        // Binary vector operator processing template
        auto vec_binary = [&]<typename T>(mlir::Value lhs, mlir::Value rhs, mlir::Value res,
            T create) {
            std::string lname = value2str(lhs);
            std::string rname = value2str(rhs);
            std::string res_name = value2str(res);
            int width = getBitWidth(res);
            for (int j=0; j<width; j++) {
                symbol_table[{res_name, j}] = (xag.*create)(symbol_table.at({lname, j}), symbol_table.at({rname, j}));
            }
        };
        

        // Process each statement in the funciton body.
        for (mlir::Operation &it : op.getRegion().getOps()) {
            if (logic::ir::NotOp notop = llvm::dyn_cast<logic::ir::NotOp>(it)) {
                std::string operand = value2str(notop.getOperand());
                std::string res = value2str(notop.getResult());
                symbol_table[{res, -1}] = xag.create_not(symbol_table.at({operand, -1}));
            }
            else if (logic::ir::NotvOp notvop = llvm::dyn_cast<logic::ir::NotvOp>(it)) {
                std::string operand = value2str(notvop.getOperand());
                mlir::Value result = notvop.getResult();
                std::string res = value2str(result);
                int width = getBitWidth(result);
                for (int i=0; i<width; i++) {
                    symbol_table[{res, i}] = xag.create_not(symbol_table.at({operand, i}));
                }
            }
            else if (logic::ir::AndOp binop = llvm::dyn_cast<logic::ir::AndOp>(it)) {
                binary(binop.getLhs(), binop.getRhs(), binop.getResult(), &mockturtle::xag_network::create_and);
            }
            else if (logic::ir::OrOp binop = llvm::dyn_cast<logic::ir::OrOp>(it)) {
                binary(binop.getLhs(), binop.getRhs(), binop.getResult(), &mockturtle::xag_network::create_or);
            }
            else if (logic::ir::XorOp binop = llvm::dyn_cast<logic::ir::XorOp>(it)) {
                binary(binop.getLhs(), binop.getRhs(), binop.getResult(), &mockturtle::xag_network::create_xor);
            }
            else if (logic::ir::XnorOp binop = llvm::dyn_cast<logic::ir::XnorOp>(it)) {
                binary(binop.getLhs(), binop.getRhs(), binop.getResult(), &mockturtle::xag_network::create_xnor);
            }
            else if (logic::ir::AndvOp binop = llvm::dyn_cast<logic::ir::AndvOp>(it)) {
                vec_binary(binop.getLhs(), binop.getRhs(), binop.getResult(), &mockturtle::xag_network::create_and);
            }
            else if (logic::ir::OrvOp binop = llvm::dyn_cast<logic::ir::OrvOp>(it)) {
                vec_binary(binop.getLhs(), binop.getRhs(), binop.getResult(), &mockturtle::xag_network::create_or);
            }
            else if (logic::ir::XorvOp binop = llvm::dyn_cast<logic::ir::XorvOp>(it)) {
                vec_binary(binop.getLhs(), binop.getRhs(), binop.getResult(), &mockturtle::xag_network::create_xor);
            }
            // Only process boolean values (with type `i1`), leaving other arith.constant (array index) untouched.
            else if (mlir::arith::ConstantOp con = llvm::dyn_cast<mlir::arith::ConstantOp>(it)) {
                mlir::BoolAttr attr = con.getValue().dyn_cast_or_null<mlir::BoolAttr>();
                if (attr) {
                    symbol_table[{value2str(con.getResult()), -1}] = xag.get_constant(attr.getValue());
                }
            }
            else if (mlir::memref::LoadOp load = llvm::dyn_cast<mlir::memref::LoadOp>(it)) {
                mlir::Value voffset = *load.getIndices().begin();
                mlir::arith::ConstantOp ooffset = voffset.getDefiningOp<mlir::arith::ConstantOp>();
                int ioffset = ooffset.getValue().dyn_cast<mlir::IntegerAttr>().getInt();
                std::string res = value2str(load.getResult());
                std::string memref = value2str(load.getMemref());
                symbol_table[{res, -1}] = symbol_table.at({memref, ioffset});
            }
            else if (mlir::memref::StoreOp store = llvm::dyn_cast<mlir::memref::StoreOp>(it)) {
                mlir::Value voffset = *store.getIndices().begin();
                mlir::arith::ConstantOp ooffset = voffset.getDefiningOp<mlir::arith::ConstantOp>();
                int ioffset = ooffset.getValue().dyn_cast<mlir::IntegerAttr>().getInt();
                std::string val = value2str(store.getValue());
                std::string memref = value2str(store.getMemref());
                symbol_table[{memref, ioffset}] = symbol_table.at({val, -1});
            }
            else if (logic::ir::ReturnOp ret = llvm::dyn_cast<logic::ir::ReturnOp>(it)) {
                mlir::Value oprand = ret.getOperand(0);
                std::string str = value2str(oprand);
                int width = getBitWidth(oprand);
                for (int i=0; i<width; i++) {
                    mockturtle::xag_network::signal sig = symbol_table.at({str, i});
                    xag.create_po(sig);
                }
            }
        }

        // Convert XAG to quantum circuit. 
        // caterpillar::eager_mapping_strategy<mockturtle::xag_network> strategy;
        caterpillar::greedy_pebbling_mapping_strategy<mockturtle::xag_network> strategy;
        tweedledum::netlist<caterpillar::stg_gate> circ;
        caterpillar::logic_network_synthesis_stats stats;
        tweedledum::stg_from_pprm stg_fn;
        caterpillar::logic_network_synthesis_params ps;
        caterpillar::detail::logic_network_synthesis_impl_oracle<tweedledum::netlist<caterpillar::stg_gate>, 
            mockturtle::xag_network, tweedledum::stg_from_pprm> impl( circ, xag, strategy, stg_fn, ps, stats );
        impl.run();
        
        // Construct MLIR-style circuit. 
        mlir::MLIRContext *ctx = op.getContext();
        mlir::Location loc = op.getLoc(); // The location of the oracle function in the source code.

        // Construct function signature.
        mlir::SmallVector<::mlir::Type> argtypes;
        mlir::SmallVector<::mlir::Type> returntypes;
        isq::ir::QStateType qstate = isq::ir::QStateType::get(ctx);
        mlir::AffineExpr d0, s0, s1;
        mlir::bindDims(ctx, d0);
        mlir::bindSymbols(ctx, s0, s1);
        auto affine_map = mlir::AffineMap::get(1, 2, d0 * s1 + s0);
        for (int i=0; i<input_num; i++) {
            mlir::Value arg = op.getArgument(i);
            int width = getBitWidth(arg);
            mlir::MemRefType memref_i_qstate = mlir::MemRefType::get(mlir::ArrayRef<int64_t>{width}, qstate, affine_map);
            argtypes.push_back(memref_i_qstate);
        }
        auto po_num = xag.num_pos();
        mlir::MemRefType memref_o_qstate = mlir::MemRefType::get(mlir::ArrayRef<int64_t>{po_num}, qstate, affine_map);
        argtypes.push_back(memref_o_qstate);
        mlir::FunctionType functype = mlir::FunctionType::get(ctx, argtypes, returntypes);

        // Debug infomation. 
        /*
        std::ofstream fout("/mnt/c/Users/14796/Desktop/workspace/isqv2/debugoutput.txt");
        std::streambuf *oldcout;
        oldcout = std::cout.rdbuf(fout.rdbuf());
        std::cout << "******xag description*******" << std::endl;
        xag.foreach_node( [&]( auto node ) {
            std::cout << "index: " << xag.node_to_index(node) << std::endl;
            xag.foreach_fanin(node, [&]( auto child ) {
                std::cout << "  child: " << child.index << std::endl;
                std::cout << "  complemented: " << (child.complement ? "y" : "n") << std::endl;
            });
        } );
        xag.foreach_pi( [&]( auto pi ) {
            std::cout << "pi: " << pi << std::endl;
        } );
        xag.foreach_po( [&]( auto node, auto index ) {
            std::cout << "po: " << xag.get_node(node) << (((mockturtle::xag_network::signal)node.data).complement ? " complemented" : "") << std::endl;
            xag.foreach_fanin(index, [&]( auto child ) {
                std::cout << "  child: " << child.index << std::endl;
                std::cout << "  complemented: " << (child.complement ? "y" : "n") << std::endl;
            });
        } );
        
        std::cout << "******circuit description*******" << std::endl;
        std::cout << "num_gates: " << circ.num_gates() << std::endl;
        circ.foreach_cgate( [&]( auto n ) {
            std::cout << n.gate << std::endl;
            n.gate.foreach_control( [&]( auto c ) {
                std::cout << "  control: " << c << " " << (c.is_complemented() ? "complemented" : "") << std::endl;
            } );
            n.gate.foreach_target( [&]( auto t ) {
                std::cout << "  target: " << t << " " << (t.is_complemented() ? "complemented" : "") << std::endl;
            } );
        } );
        std::cout << "num_qubits: " << circ.num_qubits() << std::endl;
        */
        // Create a FuncOp that represent the quantum circuit.
        mlir::func::FuncOp funcop = rewriter.create<mlir::func::FuncOp>(loc, op.getSymName(), functype);
        mlir::Block *entry_block = funcop.addEntryBlock(); // Arguments are automatically created based on the function signature.
        mlir::OpBuilder builder(entry_block, entry_block->begin());
        
        // Load arguments. 
        std::vector<mlir::Value> wires;
        std::unordered_map<uint32_t, int> qubit_to_wire;
        for (int i = 0; i <= input_num; i++) {
            mlir::BlockArgument arg = entry_block->getArgument(i);
            int width = getBitWidth(arg);
            std::string str = (i == input_num ? "" : value2str(op.getArgument(i)));
            for (int j = 0; j < width; j++) {
                mlir::arith::ConstantIndexOp index = builder.create<mlir::arith::ConstantIndexOp>(loc, j);
                mlir::memref::LoadOp load = builder.create<mlir::memref::LoadOp>(loc, qstate, arg, mlir::ValueRange{index});
                wires.push_back(load);
                // Construct qubit-to-wire mapping. 
                auto xagnode = symbol_table[{str, j}];
                uint32_t qubit = (i == input_num ? stats.o_indexes[j] : stats.i_indexes[xag.pi_index(xagnode.index)]);
                qubit_to_wire[qubit] = wires.size() - 1;
                //std::cout << (i == input_num ? "  output: " : "  input: ") << qubit << " to " << wires.size() - 1 << std::endl;
            }
        }

        // Ancilla allocation. 
        int ancilla_num = circ.num_qubits() - qubit_to_wire.size();
        int lowest_wire = wires.size();
        mlir::MemRefType memref_ancilla_qstate = mlir::MemRefType::get(mlir::ArrayRef<int64_t>{ancilla_num}, qstate);
        mlir::memref::AllocOp ancillas = builder.create<mlir::memref::AllocOp>(loc, memref_ancilla_qstate);
        auto memrefTy = ancillas.getType();
        if (!memrefTy.getElementType().isa<QStateType>()) return mlir::failure();
        for (int i = 0; i < ancilla_num; i++) {
            mlir::arith::ConstantIndexOp index = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
            mlir::memref::LoadOp ancilla = builder.create<mlir::memref::LoadOp>(loc, qstate, ancillas, mlir::ValueRange{index});
            wires.push_back(ancilla);
        }
        circ.foreach_cqubit( [&] ( tweedledum::qubit_id qubit_id ) {
            uint32_t qubit = qubit_id.index();
            if (qubit_to_wire.find(qubit) == qubit_to_wire.end()) {
                qubit_to_wire[qubit] = lowest_wire++;
                //std::cout << "  ancilla: " << qubit << " to " << lowest_wire - 1 << std::endl;
            }
        } );
        
        // Load the quantum gates. The last argument is the parameters of the gate, e.g., `theta` for Rz(theta, q);
        mlir::Value x_gate = builder.create<isq::ir::UseGateOp>(loc, isq::ir::GateType::get(ctx, 1, GateTrait::General),
            mlir::FlatSymbolRefAttr::get(ctx, "X"), mlir::ValueRange{}).getResult();
        mlir::Value cnot_gate = builder.create<isq::ir::UseGateOp>(loc, isq::ir::GateType::get(ctx, 2, GateTrait::General),
            mlir::FlatSymbolRefAttr::get(ctx, "CNOT"), mlir::ValueRange{}).getResult();
        mlir::Value toffoli_gate = builder.create<isq::ir::UseGateOp>(loc, isq::ir::GateType::get(ctx, 3, GateTrait::General),
            mlir::FlatSymbolRefAttr::get(ctx, "Toffoli"), mlir::ValueRange{}).getResult();

        // Gates application template
        auto apply_x = [&](tweedledum::qubit_id index) {
            isq::ir::ApplyGateOp applied_x = builder.create<isq::ir::ApplyGateOp>(loc, mlir::ArrayRef<mlir::Type>{qstate},
                x_gate, mlir::ArrayRef<mlir::Value>({wires[qubit_to_wire[index.index()]]}));
            wires[qubit_to_wire[index.index()]] = applied_x.getResult(0);
        };

        auto apply_cnot = [&](tweedledum::qubit_id cindex, tweedledum::qubit_id tindex) {
            if (cindex.is_complemented()) apply_x(cindex);
            isq::ir::ApplyGateOp applied_cnot = builder.create<isq::ir::ApplyGateOp>(loc, mlir::ArrayRef<mlir::Type>{qstate, qstate},
                cnot_gate, mlir::ArrayRef<mlir::Value>({wires[qubit_to_wire[cindex.index()]], wires[qubit_to_wire[tindex.index()]]}));
            wires[qubit_to_wire[cindex.index()]] = applied_cnot.getResult(0);
            wires[qubit_to_wire[tindex.index()]] = applied_cnot.getResult(1);
            if (cindex.is_complemented()) apply_x(cindex);
        };

        auto apply_toffoli = [&](tweedledum::qubit_id cindex_1, tweedledum::qubit_id cindex_2, tweedledum::qubit_id tindex) {
            if (cindex_1.is_complemented()) apply_x(cindex_1);
            if (cindex_2.is_complemented()) apply_x(cindex_2);
            isq::ir::ApplyGateOp applied_toffoli = builder.create<isq::ir::ApplyGateOp>(loc, mlir::ArrayRef<mlir::Type>{qstate, qstate, qstate},
                toffoli_gate, mlir::ArrayRef<mlir::Value>({wires[qubit_to_wire[cindex_1.index()]], wires[qubit_to_wire[cindex_2.index()]], wires[qubit_to_wire[tindex.index()]]}));
            wires[qubit_to_wire[cindex_1.index()]] = applied_toffoli.getResult(0);
            wires[qubit_to_wire[cindex_2.index()]] = applied_toffoli.getResult(1);
            wires[qubit_to_wire[tindex.index()]] = applied_toffoli.getResult(2);
            if (cindex_1.is_complemented()) apply_x(cindex_1);
            if (cindex_2.is_complemented()) apply_x(cindex_2);
        };

        // Apply gates to qstates. The last argument is the qstates to be applied on.
        circ.foreach_cgate( [&]( auto n ) {
            if (n.gate.is(tweedledum::gate_set::pauli_x)) {
                // std::cout << "apply x gate on " << n.gate.targets()[0] << std::endl;
                apply_x(n.gate.targets()[0]);
            } else if (n.gate.is(tweedledum::gate_set::cx)) {
                // std::cout << "apply cnot gate from " << n.gate.controls()[0] << " to " << n.gate.targets()[0] << std::endl;
                apply_cnot(n.gate.controls()[0], n.gate.targets()[0]);
            } else if (n.gate.is(tweedledum::gate_set::mcx)) {
                // std::cout << "apply toffoli gate from " << n.gate.controls()[0] << ", " <<  n.gate.controls()[1] << " to " << n.gate.targets()[0] << std::endl;
                apply_toffoli(n.gate.controls()[0], n.gate.controls()[1], n.gate.targets()[0]);
            } else {
                // ???
            }
        } );

        // Store qstates back to registers (i.e., the Memref<!isq.qstate> struct).
        int in_index = 0;
        for (int i = 0; i <= input_num; i++) {
            mlir::BlockArgument arg = entry_block->getArgument(i);
            int width = getBitWidth(arg);
            for (int j = 0; j < width; j++) {
                mlir::arith::ConstantIndexOp index = builder.create<mlir::arith::ConstantIndexOp>(loc, j);
                builder.create<mlir::memref::StoreOp>(loc, wires[in_index++], arg, mlir::ValueRange{index});
            }
        }
        builder.create<mlir::memref::DeallocOp>(loc, ancillas);
        builder.create<mlir::func::ReturnOp>(loc); // dummy terminator

        //std::cout.rdbuf(oldcout);
        //fout.close();

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
        rewriter.create<mlir::func::CallOp>(mlir::UnknownLoc::get(ctx), op.getCallee(), (mlir::TypeRange){}, op.getOperands());

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

class RuleReplaceLogicReturn : public mlir::OpRewritePattern<logic::ir::ReturnOp> {
public:
    RuleReplaceLogicReturn(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::ReturnOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::ReturnOp op, mlir::PatternRewriter &rewriter) const override {
        auto ctx = op.getContext();
        rewriter.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(ctx));

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
        rewriter.replaceOpWithNewOp<isq::ir::ApplyGateOp, mlir::ArrayRef<mlir::Type>, ::mlir::Value, ::mlir::ValueRange>(op, types, op.getGate(), op.getArgs());
        return mlir::success();
    }
};

class RuleReplaceLogicUse : public mlir::OpRewritePattern<logic::ir::UseGateOp> {
public:
    RuleReplaceLogicUse(mlir::MLIRContext *ctx): mlir::OpRewritePattern<logic::ir::UseGateOp>(ctx, 1) {}
    mlir::LogicalResult matchAndRewrite(logic::ir::UseGateOp op, mlir::PatternRewriter &rewriter) const override {
        auto ctx = op.getContext();
        //auto gate_type = ;
        //rewriter.create<isq::ir::UseGateOp>(mlir::UnknownLoc::get(ctx), op.getResult(), op.getName(), op.getParameters());
        rewriter.replaceOpWithNewOp<isq::ir::UseGateOp, isq::ir::GateType, ::mlir::SymbolRefAttr, ::mlir::ValueRange>(op, isq::ir::GateType::get(ctx, 1, GateTrait::General), op.getName(), op.getParameters());
        return mlir::success();
    }
};

struct LogicToISQPass : public mlir::PassWrapper<LogicToISQPass, mlir::OperationPass<mlir::ModuleOp>> {
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();

        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<RuleReplaceLogicFunc>(ctx);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);
        do{
            mlir::RewritePatternSet rps(ctx);
            rps.add<RuleReplaceLogicReturn>(ctx);
            rps.add<RuleReplaceLogicApply>(ctx);
            rps.add<RuleReplaceLogicUse>(ctx);
            mlir::FrozenRewritePatternSet frps(std::move(rps));
            (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
        }while(0);

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
    mlir::PassRegistration<logic_to_isq::LogicToISQPass>();
}

}
}
}