#include "isq/GateDefTypes.h"
#include "isq/Operations.h"
#include "isq/QAttrs.h"
#include "isq/QStructs.h"
#include "isq/QSynthesis.h"
#include "isq/QTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include <optional>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <set>
#include <iostream>

namespace isq{
namespace ir{
namespace passes{
namespace{

std::vector<std::vector<std::complex<double>>> appendMatrix(const std::vector<std::vector<std::complex<double>>>& mat, ::mlir::ArrayRef<bool> ctrl, bool adj){
    auto mat_qubit_num = (int)std::log2(mat.size());
    auto new_mat_size = ((1<<ctrl.size()) * mat.size());
    std::vector<std::vector<std::complex<double>>> new_matrix;
    new_matrix.resize(new_mat_size);
    for(auto i=0; i<new_mat_size; i++){
        new_matrix[i].resize(new_mat_size);
        new_matrix[i][i]=1.0;
    }
    uint64_t mat_mask = 0;
    for(auto i=0; i<ctrl.size(); i++){
        mat_mask = (mat_mask<<1) | (ctrl[i]?1:0);
    }
    mat_mask = mat_mask << mat_qubit_num;
    for(auto i=0; i<(1<<mat_qubit_num); i++){
        for(auto j=0; j<(1<<mat_qubit_num); j++){
            if(adj){
                new_matrix[i|mat_mask][j|mat_mask] = std::conj(mat[j][i]);
            }else{
                new_matrix[i|mat_mask][j|mat_mask] = mat[i][j];
            }
            
        }
    }
    return new_matrix;
}

std::vector<std::vector<std::complex<double>>> getBaseMat(const std::string gate_name){
    std::vector<std::vector<std::complex<double>>> base_mat;
    if (gate_name == "h"){
        base_mat = {
            {std::complex<double>(std::sqrt(0.5), 0.),std::complex<double>(std::sqrt(0.5), 0.)},
            {std::complex<double>(std::sqrt(0.5), 0.),std::complex<double>(-std::sqrt(0.5), 0.)}
        };
    }
    else if (gate_name == "x"){
        base_mat = {
            {std::complex<double>(0., 0.),std::complex<double>(1., 0.)},
            {std::complex<double>(1., 0.),std::complex<double>(0., 0.)}
        };
        
    }
    else if (gate_name == "y"){
        base_mat = {
            {std::complex<double>(0., 0.),std::complex<double>(0., -1.)},
            {std::complex<double>(0., 1.),std::complex<double>(0., 0.)}
        };
    }
    else if (gate_name == "z"){
        base_mat = {
            {std::complex<double>(1., 0.),std::complex<double>(0., 0.)},
            {std::complex<double>(0., 0.),std::complex<double>(-1., 0.)}
        };
    }
    else if (gate_name == "s"){
        base_mat = {
            {std::complex<double>(1., 0.),std::complex<double>(0., 0.)},
            {std::complex<double>(0., 0.),std::complex<double>(0., 1.)}
        };
    }
    else if (gate_name == "t"){
        base_mat = {
            {std::complex<double>(1., 0.),std::complex<double>(0., 0.)},
            {std::complex<double>(0., 0.),std::complex<double>(std::sqrt(0.5), std::sqrt(0.5))}
        };
    }
    else{
        base_mat = {
            {std::complex<double>(1,0.),std::complex<double>(0.,0.),std::complex<double>(0.,0.),std::complex<double>(0.,0.)},
            {std::complex<double>(0.,0.),std::complex<double>(1.,0.),std::complex<double>(0.,0.),std::complex<double>(0.,0.)},
            {std::complex<double>(0.,0.),std::complex<double>(0.,0.),std::complex<double>(0.,0.),std::complex<double>(1.,0.)},
            {std::complex<double>(0.,0.),std::complex<double>(0.,0.),std::complex<double>(1.,0.),std::complex<double>(0.,0.)}
        };
    }

    return base_mat;
}

mlir::LogicalResult insertToffoli(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op);

mlir::LogicalResult insertGate(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op, std::string gate_name, std::string func_name, int mat_qubit_num, bool is_private){
    
    auto ctx = rewriter.getContext();

    auto new_defgate_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, gate_name));
    auto new_defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(op, new_defgate_sym);
    
    if (!new_defgate){
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(rootModule.getBody());
        mlir::SmallVector<mlir::Attribute> new_mat_defs;
        new_mat_defs.push_back(GateDefinition::get(mlir::StringAttr::get(ctx, "qir"), mlir::StringAttr::get(ctx, func_name), ctx));
        rewriter.create<DefgateOp>(::mlir::UnknownLoc::get(ctx), mlir::TypeAttr::get(GateType::get(ctx, mat_qubit_num, GateTrait::General)), mlir::StringAttr::get(ctx, gate_name), mlir::StringAttr::get(ctx, "nested"), ::mlir::ArrayAttr{}, ::mlir::ArrayAttr::get(ctx, new_mat_defs), ::mlir::ArrayAttr::get(ctx, ::llvm::ArrayRef<::mlir::Attribute>{}));
                
        auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, func_name));
        auto new_func = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
        if (!new_func){
            
            if (is_private){
                llvm::SmallVector<mlir::Type> argsType;
                for (int i = 0; i < mat_qubit_num; i++){
                    argsType.push_back(QIRQubitType::get(rewriter.getContext()));
                }
                auto func_type = rewriter.getFunctionType(argsType, llvm::None);
                rewriter.create<mlir::FuncOp>(::mlir::UnknownLoc::get(ctx), func_name, func_type, mlir::StringAttr::get(ctx, "private"));
            }else{
                llvm::SmallVector<mlir::Type> argsType;
                for (int i = 0; i < mat_qubit_num; i++){
                    argsType.push_back(QStateType::get(rewriter.getContext()));
                }
                auto func_type = rewriter.getFunctionType(argsType, argsType);
                rewriter.create<mlir::FuncOp>(::mlir::UnknownLoc::get(ctx), func_name, func_type);
                if (func_name == "__isq__builtin__toffoli"){
                    if (mlir::failed(insertToffoli(rewriter, rootModule, op))){
                        rewriter.restoreInsertionPoint(ip);
                        return mlir::failure();
                    }
                }
            }
        }
        rewriter.restoreInsertionPoint(ip);
    }

    return mlir::success();
}

mlir::LogicalResult insertRotateFunc(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op, std::string gate_name, std::string func_name, int quantum_size, int double_size){
    auto ctx = rewriter.getContext();
    auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, func_name));
    auto funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
    if (!funcop) {
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(rootModule.getBody());
        llvm::SmallVector<mlir::Type> argsType;
        llvm::SmallVector<mlir::Type> resType;
        for (int i = 0; i < quantum_size; i++){
            argsType.push_back(QStateType::get(ctx));
            resType.push_back(QStateType::get(ctx));
        }
        for (int i = 0; i < double_size; i++){
            argsType.push_back(rewriter.getF64Type());
        }

        auto func_type = rewriter.getFunctionType(argsType, resType);
        auto funcop = rewriter.create<mlir::FuncOp>(::mlir::UnknownLoc::get(ctx), func_name, func_type);

        if (quantum_size == 1){
            auto entry_block = funcop.addEntryBlock();
            rewriter.setInsertionPointToStart(entry_block);

            mlir::SmallVector<mlir::Value> args;
            args.append(entry_block->args_begin(), entry_block->args_end());
            
            auto u3_builtin = "__isq__builtin__u3";
            ::mlir::SmallVector<mlir::Value> theta_v;

            if (gate_name == "rx"){
                auto pi_2 = rewriter.create<mlir::arith::ConstantFloatOp>(
                    ::mlir::UnknownLoc::get(ctx),
                    ::llvm::APFloat(M_PI / 2),
                    ::mlir::Float64Type::get(ctx)
                );
                auto neg_pi_2 = rewriter.create<mlir::arith::ConstantFloatOp>(
                    ::mlir::UnknownLoc::get(ctx),
                    ::llvm::APFloat(-M_PI / 2),
                    ::mlir::Float64Type::get(ctx)
                );
                theta_v = {args[1], neg_pi_2, pi_2};
            }else if (gate_name == "ry"){
                auto zero = rewriter.create<mlir::arith::ConstantFloatOp>(
                    ::mlir::UnknownLoc::get(ctx),
                    ::llvm::APFloat(0.0),
                    ::mlir::Float64Type::get(ctx)
                );
                theta_v = {args[1], zero, zero};
            }else{
                auto zero = rewriter.create<mlir::arith::ConstantFloatOp>(
                    ::mlir::UnknownLoc::get(ctx),
                    ::llvm::APFloat(0.0),
                    ::mlir::Float64Type::get(ctx)
                );
                theta_v = {zero, zero, args[1]};
            }
            
            auto use_u3_gate = rewriter.create<UseGateOp>(
                ::mlir::UnknownLoc::get(ctx),
                ir::GateType::get(ctx, 1, GateTrait::General),
                mlir::FlatSymbolRefAttr::get(ctx, u3_builtin),
                theta_v
            );
            
            auto qst = QStateType::get(rewriter.getContext());
            auto apply_u3_gate = rewriter.create<ApplyGateOp>(
                ::mlir::UnknownLoc::get(ctx),
                ::mlir::ArrayRef<::mlir::Type>{qst},
                use_u3_gate.result(),
                ::mlir::ArrayRef{args[0]}
            );
            args[0]=apply_u3_gate.getResult(0);
            rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), llvm::ArrayRef<mlir::Value>{args[0]});
        }
        rewriter.restoreInsertionPoint(ip);
    }

    return mlir::success();
    
}

mlir::LogicalResult decomposeMatrix(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op, std::string decomposed_name, std::string gate_name, std::vector<std::vector<std::complex<double>>>& mat) {
    
    auto n = (int) std::log2(mat.size());
    
    mlir::PatternRewriter::InsertionGuard guard(rewriter);
    auto ctx = rewriter.getContext();
    auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, decomposed_name));
    auto funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
    if (funcop) return mlir::success();
    
    if (mlir::failed(insertGate(rewriter, rootModule, op, gate_name, decomposed_name, n, false))){
        return mlir::failure();
    }
    funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
    auto entry_block = funcop.addEntryBlock();
    rewriter.setInsertionPointToStart(entry_block);
    mlir::SmallVector<mlir::Value> qubits;
    qubits.append(entry_block->args_begin(), entry_block->args_end());

    double eps = 1e-6;
    synthesis::UnitaryVector v;
    for(auto& row: mat){
        for(auto& elem: row){
            v.push_back(std::make_pair(elem.real(), elem.imag()));
        }
    }
    synthesis::QSynthesis A(n, v, eps);
    auto sim_gates = synthesis::simplify(A.gates);
    if(!synthesis::verify(n, v, sim_gates, A.phase)){
        return ::mlir::failure();
    }

    for (int j=0; j< sim_gates.size(); j++) {
        auto type = std::get<0>(sim_gates[j]);
        auto pos = std::get<1>(sim_gates[j]);
        if (type == synthesis::GateType::CNOT){
            auto cnot_builtin = "__isq__builtin__cnot";
            auto use_cnot_gate = rewriter.create<UseGateOp>(
                ::mlir::UnknownLoc::get(ctx),
                ir::GateType::get(ctx, 2, GateTrait::General),
                mlir::FlatSymbolRefAttr::get(ctx, cnot_builtin),
                ::mlir::ValueRange{}
            );
            auto qst = QStateType::get(rewriter.getContext());
            auto apply_cnot_gate = rewriter.create<ApplyGateOp>(
                ::mlir::UnknownLoc::get(ctx),
                ::mlir::ArrayRef<::mlir::Type>{qst, qst},
                use_cnot_gate.result(),
                ::mlir::ArrayRef{qubits[pos[0]], qubits[pos[1]]}
            );
            qubits[pos[0]]=apply_cnot_gate.getResult(0);
            qubits[pos[1]]=apply_cnot_gate.getResult(1);
        }else{
            double theta[3] = {std::get<2>(sim_gates[j]), std::get<3>(sim_gates[j]), std::get<4>(sim_gates[j])};
            auto u3_builtin = "__isq__builtin__u3";
            ::mlir::SmallVector<mlir::Value> theta_v;
            for(auto i=0; i<3; i++){
                auto v = rewriter.create<mlir::arith::ConstantFloatOp>(
                    ::mlir::UnknownLoc::get(ctx),
                    ::llvm::APFloat(theta[i]),
                    ::mlir::Float64Type::get(ctx)
                );
                theta_v.push_back(v.getResult());
            }
            auto use_u3_gate = rewriter.create<UseGateOp>(
                ::mlir::UnknownLoc::get(ctx),
                ir::GateType::get(ctx, 1, GateTrait::General),
                mlir::FlatSymbolRefAttr::get(ctx, u3_builtin),
                theta_v
            );
            auto qst = QStateType::get(rewriter.getContext());
            auto apply_u3_gate = rewriter.create<ApplyGateOp>(
                ::mlir::UnknownLoc::get(ctx),
                ::mlir::ArrayRef<::mlir::Type>{qst},
                use_u3_gate.result(),
                ::mlir::ArrayRef{qubits[pos[0]]}
            );
            qubits[pos[0]]=apply_u3_gate.getResult(0);
        }
    }

    rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
    return mlir::success();
}

mlir::LogicalResult addU3Gate(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op, mlir::SmallVector<mlir::Value>& qubits, synthesis::GateLocation pos, mlir::SmallVector<mlir::Value>& theta_v){

    auto ctx = rewriter.getContext();
    
    auto u3_builtin = "__isq__builtin__u3";

    auto use_u3_gate = rewriter.create<UseGateOp>(
        ::mlir::UnknownLoc::get(ctx),
        ir::GateType::get(ctx, 1, GateTrait::General),
        mlir::FlatSymbolRefAttr::get(ctx, u3_builtin),
        theta_v
    );
    auto apply_u3_gate = rewriter.create<ApplyGateOp>(
        ::mlir::UnknownLoc::get(ctx),
        ::mlir::ArrayRef<::mlir::Type>{QStateType::get(rewriter.getContext())},
        use_u3_gate.result(),
        ::mlir::ArrayRef{qubits[pos[0]]}
    );
    qubits[pos[0]]=apply_u3_gate.getResult(0);
    
    return mlir::success();
}

mlir::LogicalResult addBaseGate(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op, std::string gate_name, std::string func_name, mlir::SmallVector<mlir::Value>& qubits, synthesis::GateLocation pos){

    auto ctx = rewriter.getContext();
    
    if (mlir::failed(insertGate(rewriter, rootModule, op, gate_name, func_name, pos.size(), true))){
        return mlir::failure();
    }

    auto use_gate = rewriter.create<UseGateOp>(
        ::mlir::UnknownLoc::get(ctx),
        ir::GateType::get(ctx, pos.size(), GateTrait::General),
        mlir::FlatSymbolRefAttr::get(ctx, gate_name),
        ::mlir::ValueRange{}
    );

    auto qst = QStateType::get(rewriter.getContext());
    llvm::SmallVector<mlir::Type> res;
    llvm::SmallVector<mlir::Value> operand;
    for (int j = 0; j < pos.size(); j++){
        res.push_back(qst);
        operand.push_back(qubits[pos[j]]);
    }
    auto apply_gate = rewriter.create<ApplyGateOp>(
        ::mlir::UnknownLoc::get(ctx),
        res,
        use_gate.result(),
        operand
    );
    
    for (int j = 0; j < pos.size(); j++)
        qubits[pos[j]]=apply_gate.getResult(j);
    
    return mlir::success();
}

mlir::LogicalResult addGates(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op, synthesis::DecomposedGates& sim_gates, mlir::SmallVector<mlir::Value>& qubits){
    
    auto ctx = rewriter.getContext();

    for (int j=0; j< sim_gates.size(); j++) {
        auto type = std::get<0>(sim_gates[j]);
        auto pos = std::get<1>(sim_gates[j]);

        if (type == synthesis::GateType::H || type == synthesis::GateType::X){
            auto gate_name = "__quantum__qis__h__body_gate";
            auto func_name = "__quantum__qis__h__body";
            if (type == synthesis::GateType::X){
                gate_name = "__quantum__qis__x__body_gate";
                func_name = "__quantum__qis__x__body";
            }
            if (mlir::failed(addBaseGate(rewriter, rootModule, op, gate_name, func_name, qubits, pos))){
                return mlir::failure();
            }
        }
        else if (type == synthesis::GateType::TOFFOLI){
            auto gate_name = "__isq__builtin__toffoli_gate";
            auto func_name = "__isq__builtin__toffoli";
            if (mlir::failed(insertGate(rewriter, rootModule, op, gate_name, func_name, 3,  false))){
                return mlir::failure();
            }
            auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, func_name));
            auto funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
            auto callop = rewriter.create<mlir::CallOp>(::mlir::UnknownLoc::get(ctx), funcop, ::mlir::ArrayRef{qubits[pos[0]], qubits[pos[1]], qubits[pos[2]]});
            qubits[pos[0]]=callop.getResult(0);
            qubits[pos[1]]=callop.getResult(1);
            qubits[pos[2]]=callop.getResult(2);
        }
        else if (type == synthesis::GateType::CNOT){
            auto cnot_builtin = "__isq__builtin__cnot";
            auto use_cnot_gate = rewriter.create<UseGateOp>(
                ::mlir::UnknownLoc::get(ctx),
                ir::GateType::get(ctx, 2, GateTrait::General),
                mlir::FlatSymbolRefAttr::get(ctx, cnot_builtin),
                ::mlir::ValueRange{}
            );
            auto qst = QStateType::get(rewriter.getContext());
            auto apply_cnot_gate = rewriter.create<ApplyGateOp>(
                ::mlir::UnknownLoc::get(ctx),
                ::mlir::ArrayRef<::mlir::Type>{qst, qst},
                use_cnot_gate.result(),
                ::mlir::ArrayRef{qubits[pos[0]], qubits[pos[1]]}
            );
            qubits[pos[0]]=apply_cnot_gate.getResult(0);
            qubits[pos[1]]=apply_cnot_gate.getResult(1);
        }else{
            double theta[3] = {std::get<2>(sim_gates[j]), std::get<3>(sim_gates[j]), std::get<4>(sim_gates[j])};
            auto u3_builtin = "__isq__builtin__u3";
            ::mlir::SmallVector<mlir::Value> theta_v;
            for(auto i=0; i<3; i++){
                auto v = rewriter.create<mlir::arith::ConstantFloatOp>(
                    ::mlir::UnknownLoc::get(ctx),
                    ::llvm::APFloat(theta[i]),
                    ::mlir::Float64Type::get(ctx)
                );
                theta_v.push_back(v.getResult());
            }
            auto use_u3_gate = rewriter.create<UseGateOp>(
                ::mlir::UnknownLoc::get(ctx),
                ir::GateType::get(ctx, 1, GateTrait::General),
                mlir::FlatSymbolRefAttr::get(ctx, u3_builtin),
                theta_v
            );
            auto qst = QStateType::get(rewriter.getContext());
            auto apply_u3_gate = rewriter.create<ApplyGateOp>(
                ::mlir::UnknownLoc::get(ctx),
                ::mlir::ArrayRef<::mlir::Type>{qst},
                use_u3_gate.result(),
                ::mlir::ArrayRef{qubits[pos[0]]}
            );
            qubits[pos[0]]=apply_u3_gate.getResult(0);
        }
    }

    return mlir::success();
}


mlir::LogicalResult insertToffoli(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op){
    
    mlir::PatternRewriter::InsertionGuard guard(rewriter);
    auto ctx = rewriter.getContext();
    auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, "__isq__builtin__toffoli"));
    auto funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
    if (!funcop) return mlir::failure();
    auto entry_block = funcop.addEntryBlock();
    rewriter.setInsertionPointToStart(entry_block);
    mlir::SmallVector<mlir::Value> qubits;
    qubits.append(entry_block->args_begin(), entry_block->args_end());

    std::vector<std::tuple<std::string, std::string, std::vector<int>>> gates = {
        {"__quantum__qis__h__body_gate", "__quantum__qis__h__body", {2}},
        {"__quantum__qis__cnot_gate", "__quantum__qis__cnot",  {1, 2}},
        {"__quantum__qis__t__adj_gate", "__quantum__qis__t__adj", {2}},
        {"__quantum__qis__cnot_gate", "__quantum__qis__cnot", {0, 2}},
        {"__quantum__qis__t__body_gate", "__quantum__qis__t__body", {2}},
        {"__quantum__qis__cnot_gate", "__quantum__qis__cnot", {1, 2}},
        {"__quantum__qis__t__adj_gate", "__quantum__qis__t__adj", {2}},
        {"__quantum__qis__cnot_gate", "__quantum__qis__cnot", {0, 2}},
        {"__quantum__qis__t__body_gate", "__quantum__qis__t__body", {2}},
        {"__quantum__qis__t__adj_gate", "__quantum__qis__t__adj", {1}},
        {"__quantum__qis__h__body_gate", "__quantum__qis__h__body", {2}},
        {"__quantum__qis__cnot_gate", "__quantum__qis__cnot", {0, 1}},
        {"__quantum__qis__t__adj_gate", "__quantum__qis__t__adj", {1}},
        {"__quantum__qis__cnot_gate", "__quantum__qis__cnot", {0, 1}},
        {"__quantum__qis__t__body_gate", "__quantum__qis__t__body", {0}},
        {"__quantum__qis__s__body_gate", "__quantum__qis__s__body", {1}}
    };

    for(int i = 0; i < gates.size(); i++){
        auto gate_name = std::get<0>(gates[i]);
        auto func_name = std::get<1>(gates[i]);
        auto pos = std::get<2>(gates[i]);
        if (mlir::failed(addBaseGate(rewriter, rootModule, op, gate_name, func_name, qubits, pos))){
            rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
            return mlir::failure();
        }
    }
    rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
    return mlir::success();

}

mlir::LogicalResult mcdecomposeMatrix(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op,  std::string decomposed_name, std::string gate_name, std::vector<std::vector<std::complex<double>>>& mat, ::mlir::ArrayRef<bool> ctrl, bool adj) {
    
    int n = ctrl.size() + 1;

    mlir::PatternRewriter::InsertionGuard guard(rewriter);
    auto ctx = rewriter.getContext();
    auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, decomposed_name));
    auto funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
    
    if (funcop) return mlir::success();

    if (mlir::failed(insertGate(rewriter, rootModule, op, gate_name, decomposed_name, n, false))){
        return mlir::failure();
    }
    funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
    auto entry_block = funcop.addEntryBlock();
    rewriter.setInsertionPointToStart(entry_block);
    mlir::SmallVector<mlir::Value> qubits;
    qubits.append(entry_block->args_begin(), entry_block->args_end());
    

    std::string ctrl_s = "";
    for (auto c: ctrl){
        ctrl_s += c?"t":"f";
    }
    
    double eps = 1e-6;
    synthesis::UnitaryVector v;
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            if (adj){
                v.push_back(std::make_pair(mat[(i+(i^j)) % 2][(j+(i^j)) % 2].real(), -1*mat[(i+(i^j)) % 2][(j+(i^j)) % 2].imag()));
            }else{
                v.push_back(std::make_pair(mat[i][j].real(), mat[i][j].imag()));
            }
        }
    }
    
    auto sim_gates = synthesis::mcdecompose_u(v, ctrl_s);
    

    if (mlir::failed(addGates(rewriter, rootModule, op, sim_gates, qubits))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    } 

    rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
    return mlir::success();
}

// ctrl-rx = H - ctrl-rz - H
// ctrl-rz can decompose to rz(theta/2) ctrl-x rz(-theta/2) ctrl-x 
mlir::LogicalResult mcdecomposeRotate(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op,  std::string decomposed_name, std::string gate_name, ::mlir::ArrayRef<bool> ctrl, bool adj){

    int n = ctrl.size() + 1;
    
    mlir::PatternRewriter::InsertionGuard guard(rewriter);
    auto ctx = rewriter.getContext();
    auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, decomposed_name));
    auto funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
    if (funcop) return mlir::success();

    if (mlir::failed(insertRotateFunc(rewriter, rootModule, op, gate_name, decomposed_name, n, 1))){
        return mlir::failure();
    }

    funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
    auto entry_block = funcop.addEntryBlock();
    rewriter.setInsertionPointToStart(entry_block);
    mlir::SmallVector<mlir::Value> qubits;
    qubits.append(entry_block->args_begin(), entry_block->args_end());
    auto theta = qubits[qubits.size()-1]; // get last double arg
    qubits.pop_back(); // pop last double arg 
    
    std::string ctrl_s = "";
    for (int i = 0; i < ctrl.size(); i++){
        ctrl_s += "t";
        if (!ctrl[i]){
            std::string gate_name = "__quantum__qis__x__body_gate";
            std::string func_name = "__quantum__qis__x__body";
            if (mlir::failed(addBaseGate(rewriter, rootModule, op, gate_name, func_name, qubits, {i}))){
                rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
                return mlir::failure();
            }
        }
    }

    auto mat = getBaseMat("x");

    synthesis::UnitaryVector v;
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            v.push_back(std::make_pair(mat[i][j].real(), mat[i][j].imag()));
        }
    }

    auto sim_gates = synthesis::mcdecompose_u(v, ctrl_s);

    // get theta/2 and -theta / 2
    auto half = rewriter.create<mlir::arith::ConstantFloatOp>(
        ::mlir::UnknownLoc::get(ctx),
        ::llvm::APFloat(0.5),
        ::mlir::Float64Type::get(ctx)
    );
    theta = rewriter.create<mlir::arith::MulFOp>(
        ::mlir::UnknownLoc::get(ctx),
        theta,
        half
    );

    auto negone = rewriter.create<mlir::arith::ConstantFloatOp>(
        ::mlir::UnknownLoc::get(ctx),
        ::llvm::APFloat(-1.0),
        ::mlir::Float64Type::get(ctx)
    );
    auto neg_theta = rewriter.create<mlir::arith::MulFOp>(
        ::mlir::UnknownLoc::get(ctx),
        theta,
        negone
    );

    std::string matrix_name_low = gate_name;
    std::string new_matrix_name = "__isq__builtin__" + matrix_name_low;

    if (gate_name == "rx"){
        std::string gate_name = "__quantum__qis__h__body_gate";
        std::string func_name = "__quantum__qis__h__body";
        if (mlir::failed(addBaseGate(rewriter, rootModule, op, gate_name, func_name, qubits, {n-1}))){
            rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
            return mlir::failure();
        }
        matrix_name_low = "rz";
        new_matrix_name = "__isq__builtin__rz";
    }
    

    if (mlir::failed(insertRotateFunc(rewriter, rootModule, op, matrix_name_low, new_matrix_name, 1, 1))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    }
    
    new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, new_matrix_name));
    funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
    mlir::CallOp callop;
    if (adj){
        callop = rewriter.create<mlir::CallOp>(::mlir::UnknownLoc::get(ctx), funcop, ::mlir::ArrayRef<mlir::Value>{qubits[n-1], neg_theta});
    }else{ 
        callop = rewriter.create<mlir::CallOp>(::mlir::UnknownLoc::get(ctx), funcop, ::mlir::ArrayRef<mlir::Value>{qubits[n-1], theta});
    }
    qubits[n-1] = callop.getResult(0);

    if (mlir::failed(addGates(rewriter, rootModule, op, sim_gates, qubits))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    }

    if (adj){
        callop = rewriter.create<mlir::CallOp>(::mlir::UnknownLoc::get(ctx), funcop, ::mlir::ArrayRef<mlir::Value>{qubits[n-1], theta});
    }else{
        callop = rewriter.create<mlir::CallOp>(::mlir::UnknownLoc::get(ctx), funcop, ::mlir::ArrayRef<mlir::Value>{qubits[n-1], neg_theta});
    }
    qubits[n-1] = callop.getResult(0);

    if (mlir::failed(addGates(rewriter, rootModule, op, sim_gates, qubits))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    }

    if (gate_name == "rx"){
        std::string gate_name = "__quantum__qis__h__body_gate";
        std::string func_name = "__quantum__qis__h__body";
        if (mlir::failed(addBaseGate(rewriter, rootModule, op, gate_name, func_name, qubits, {n-1}))){
            rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
            return mlir::failure();
        }
    }

    for (int i = 0; i < ctrl.size(); i++){
        if (!ctrl[i]){
            std::string gate_name = "__quantum__qis__x__body_gate";
            std::string func_name = "__quantum__qis__x__body";
            if (mlir::failed(addBaseGate(rewriter, rootModule, op, gate_name, func_name, qubits, {i}))){
                rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
                return mlir::failure();
            }
        }
    }
    
    rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
    return mlir::success();
}

// u3 = ei(φ+λ)/2 Rz(φ)Ry(θ)Rz(λ)
mlir::LogicalResult mcdecomposeU3(mlir::PatternRewriter& rewriter, mlir::ModuleOp rootModule, mlir::Operation* op,  std::string decomposed_name, ::mlir::ArrayRef<bool> ctrl, bool adj){

    int n = ctrl.size() + 1;
    
    mlir::PatternRewriter::InsertionGuard guard(rewriter);
    auto ctx = rewriter.getContext();
    auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, decomposed_name));
    auto funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
    if (funcop) return mlir::success();

    if (mlir::failed(insertRotateFunc(rewriter, rootModule, op, "u3", decomposed_name, n, 3))){
        return mlir::failure();
    }

    funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
    auto entry_block = funcop.addEntryBlock();
    rewriter.setInsertionPointToStart(entry_block);
    mlir::SmallVector<mlir::Value> qubits;
    qubits.append(entry_block->args_begin(), entry_block->args_end());
    
    std::string ctrl_s = "";
    for (int i = 0; i < ctrl.size(); i++){
        ctrl_s += "t";
        if (!ctrl[i]){
            std::string gate_name = "__quantum__qis__x__body_gate";
            std::string func_name = "__quantum__qis__x__body";
            if (mlir::failed(addBaseGate(rewriter, rootModule, op, gate_name, func_name, qubits, {i}))){
                rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
                return mlir::failure();
            }
        }
    }

    auto mat = getBaseMat("x");

    synthesis::UnitaryVector v;
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            v.push_back(std::make_pair(mat[i][j].real(), mat[i][j].imag()));
        }
    }

    auto sim_gates = synthesis::mcdecompose_u(v, ctrl_s);

    auto u3_builtin = "__isq__builtin__u3";
    auto zero = rewriter.create<mlir::arith::ConstantFloatOp>(
        ::mlir::UnknownLoc::get(ctx),
        ::llvm::APFloat(0.0),
        ::mlir::Float64Type::get(ctx)
    );
    auto half = rewriter.create<mlir::arith::ConstantFloatOp>(
        ::mlir::UnknownLoc::get(ctx),
        ::llvm::APFloat(0.5),
        ::mlir::Float64Type::get(ctx)
    );
    auto neghalf = rewriter.create<mlir::arith::ConstantFloatOp>(
        ::mlir::UnknownLoc::get(ctx),
        ::llvm::APFloat(-0.5),
        ::mlir::Float64Type::get(ctx)
    );
    auto negone = rewriter.create<mlir::arith::ConstantFloatOp>(
        ::mlir::UnknownLoc::get(ctx),
        ::llvm::APFloat(-1.0),
        ::mlir::Float64Type::get(ctx)
    );
    
    auto lam = qubits[qubits.size()-1];
    auto phi = qubits[qubits.size()-2];
    auto theta = qubits[qubits.size()-3];
    if (adj){
        lam = rewriter.create<mlir::arith::MulFOp>(::mlir::UnknownLoc::get(ctx), qubits[qubits.size()-2], negone);
        phi = rewriter.create<mlir::arith::MulFOp>(::mlir::UnknownLoc::get(ctx), qubits[qubits.size()-1], negone);
        theta = rewriter.create<mlir::arith::MulFOp>(::mlir::UnknownLoc::get(ctx), qubits[qubits.size()-3], negone);
    }
    qubits.pop_back();
    qubits.pop_back();
    qubits.pop_back();

    // decompose U = eiθ*AXBXC
    // C = Rz((λ-φ) / 2) = u3(0, (λ-φ) / 2, 0)
    auto c_sub = rewriter.create<mlir::arith::SubFOp>(::mlir::UnknownLoc::get(ctx), lam, phi);
    auto c_angle = rewriter.create<mlir::arith::MulFOp>(::mlir::UnknownLoc::get(ctx), c_sub, half);
    auto c_theta_v = mlir::SmallVector<mlir::Value>{zero, c_angle, zero};
    if (mlir::failed(addU3Gate(rewriter, rootModule, op, qubits, {n-1}, c_theta_v))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    }
    
    // ctrl(n-1) x
    if (mlir::failed(addGates(rewriter, rootModule, op, sim_gates, qubits))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    }

    // B = Ry(-θ/2)Rz(-(λ+φ) / 2) = u3(-θ/2, 0, -(λ+φ) / 2)
    auto b_theta = rewriter.create<mlir::arith::MulFOp>(::mlir::UnknownLoc::get(ctx), theta, neghalf);
    auto b_add = rewriter.create<mlir::arith::AddFOp>(::mlir::UnknownLoc::get(ctx), lam, phi);
    auto b_angle = rewriter.create<mlir::arith::MulFOp>(::mlir::UnknownLoc::get(ctx), b_add, neghalf);
    auto b_theta_v = mlir::SmallVector<mlir::Value>{b_theta, zero, b_angle};
    if (mlir::failed(addU3Gate(rewriter, rootModule, op, qubits, {n-1}, b_theta_v))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    }

    // ctrl(n-1) x
    if (mlir::failed(addGates(rewriter, rootModule, op, sim_gates, qubits))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    }

    // A = Rz(φ)Ry(θ/2) = u3(θ/2, φ, 0)
    auto a_theta = rewriter.create<mlir::arith::MulFOp>(::mlir::UnknownLoc::get(ctx), theta, half);
    auto a_theta_v = mlir::SmallVector<mlir::Value>{a_theta, phi, zero};
    if (mlir::failed(addU3Gate(rewriter, rootModule, op, qubits, {n-1}, a_theta_v))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    }

    // ctrl(n-2) Z((λ+φ) / 2)
    
    auto addone_gates = synthesis::mcdecompose_addone(n);
    if (mlir::failed(addGates(rewriter, rootModule, op, addone_gates, qubits))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    }
    auto neg_z_angle = rewriter.create<mlir::arith::MulFOp>(::mlir::UnknownLoc::get(ctx), b_add, neghalf);
    for (int i = n-2; i > 0; i--){
        neg_z_angle = rewriter.create<mlir::arith::MulFOp>(::mlir::UnknownLoc::get(ctx), neg_z_angle, half);
        auto z_theta_v = mlir::SmallVector<mlir::Value>{zero, neg_z_angle, zero};
        if (mlir::failed(addU3Gate(rewriter, rootModule, op, qubits, {i}, z_theta_v))){
            rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
            return mlir::failure();
        }
    }
    reverse(addone_gates.begin(), addone_gates.end());
    if (mlir::failed(addGates(rewriter, rootModule, op, addone_gates, qubits))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    }
    auto z_angle = rewriter.create<mlir::arith::MulFOp>(::mlir::UnknownLoc::get(ctx), b_add, half);
    for (int i = n-2; i > 0; i--){
        z_angle = rewriter.create<mlir::arith::MulFOp>(::mlir::UnknownLoc::get(ctx), z_angle, half);
        auto z_theta_v = mlir::SmallVector<mlir::Value>{zero, z_angle, zero};
        if (mlir::failed(addU3Gate(rewriter, rootModule, op, qubits, {i}, z_theta_v))){
            rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
            return mlir::failure();
        }
    }
    auto z_theta_v = mlir::SmallVector<mlir::Value>{zero, z_angle, zero};
    if (mlir::failed(addU3Gate(rewriter, rootModule, op, qubits, {0}, z_theta_v))){
        rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
        return mlir::failure();
    }

    for (int i = 0; i < ctrl.size(); i++){
        if (!ctrl[i]){
            std::string gate_name = "__quantum__qis__x__body_gate";
            std::string func_name = "__quantum__qis__x__body";
            if (mlir::failed(addBaseGate(rewriter, rootModule, op, gate_name, func_name, qubits, {i}))){
                rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
                return mlir::failure();
            }
        }
    }

    rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);
    return mlir::success();
}

struct QuantumGateRewriteRule : public mlir::OpRewritePattern<isq::ir::ApplyGateOp>{
    mlir::ModuleOp rootModule;
    QuantumGateRewriteRule(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx, 1), rootModule(module){

    }
    mlir::LogicalResult matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const override{
        // Check if it is a use-decorate-apply pattern.
        //std::cout << "start apply op -> call\n";
        auto decorate_op = mlir::dyn_cast_or_null<DecorateOp>(op.gate().getDefiningOp());
        UseGateOp use_op;
        if(!decorate_op) {
            use_op = mlir::dyn_cast_or_null<UseGateOp>(op.gate().getDefiningOp());
        }else{
            use_op = mlir::dyn_cast_or_null<UseGateOp>(decorate_op.args().getDefiningOp());
        }
        if(!use_op) return mlir::failure();
        
        if(use_op.parameters().size()>0){
            return mlir::failure(); // Only matrix-gates are supported.
        }
        
        //std::cout << "use_op ok\n";
        
        auto defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(use_op.getOperation(), use_op.name());
        assert(defgate);
        
            
        //std::cout << "defgate ok\n";
        std::set<std::string> base_gate = {"h", "x", "y", "z", "s", "t"};
        auto matrix_name = std::string(defgate.sym_name());
        std::string matrix_name_low = matrix_name;
        transform(matrix_name.begin(), matrix_name.end(), matrix_name_low.begin(), [](unsigned char c) { return tolower(c); });
        
        //std::cout << "matrix name low: " << matrix_name_low << std::endl;
        std::vector<std::vector<std::complex<double>>> old_mat;
        if (base_gate.count(matrix_name_low) == 1 || matrix_name_low == "cnot"){
            old_mat = getBaseMat(matrix_name_low);
        }else{
            if(!defgate.definition()) return mlir::failure();
            bool hasmat = false;
            int id = 0;
            for(auto def: defgate.definition()->getAsRange<GateDefinition>()){
                auto d = AllGateDefs::parseGateDefinition(defgate, id, defgate.type(), def);
                if(d==std::nullopt) return mlir::failure();
                auto mat = llvm::dyn_cast_or_null<MatrixDefinition>(&**d);
                if(!mat){
                    id++;
                    continue;
                }
                old_mat = mat->getMatrix();
                hasmat = true;
                break;
            }
            if (!hasmat) return mlir::failure();
        }
        //std::cout << "get ori_mat ok\n";
                
        // construct function name.
        std::string new_matrix_name = matrix_name_low;
        std::string isq_buildin = "__isq__builtin__";
        std::string quantum_qis = "__quantum__qis__";
        // has no ctrl (base gate need not decompose later, start with "__quantum__qis")
        if (!decorate_op || decorate_op.ctrl().size() == 0){
            // base single gate
            if (base_gate.count(new_matrix_name) == 1){
                new_matrix_name = quantum_qis+new_matrix_name;
                if (decorate_op && decorate_op.adjoint()){
                    new_matrix_name += "__adj";
                }else{
                    new_matrix_name += "__body";
                }
            }else{
                // base cnot 
                if (new_matrix_name == "cnot" && (!decorate_op || !decorate_op.adjoint())){
                    new_matrix_name = quantum_qis+new_matrix_name;
                }else{ // other gate or inv cnot (need decompose later)
                    new_matrix_name = isq_buildin+new_matrix_name;
                    if (decorate_op && decorate_op.adjoint()){
                        new_matrix_name += "_adj";
                    }
                }
            }
        } // has ctrl (all gate need decompose later, start with "__isq__buildin")
        else{
            new_matrix_name = isq_buildin+new_matrix_name;
            new_matrix_name+="_ctrl_";
            for(auto c: decorate_op.ctrl().getAsValueRange<mlir::BoolAttr>()){
                new_matrix_name+= c?"1":"0";
            }
            if(decorate_op.adjoint()){
                new_matrix_name += "_adj";
            }
        }
        std::string new_defgate_name = new_matrix_name + "_gate";

        //std::cout << "new gate name ok, is: " << new_matrix_name << std::endl;
        // ceate new defgate
        auto ctx = rewriter.getContext();
        auto mat_qubit_num = defgate.type().getSize();
        if (decorate_op)
            mat_qubit_num += decorate_op.ctrl().size();
            
        bool is_private = false;
        if (new_matrix_name.compare(0, quantum_qis.size(), quantum_qis) == 0){
            is_private = true;
        }
        
        
        //std::cout << "new gate create ok\n";
        // decompose
        if (!is_private)
        {
            ::mlir::SmallVector<bool> ctrl;
            bool adj = false;
            if (decorate_op){
                for(auto b: decorate_op.ctrl().getAsValueRange<mlir::BoolAttr>()){
                    ctrl.push_back(b);
                }
                adj = decorate_op.adjoint();
            }
                        
            if (ctrl.size() > 0 && defgate.type().getSize() == 1){
                    
                if (mlir::failed(mcdecomposeMatrix(rewriter, rootModule, op, new_matrix_name, new_defgate_name, old_mat, ctrl, adj))){
                    return mlir::failure();
                }
            }else{
                auto new_matrix = appendMatrix(old_mat, ctrl, adj);
                if (mlir::failed(decomposeMatrix(rewriter, rootModule, op, new_matrix_name, new_defgate_name, new_matrix))){
                    return mlir::failure();
                }
            }
        }

        //std::cout << "decompose ok\n";
        
        // replace with call op, private func need create a buildin func?
        if (is_private){

            if (mlir::failed(insertGate(rewriter, rootModule, op, new_defgate_name, new_matrix_name, mat_qubit_num, is_private))){
                return mlir::failure();
            }
            /*
            auto ip = rewriter.saveInsertionPoint();
            rewriter.setInsertionPoint(op);
            auto new_use_gate = rewriter.create<UseGateOp>(op->getLoc(), GateType::get(ctx, mat_qubit_num, GateTrait::General), mlir::FlatSymbolRefAttr::get(ctx, new_defgate_name), ::mlir::ValueRange{});
            rewriter.restoreInsertionPoint(ip);
            rewriter.replaceOpWithNewOp<ApplyGateOp>(op.getOperation(), op->getResultTypes(), new_use_gate.result(), op.args());
            */
            
            std::string buildin_func = "__isq__builtin" + new_matrix_name;
            auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, buildin_func));
            auto new_func = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
            if (!new_func){
                auto ip = rewriter.saveInsertionPoint();
                auto rootM = rootModule;
                rewriter.setInsertionPointToStart(rootM.getBody());
                llvm::SmallVector<mlir::Type> argsType;
                for (int i = 0; i < mat_qubit_num; i++){
                    argsType.push_back(QStateType::get(rewriter.getContext()));
                }
                auto func_type = rewriter.getFunctionType(argsType, argsType);
                auto funcop = rewriter.create<mlir::FuncOp>(::mlir::UnknownLoc::get(ctx), buildin_func, func_type);
                auto entry_block = funcop.addEntryBlock();
                rewriter.setInsertionPointToStart(entry_block);
                mlir::SmallVector<mlir::Value> qubits;
                qubits.append(entry_block->args_begin(), entry_block->args_end());
                
                auto use_gate = rewriter.create<UseGateOp>(
                    ::mlir::UnknownLoc::get(ctx),
                    ir::GateType::get(ctx, mat_qubit_num, GateTrait::General),
                    mlir::FlatSymbolRefAttr::get(ctx, new_defgate_name),
                    ::mlir::ValueRange{}
                );
                
                auto qst = QStateType::get(rewriter.getContext());
                llvm::SmallVector<mlir::Type> res;
                for (int j = 0; j < mat_qubit_num; j++){
                    res.push_back(qst);
                }
                auto apply_gate = rewriter.create<ApplyGateOp>(
                    ::mlir::UnknownLoc::get(ctx),
                    res,
                    use_gate.result(),
                    qubits
                );
                
                for (int j = 0; j < mat_qubit_num; j++)
                    qubits[j]=apply_gate.getResult(j);
                rewriter.create<mlir::ReturnOp>(::mlir::UnknownLoc::get(rewriter.getContext()), qubits);

                rewriter.restoreInsertionPoint(ip);

                new_func = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
            }

            rewriter.replaceOpWithNewOp<mlir::CallOp>(op.getOperation(), new_func, op.args());

        }else{
            auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, new_matrix_name));
            auto funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);
        
            rewriter.replaceOpWithNewOp<mlir::CallOp>(op.getOperation(), funcop, op.args());
        }
        //std::cout << "call op create ok\n";
        //std::cout << "----------------------\n";
        return mlir::success();

    }
};


struct QuantumRotateGateRewriteRule : public mlir::OpRewritePattern<isq::ir::ApplyRotateGateOp>{
    mlir::ModuleOp rootModule;
    QuantumRotateGateRewriteRule(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<isq::ir::ApplyRotateGateOp>(ctx, 1), rootModule(module){

    }
    mlir::LogicalResult matchAndRewrite(isq::ir::ApplyRotateGateOp op,  mlir::PatternRewriter &rewriter) const override{
        // Check if it is a use-decorate-apply pattern.
        //std::cout << "start apply op -> call\n";
        auto decorate_op = mlir::dyn_cast_or_null<DecorateOp>(op.gate().getDefiningOp());
        UseGateOp use_op;
        if(!decorate_op) {
            use_op = mlir::dyn_cast_or_null<UseGateOp>(op.gate().getDefiningOp());
        }else{
            use_op = mlir::dyn_cast_or_null<UseGateOp>(decorate_op.args().getDefiningOp());
        }
        if(!use_op) return mlir::failure();
        
        if(use_op.parameters().size()>0){
            return mlir::failure(); // Only matrix-gates are supported.
        }
        
        
        auto defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(use_op.getOperation(), use_op.name());
        assert(defgate);

        auto theta = op.rotation();
        auto ctx = rewriter.getContext();
        
        auto matrix_name = std::string(defgate.sym_name());
        std::string matrix_name_low = matrix_name;
        transform(matrix_name.begin(), matrix_name.end(), matrix_name_low.begin(), [](unsigned char c) { return tolower(c); });
                    
        // construct function name.
        std::string new_matrix_name = "__isq__builtin__" + matrix_name_low;
        // has no ctrl, directly use u3
        if (!decorate_op || decorate_op.ctrl().size() == 0){
            auto ip = rewriter.saveInsertionPoint();
            rewriter.setInsertionPoint(op);
            if (decorate_op && decorate_op.adjoint()){
                auto negone = rewriter.create<mlir::arith::ConstantFloatOp>(
                    ::mlir::UnknownLoc::get(ctx),
                    ::llvm::APFloat(-1.0),
                    ::mlir::Float64Type::get(ctx)
                );
                theta = rewriter.create<mlir::arith::MulFOp>(
                    ::mlir::UnknownLoc::get(ctx),
                    theta,
                    negone
                );
            }
            rewriter.restoreInsertionPoint(ip);
            
            if (mlir::failed(insertRotateFunc(rewriter, rootModule, op, matrix_name_low, new_matrix_name, 1, 1))) return mlir::failure();

        }else{
            new_matrix_name += "_ctrl_";
            ::mlir::SmallVector<bool> ctrl;

            for (auto c: decorate_op.ctrl().getAsValueRange<mlir::BoolAttr>()){
                new_matrix_name += c?"1":"0";
                ctrl.push_back(c);
            }

            if (decorate_op.adjoint()) new_matrix_name += "_adj";


            if (mlir::failed(mcdecomposeRotate(rewriter, rootModule, op, new_matrix_name, matrix_name_low, ctrl, decorate_op.adjoint()))) return mlir::failure();

        }

        auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, new_matrix_name));
        auto funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);

        mlir::SmallVector<mlir::Value> args;
        args.append(op.args().begin(), op.args().end());
        args.push_back(theta);
        rewriter.replaceOpWithNewOp<mlir::CallOp>(op.getOperation(), funcop, args);
        
        return mlir::success();

    }
};

struct QuantumU3GateRewriteRule : public mlir::OpRewritePattern<isq::ir::ApplyGateOp>{
    mlir::ModuleOp rootModule;
    QuantumU3GateRewriteRule(mlir::MLIRContext* ctx, mlir::ModuleOp module): mlir::OpRewritePattern<isq::ir::ApplyGateOp>(ctx, 1), rootModule(module){

    }
    mlir::LogicalResult matchAndRewrite(isq::ir::ApplyGateOp op,  mlir::PatternRewriter &rewriter) const override{
        // Check if it is a use-decorate-apply pattern.
        //std::cout << "start apply op -> call\n";
        auto decorate_op = mlir::dyn_cast_or_null<DecorateOp>(op.gate().getDefiningOp());
        if(!decorate_op) return mlir::failure();
        auto use_op = mlir::dyn_cast_or_null<UseGateOp>(decorate_op.args().getDefiningOp());
        if(!use_op) return mlir::failure();
        
        if(use_op.parameters().size() != 3){
            return mlir::failure(); // Only u3-gates are supported.
        }
        
        auto defgate = mlir::SymbolTable::lookupNearestSymbolFrom<DefgateOp>(use_op.getOperation(), use_op.name());
        assert(defgate);

        
        std::string new_func_name = "__isq__buildtin__u3_ctrl_";
        ::mlir::SmallVector<bool> ctrl;
        for (auto c: decorate_op.ctrl().getAsValueRange<mlir::BoolAttr>()){
            new_func_name += c?"1":"0";
            ctrl.push_back(c);
        }
        if (decorate_op.adjoint()) new_func_name += "_adj";


        if (mlir::failed(mcdecomposeU3(rewriter, rootModule, op, new_func_name, ctrl, decorate_op.adjoint()))) return mlir::failure();

        auto ctx = rewriter.getContext();
        auto new_func_sym = mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, new_func_name));
        auto funcop = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(op, new_func_sym);

        mlir::SmallVector<mlir::Value> args;
        args.append(op.args().begin(), op.args().end());
        args.append(use_op.parameters().begin(), use_op.parameters().end());
        rewriter.replaceOpWithNewOp<mlir::CallOp>(op.getOperation(), funcop, args);
        
        return mlir::success();

    }
};


}

struct QuantumGatePass : public mlir::PassWrapper<QuantumGatePass, mlir::OperationPass<mlir::ModuleOp>>{
    void runOnOperation() override {
        mlir::ModuleOp m = this->getOperation();
        auto ctx = m->getContext();
        mlir::RewritePatternSet rps(ctx);
        rps.add<QuantumGateRewriteRule>(ctx, m);
        rps.add<QuantumRotateGateRewriteRule>(ctx, m);
        rps.add<QuantumU3GateRewriteRule>(ctx, m);
        mlir::FrozenRewritePatternSet frps(std::move(rps));
        (void)mlir::applyPatternsAndFoldGreedily(m.getOperation(), frps);
    }
  mlir::StringRef getArgument() const final {
    return "isq-quantum-gates-pass";
  }
  mlir::StringRef getDescription() const final {
    return  "pass quantum gates to function call.";
  }
};

void registerQuantumGatePass(){
    mlir::PassRegistration<QuantumGatePass>();
}

}
}
}