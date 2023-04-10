#ifndef _ISQ_GATEDEFTYPES_H
#define _ISQ_GATEDEFTYPES_H
#include "isq/Operations.h"
#include "isq/QAttrs.h"
<<<<<<< HEAD
#include "isq/QStructs.h"
=======
>>>>>>> merge
#include "isq/QTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/StringRef.h"
<<<<<<< HEAD
#include <memory>
#include <optional>
#include "isq/Math.h"
namespace isq{
namespace ir{

GateDefinition createMatrixDef(mlir::MLIRContext* ctx, const std::vector<std::vector<std::complex<double>>>&);

=======
#include <llvm/ADT/APFloat.h>
#include <memory>
#include <optional>
#include "isq/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace isq{
namespace ir{


template<isq::ir::math::MatDouble Mat>
static DenseComplexF64MatrixAttr fromMatrixImpl(mlir::MLIRContext* ctx, const Mat& mat){
    mlir::SmallVector<mlir::SmallVector<std::complex<double>>> data;
    for(auto& row: mat){
        mlir::SmallVector<std::complex<double>> curr_row;
        for(auto& value: row){
            curr_row.push_back(value);
        }
        data.push_back(std::move(curr_row));
    }
    return DenseComplexF64MatrixAttr::get(ctx, data);
    /*
    mlir::SmallVector<std::complex<llvm::APFloat>> data;
    for(auto& row: mat){
        for(auto& value: row){
            data.push_back(value);
        }
    }
    auto shape = mlir::RankedTensorType::get({2, 2}, mlir::ComplexType::get(mlir::Float64Type::get(ctx)));
    auto dense = mlir::DenseElementsAttr::get(shape, data);
    return DenseComplexF64MatrixAttr::get(ctx, dense);
    */
}


template<isq::ir::math::MatDouble Mat>
GateDefinition createMatrixDef(mlir::MLIRContext* ctx, const Mat & mat){
    return (GateDefinition::get(ctx, mlir::StringAttr::get(ctx, "unitary"), fromMatrixImpl(ctx,  mat)));
}
>>>>>>> merge
class GateDefinitionAttribute{
public:
    enum GateDefinitionKind{
        GD_MATRIX,
        GD_DECOMPOSITION,
        GD_DECOMPOSITION_RAW,
        GD_QIR,
        GD_CLASSICAL_ORACLE,
        GD_ORACLE_TABLE,
    };
    GateDefinitionAttribute(GateDefinitionKind kind): kind(kind) {}
private:
    const GateDefinitionKind kind;
public:
    GateDefinitionKind getKind() const {
        return kind;
    }
    virtual ~GateDefinitionAttribute(){

    }
};

// Define by matrix.
class MatrixDefinition: public GateDefinitionAttribute{
private:
<<<<<<< HEAD
    std::vector<std::vector<std::complex<double>>> mat;
=======
    DenseComplexF64MatrixAttr::MatrixVal mat;
>>>>>>> merge
public:
    MatrixDefinition(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType gateType, ::mlir::Attribute value);
    static bool classof(const GateDefinitionAttribute *attr) {
        return attr->getKind() == GateDefinitionAttribute::GD_MATRIX;
    }
    static ::mlir::StringRef defKindName() {
        return "unitary";
    }
    static ::mlir::LogicalResult verify(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::mlir::Attribute attribute);
    static ::mlir::LogicalResult verifySymTable(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::mlir::Attribute attribute, ::mlir::SymbolTableCollection &symbolTable);
<<<<<<< HEAD
    const std::vector<std::vector<std::complex<double>>>& getMatrix() const;
=======
    const DenseComplexF64MatrixAttr::MatrixVal& getMatrix() const;
>>>>>>> merge
};

// Define by decomposition.
class DecompositionDefinition: public GateDefinitionAttribute{
private:
<<<<<<< HEAD
    mlir::FuncOp decomposition;
public:
    ::mlir::FuncOp getDecomposedFunc();
=======
    mlir::func::FuncOp decomposition;
public:
    mlir::func::FuncOp getDecomposedFunc();
>>>>>>> merge
    DecompositionDefinition(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType gateType, ::mlir::Attribute value);
    static bool classof(const GateDefinitionAttribute *attr) {
        return attr->getKind() == GateDefinitionAttribute::GD_DECOMPOSITION;
    }
    static ::mlir::StringRef defKindName() {
        return "decomposition";
    }
    static ::mlir::LogicalResult verify(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::mlir::Attribute value);
    static ::mlir::LogicalResult verifySymTable(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::mlir::Attribute attribute, ::mlir::SymbolTableCollection &symbolTable);
};
// Define by decomposition.
class DecompositionRawDefinition: public GateDefinitionAttribute{
private:
<<<<<<< HEAD
    mlir::FuncOp decomposition;
public:
    ::mlir::FuncOp getDecomposedFunc();
=======
    mlir::func::FuncOp decomposition;
public:
    mlir::func::FuncOp getDecomposedFunc();
>>>>>>> merge
    DecompositionRawDefinition(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType gateType, ::mlir::Attribute value);
    static bool classof(const GateDefinitionAttribute *attr) {
        return attr->getKind() == GateDefinitionAttribute::GD_DECOMPOSITION_RAW;
    }
    static ::mlir::StringRef defKindName() {
        return "decomposition_raw";
    }
    static ::mlir::LogicalResult verify(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::mlir::Attribute value);
    static ::mlir::LogicalResult verifySymTable(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::mlir::Attribute attribute, ::mlir::SymbolTableCollection &symbolTable);
};

// Define by QIR primitive. This allows lowering to QIR.
class QIRDefinition: public GateDefinitionAttribute{
private:
    ::mlir::FlatSymbolRefAttr qir_name;
public:
    ::mlir::FlatSymbolRefAttr getQIRName();
    QIRDefinition(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType gateType, ::mlir::Attribute value);
    static bool classof(const GateDefinitionAttribute *attr) {
        return attr->getKind() == GateDefinitionAttribute::GD_QIR;
    }
    static ::mlir::StringRef defKindName() {
        return "qir";
    }
    static ::mlir::LogicalResult verify(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::mlir::Attribute value);
    static ::mlir::LogicalResult verifySymTable(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::mlir::Attribute attribute, ::mlir::SymbolTableCollection &symbolTable);
};

// Define by oracle.
class OracleTableDefinition: public GateDefinitionAttribute{
private:
    std::vector<std::vector<int>> value;
public:
    OracleTableDefinition(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType GateType, ::mlir::Attribute value);
    static bool classof(const GateDefinitionAttribute *attr){
        return attr->getKind() == GateDefinitionAttribute::GD_ORACLE_TABLE;
    }

    static ::mlir::StringRef defKindName(){
        return "oracle_table";
    }
    static ::mlir::LogicalResult verify(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::mlir::Attribute value);
    static ::mlir::LogicalResult verifySymTable(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::mlir::Attribute attribute, ::mlir::SymbolTableCollection &symbolTable);
    const std::vector<std::vector<int>>& getValue() const;
};

// Helpers
template<typename T>
std::optional<std::shared_ptr<GateDefinitionAttribute>> inline parseGateDefinitionAs(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType gateType, ::isq::ir::GateDefinition def){
<<<<<<< HEAD
    if(def.type().strref() != T::defKindName()){
        return std::nullopt;
    }
    if(::mlir::failed(T::verify(op, id, gateType, def.value()))){
        return std::nullopt;
    }
    return std::make_shared<T>(op, id, gateType, def.value());
=======
    if(def.getType().strref() != T::defKindName()){
        return std::nullopt;
    }
    if(::mlir::failed(T::verify(op, id, gateType, def.getValue()))){
        return std::nullopt;
    }
    return std::make_shared<T>(op, id, gateType, def.getValue());
>>>>>>> merge
}

template<typename ... T>
struct GateDefParser{};
template<> struct GateDefParser<>{
public:
    static std::optional<std::shared_ptr<GateDefinitionAttribute>> parseGateDefinition(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType gateType, ::isq::ir::GateDefinition def){
        op->emitError() << "Definition #" << id
<<<<<<< HEAD
                << " has unrecognized type \""<<def.type().strref()<<"\".";
=======
                << " has unrecognized type \""<<def.getType().strref()<<"\".";
>>>>>>> merge
        return std::nullopt;
    };
    static ::mlir::LogicalResult verifySymTable(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::isq::ir::GateDefinition def, ::mlir::SymbolTableCollection &symbolTable){
        op->emitError() << "Definition #" << id
<<<<<<< HEAD
                << " has unrecognized type \""<<def.type().strref()<<"\".";
=======
                << " has unrecognized type \""<<def.getType().strref()<<"\".";
>>>>>>> merge
        return ::mlir::failure();
    }
};
template<typename T, typename ...U>
struct GateDefParser<T, U...>{
public:
    static std::optional<std::shared_ptr<GateDefinitionAttribute>> parseGateDefinition(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType gateType, ::isq::ir::GateDefinition def){
<<<<<<< HEAD
        if(def.type().strref() == T::defKindName()){
=======
        if(def.getType().strref() == T::defKindName()){
>>>>>>> merge
            return parseGateDefinitionAs<T>(op, id, gateType, def);
        }else{
            return GateDefParser<U...>::parseGateDefinition(op, id, gateType, def);
        }
    };
    static ::mlir::LogicalResult verifySymTable(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::isq::ir::GateDefinition def, ::mlir::SymbolTableCollection &symbolTable){
<<<<<<< HEAD
        if(def.type().strref() == T::defKindName()){
            return T::verifySymTable(op, id, ty, def.value(), symbolTable);
=======
        if(def.getType().strref() == T::defKindName()){
            return T::verifySymTable(op, id, ty, def.getValue(), symbolTable);
>>>>>>> merge
        }else{
            return GateDefParser<U...>::verifySymTable(op, id, ty, def, symbolTable);
        }
        
    };
};

using AllGateDefs = GateDefParser<
    MatrixDefinition, 
    DecompositionDefinition,
    DecompositionRawDefinition,
    QIRDefinition,
    OracleTableDefinition
>;

}
}
#endif