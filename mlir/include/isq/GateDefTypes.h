#ifndef _ISQ_GATEDEFTYPES_H
#define _ISQ_GATEDEFTYPES_H
#include "isq/Operations.h"
#include "isq/QAttrs.h"
#include "isq/QStructs.h"
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
#include <memory>
#include <optional>
#include "isq/Math.h"
namespace isq{
namespace ir{

GateDefinition createMatrixDef(mlir::MLIRContext* ctx, const std::vector<std::vector<std::complex<double>>>&);

class GateDefinitionAttribute{
public:
    enum GateDefinitionKind{
        GD_MATRIX,
        GD_DECOMPOSITION,
        GD_DECOMPOSITION_RAW,
        GD_QIR,
        GD_CLASSICAL_ORACLE,
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
    std::vector<std::vector<std::complex<double>>> mat;
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
    const std::vector<std::vector<std::complex<double>>>& getMatrix() const;
};

// Define by decomposition.
class DecompositionDefinition: public GateDefinitionAttribute{
private:
    mlir::FuncOp decomposition;
public:
    ::mlir::FuncOp getDecomposedFunc();
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
    mlir::FuncOp decomposition;
public:
    ::mlir::FuncOp getDecomposedFunc();
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


// Helpers
template<typename T>
std::optional<std::shared_ptr<GateDefinitionAttribute>> inline parseGateDefinitionAs(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType gateType, ::isq::ir::GateDefinition def){
    if(def.type().strref() != T::defKindName()){
        return std::nullopt;
    }
    if(::mlir::failed(T::verify(op, id, gateType, def.value()))){
        return std::nullopt;
    }
    return std::make_shared<T>(op, id, gateType, def.value());
}

template<typename ... T>
struct GateDefParser{};
template<> struct GateDefParser<>{
public:
    static std::optional<std::shared_ptr<GateDefinitionAttribute>> parseGateDefinition(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType gateType, ::isq::ir::GateDefinition def){
        op->emitError() << "Definition #" << id
                << " has unrecognized type \""<<def.type().strref()<<"\".";
        return std::nullopt;
    };
    static ::mlir::LogicalResult verifySymTable(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::isq::ir::GateDefinition def, ::mlir::SymbolTableCollection &symbolTable){
        op->emitError() << "Definition #" << id
                << " has unrecognized type \""<<def.type().strref()<<"\".";
        return ::mlir::failure();
    }
};
template<typename T, typename ...U>
struct GateDefParser<T, U...>{
public:
    static std::optional<std::shared_ptr<GateDefinitionAttribute>> parseGateDefinition(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType gateType, ::isq::ir::GateDefinition def){
        if(def.type().strref() == T::defKindName()){
            return parseGateDefinitionAs<T>(op, id, gateType, def);
        }else{
            return GateDefParser<U...>::parseGateDefinition(op, id, gateType, def);
        }
    };
    static ::mlir::LogicalResult verifySymTable(::isq::ir::DefgateOp op, int id, ::isq::ir::GateType ty, ::isq::ir::GateDefinition def, ::mlir::SymbolTableCollection &symbolTable){
        if(def.type().strref() == T::defKindName()){
            return T::verifySymTable(op, id, ty, def.value(), symbolTable);
        }else{
            return GateDefParser<U...>::verifySymTable(op, id, ty, def, symbolTable);
        }
        
    };
};

using AllGateDefs = GateDefParser<
    MatrixDefinition, 
    DecompositionDefinition,
    DecompositionRawDefinition,
    QIRDefinition
>;

}
}
#endif