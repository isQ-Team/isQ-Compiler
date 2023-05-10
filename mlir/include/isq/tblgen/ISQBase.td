#ifndef ISQ_TBLGEN_ISQBASE_TD_
#define ISQ_TBLGEN_ISQBASE_TD_

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
//include "mlir/Dialect/Shape/IR/ShapeBase.td"

////////////////////////////////////
// Dialect
////////////////////////////////////

def ISQ_Dialect: Dialect{
    let name = "isq";
    let cppNamespace = "::isq::ir";
    let summary = "isQ-IR Dialect.";
    let description = [{
        MLIR Dialect for expressing and optimization ranging from low-level quantum circuits, to high level quantum programs with complicated control structures.
    }];
    let dependentDialects = ["::mlir::memref::MemRefDialect", "::mlir::AffineDialect", "::mlir::arith::ArithmeticDialect", "::mlir::scf::SCFDialect", "::mlir::LLVM::LLVMDialect",
    "::mlir::func::FuncDialect", "::mlir::cf::ControlFlowDialect",
    "::mlir::complex::ComplexDialect"];
    let hasCanonicalizer = 1;
    let useDefaultTypePrinterParser = 0;
    let useDefaultAttributePrinterParser = 0;
    let extraClassDeclaration = [{
        ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
        void printType(::mlir::Type type,
                        ::mlir::DialectAsmPrinter &printer) const override;
        ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                                ::mlir::Type type) const override;
        void printAttribute(::mlir::Attribute attr,
                                        ::mlir::DialectAsmPrinter &os) const override;
    }];
}

////////////////////////////////////
// Types
////////////////////////////////////

class ISQ_Type<string name> : TypeDef<ISQ_Dialect, name>{}
class ISQ_Op<string mnemonic, list<Trait> traits = []> :
    Op<ISQ_Dialect, mnemonic, traits>{
    
}

def ISQ_QState : TypeDef<ISQ_Dialect, "QState", [MemRefElementTypeInterface]>{
    let description = [{
        Represents an intermediate qubit SSA state. Written `!isq.qstate`.

        **Representation of qubit**: we use `memref<!isq.qstate>` to represent qubits.

        **Qubit state lifetime rule**: every `!isq.qstate` SSA value must be used **exactly once**. 
        The source of an `!isq.qstate` should be a memory-reading operation like `affine.load`, 
        and the drain should be a memory-writing operation like `affine.store`.
        The source `load` and drain `store` must refer to exactly the same memory location.
        An `!isq.qstate` SSA value is called **alive** between its definition and its usage.

        **Qubit state aliasing rule**: at any moment two alive `!isq.qstate` SSA values must refer to two different qubits.
        This is similar to the aliasing rule for `&mut` references in Rust or the functionality of `__restrict`(`restrict`) keyword in C.
        In this sense, all multi-qubit calls are naturally legal.

        This SSA representation is inspired by [David et al.](https://arxiv.org/abs/2101.11030).
    }];
    let summary = [{Qubit state}];
    let mnemonic = "qstate";
    let hasCustomAssemblyFormat = 1;
    /*
    let printer = [{
        QStateParsePrint p;
        p.printType(*this, $_printer);
    }];
    let parser = [{
        QStateParsePrint p;
        return p.parseType($_parser);
    }];
    */
}
def ISQ_Gate : TypeDef<ISQ_Dialect, "Gate", [MemRefElementTypeInterface]>{
    let description = [{
        Represents **reference** to a unitary (pure) quantum gate. This enables gate-level optimization of quantum programs.

        A gate type is parametrized with two parameters: **size** and **traits**.
        - **Size** indicates the size of the gate.
        - **Traits** indicates useful properties for optimization of the gate, for example, if the gate is diagonal a gate, or the gate is symmetric over its operands.

        A gate type can be written in the form `!isq.gate<size[, traits]>`. Currently supported traits include:
        - `hermitian`, indicating the gate is self-adjoint, i.e. \f$U^\dagger = U\f$.
        - `diagonal`, indicating the gate is diagonal (under computational basis).
        - `antidiagonal`, indicating the gate is antidiagonal (under computational basis). This enables switching between antidiagonal gates and \f$CZ\f$ gates, or `EjectPhasedPaulis` optimization.
        - `symmetric`, indicating the gate is symmetric over its **all** operands. Single qubit gates are intrinsically symmetric whether they are labeled or not.

        Note that we allow loading gates at runtime.
    }];
    let parameters = (ins "int64_t":$size,"GateTrait":$hints);
    let summary = [{Gate reference}];
    let mnemonic = "gate";
    let hasCustomAssemblyFormat = 1;
    /*
    let printer = [{
        GateParsePrint p;
        p.printType(*this, $_printer);
    }];
    let parser = [{
        GateParsePrint p;
        return p.parseType($_parser);
    }];
    */
    let extraClassDeclaration = [{
        ::mlir::SmallVector<mlir::Type> getApplyParameterType(GateType ty);
    }];
}
def ISQ_QIRQubit : TypeDef<ISQ_Dialect, "QIRQubit", [MemRefElementTypeInterface]>{
    let description = [{
        Represents an opaque value `%Qubit` in QIR(LLVM).
    }];
    let summary = [{QIR Opaque Qubit value.}];
    let mnemonic = "qir.qubit";
    let hasCustomAssemblyFormat = 1;
    /*
    let printer = [{
        QIRQubitParsePrint p;
        p.printType(*this, $_printer);
    }];
    let parser = [{
        QIRQubitParsePrint p;
        return p.parseType($_parser);
    }];
    */
}
def ISQ_QIRResult : TypeDef<ISQ_Dialect, "QIRResult", [MemRefElementTypeInterface]>{
    let description = [{
        Represents an opaque value `%Result` in QIR(LLVM).
    }];
    let summary = [{QIR Opaque measurement result value.}];
    let mnemonic = "qir.result";
    let hasCustomAssemblyFormat = 1;
    /*
    let printer = [{
        QIRResultParsePrint p;
        p.printType(*this, $_printer);
    }];
    let parser = [{
        QIRResultParsePrint p;
        return p.parseType($_parser);
    }];
    */
}
/*
def ISQ_QOp: DialectType<ISQ_Dialect, CPred<"$_self.isa<::isq::ir::QOpType>()">, "General quantum operation type">{
    let description = [{
        (Deprecated, since QOp will remain symbol and not materialized.)
        Represents a general quantum operation on several qubits, with certain classical input/output parameters.
        Think of this type as a black-box with the same number of input wires and output wires.

        A quantum operation type can be written in the form `!isq.qop<size, (operands)->results>`. Qstates and `operands` will be sent into this operation, and `results` come out.
    }];
}
*/

////////////////////////////////////
// Attributes
////////////////////////////////////

class ISQ_Attr<string name, list<Trait> traits = []> : AttrDef<ISQ_Dialect, name, traits>;


def GateTypeAttr: TypeAttrBase<"::isq::ir::GateType", "Gate type attribute">;
def FuncTypeAttr: TypeAttrBase<"::mlir::FunctionType", "Function type attribute, used as tuple of operands and results">;


// Gate traits

def GateTraitGeneral: I32BitEnumAttrCase<"General", 0, "general">;
def GateTraitDiagonal: I32BitEnumAttrCase<"Diagonal", 1, "diagonal">;
def GateTraitAntidiagonal: I32BitEnumAttrCase<"Antidiagonal", 2, "antidiagonal">;
def GateTraitSymmetric: I32BitEnumAttrCase<"Symmetric", 4, "symmetric">;
def GateTraitHermitian: I32BitEnumAttrCase<"Hermitian", 8, "hermitian">;
def GateTraitPhase: I32BitEnumAttrCase<"Phase", 16, "phase">;

def GateTrait: I32BitEnumAttr<"GateTrait", "Gate traits/hints", [GateTraitGeneral, GateTraitDiagonal, GateTraitAntidiagonal, GateTraitSymmetric, GateTraitHermitian, GateTraitPhase]>;
// Complex matrix.
def DenseComplexF64MatrixAttr : AttrDef<ISQ_Dialect, "DenseComplexF64Matrix">{
    let parameters = (ins
        "mlir::DenseElementsAttr": $body
    );
    let mnemonic = "matrix";
    let assemblyFormat = "`<` $body `>`";
    let builders = [
        AttrBuilder<(ins "const MatrixVal&": $matrix)>
    ];
    let extraClassDeclaration = [{
        using MatrixVal = llvm::SmallVector<llvm::SmallVector<std::complex<double>>>;
        MatrixVal toMatrixVal();
    }];
    let extraClassDefinition = [{
        template<isq::ir::math::MatDouble T> 
        static DenseComplexF64MatrixAttr fromMatrix(mlir::MLIRContext* ctx, const T& mat){
            return fromMatrixImpl(ctx, mat);
        }
    }];

}
// Gate definition
def ISQ_GateDefinitionAttr : AttrDef<ISQ_Dialect, "GateDefinition">{
    let parameters = (ins 
        "mlir::StringAttr": $type, 
        "mlir::Attribute": $value);
    let mnemonic = "gatedef";
    let assemblyFormat = "`<` struct(params) `>`";
}

def ISQ_GateDefinitionList : TypedArrayAttrBase<ISQ_GateDefinitionAttr, "Gate definition array attribute">;



def DenseComplexF64ArrayAttr : DenseArrayAttrBase<"DenseComplexF64ArrayAttr", "std::complex<double>", "complexF64">;

#endif
