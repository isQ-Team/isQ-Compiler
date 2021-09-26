include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
//include "mlir/Dialect/Shape/IR/ShapeBase.td"

////////////////////////////////////
// Dialect
////////////////////////////////////

def ISQ_Dialect: Dialect{
    let name = "isq";
    let cppNamespace = "::isq::ir";
    let summary = "ISQ Quantum MLIR Dialect.";
    let description = [{
        MLIR Dialect for expressing and optimization from low-level quantum circuits, to high level quantum programs with complicated control structures.
    }];
    let dependentDialects = ["::mlir::memref::MemRefDialect"];
}

////////////////////////////////////
// Types
////////////////////////////////////

class ISQ_Type<string name> : TypeDef<ISQ_Dialect, name>{}
class ISQ_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ISQ_Dialect, mnemonic, traits>{
    
}

// We use initialize-list-and-store to perform operations.
def ISQ_QState : TypeDef<ISQ_Dialect, "QState", [MemRefElementTypeInterface]>{
    let description = [{
        Represents an intermediate qubit SSA state. Written `!isq.qstate`.

        **Representation of qubit**: we use `memref<!isq.qstate>` to represent qubits.

        **Qubit state lifetime rule**: every `!isq.qstate` SSA value must be used **exactly once**. 
        The source of an `!isq.qstate` should be a memory-reading operation like `affine.load`, 
        and the drain should be a memory-writing operation like `affine.store`.
        An `!isq.qstate` SSA value is called **alive** between its definition and its usage.

        **Qubit state aliasing rule**: at any moment two alive `!isq.qstate` SSA values must refer to two different qubits.
        This is similar to the aliasing rule for `&mut` references in Rust or the functionality of `__restrict`(`restrict`) keyword in C.
        In this sense, all multi-qubit calls are naturally legit.

        This SSA representation is inspired by [David et al.](https://arxiv.org/abs/2101.11030).
    }];
    let summary = [{Qubit state}];
    let mnemonic = "qstate";

    let printer = [{
        QStateParsePrint p;
        p.printType(*this, $_printer);
    }];
    let parser = [{
        QStateParsePrint p;
        return p.parseType($_parser);
    }];
    
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
        - `symmetric`, indicating the gate is symmetric over its **all** operands.

        Note that we allow loading and storing gates at runtime.
    }];
    let parameters = (ins "int64_t":$size,"GateTrait":$hints);
    let summary = [{Gate reference}];
    let mnemonic = "gate";
    let printer = [{
        GateParsePrint p;
        p.printType(*this, $_printer);
    }];
    let parser = [{
        GateParsePrint p;
        return p.parseType($_parser);
    }];
    let extraClassDeclaration = [{
        ::mlir::SmallVector<mlir::Type> getApplyParameterType(GateType ty);
    }];
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

def GateTypeAttr: TypeAttrBase<"::isq::ir::GateType", "Gate type as attribute">;
def FuncTypeAttr: TypeAttrBase<"::mlir::FunctionType", "Function type attribute, for operands and results">;
//def ShapeAttr: TypeAttrBase<"::mlir::shape::ShapeType", "Shape type attribute.">;

////////////////////////////////////
// Operations
////////////////////////////////////

def ISQ_DefgateOp : ISQ_Op<"defgate", [Symbol]>{
    let summary = "Defines or declares an array of gates.";
    let description = [{
        Declares or defines an array of gate references.

        Defgate comes with an optional shape attribute.
        If the shape is present, the gate will be used as an array;
        otherwise the shape will be used as a single gate.

        Defgate allows multiple and overlapping definition of one gate.
        A defgate without definition is seen as declaration.
    }];
    let arguments = (ins
        GateTypeAttr: $type,
        SymbolNameAttr: $sym_name,
        OptionalAttr<I64ArrayAttr>: $shape,
        OptionalAttr<ArrayAttr>: $definition
    );
    let results = (outs);
    let assemblyFormat = [{
        $sym_name attr-dict `:` $type
    }];
    let extraClassDeclaration = [{
        bool isGateArray();
        bool isDeclaration();
        ::mlir::Type getTypeWhenUsed();
    }];
}

def ISQ_DeclareQOpOp : ISQ_Op<"declare_qop", [Symbol]>{
    let summary = "Declares general quantum operations";
    let description = [{
        Declares a general quantum operation that is most possibly not pure, i.e. a measurement.
    }];
    let arguments = (ins
        SymbolNameAttr: $sym_name,
        UI64Attr: $size,
        FuncTypeAttr: $signature
    );
    let results = (outs);
    let assemblyFormat = [{
        $sym_name attr-dict `:` `[` $size `]` $signature
    }];
    let extraClassDeclaration = [{
        ::mlir::Type getTypeWhenUsed();
    }];

}
def ISQ_UseGateOp : ISQ_Op<"use", [NoSideEffect, DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let summary = "Use gate array";
    let description = [{
        Fetching a declared/defined global gate reference array. Returned as `memref<!isq.gate<...>>`.
        Gate arrays are readonly and writing into gate array is currently not allowed.
    }];
    let arguments = (ins 
    SymbolRefAttr: $name
    );
    let results = (outs
    AnyType: $result);
    let assemblyFormat = [{
        $name attr-dict `:` type(results)
    }];
}
def ISQ_ApplyGateOp : ISQ_Op<"apply", [NoSideEffect,
    TypesMatchWith<
        "args equivalence", "gate", "args", "::isq::ir::getGateParameterType($_self.cast<::isq::ir::GateType>())"
    >,TypesMatchWith<
        "output equivalence", "gate", "r", "::isq::ir::getGateParameterType($_self.cast<::isq::ir::GateType>())"
    >]>{
    let summary = "Applies a quantum gate onto qstates";
    let description = [{
        Performing a gate reference onto qstates. Qstate outputs are in the same order as inputs.
    }];
    let arguments = (ins
        ISQ_Gate: $gate,
        Variadic<ISQ_QState>: $args
    );
    let results = (outs Variadic<ISQ_QState>: $r);
    let assemblyFormat = [{
        $gate `(` $args `)` attr-dict `:` type($gate)
    }];
}
def ISQ_CallQOpOp: ISQ_Op<"call_qop", [NoSideEffect, DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let summary = "Applies a quantum operation onto qstates";
    let description = [{
        Performing a quantum operation onto qstates by given symbol and input, yielding outputs.
        Quantum operation arrays are currently disallowed.
    }];
    
    let arguments = (ins
        SymbolRefAttr: $callee,
        Variadic<AnyType>: $args
    );
    let results = (outs Variadic<AnyType>);
    let assemblyFormat = [{
        $callee `(` $args `)` attr-dict `:` functional-type($args, results)
    }];

}
/*
def ISQ_ApplyOp : ISQ_Op<"apply", [NoSideEffect, TypesMatchWith<
        "args equivalence", "op", "args", "$_self.cast<::isq::ir::QOpType>().getFuncType().getInputs()"
    >,TypesMatchWith<
        "output equivalence", "op", "output", "$_self.cast<::isq::ir::QOpType>().getFuncType().getResults()"
    >]>{
    let summary = "Applies a quantum operation onto a quantum state.";
    let description = [{
        Performs QOp on QState to generate new QState.
    }];
    let arguments = (ins
        ISQ_QOp: $op,
        Variadic<AnyType>: $args
    );
    let results = (outs Variadic<AnyType>: $output);
    
    //let parser = [{return ::isq::ir::parse$cppClass(parser, result);}];
    //let printer = [{return ::isq::ir::print(p, *this);}];
    //let verifier = [{return ::isq::ir::verify(*this);}];
    
    let assemblyFormat = [{
        $op `(` $args `)` attr-dict `:` type($op)
    }];
    let hasCanonicalizer = 1;
}

def ISQ_UseOp: ISQ_Op<"use_gate", [NoSideEffect, TypesMatchWith<
    "op type inference", "gate", "op", "::isq::ir::gateToQOp($_self.cast<::isq::ir::GateType>())"
    >]>{
    let summary = "Uses a gate as an operation.";
    let arguments = (ins
        ISQ_Gate: $gate
    );
    let results = (outs ISQ_QOp: $op);
    let assemblyFormat = [{
        $gate attr-dict `:` type($gate)
    }];
}
*/

//def QOpTypeAttr: TypeAttrBase<"::isq::ir::QOpType", "Quantum operation type as attribute">;
/*
def ISQ_GateOp : ISQ_Op<"gate", [NoSideEffect]>{
    let summary = "Declare/define a gate.";
    let description = [{
        Declaring a predefined gate from attributes.
    }];
    let arguments = (ins
        StrAttr: $name,
        GateTypeAttr: $gate_type
    );
    let results = (outs ISQ_Gate: $outgate);
    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];
    let verifier = [{
        return ::isq::ir::verify(*this);
    }];
}
*/
/*

def ISQ_DeclareOp: ISQ_Op<"declare_op", [NoSideEffect]>{
    let summary = "Declare an external quantum operation.";
    let arguments = (ins
        StrAttr: $name,
        QOpTypeAttr: $op_type
    );
    let results = (outs ISQ_QOp: $out_op);
    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];
    let verifier = [{
        return ::isq::ir::verify(*this);
    }];
}*/

def ISQ_DowngradeGateOp: ISQ_Op<"downgrade", [NoSideEffect]>{
    let summary = "Downgrade a gate by removing its traits.`";
    let description = [{
        This operation is used to perform safe gate reference type casting.
        For example, if we want to pass the $CZ$ gate with type `!isq.gate<2, hermitian, symmetric>` to a function
        that accepts `!isq.gate<2>` only, this operation is needed for a downcast.
    }];
    let arguments = (ins ISQ_Gate: $args);
    let results = (outs ISQ_Gate);
    let assemblyFormat = [{
        `(` $args `:` type($args) `)` attr-dict `:` type(results)
    }];
    let verifier = [{
        return ::isq::ir::verify(*this);
    }];
}



// Gate traits

def GateTraitGeneral: BitEnumAttrCase<"General", 0, "general">;
def GateTraitDiagonal: BitEnumAttrCase<"Diagonal", 1, "diagonal">;
def GateTraitAntidiagonal: BitEnumAttrCase<"Antidiagonal", 2, "antidiagonal">;
def GateTraitSymmetric: BitEnumAttrCase<"Symmetric", 4, "symmetric">;
def GateTraitHermitian: BitEnumAttrCase<"Hermitian", 8, "hermitian">;

def GateTrait: BitEnumAttr<"GateTrait", "Gate traits/hints", [GateTraitGeneral, GateTraitDiagonal, GateTraitAntidiagonal, GateTraitSymmetric, GateTraitHermitian]>;

/*
def EliminateHermitianPairPattern: Pat<
(ISQ_ApplyOp 
    (ISQ_UseOp (ISQ_GateOp $name1, $attr1)), 
    (ISQ_ApplyOp (ISQ_UseOp (ISQ_GateOp $name2, $attr2)), $qubits)), (replaceWithValue $qubits)>;
*/