/*
Data types:
!isq.qstate: represents an intermediate state of qubit.
!isq.gate<n: int, [traits: symmetric | diagonal | antiDiagonal | hermitian]=[]>: represents a gate, the basic unit of execution.
!isq.qop<arguments->return type> # represents a generic operation. 


Operations:
# use memref dialect for allocation, read and store.
isq.apply : (func)->!isq.qop # referencing a named operation.

Gate primitives:
# use memref dialect for gate allocation.
isq.gate : {name: string, ty: !isq.gate<size, hints> }->!isq.gate<size, hints>
isq.generic: {name: string, size: int, outcome: ty}->!isq.qop<{qubit states}->{qubit states, ty}>
isq.downgrade : !isq.gate<size, hints>->!isq.gate<size, {less_hints}>

Gate high level constructs:
isq.cnot : ()->!isq.gate<2>
isq.cz : ()->!isq.gate<2, [Symmetric]>
isq.sq : {mat<1>}()->!isq.gate<1>
isq.measure: ()->!isq.qop<(!isq.qstate->(!isq.qstate, i1))>
isq.matrix : {mat<n>}()->!isq.gate<n> # no hints, needs an inference pass.


Notations:
isq.schedule {max_width: int} [| |] # represents an explicit scheduling.
isq.tick [| |] # represents a tick in explicit scheduling. must be placed in isq.schedule and the width must match.

*/
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
def ISQ_Dialect: Dialect{
    let name = "isq";
    let cppNamespace = "::isq::ir";
    let summary = "ISQ Quantum MLIR Dialect.";
    let description = [{
        MLIR Dialect for expressing and optimization from low-level quantum circuits, to high level quantum programs with complicated control structures.
    }];
}

class ISQ_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ISQ_Dialect, mnemonic, traits>{
    
}

// We use initialize-list-and-store to perform operations.
def ISQ_QState : DialectType<ISQ_Dialect, CPred<"$_self.isa<::isq::ir::QStateType>()">, "Qubit state">{
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
}
def ISQ_Gate : DialectType<ISQ_Dialect, CPred<"$_self.isa<::isq::ir::GateType>()">, "Gate type">;
def ISQ_QOp: DialectType<ISQ_Dialect, CPred<"$_self.isa<::isq::ir::QOpType>()">, "Boxed quantum operation type">;


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
    /*
    let parser = [{return ::isq::ir::parse$cppClass(parser, result);}];
    let printer = [{return ::isq::ir::print(p, *this);}];
    let verifier = [{return ::isq::ir::verify(*this);}];
    */
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

def GateTypeAttr: TypeAttrBase<"::isq::ir::GateType", "Gate type as attribute">;
def QOpTypeAttr: TypeAttrBase<"::isq::ir::QOpType", "Quantum operation type as attribute">;
    
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
}

def ISQ_DowngradeGateOp: ISQ_Op<"downgrade", [NoSideEffect]>{
    let summary = "Downgrade a gate by removing its traits.";
    let arguments = (ins ISQ_Gate: $args);
    let results = (outs ISQ_Gate);
    let assemblyFormat = [{
        `(` $args `:` type($args) `)` attr-dict `:` type(results)
    }];
    let verifier = [{
        return ::isq::ir::verify(*this);
    }];
}



/*
def EliminateHermitianPairPattern: Pat<
(ISQ_ApplyOp 
    (ISQ_UseOp (ISQ_GateOp $name1, $attr1)), 
    (ISQ_ApplyOp (ISQ_UseOp (ISQ_GateOp $name2, $attr2)), $qubits)), (replaceWithValue $qubits)>;
*/