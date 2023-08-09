#ifndef ISQ_TBLGEN_ISQOPS_TD_
#define ISQ_TBLGEN_ISQOPS_TD_

include "isq/tblgen/ISQBase.td"
//include "mlir/Dialect/Shape/IR/ShapeBase.td"

////////////////////////////////////
// Operations
////////////////////////////////////

class ISQ_Op<string mnemonic, list<Trait> traits = []> :
    Op<ISQ_Dialect, mnemonic, traits> {
}

def ISQ_DefgateOp : ISQ_Op<"defgate", [Symbol, DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let summary = "Defines or declares an array of gates.";
    let description = [{
        Syntax:
        ```
        operation ::= $sym_name attr-dict `:` $type
        ```
        Declares or defines an array of gate references.

        Defgate comes with an optional shape attribute.
        If the shape is present, the gate will be used as an array;
        otherwise the shape will be used as a single gate.

        Defgate allows multiple and overlapping definition of one gate.
        A defgate without shape and definition is seen as declaration.
        Which definition to choose while optimization and lowering depends on the optimization passes.



        Currently supported definitions:
        - `unitary`, unitary matrix definition. Not compatible with attribute `shape`.
        - `decomposition`, a `builtin.func` to represent the decomposition. The function should have the right signature and obeys correspondence between args and results.
        - `qir`, QIR call.

        (TODO: ways to define a gate are still not defined. We may want to support:
        - Matrix representation, where complex matrix is given out directly.
        - Gate decomposition, allowing using an entire decomposition function as a single gate.
        - Controlled-gate (Functors)
        - Classical oracle conversion
        - ...
        )

        ```mlir
        // Gate declaration
        isq.defgate @cnot : !isq.gate<2, hermitian>
        // Gate definition
        isq.defgate @hadamard {definition = [ /* possible gate definition */ ]} : !isq.gate<1, hermitian>
        // Gate array declaration
        isq.defgate @one_way_equivalent_gates [10, 10] : !isq.gate<1>
        // Gate family definition
        isq.defgate @rz(f64) { definitions = [ /* external call */ ]} : !isq.gate<1, diagonal>
        ```

        The symbol always uses `nested` visbility.
    }];
    let arguments = (ins
        GateTypeAttr: $type,
        SymbolNameAttr: $sym_name,
        StrAttr: $sym_visibility,
        OptionalAttr<I64ArrayAttr>: $shape,
        OptionalAttr<ISQ_GateDefinitionList>: $definition,
        TypeArrayAttr: $parameters
    );
    let results = (outs);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    /*
    let assemblyFormat = [{
        custom<GateVisibility>($sym_visibility) $sym_name (`(` $parameters^ `)`)? ( $shape^ )? attr-dict `:` $type
    }];
    */
    /*
    let parser = [{return $cppClass::parseIR(parser, result);}];
    let printer = [{return this->printIR(p);}];
    
    let verifier = [{
        return this->verifyIR();
    }];
    */
    let extraClassDeclaration = [{
        bool isGateArray();
        bool isDeclaration();
        static ::mlir::LogicalResult parseIR(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
        void printIR(::mlir::OpAsmPrinter& p);
        static bool parseGateVisibility(::mlir::OpAsmParser& parser, ::mlir::StringAttr& v);
        void printGateVisibility(::mlir::OpAsmPrinter& p, DefgateOp op, ::mlir::StringAttr v);
        ::mlir::Type getTypeWhenUsed();
        ::mlir::LogicalResult verifyIR();
    }];
}

def ISQ_DeclareQOpOp : ISQ_Op<"declare_qop", [Symbol]>{
    let summary = "Declares general quantum operations";
    let description = [{
        Syntax:
        ```
        operation ::= $sym_name attr-dict `:` `[` $size `]` $signature
        ```

        Declares a general quantum operation that is most possibly not pure, i.e. a measurement.
        A QOp can be seen as a general operation accepting several qubits as fan-in and fan-out,
        while accepting possible (classical) inputs and outputs.


        Difference between QOp and general functions: QOp guarantees correspondence between input and output qstates.
        
        (TODO: recognizing general function as qop.)
        ```mlir
        // Computational basis measurement, accepting 1 qubit, no input parameters and outputting a boolean value.
        isq.declare_qop @measure : [1]()->i1
        // Try to measure a quantum money qubit, accepting 1 bit to specity basis and outputting 1 bit.
        isq.declare_qop @check_quantum_money_bit : [2](i1)->i1
        ```

        The symbol always uses `nested` visbility.
    }];
    let arguments = (ins
        SymbolNameAttr: $sym_name,
        StrAttr: $sym_visibility,
        UI64Attr: $size,
        FuncTypeAttr: $signature
    );
    let results = (outs);
    /*
    let assemblyFormat = [{
        $sym_name attr-dict `:` `[` $size `]` $signature
    }];
    */
    let hasCustomAssemblyFormat = 1;
    /*
    let parser = [{return $cppClass::parseIR(parser, result);}];
    let printer = [{return this->printIR(p);}];
    */
    let extraClassDeclaration = [{
        ::mlir::Type getTypeWhenUsed();
        bool isDeclaration();
        static ::mlir::LogicalResult parseIR(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
        void printIR(::mlir::OpAsmPrinter& p);
    }];

}
def ISQ_UseGateOp : ISQ_Op<"use", [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let summary = "Use gate array";
    let description = [{
        Fetching a declared/defined global gate or gate array.
        Gate arrays are readonly and writing into gate array is currently not allowed.

        The return type depends on whether the `isq.defgate` declares a gate or a gate array,
        i.e. whether the `isq.defgate` contains the `shape` attribute.

        ```mlir
        isq.defgate @one_gate : !isq.gate<1, hermitian>
        isq.defgate @gates {shape=[10,20]}: !isq.gate<1>
        isq.defgate @u3(f64, f64, f64) : !isq.gate<1>
        func @use_gates()->(){
            %g1 = isq.use @one_gate : !isq.gate<1, hermitian>
            %g = isq.use @gates : memref<10x20x!isq.gate<1>>
            %zero = arith.constant 0.0 : f64
            %identity = isq.use @u3(%zero, %zero, %zero) : (f64, f64, f64)->!isq.gate<1>
            return
        }
        ```
    }];
    let arguments = (ins 
    SymbolRefAttr: $name,
    Variadic<AnyType>: $parameters
    );
    let results = (outs
    AnyType: $result);
    
    let assemblyFormat = [{
        $name attr-dict  ( `(` $parameters^ `)` `:` `(` qualified(type($parameters)) `)` `->`):(`:`)? qualified(type(results))
    }];
    
}

def ISQ_ContribUndef : ISQ_Op<"contrib.memref.undef", [Pure]>{
    let summary = "Undef value for a memref.";
    let description = [{
        The value that you will obtain by reading from an uninitialized memory.
    }];
    let arguments = (ins
        Variadic<AnyType>: $args
    );
    let results = (outs 
        AnyType: $result
    );
}

def ISQ_ApplyGPhase : ISQ_Op<"apply_gphase", []>{
    let summary = "Auxiliary operation that applies global-phase";
    let description = [{
        Auxiliary operation that side-effectfully applies a global-phase gate.
    }];
    let arguments = (ins
        ISQ_Gate: $gate
    );
    let assemblyFormat = [{
        $gate attr-dict `:` qualified(type($gate))
    }];
}

def ISQ_AccumulateGPhase : ISQ_Op<"accumulate_gphase", []>{
    let summary = "Auxiliary operation that accumulates global phase on a sub-system onto total global phase";
    let description = [{
        Auxiliary operation that prevents a procedure-local quantum subsystem from being optimized out.
        A sufficient condition that can prevent a procedure-local quantum subsystem from getting optimized out is creating entanglement between an external system. However, an isolated local system may introduce global phase onto the system as well, thus local phase for controlled-gates.
        This operation is inserted before every dealloc op. If the subsystem is linked to another subsystem, the semantics of the operation is "no-op"; otherwise, the semantics of the operation will be two `global-phase` gates: one global-phase zeroing out current subsystem and another global-phase on the other part of the system.
    }];
    let arguments = (ins
        AnyType: $subsystem
    );
    let assemblyFormat = [{
        $subsystem attr-dict `:` qualified(type($subsystem))
    }];
}

def ISQ_ApplyGateOp : ISQ_Op<"apply", [
    Pure,
    TypesMatchWith<
        "args equivalence", "gate", "args", "::isq::ir::getGateParameterType($_self.cast<::isq::ir::GateType>())"
    >,TypesMatchWith<
        "output equivalence", "gate", "r", "::isq::ir::getGateParameterType($_self.cast<::isq::ir::GateType>())"
    >]>{
    let summary = "Applies a quantum gate onto qstates";
    let description = [{
        Performing a gate onto qstates. Qstate outputs are in the same order as inputs.

        ```mlir
        isq.defgate @hadamard : !isq.gate<1, hermitian>
        isq.defgate @cnot : !isq.gate<2, hermitian>
        func @bell(%q1: !isq.qstate, %q2: !isq.qstate)->(!isq.qstate, !isq.qstate){
            %hadamard = isq.use @hadamard : !isq.gate<1, hermitian>
            %cnot = isq.use @cnot : !isq.gate<2, hermitian>
            %q3 = isq.apply %hadamard(%q1): !isq.gate<1, hermitian>
            %q4, %q5 = isq.apply %cnot(%q3, %q2): !isq.gate<2, hermitian>
            return %q4, %q5 : !isq.qstate, !isq.qstate
        }
        ```
    }];
    let arguments = (ins
        ISQ_Gate: $gate,
        Variadic<ISQ_QState>: $args
    );
    let results = (outs Variadic<ISQ_QState>: $r);
    let assemblyFormat = [{
        $gate `(` $args `)` attr-dict `:` qualified(type($gate))
    }];
    let hasVerifier = 1;
    /*
    let verifier = [{
        return this->verifyIR();
    }];
    */
    let extraClassDeclaration = [{
        ::mlir::LogicalResult verifyIR();
    }];
    let hasCanonicalizer = 1;
}

def ISQ_CallQOpOp: ISQ_Op<"call_qop", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let summary = "Applies a quantum operation onto qstates";
    let description = [{
        Syntax:
        ```
        operation ::= $callee `(` $args `)` attr-dict `:` `[` $size `]` $signature
        ```
        Performing a quantum operation onto qstates by given symbol and input, yielding outputs.
        Quantum operation arrays are currently disallowed.

        The syntax of `call_qop` resembles a function call.
        The list of required qstates are attached to the front of argument list,
        while the outputting qstates are attached to the front of result list.

        ```mlir
        // Measurement accepts one qubit, no inputs and yields one output.
        isq.declare_qop @measure : [1]()->i1

        func @measure_and_discard_result(%q: !isq.qstate)->!isq.qstate{
            // Here, [1]()->i1 corresponds to (!isq.qstate)->(!isq.qstate, i1).
            %q1, %c = isq.call_qop @measure(%q): (!isq.qstate)->(!isq.qstate, i1)
            return %q1: !isq.qstate
        }
        ```
    }];
    
    
    let arguments = (ins
        SymbolRefAttr: $callee,
        Variadic<AnyType>: $args,
        UI64Attr: $size,
        FuncTypeAttr: $signature
    );

    let results = (outs Variadic<AnyType>);
    let extraClassDeclaration = [{
        static ::mlir::ParseResult parseIR(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
        void printIR(::mlir::OpAsmPrinter& p);
    }];
    /*
    let parser = [{return $cppClass::parseIR(parser, result);}];
    let printer = [{return this->printIR(p);}];
    */
    let hasCustomAssemblyFormat = 1;

    /*
    let assemblyFormat = [{
        $callee `(` $args `)` attr-dict `:` `[` $size `]` $signature
    }];
    */

}

def ISQ_DowngradeGateOp: ISQ_Op<"downgrade", [Pure]>{
    let summary = "Downgrade a gate by removing its traits.`";
    let description = [{
        This operation is used to perform safe gate reference type casting.
        For example, if we want to pass the $CZ$ gate with type `!isq.gate<2, hermitian, symmetric>` to a function
        that accepts `!isq.gate<2>` only, this operation is needed for a downcast.
        ```mlir
        isq.defgate @hadamard : !isq.gate<1, hermitian>
        func @apply_twice(%g: !isq.gate<1>, %q: !isq.qstate)->!isq.qstate{
            %q1 = isq.apply %g(%q): !isq.gate<1>
            %q2 = isq.apply %g(%q1): !isq.gate<1>
            return %q2: !isq.qstate
        }
        func @apply_hadamard_twice(%q: !isq.qstate)->!isq.qstate{
            %hadamard = isq.use @hadamard : !isq.gate<1, hermitian>
            // Erasing the `hermitian` trait.
            %downgraded_hadamard = isq.downgrade (%hadamard: !isq.gate<1, hermitian>) : !isq.gate<1>
            %q2 = std.call @apply_twice(%downgraded_hadamard, %q) : (!isq.gate<1>, !isq.qstate)->!isq.qstate
            return %q2: !isq.qstate
        }
        ```
    }];
    let arguments = (ins 
        ISQ_Gate: $args
    );
    let results = (outs ISQ_Gate);
    let assemblyFormat = [{
        `(` $args `:` qualified(type($args)) `)` attr-dict `:`qualified(type(results))
    }];
    /*
    let verifier = [{
        return ::isq::ir::verify(*this);
    }];
    */
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def ISQ_DecorateOp: ISQ_Op<"decorate", [Pure]>{
    let summary = "Decorate a gate by adding controlled and adjoint markers.";
    let description = [{
        This operations is used to derive a controlled / inverse gate from a used gate.
    }];
    let arguments = (
        ins ISQ_Gate: $args,
        BoolAttr: $adjoint,
        BoolArrayAttr : $ctrl
    );
    let results = (outs ISQ_Gate);
    let assemblyFormat = [{
        `(` $args `:` qualified(type($args)) `)` attr-dict `:` qualified(type(results))
    }];
    let extraClassDeclaration = [{
    public:
        static GateTrait computePostDecorateTrait(GateTrait attr, int ctrl, bool adj, bool ctrl_all_one);
    }];
    /*
    let verifier = [{
        return ::isq::ir::verify(*this);
    }];*/
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def ISQ_PassOp :ISQ_Op<"pass"> {
    let assemblyFormat = [{
        attr-dict
    }];
}

def ISQ_AssertOp: ISQ_Op<"assert">{
    let summary = "Validate classical dynamic execution conditions.";
    let description = [{
        If the condition is false, report an error and abort the program execution.
    }];
    let arguments = (ins
        BoolLike: $cond,
        UI64Attr: $error_num
    );
    let results = (outs);
    let assemblyFormat = [{
        $cond `:` qualified(type($cond)) `,` $error_num attr-dict
    }];
}

def ISQ_AssertQOp: ISQ_Op<"assertq">{
    let summary = "Validate qubits are in specified Hilbert space.";
    let description = [{
        If qubits are not in the specified space, report an error and abort the program execution.
    }];
    let arguments = (ins
        AnyType: $cond,
        DenseComplexF64MatrixAttr: $space
    );
    let results = (outs);
    let assemblyFormat = [{
        $cond `:` qualified(type($cond)) `,` $space attr-dict
    }];
}

def ISQ_InitOp: ISQ_Op<"init">{
    let summary = "Set qubits in the specified state.";
    let description = [{
        First reset all the qubits. Then apply a generated circuit to these qubits.
    }];
    let arguments = (ins
        AnyType: $qubits,
        DenseComplexF64MatrixAttr: $state
    );
    let results = (outs);
    let assemblyFormat = [{
        $qubits `:` qualified(type($qubits)) `,` $state attr-dict
    }];
}

def ISQ_InitKetOp: ISQ_Op<"init_ket">{
    let summary = "Set qubits in the specified state in |ket> representation.";
    let description = [{
        First reset all the qubits. Then apply a generated circuit to these qubits.
    }];
    let arguments = (ins
        AnyType: $qubits,
        ISQ_Ket: $state
    );
    let results = (outs);
    let assemblyFormat = [{
        $qubits `:` qualified(type($qubits)) `,` $state attr-dict
    }];
}

def ISQ_KetOp: ISQ_Op<"create_ket", [Pure]> {
    let summary = "Ket representation of a column vector.";
    let description = [{
        Representing a computational basis with a complex amplitude.
    }];
    let arguments = (ins
        Complex<AnyFloat>: $coeff,
        UI64Attr: $basis
    );
    let results = (outs ISQ_Ket);
    let assemblyFormat = [{
        $coeff `:` qualified(type($coeff)) `,` $basis attr-dict `:` qualified(type(results))
    }];
}

class Ket_Op<string mnemonic, list<Trait> traits = [Pure]> : ISQ_Op<mnemonic, traits> {
    let arguments = (ins
        ISQ_Ket: $lhs,
        ISQ_Ket: $rhs
    );
    let results = (outs ISQ_Ket);
    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` qualified(type(results))
    }];
}

def ISQ_AddOp: Ket_Op<"add"> {
    let summary = "Add operation between two ket expressions.";
    let description = [{
        Representing the sum of two ket expressions.
    }];
}

def ISQ_SubOp: Ket_Op<"sub"> {
    let summary = "Sub operation between two ket expressions.";
    let description = [{
        Representing the difference of two ket expressions.
    }];
}

/// QMPI


// Consider an MPI interface with multiple implementations.
def ISQ_QMPI_Comm : TypeDef<ISQ_Dialect, "MPIComm", [MemRefElementTypeInterface]>{
    let mnemonic = "qmpi.comm";
}
def ISQ_QMPI_World : ISQ_Op<"qmpi.world">{
    let results = (outs 
        ISQ_QMPI_Comm: $comm
    );
}

def ISQ_QMPI_Me : ISQ_Op<"qmpi.me">{
    let arguments = (ins
        ISQ_QMPI_Comm: $comm
    );
    let results = (outs 
        Builtin_Index: $rank
    );
}

def ISQ_QMPI_CommSize : ISQ_Op<"qmpi.commsize">{
    let arguments = (ins
        ISQ_QMPI_Comm: $comm
    );
    let results = (outs 
        Builtin_Index: $size
    );
}

def ISQ_QMPI_EPR : ISQ_Op<"qmpi.epr">{
    let arguments = (ins
        ISQ_QState : $qin,
        Builtin_Index : $peer,
        ISQ_QMPI_Comm: $comm
    );
    let results = (outs
        ISQ_QState: $qout
    );
}

def ISQ_QMPI_CSend : ISQ_Op<"qmpi.csend">{
    let arguments = (ins
        Builtin_Index: $value,
        Builtin_Index : $peer,
        ISQ_QMPI_Comm: $comm
    );
}
def ISQ_QMPI_CRecv : ISQ_Op<"qmpi.crecv">{
    let arguments = (ins
        Builtin_Index : $peer,
        ISQ_QMPI_Comm: $comm
    );
    let results = (outs
        Builtin_Index: $value
    );
}


class ISQ_QMPI_Blocking_P2P<string mnemonic> : ISQ_Op<mnemonic>{
    let arguments = (ins
        BoolAttr : $move,
        ISQ_QState : $qin,
        Builtin_Index : $peer,
        ISQ_QMPI_Comm: $comm
    );
    let results = (outs
        ISQ_QState: $qout
    );
}
def ISQ_QMPI_Blocking_Send : ISQ_QMPI_Blocking_P2P<"qmpi.blocking.send">;
def ISQ_QMPI_Blocking_Unsend : ISQ_QMPI_Blocking_P2P<"qmpi.blocking.unsend">;
def ISQ_QMPI_Blocking_Recv : ISQ_QMPI_Blocking_P2P<"qmpi.blocking.recv">;
def ISQ_QMPI_Blocking_Unrecv : ISQ_QMPI_Blocking_P2P<"qmpi.blocking.unrecv">;

def ISQ_QMPI_Blocking_Bcast : ISQ_Op<"qmpi.blocking.bcast">{
    let arguments = (ins
        ISQ_QState : $qin,
        Builtin_Index : $root,
        ISQ_QMPI_Comm: $comm
    );
    let results = (outs
        ISQ_QState: $qout
    );
}
def ISQ_QMPI_Blocking_Gather : ISQ_Op<"qmpi.blocking.gather">{
    let arguments = (ins
        BoolAttr : $move,
        ISQ_QState : $qsend,
        MemRefRankOf<[ISQ_QState], [1]> : $qrecvbuf,
        Builtin_Index : $root,
        ISQ_QMPI_Comm: $comm
    );
    let results = (outs
        ISQ_QState: $qsendout
    );
}
def ISQ_QMPI_Blocking_Scatter : ISQ_Op<"qmpi.blocking.scatter">{
    let arguments = (ins
        BoolAttr : $move,
        ISQ_QState : $qrecv,
        MemRefRankOf<[ISQ_QState], [1]> : $qsend,
        Builtin_Index : $root,
        ISQ_QMPI_Comm: $comm
    );
    let results = (outs
        ISQ_QState: $qrecvout
    );
}
def ISQ_QMPI_Blocking_AllToAll : ISQ_Op<"qmpi.blocking.alltoall">{
    let arguments = (ins
        ISQ_QState : $qin,
        MemRefRankOf<[ISQ_QState], [1]> : $buffer,
        ISQ_QMPI_Comm: $comm
    );
    let results = (outs
        ISQ_QState : $qout
    );
}
def ISQ_QMPI_Blocking_AllToAllMove : ISQ_Op<"qmpi.blocking.alltoall_move">{
    let arguments = (ins
        MemRefRankOf<[ISQ_QState], [1]> : $buffer,
        ISQ_QMPI_Comm: $comm
    );
    let results = (outs
    );
}

#endif
