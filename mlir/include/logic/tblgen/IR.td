include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "isq/tblgen/ISQBase.td"

////////////////////////////////////
// Dialect
////////////////////////////////////

def Logic_Dialect: Dialect {
    let name = "logic";
    let cppNamespace = "::logic::ir";
    let summary = "Logic MLIR Dialect.";
    let dependentDialects = ["::isq::ir::ISQDialect"];
    let description = [{
        MLIR Dialect for expressing logic operations between boolean vectors.
    }];
}

class Logic_Op<string mnemonic, list<Trait> traits = []> :
    Op<Logic_Dialect, mnemonic, traits> {}

def Logic_FuncOp : Logic_Op<"func", [FunctionOpInterface]> {
    let summary = "Define a logic function.";
    let arguments = (ins
        SymbolNameAttr: $sym_name,
        TypeAttrOf<FunctionType>: $function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs,
        OptionalAttr<StrAttr>:$sym_visibility
    );
    let regions = (region AnyRegion: $body);
    let extraClassDeclaration = [{

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

        /// Returns the results types that the callable region produces when
        /// executed.
        ::mlir::ArrayRef<::mlir::Type> getCallableResults() { return getFunctionType().getResults(); }

        /// Returns the argument types of this function.
        ::mlir::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ::mlir::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
    }];
    let assemblyFormat = [{
        $sym_name `:` $function_type $body attr-dict
    }];
}

// Base class for unary and binary logic ops
class Logic_LogicalOp<string mnemonic, list<Trait> traits = []>:
    Logic_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]>;

class Logic_BinaryOp<string mnemonic, list<Trait> traits = []>:
    Logic_LogicalOp<mnemonic, traits> {
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class Logic_BitBinaryOp<string mnemonic, list<Trait> traits = []>:
    Logic_BinaryOp<mnemonic, traits>,
    Arguments<(ins BoolLike: $lhs, BoolLike: $rhs)>,
    Results<(outs BoolLike: $result)>;

def Logic_AndOp : Logic_BitBinaryOp<"and"> {
    let summary = "AND operation between two bool values";
}

def Logic_OrOp : Logic_BitBinaryOp<"or"> {
    let summary = "OR operation between two bool values";
}

def Logic_XorOp : Logic_BitBinaryOp<"xor"> {
    let summary = "XOR operation between two bool values";
}

def Logic_XnorOp : Logic_BitBinaryOp<"xnor"> {
    let summary = "XNOR operation between two bool values";
}

class Logic_VecBinaryOp<string mnemonic, list<Trait> traits = []>:
    Logic_BinaryOp<mnemonic, traits>,
    Arguments<(ins AnyType: $lhs, AnyType: $rhs)>,
    Results<(outs AnyType: $result)>;

def Logic_AndvOp : Logic_VecBinaryOp<"andv"> {
    let summary = "bit-wise AND operation for bit array";
}

def Logic_OrvOp : Logic_VecBinaryOp<"orv"> {
    let summary = "bit-wise OR operation for bit array";
}

def Logic_XorvOp : Logic_VecBinaryOp<"xorv"> {
    let summary = "bit-wise XOR operation for bit array";
}

def Logic_NotOp : Logic_LogicalOp<"not"> {
    let arguments = (ins BoolLike: $operand);
    let results = (outs BoolLike: $result);
    let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def Logic_NotvOp : Logic_LogicalOp<"notv"> {
    let arguments = (ins I1MemRef: $operand);
    let results = (outs I1MemRef: $result);
    let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def Logic_CallOp : Logic_Op<"call"> {
    let summary = "call operation";
    let arguments = (ins
        FlatSymbolRefAttr: $callee,
        Variadic<AnyType>: $operands
    );
    let results = (outs Variadic<AnyType>);
    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];
}

def Logic_ReturnOp : Logic_Op<"return", [ReturnLike, Terminator]> {
    let summary = "return operation";
    let arguments = (ins Variadic<AnyType>:$operands);
    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Logic_UseGateOp : Logic_Op<"use"> {
    let summary = "Use logic gate";
    let arguments = (ins
        SymbolRefAttr: $name,
        Variadic<AnyType>: $parameters
    );
    let results = (outs AnyType: $result);
    let assemblyFormat = [{
        $name attr-dict  ( `(` $parameters^ `)` `:` `(` qualified(type($parameters)) `)` `->`):(`:`)? qualified(type(results))
    }];
}

def Logic_ApplyGateOp : Logic_Op<"apply", [
        Pure,
        TypesMatchWith<
            "args equivalence", "gate", "args", "::isq::ir::getGateParameterType($_self.cast<::isq::ir::GateType>())"
        >,TypesMatchWith<
            "output equivalence", "gate", "r", "::isq::ir::getGateParameterType($_self.cast<::isq::ir::GateType>())"
        >
    ]> {
    let summary = "Apply a logic gate onto qstates";
    let arguments = (ins
        ISQ_Gate: $gate,
        Variadic<ISQ_QState>: $args
    );
    let results = (outs Variadic<ISQ_QState>: $r);
    let assemblyFormat = [{
        $gate `(` $args `)` attr-dict `:` qualified(type($gate))
    }];
}
