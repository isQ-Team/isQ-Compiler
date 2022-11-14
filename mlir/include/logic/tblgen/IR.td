include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

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

class Logic_Op<string mnemonic> : Op<Logic_Dialect, mnemonic> {}

def Logic_FuncOp : Logic_Op<"func"> {
    let summary = "Define a logic function.";
    let arguments = (ins
        SymbolNameAttr: $sym_name,
        TypeAttr: $type
    );
    let regions = (region AnyRegion: $body);
    let assemblyFormat = [{
        $sym_name `:` $type $body attr-dict
    }];
}
