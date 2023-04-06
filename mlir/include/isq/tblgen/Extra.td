/**
 * This file is for defining operations that are not part of isQ dialects.
 * e.g. operations that you believe should be in upstream, but in fact not.
*/
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"
//include "mlir/Dialect/Shape/IR/ShapeBase.td"


def ISQExtra_Dialect : Dialect{
    let name = "isqextra";
    let cppNamespace = "::isq::extra";
    let summary = "Extra dialect used by isQ.";
    let description = [{
        Extra dialect for isQ.
    }];
}

class ISQExtra_Type<string name> : TypeDef<ISQExtra_Dialect, name>{}
class ISQExtra_Op<string mnemonic, list<Trait> traits = []>:
    Op<ISQExtra_Dialect, mnemonic, traits>{

}


// A VLIW bundle of operations executed in parallel.
// Every result is attached with a duration, such that the result value is only ``available`` after this duration.
def ISQExtra_Bundle : ISQExtra_Op<"bundle", []>{
    let regions = (region VariadicRegion<SizedRegion<1>>:$body);
    let results = (outs Variadic<AnyType>);
}

/**
 * A schedule of VLIW bundles to execute.
 */
def ISQExtra_Schedule : ISQExtra_Op<"schedule", []>{
    let regions = (region SizedRegion<1>);
    let results = (outs Variadic<AnyType>);
}

def ISQExtra_ScheduleYield : ISQExtra_Op<"schedule.yield"> {
    let arguments = (ins Variadic<AnyType>);
}
def ISQExtra_BundleYield : ISQExtra_Op<"bundle.yield">{
    let arguments = (ins Variadic<AnyType>);
}
