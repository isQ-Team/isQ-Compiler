isq.defgate @__isq__builtin__u3(f64, f64, f64) {definition = [{type = "qir", value = "__quantum__qis__u3"}]} : !isq.gate<1>
isq.defgate @__isq__builtin__cnot {definition = [{type = "qir", value = "__quantum__qis__cnot"}]} : !isq.gate<2>
func private @__quantum__rt__qubit_release(!isq.qir.qubit)
func private @__quantum__rt__qubit_allocate() -> !isq.qir.qubit
func private @__quantum__qis__u3(f64, f64, f64, !isq.qir.qubit)
func private @__quantum__qis__cnot(!isq.qir.qubit, !isq.qir.qubit)

isq.declare_qop @__isq__builtin__measure : [1]()->i1
isq.declare_qop @__isq__builtin__reset : [1]()->()
isq.declare_qop @__isq__builtin__print_int : [0](index)->()
isq.declare_qop @__isq__builtin__print_double : [0](f64)->()

memref.global @q : memref<3x!isq.qstate> = uninitialized 
memref.global @p : memref<1x!isq.qstate> = uninitialized

func @test(){
    %a = memref.alloc() : memref<10x!isq.qstate>
    memref.dealloc %a : memref<10x!isq.qstate>
    return
}

func @qir_op_test(%a: !isq.qstate, %b: !isq.qstate, %c: index, %d: f64)->(!isq.qstate, !isq.qstate, i1){
    %a1, %meas = isq.call_qop @__isq__builtin__measure(%a) : [1]()->i1
    %b1 = isq.call_qop @__isq__builtin__reset(%b) : [1]()->()
    isq.call_qop @__isq__builtin__print_int(%c) : [0](index)->()
    isq.call_qop @__isq__builtin__print_double(%d) : [0](f64)->()
    return %a1, %b1, %meas : !isq.qstate, !isq.qstate, i1
}


func @qir_gate_test(%a: !isq.qstate, %b: !isq.qstate)->(!isq.qstate, !isq.qstate){
    %CNOT = isq.use @__isq__builtin__cnot : !isq.gate<2>
    %a1, %b1 = isq.apply %CNOT(%a, %b) : !isq.gate<2>
    %a2, %b2 = isq.apply %CNOT(%b1, %a1) : !isq.gate<2>
    %a3, %b3 = isq.apply %CNOT(%a2, %b2) : !isq.gate<2>
    return %a3, %b3 : !isq.qstate, !isq.qstate
}

func @__isq__global_initialize() {
return
}
func @__isq__global_finalize() {
return
}