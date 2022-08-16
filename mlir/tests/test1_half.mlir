#map0 = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0 + 2)>
module {
  func @H__qsd__decomposition(%arg0: !isq.qstate) -> !isq.qstate {
    return %arg0 : !isq.qstate
  }
  isq.declare_qop@__isq__builtin__measure : [1] () -> i1
  isq.declare_qop@__isq__builtin__reset : [1] () -> ()
  isq.declare_qop@__isq__builtin__print_int : [0] (index) -> ()
  isq.declare_qop@__isq__builtin__print_double : [0] (f64) -> ()
  isq.defgate @__isq__builtin__u3(f64, f64, f64) {definition = [{type = "qir", value = "__quantum__qis__u3"}]} : !isq.gate<1>
  isq.defgate @__isq__builtin__cnot {definition = [{type = "qir", value = "__quantum__qis__cnot"}]} : !isq.gate<2>
  func private @__quantum__qis__u3(f64, f64, f64, !isq.qir.qubit)
  func private @__quantum__qis__cnot(!isq.qir.qubit, !isq.qir.qubit)

  func @test2(%arg0: memref<?x!isq.qstate>, %arg1: index) {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<1xindex>
    memref.store %arg1, %0[%c0] : memref<1xindex>
    %1 = memref.load %0[%c0] : memref<1xindex>
    %2 = memref.subview %arg0[%1] [1] [1] : memref<?x!isq.qstate> to memref<1x!isq.qstate, #map0>
    %3 = memref.load %2[%c0] : memref<1x!isq.qstate, #map0>
    %4 = call @H__qsd__decomposition(%3) : (!isq.qstate) -> !isq.qstate
    memref.store %4, %2[%c0] : memref<1x!isq.qstate, #map0>
    memref.dealloc %0 : memref<1xindex>
    return
  }

  func @__isq__global_initialize() {
    return
  }
  func @__isq__global_finalize() {
    return
  }

}

