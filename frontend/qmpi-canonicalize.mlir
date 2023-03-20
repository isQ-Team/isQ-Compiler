#map = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  isq.declare_qop @__isq__builtin__measure : [1] () -> i1
  isq.declare_qop @__isq__builtin__reset : [1] () -> ()
  isq.declare_qop @__isq__builtin__bp : [0] (index) -> ()
  isq.declare_qop @__isq__builtin__print_int : [0] (index) -> ()
  isq.declare_qop @__isq__builtin__print_double : [0] (f64) -> ()
  isq.declare_qop @__isq__qmpiprim__me : [0] () -> index
  isq.declare_qop @__isq__qmpiprim__size : [0] () -> index
  isq.declare_qop @__isq__qmpiprim__epr : [1] (index) -> ()
  isq.declare_qop @__isq__qmpiprim__csend : [0] (i1, index) -> ()
  isq.declare_qop @__isq__qmpiprim__crecv : [0] (index) -> i1
  isq.defgate @qmpi.H {definition = [#isq.gatedef<type = "unitary", value = #isq.matrix<dense<[[(0.70710678118654746,0.000000e+00), (0.70710678118654746,0.000000e+00)], [(0.70710678118654746,0.000000e+00), (-0.70710678118654746,-0.000000e+00)]]> : tensor<2x2xcomplex<f64>>>>]} : !isq.gate<1>
  isq.defgate @qmpi.X {definition = [#isq.gatedef<type = "unitary", value = #isq.matrix<dense<[[(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)], [(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]]> : tensor<2x2xcomplex<f64>>>>]} : !isq.gate<1>
  isq.defgate @qmpi.Z {definition = [#isq.gatedef<type = "unitary", value = #isq.matrix<dense<[[(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (-1.000000e+00,-0.000000e+00)]]> : tensor<2x2xcomplex<f64>>>>]} : !isq.gate<1>
  func.func @qmpi.qmpi_me()->index {
    %0 = isq.call_qop @__isq__qmpiprim__me() : [0]()->index
    return %0 : index
  }
  
  func.func @qmpi.qmpi_size() -> index {
    %0 = isq.call_qop @__isq__qmpiprim__size() : [0]()->index
    return %0 : index
  }
  func.func @qmpi.qmpi_epr(%arg0: memref<1x!isq.qstate, #map>, %arg1: index) {
    %q = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %q1 = isq.call_qop @__isq__qmpiprim__epr(%q, %arg1) : [1](index)->()
    return
  }
  func.func @qmpi.qmpi_csend(%arg0: i1, %arg1: index) {
    isq.call_qop @__isq__qmpiprim__csend(%arg0, %arg1) : [0](i1, index)->()
    return
  }
  func.func @qmpi.qmpi_crecv(%arg0: index) -> i1 {
    %0 = isq.call_qop @__isq__qmpiprim__crecv(%arg0) : [0](index)->i1
    return %0 : i1
  }
  func.func @qmpi.qmpi_send(%arg0: memref<1x!isq.qstate, #map>, %arg1: index) {
    %false = arith.constant false
    %0 = memref.alloc() : memref<1xindex>
    affine.store %arg1, %0[0] : memref<1xindex>
    %1 = memref.alloc() : memref<1xi1>
    affine.store %false, %1[0] : memref<1xi1>
    %2 = memref.alloc() : memref<1x!isq.qstate>
    %3 = memref.cast %2 : memref<1x!isq.qstate> to memref<1x!isq.qstate, #map>
    %4 = affine.load %0[0] : memref<1xindex>
    call @qmpi.qmpi_epr(%3, %4) : (memref<1x!isq.qstate, #map>, index) -> ()
    %5 = isq.use @qmpi.X : !isq.gate<1>
    %6 = isq.decorate(%5 : !isq.gate<1>) {adjoint = false, ctrl = [true]} : !isq.gate<2>
    %7 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %8 = affine.load %2[0] : memref<1x!isq.qstate>
    %9:2 = isq.apply %6(%7, %8) : !isq.gate<2>
    affine.store %9#0, %arg0[0] : memref<1x!isq.qstate, #map>
    affine.store %9#1, %2[0] : memref<1x!isq.qstate>
    %10 = affine.load %2[0] : memref<1x!isq.qstate>
    %11:2 = isq.call_qop @__isq__builtin__measure(%10) : [1] () -> i1
    affine.store %11#0, %2[0] : memref<1x!isq.qstate>
    %12 = affine.load %0[0] : memref<1xindex>
    call @qmpi.qmpi_csend(%11#1, %12) : (i1, index) -> ()
    isq.accumulate_gphase %2 : memref<1x!isq.qstate>
    memref.dealloc %2 : memref<1x!isq.qstate>
    isq.accumulate_gphase %1 : memref<1xi1>
    memref.dealloc %1 : memref<1xi1>
    isq.accumulate_gphase %0 : memref<1xindex>
    memref.dealloc %0 : memref<1xindex>
    return
  }
  func.func @qmpi.qmpi_recv(%arg0: memref<1x!isq.qstate, #map>, %arg1: index) {
    %false = arith.constant false
    %0 = memref.alloc() : memref<1xindex>
    affine.store %arg1, %0[0] : memref<1xindex>
    %1 = memref.alloc() : memref<1xi1>
    affine.store %false, %1[0] : memref<1xi1>
    %2 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %3 = isq.call_qop @__isq__builtin__reset(%2) : [1] () -> ()
    affine.store %3, %arg0[0] : memref<1x!isq.qstate, #map>
    %4 = affine.load %0[0] : memref<1xindex>
    call @qmpi.qmpi_epr(%arg0, %4) : (memref<1x!isq.qstate, #map>, index) -> ()
    %5 = memref.alloc() : memref<1xi1>
    affine.store %false, %5[0] : memref<1xi1>
    %6 = affine.load %0[0] : memref<1xindex>
    %7 = call @qmpi.qmpi_crecv(%6) : (index) -> i1
    scf.if %7 {
      %8 = memref.alloc() : memref<1xi1>
      affine.store %false, %8[0] : memref<1xi1>
      %9 = isq.use @qmpi.X : !isq.gate<1>
      %10 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
      %11 = isq.apply %9(%10) : !isq.gate<1>
      affine.store %11, %arg0[0] : memref<1x!isq.qstate, #map>
      isq.accumulate_gphase %8 : memref<1xi1>
      memref.dealloc %8 : memref<1xi1>
    }
    isq.accumulate_gphase %5 : memref<1xi1>
    memref.dealloc %5 : memref<1xi1>
    isq.accumulate_gphase %1 : memref<1xi1>
    memref.dealloc %1 : memref<1xi1>
    isq.accumulate_gphase %0 : memref<1xindex>
    memref.dealloc %0 : memref<1xindex>
    return
  }
  func.func @qmpi.qmpi_unsend(%arg0: memref<1x!isq.qstate, #map>, %arg1: index) {
    %false = arith.constant false
    %0 = memref.alloc() : memref<1xindex>
    affine.store %arg1, %0[0] : memref<1xindex>
    %1 = memref.alloc() : memref<1xi1>
    affine.store %false, %1[0] : memref<1xi1>
    %2 = memref.alloc() : memref<1xi1>
    affine.store %false, %2[0] : memref<1xi1>
    %3 = affine.load %0[0] : memref<1xindex>
    %4 = call @qmpi.qmpi_crecv(%3) : (index) -> i1
    scf.if %4 {
      %5 = memref.alloc() : memref<1xi1>
      affine.store %false, %5[0] : memref<1xi1>
      %6 = isq.use @qmpi.Z : !isq.gate<1>
      %7 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
      %8 = isq.apply %6(%7) : !isq.gate<1>
      affine.store %8, %arg0[0] : memref<1x!isq.qstate, #map>
      isq.accumulate_gphase %5 : memref<1xi1>
      memref.dealloc %5 : memref<1xi1>
    }
    isq.accumulate_gphase %2 : memref<1xi1>
    memref.dealloc %2 : memref<1xi1>
    isq.accumulate_gphase %1 : memref<1xi1>
    memref.dealloc %1 : memref<1xi1>
    isq.accumulate_gphase %0 : memref<1xindex>
    memref.dealloc %0 : memref<1xindex>
    return
  }
  func.func @qmpi.qmpi_unrecv(%arg0: memref<1x!isq.qstate, #map>, %arg1: index) {
    %false = arith.constant false
    %0 = memref.alloc() : memref<1xindex>
    affine.store %arg1, %0[0] : memref<1xindex>
    %1 = memref.alloc() : memref<1xi1>
    affine.store %false, %1[0] : memref<1xi1>
    %2 = isq.use @qmpi.H : !isq.gate<1>
    %3 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %4 = isq.apply %2(%3) : !isq.gate<1>
    affine.store %4, %arg0[0] : memref<1x!isq.qstate, #map>
    %5 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %6:2 = isq.call_qop @__isq__builtin__measure(%5) : [1] () -> i1
    affine.store %6#0, %arg0[0] : memref<1x!isq.qstate, #map>
    %7 = affine.load %0[0] : memref<1xindex>
    call @qmpi.qmpi_csend(%6#1, %7) : (i1, index) -> ()
    isq.accumulate_gphase %1 : memref<1xi1>
    memref.dealloc %1 : memref<1xi1>
    isq.accumulate_gphase %0 : memref<1xindex>
    memref.dealloc %0 : memref<1xindex>
    return
  }
  func.func @qmpi.qmpi_send_move(%arg0: memref<1x!isq.qstate, #map>, %arg1: index) {
    %false = arith.constant false
    %0 = memref.alloc() : memref<1xindex>
    affine.store %arg1, %0[0] : memref<1xindex>
    %1 = memref.alloc() : memref<1xi1>
    affine.store %false, %1[0] : memref<1xi1>
    %2 = memref.alloc() : memref<1x!isq.qstate>
    %3 = memref.cast %2 : memref<1x!isq.qstate> to memref<1x!isq.qstate, #map>
    %4 = affine.load %0[0] : memref<1xindex>
    call @qmpi.qmpi_epr(%3, %4) : (memref<1x!isq.qstate, #map>, index) -> ()
    %5 = isq.use @qmpi.X : !isq.gate<1>
    %6 = isq.decorate(%5 : !isq.gate<1>) {adjoint = false, ctrl = [true]} : !isq.gate<2>
    %7 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %8 = affine.load %2[0] : memref<1x!isq.qstate>
    %9:2 = isq.apply %6(%7, %8) : !isq.gate<2>
    affine.store %9#0, %arg0[0] : memref<1x!isq.qstate, #map>
    affine.store %9#1, %2[0] : memref<1x!isq.qstate>
    %10 = isq.use @qmpi.H : !isq.gate<1>
    %11 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %12 = isq.apply %10(%11) : !isq.gate<1>
    affine.store %12, %arg0[0] : memref<1x!isq.qstate, #map>
    %13 = affine.load %2[0] : memref<1x!isq.qstate>
    %14:2 = isq.call_qop @__isq__builtin__measure(%13) : [1] () -> i1
    affine.store %14#0, %2[0] : memref<1x!isq.qstate>
    %15 = affine.load %0[0] : memref<1xindex>
    call @qmpi.qmpi_csend(%14#1, %15) : (i1, index) -> ()
    %16 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %17:2 = isq.call_qop @__isq__builtin__measure(%16) : [1] () -> i1
    affine.store %17#0, %arg0[0] : memref<1x!isq.qstate, #map>
    %18 = affine.load %0[0] : memref<1xindex>
    call @qmpi.qmpi_csend(%17#1, %18) : (i1, index) -> ()
    %19 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %20 = isq.call_qop @__isq__builtin__reset(%19) : [1] () -> ()
    affine.store %20, %arg0[0] : memref<1x!isq.qstate, #map>
    %21 = affine.load %2[0] : memref<1x!isq.qstate>
    %22 = isq.call_qop @__isq__builtin__reset(%21) : [1] () -> ()
    affine.store %22, %2[0] : memref<1x!isq.qstate>
    isq.accumulate_gphase %2 : memref<1x!isq.qstate>
    memref.dealloc %2 : memref<1x!isq.qstate>
    isq.accumulate_gphase %1 : memref<1xi1>
    memref.dealloc %1 : memref<1xi1>
    isq.accumulate_gphase %0 : memref<1xindex>
    memref.dealloc %0 : memref<1xindex>
    return
  }
  func.func @qmpi.qmpi_recv_move(%arg0: memref<1x!isq.qstate, #map>, %arg1: index) {
    %false = arith.constant false
    %0 = memref.alloc() : memref<1xindex>
    affine.store %arg1, %0[0] : memref<1xindex>
    %1 = memref.alloc() : memref<1xi1>
    affine.store %false, %1[0] : memref<1xi1>
    %2 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %3 = isq.call_qop @__isq__builtin__reset(%2) : [1] () -> ()
    affine.store %3, %arg0[0] : memref<1x!isq.qstate, #map>
    %4 = affine.load %0[0] : memref<1xindex>
    call @qmpi.qmpi_epr(%arg0, %4) : (memref<1x!isq.qstate, #map>, index) -> ()
    %5 = memref.alloc() : memref<1xi1>
    affine.store %false, %5[0] : memref<1xi1>
    %6 = affine.load %0[0] : memref<1xindex>
    %7 = call @qmpi.qmpi_crecv(%6) : (index) -> i1
    scf.if %7 {
      %8 = memref.alloc() : memref<1xi1>
      affine.store %false, %8[0] : memref<1xi1>
      %9 = isq.use @qmpi.Z : !isq.gate<1>
      %10 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
      %11 = isq.apply %9(%10) : !isq.gate<1>
      affine.store %11, %arg0[0] : memref<1x!isq.qstate, #map>
      isq.accumulate_gphase %8 : memref<1xi1>
      memref.dealloc %8 : memref<1xi1>
    }
    isq.accumulate_gphase %5 : memref<1xi1>
    memref.dealloc %5 : memref<1xi1>
    isq.accumulate_gphase %1 : memref<1xi1>
    memref.dealloc %1 : memref<1xi1>
    isq.accumulate_gphase %0 : memref<1xindex>
    memref.dealloc %0 : memref<1xindex>
    return
  }
  func.func @qmpi.qmpi_unsend_move(%arg0: memref<1x!isq.qstate, #map>, %arg1: index) {
    %false = arith.constant false
    %0 = memref.alloc() : memref<1xindex>
    affine.store %arg1, %0[0] : memref<1xindex>
    %1 = memref.alloc() : memref<1xi1>
    affine.store %false, %1[0] : memref<1xi1>
    %2 = affine.load %0[0] : memref<1xindex>
    call @qmpi.qmpi_recv_move(%arg0, %2) : (memref<1x!isq.qstate, #map>, index) -> ()
    isq.accumulate_gphase %1 : memref<1xi1>
    memref.dealloc %1 : memref<1xi1>
    isq.accumulate_gphase %0 : memref<1xindex>
    memref.dealloc %0 : memref<1xindex>
    return
  }
  func.func @qmpi.qmpi_unrecv_move(%arg0: memref<1x!isq.qstate, #map>, %arg1: index) {
    %false = arith.constant false
    %0 = memref.alloc() : memref<1xindex>
    affine.store %arg1, %0[0] : memref<1xindex>
    %1 = memref.alloc() : memref<1xi1>
    affine.store %false, %1[0] : memref<1xi1>
    %2 = affine.load %0[0] : memref<1xindex>
    call @qmpi.qmpi_send_move(%arg0, %2) : (memref<1x!isq.qstate, #map>, index) -> ()
    isq.accumulate_gphase %1 : memref<1xi1>
    memref.dealloc %1 : memref<1xi1>
    isq.accumulate_gphase %0 : memref<1xindex>
    memref.dealloc %0 : memref<1xindex>
    return
  }
  func.func @qmpi.log2(%arg0: index) -> index {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<1xindex>
    %1 = memref.alloc() : memref<1xindex>
    affine.store %arg0, %1[0] : memref<1xindex>
    %2 = memref.alloc() : memref<1xi1>
    affine.store %false, %2[0] : memref<1xi1>
    %3 = memref.alloc() : memref<1xindex>
    affine.store %c1, %3[0] : memref<1xindex>
    %4 = memref.alloc() : memref<1xindex>
    affine.store %c0, %4[0] : memref<1xindex>
    %5 = memref.alloc() : memref<1xi1>
    affine.store %false, %5[0] : memref<1xi1>
    scf.while : () -> () {
      %8 = scf.execute_region -> i1 {
        %9 = affine.load %5[0] : memref<1xi1>
        cf.cond_br %9, ^bb2, ^bb1
      ^bb1:  // pred: ^bb0
        %10 = affine.load %3[0] : memref<1xindex>
        %11 = affine.load %1[0] : memref<1xindex>
        %12 = arith.cmpi slt, %10, %11 : index
        scf.yield %12 : i1
      ^bb2:  // pred: ^bb0
        scf.yield %false : i1
      }
      scf.condition(%8)
    } do {
      %8 = memref.alloc() : memref<1xi1>
      affine.store %false, %8[0] : memref<1xi1>
      %9 = memref.alloc() : memref<1xi1>
      affine.store %false, %9[0] : memref<1xi1>
      %10 = affine.load %3[0] : memref<1xindex>
      %11 = arith.muli %10, %c2 : index
      affine.store %11, %3[0] : memref<1xindex>
      %12 = affine.load %4[0] : memref<1xindex>
      %13 = arith.addi %12, %c1 : index
      affine.store %13, %4[0] : memref<1xindex>
      isq.accumulate_gphase %9 : memref<1xi1>
      memref.dealloc %9 : memref<1xi1>
      isq.accumulate_gphase %8 : memref<1xi1>
      memref.dealloc %8 : memref<1xi1>
      scf.yield
    }
    %6 = affine.load %4[0] : memref<1xindex>
    affine.store %6, %0[0] : memref<1xindex>
    isq.accumulate_gphase %5 : memref<1xi1>
    memref.dealloc %5 : memref<1xi1>
    isq.accumulate_gphase %4 : memref<1xindex>
    memref.dealloc %4 : memref<1xindex>
    isq.accumulate_gphase %3 : memref<1xindex>
    memref.dealloc %3 : memref<1xindex>
    isq.accumulate_gphase %2 : memref<1xi1>
    memref.dealloc %2 : memref<1xi1>
    isq.accumulate_gphase %1 : memref<1xindex>
    memref.dealloc %1 : memref<1xindex>
    %7 = affine.load %0[0] : memref<1xindex>
    isq.accumulate_gphase %0 : memref<1xindex>
    memref.dealloc %0 : memref<1xindex>
    return %7 : index
  }
  func.func @qmpi.qmpi_poorman_cat(%arg0: memref<1x!isq.qstate, #map>, %arg1: index, %arg2: index, %arg3: index) {
    %false = arith.constant false
    %0 = memref.alloc() : memref<1xindex>
    affine.store %arg1, %0[0] : memref<1xindex>
    %1 = memref.alloc() : memref<1xindex>
    affine.store %arg2, %1[0] : memref<1xindex>
    %2 = memref.alloc() : memref<1xindex>
    affine.store %arg3, %2[0] : memref<1xindex>
    %3 = memref.alloc() : memref<1xi1>
    affine.store %false, %3[0] : memref<1xi1>
    isq.accumulate_gphase %3 : memref<1xi1>
    memref.dealloc %3 : memref<1xi1>
    isq.accumulate_gphase %2 : memref<1xindex>
    memref.dealloc %2 : memref<1xindex>
    isq.accumulate_gphase %1 : memref<1xindex>
    memref.dealloc %1 : memref<1xindex>
    isq.accumulate_gphase %0 : memref<1xindex>
    memref.dealloc %0 : memref<1xindex>
    return
  }
  func.func @qmpi.qmpi_bcast(%arg0: memref<1x!isq.qstate, #map>, %arg1: index) {
    %false = arith.constant false
    %0 = memref.alloc() : memref<1xindex>
    affine.store %arg1, %0[0] : memref<1xindex>
    %1 = memref.alloc() : memref<1xi1>
    affine.store %false, %1[0] : memref<1xi1>
    %2 = call @qmpi.qmpi_me() : () -> index
    %3 = memref.alloc() : memref<1xindex>
    affine.store %2, %3[0] : memref<1xindex>
    %4 = memref.alloc() : memref<1xi1>
    affine.store %false, %4[0] : memref<1xi1>
    %5 = affine.load %3[0] : memref<1xindex>
    %6 = affine.load %0[0] : memref<1xindex>
    %7 = arith.cmpi ne, %5, %6 : index
    scf.if %7 {
      %23 = memref.alloc() : memref<1xi1>
      affine.store %false, %23[0] : memref<1xi1>
      %24 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
      %25 = isq.call_qop @__isq__builtin__reset(%24) : [1] () -> ()
      affine.store %25, %arg0[0] : memref<1x!isq.qstate, #map>
      isq.accumulate_gphase %23 : memref<1xi1>
      memref.dealloc %23 : memref<1xi1>
    }
    %8 = affine.load %1[0] : memref<1xi1>
    cf.cond_br %8, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %9 = call @qmpi.qmpi_size() : () -> index
    %10 = memref.alloc() : memref<1xindex>
    affine.store %9, %10[0] : memref<1xindex>
    %11 = affine.load %3[0] : memref<1xindex>
    %12 = affine.load %10[0] : memref<1xindex>
    %13 = arith.addi %11, %12 : index
    %14 = affine.load %0[0] : memref<1xindex>
    %15 = arith.subi %13, %14 : index
    %16 = affine.load %10[0] : memref<1xindex>
    %17 = arith.remsi %15, %16 : index
    %18 = memref.alloc() : memref<1xindex>
    affine.store %17, %18[0] : memref<1xindex>
    %19 = affine.load %10[0] : memref<1xindex>
    %20 = call @qmpi.log2(%19) : (index) -> index
    %21 = memref.alloc() : memref<1xindex>
    affine.store %20, %21[0] : memref<1xindex>
    %22 = affine.load %21[0] : memref<1xindex>
    affine.for %arg2 = 0 to %22 {
      %23 = memref.alloc() : memref<1xi1>
      affine.store %false, %23[0] : memref<1xi1>
      %24 = memref.alloc() : memref<1xi1>
      affine.store %false, %24[0] : memref<1xi1>
      isq.accumulate_gphase %24 : memref<1xi1>
      memref.dealloc %24 : memref<1xi1>
      isq.accumulate_gphase %23 : memref<1xi1>
      memref.dealloc %23 : memref<1xi1>
    }
    isq.accumulate_gphase %21 : memref<1xindex>
    memref.dealloc %21 : memref<1xindex>
    isq.accumulate_gphase %18 : memref<1xindex>
    memref.dealloc %18 : memref<1xindex>
    isq.accumulate_gphase %10 : memref<1xindex>
    memref.dealloc %10 : memref<1xindex>
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    isq.accumulate_gphase %4 : memref<1xi1>
    memref.dealloc %4 : memref<1xi1>
    isq.accumulate_gphase %3 : memref<1xindex>
    memref.dealloc %3 : memref<1xindex>
    isq.accumulate_gphase %1 : memref<1xi1>
    memref.dealloc %1 : memref<1xi1>
    isq.accumulate_gphase %0 : memref<1xindex>
    memref.dealloc %0 : memref<1xindex>
    return
  }
  func.func @__isq__main(%arg0: memref<?xindex>, %arg1: memref<?xf64>) {
    %false = arith.constant false
    %0 = memref.alloc() : memref<1xi1>
    affine.store %false, %0[0] : memref<1xi1>
    isq.accumulate_gphase %0 : memref<1xi1>
    memref.dealloc %0 : memref<1xi1>
    return
  }
  func.func @__isq__global_initialize() {
    return
  }
  func.func @__isq__global_finalize() {
    return
  }
  func.func @__isq__entry(%arg0: memref<?xindex>, %arg1: memref<?xf64>) {
    call @__isq__global_initialize() : () -> ()
    call @__isq__main(%arg0, %arg1) : (memref<?xindex>, memref<?xf64>) -> ()
    call @__isq__global_finalize() : () -> ()
    return
  }
}

