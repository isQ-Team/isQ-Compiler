#map = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  module @isq_builtin {
    isq.declare_qop@measure : [1] () -> i1
    isq.declare_qop@reset : [1] () -> ()
    isq.declare_qop@print_int : [0] (index) -> ()
    isq.declare_qop@print_double : [0] (f64) -> ()
  }
  isq.defgate@Rs {definition = [{type = "unitary", value = [[#isq.complex<5.000000e-01, 0.86602539999999995>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>]]}]} : !isq.gate<2>
  isq.defgate@Rs2 {definition = [{type = "unitary", value = [[#isq.complex<5.000000e-01, -0.86602539999999995>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>]]}]} : !isq.gate<2>
  isq.defgate@Rt {definition = [{type = "unitary", value = [[#isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<5.000000e-01, 0.86602539999999995>]]}]} : !isq.gate<2>
  isq.defgate@Rt2 {definition = [{type = "unitary", value = [[#isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<5.000000e-01, -0.86602539999999995>]]}]} : !isq.gate<2>
  isq.defgate@CNOT {definition = [{type = "unitary", value = [[#isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>]]}]} : !isq.gate<2>
  isq.defgate@H {definition = [{type = "unitary", value = [[#isq.complex<0.70710678118654757, 0.000000e+00>, #isq.complex<0.70710678118654757, 0.000000e+00>], [#isq.complex<0.70710678118654757, 0.000000e+00>, #isq.complex<-0.70710678118654757, -0.000000e+00>]]}]} : !isq.gate<1>
  memref.global @a : memref<1xindex, #map> = uninitialized
  memref.global @b : memref<1xindex, #map> = uninitialized
  memref.global @c : memref<1xindex, #map> = uninitialized
  memref.global @q : memref<3x!isq.qstate> = uninitialized
  memref.global @p : memref<1x!isq.qstate, #map> = uninitialized
  func @test(%arg0: memref<1x!isq.qstate, #map>, %arg1: memref<1x!isq.qstate, #map>, %arg2: index) -> index {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc()[%c0] : memref<1xindex, #map>
    %c0_0 = arith.constant 0 : index
    %1 = memref.alloc()[%c0_0] : memref<1xindex, #map>
    affine.store %arg2, %1[0] : memref<1xindex, #map>
    %false = arith.constant false
    %c0_1 = arith.constant 0 : index
    %2 = memref.alloc()[%c0_1] : memref<1xi1, #map>
    affine.store %false, %2[0] : memref<1xi1, #map>
    %3 = isq.use @H : !isq.gate<1>
    %4 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %5 = isq.apply %3(%4) : !isq.gate<1>
    affine.store %5, %arg0[0] : memref<1x!isq.qstate, #map>
    %c0_2 = arith.constant 0 : index
    %6 = memref.alloc()[%c0_2] : memref<1x!isq.qstate, #map>
    %7 = isq.use @CNOT : !isq.gate<2>
    %8 = affine.load %6[0] : memref<1x!isq.qstate, #map>
    %9 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %10:2 = isq.apply %7(%8, %9) : !isq.gate<2>
    affine.store %10#0, %6[0] : memref<1x!isq.qstate, #map>
    affine.store %10#1, %arg0[0] : memref<1x!isq.qstate, #map>
    %11 = isq.use @H : !isq.gate<1>
    %12 = affine.load %arg0[0] : memref<1x!isq.qstate, #map>
    %13 = isq.apply %11(%12) : !isq.gate<1>
    affine.store %13, %arg0[0] : memref<1x!isq.qstate, #map>
    %c2 = arith.constant 2 : index
    affine.store %c2, %0[0] : memref<1xindex, #map>
    br ^bb2
  ^bb1:  // no predecessors
    br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    memref.dealloc %6 : memref<1x!isq.qstate, #map>
    br ^bb3
  ^bb3:  // pred: ^bb2
    memref.dealloc %2 : memref<1xi1, #map>
    br ^bb4
  ^bb4:  // pred: ^bb3
    memref.dealloc %1 : memref<1xindex, #map>
    br ^bb5
  ^bb5:  // pred: ^bb4
    %14 = affine.load %0[0] : memref<1xindex, #map>
    memref.dealloc %0 : memref<1xindex, #map>
    return %14 : index
  }
  func @test2(%arg0: memref<?x!isq.qstate>, %arg1: index) {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc()[%c0] : memref<1xindex, #map>
    affine.store %arg1, %0[0] : memref<1xindex, #map>
    %false = arith.constant false
    %c0_0 = arith.constant 0 : index
    %1 = memref.alloc()[%c0_0] : memref<1xi1, #map>
    affine.store %false, %1[0] : memref<1xi1, #map>
    %2 = affine.load %0[0] : memref<1xindex, #map>
    %3 = memref.subview %arg0[%2] [1] [1] : memref<?x!isq.qstate> to memref<1x!isq.qstate, #map>
    %4 = isq.use @H : !isq.gate<1>
    %5 = affine.load %3[0] : memref<1x!isq.qstate, #map>
    %6 = isq.apply %4(%5) : !isq.gate<1>
    affine.store %6, %3[0] : memref<1x!isq.qstate, #map>
    br ^bb1
  ^bb1:  // pred: ^bb0
    memref.dealloc %1 : memref<1xi1, #map>
    br ^bb2
  ^bb2:  // pred: ^bb1
    memref.dealloc %0 : memref<1xindex, #map>
    br ^bb3
  ^bb3:  // pred: ^bb2
    return
  }
  func @test_main() {
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %0 = memref.alloc()[%c0] : memref<1xi1, #map>
    affine.store %false, %0[0] : memref<1xi1, #map>
    %c0_0 = arith.constant 0 : index
    %1 = memref.alloc()[%c0_0] : memref<1xindex, #map>
    %c0_1 = arith.constant 0 : index
    %2 = memref.alloc()[%c0_1] : memref<1xindex, #map>
    %false_2 = arith.constant false
    %c0_3 = arith.constant 0 : index
    %3 = memref.alloc()[%c0_3] : memref<1xi1, #map>
    affine.store %false_2, %3[0] : memref<1xi1, #map>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %4 = arith.cmpi slt, %c1, %c2 : index
    scf.if %4 {
      scf.execute_region {
        %105 = memref.get_global @a : memref<1xindex, #map>
        %106 = affine.load %105[0] : memref<1xindex, #map>
        %c3 = arith.constant 3 : index
        %c2_26 = arith.constant 2 : index
        %107 = arith.muli %c3, %c2_26 : index
        %108 = arith.addi %106, %107 : index
        %109 = memref.get_global @b : memref<1xindex, #map>
        %110 = affine.load %109[0] : memref<1xindex, #map>
        %111 = memref.get_global @c : memref<1xindex, #map>
        %112 = affine.load %111[0] : memref<1xindex, #map>
        %113 = arith.addi %110, %112 : index
        %114 = arith.muli %108, %113 : index
        affine.store %114, %1[0] : memref<1xindex, #map>
        br ^bb1
      ^bb1:  // pred: ^bb0
        scf.yield
      }
    } else {
      scf.execute_region {
        %c0_26 = arith.constant 0 : index
        %105 = memref.alloc()[%c0_26] : memref<1xindex, #map>
        %106 = memref.get_global @c : memref<1xindex, #map>
        %107 = affine.load %106[0] : memref<1xindex, #map>
        %c1_27 = arith.constant 1 : index
        %108 = arith.addi %107, %c1_27 : index
        affine.store %108, %105[0] : memref<1xindex, #map>
        br ^bb1
      ^bb1:  // pred: ^bb0
        memref.dealloc %105 : memref<1xindex, #map>
        br ^bb2
      ^bb2:  // pred: ^bb1
        scf.yield
      }
    }
    %5 = affine.load %0[0] : memref<1xi1, #map>
    cond_br %5, ^bb7, ^bb1
  ^bb1:  // pred: ^bb0
    %false_4 = arith.constant false
    %c0_5 = arith.constant 0 : index
    %6 = memref.alloc()[%c0_5] : memref<1xi1, #map>
    affine.store %false_4, %6[0] : memref<1xi1, #map>
    %c1_6 = arith.constant 1 : index
    %7 = memref.get_global @b : memref<1xindex, #map>
    %8 = affine.load %7[0] : memref<1xindex, #map>
    affine.for %arg0 = %c1_6 to %8 {
      scf.execute_region {
        br ^bb1
      ^bb1:  // pred: ^bb0
        scf.yield
      }
    }
    %9 = affine.load %0[0] : memref<1xi1, #map>
    cond_br %9, ^bb6, ^bb2
  ^bb2:  // pred: ^bb1
    %10 = memref.alloc() : memref<5xindex>
    %11 = memref.get_global @a : memref<1xindex, #map>
    %12 = memref.get_global @c : memref<1xindex, #map>
    %13 = affine.load %12[0] : memref<1xindex, #map>
    %14 = memref.subview %10[%13] [1] [1] : memref<5xindex> to memref<1xindex, #map>
    %15 = affine.load %14[0] : memref<1xindex, #map>
    %c2_7 = arith.constant 2 : index
    %16 = arith.addi %15, %c2_7 : index
    affine.store %16, %11[0] : memref<1xindex, #map>
    %17 = memref.get_global @b : memref<1xindex, #map>
    %18 = memref.get_global @p : memref<1x!isq.qstate, #map>
    %19 = memref.get_global @p : memref<1x!isq.qstate, #map>
    %20 = affine.load %1[0] : memref<1xindex, #map>
    %21 = call @test(%18, %19, %20) : (memref<1x!isq.qstate, #map>, memref<1x!isq.qstate, #map>, index) -> index
    affine.store %21, %17[0] : memref<1xindex, #map>
    %22 = memref.get_global @a : memref<1xindex, #map>
    %23 = memref.get_global @q : memref<3x!isq.qstate>
    %c0_8 = arith.constant 0 : index
    %24 = memref.subview %23[%c0_8] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %25 = affine.load %24[0] : memref<1x!isq.qstate, #map>
    %26:2 = isq.call_qop@isq_builtin::@measure(%25) : [1] () -> i1
    affine.store %26#0, %24[0] : memref<1x!isq.qstate, #map>
    %27 = arith.extui %26#1 : i1 to i2
    %28 = arith.index_cast %27 : i2 to index
    affine.store %28, %22[0] : memref<1xindex, #map>
    %29 = memref.get_global @p : memref<1x!isq.qstate, #map>
    %30 = memref.get_global @q : memref<3x!isq.qstate>
    %c0_9 = arith.constant 0 : index
    %31 = memref.subview %30[%c0_9] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %32 = isq.use @CNOT : !isq.gate<2>
    %33 = affine.load %29[0] : memref<1x!isq.qstate, #map>
    %34 = affine.load %31[0] : memref<1x!isq.qstate, #map>
    %35:2 = isq.apply %32(%33, %34) : !isq.gate<2>
    affine.store %35#0, %29[0] : memref<1x!isq.qstate, #map>
    affine.store %35#1, %31[0] : memref<1x!isq.qstate, #map>
    %36 = memref.get_global @q : memref<3x!isq.qstate>
    %c0_10 = arith.constant 0 : index
    %37 = memref.subview %36[%c0_10] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %38 = memref.get_global @q : memref<3x!isq.qstate>
    %c1_11 = arith.constant 1 : index
    %39 = memref.subview %38[%c1_11] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %40 = memref.get_global @p : memref<1x!isq.qstate, #map>
    %41 = isq.use @H : !isq.gate<1>
    %42 = isq.decorate(%41 : !isq.gate<1>) {adjoint = true, ctrl = [true, true]} : !isq.gate<3>
    %43 = affine.load %37[0] : memref<1x!isq.qstate, #map>
    %44 = affine.load %39[0] : memref<1x!isq.qstate, #map>
    %45 = affine.load %40[0] : memref<1x!isq.qstate, #map>
    %46:3 = isq.apply %42(%43, %44, %45) : !isq.gate<3>
    affine.store %46#0, %37[0] : memref<1x!isq.qstate, #map>
    affine.store %46#1, %39[0] : memref<1x!isq.qstate, #map>
    affine.store %46#2, %40[0] : memref<1x!isq.qstate, #map>
    %47 = memref.get_global @q : memref<3x!isq.qstate>
    %c1_12 = arith.constant 1 : index
    %48 = memref.subview %47[%c1_12] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %49 = memref.get_global @q : memref<3x!isq.qstate>
    %c0_13 = arith.constant 0 : index
    %50 = memref.subview %49[%c0_13] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %51 = memref.get_global @p : memref<1x!isq.qstate, #map>
    %52 = isq.use @H : !isq.gate<1>
    %53 = isq.decorate(%52 : !isq.gate<1>) {adjoint = false, ctrl = [true, true]} : !isq.gate<3>
    %54 = affine.load %48[0] : memref<1x!isq.qstate, #map>
    %55 = affine.load %50[0] : memref<1x!isq.qstate, #map>
    %56 = affine.load %51[0] : memref<1x!isq.qstate, #map>
    %57:3 = isq.apply %53(%54, %55, %56) : !isq.gate<3>
    affine.store %57#0, %48[0] : memref<1x!isq.qstate, #map>
    affine.store %57#1, %50[0] : memref<1x!isq.qstate, #map>
    affine.store %57#2, %51[0] : memref<1x!isq.qstate, #map>
    %58 = memref.get_global @q : memref<3x!isq.qstate>
    %c0_14 = arith.constant 0 : index
    %59 = memref.subview %58[%c0_14] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %60 = memref.get_global @p : memref<1x!isq.qstate, #map>
    %61 = isq.use @H : !isq.gate<1>
    %62 = isq.decorate(%61 : !isq.gate<1>) {adjoint = true, ctrl = [false]} : !isq.gate<2>
    %63 = affine.load %59[0] : memref<1x!isq.qstate, #map>
    %64 = affine.load %60[0] : memref<1x!isq.qstate, #map>
    %65:2 = isq.apply %62(%63, %64) : !isq.gate<2>
    affine.store %65#0, %59[0] : memref<1x!isq.qstate, #map>
    affine.store %65#1, %60[0] : memref<1x!isq.qstate, #map>
    %66 = memref.get_global @q : memref<3x!isq.qstate>
    %c0_15 = arith.constant 0 : index
    %67 = memref.subview %66[%c0_15] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %68 = memref.get_global @q : memref<3x!isq.qstate>
    %c1_16 = arith.constant 1 : index
    %69 = memref.subview %68[%c1_16] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %70 = memref.get_global @q : memref<3x!isq.qstate>
    %c2_17 = arith.constant 2 : index
    %71 = memref.subview %70[%c2_17] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %72 = memref.get_global @p : memref<1x!isq.qstate, #map>
    %73 = isq.use @H : !isq.gate<1>
    %74 = isq.decorate(%73 : !isq.gate<1>) {adjoint = true, ctrl = [true, false, false]} : !isq.gate<4>
    %75 = affine.load %67[0] : memref<1x!isq.qstate, #map>
    %76 = affine.load %69[0] : memref<1x!isq.qstate, #map>
    %77 = affine.load %71[0] : memref<1x!isq.qstate, #map>
    %78 = affine.load %72[0] : memref<1x!isq.qstate, #map>
    %79:4 = isq.apply %74(%75, %76, %77, %78) : !isq.gate<4>
    affine.store %79#0, %67[0] : memref<1x!isq.qstate, #map>
    affine.store %79#1, %69[0] : memref<1x!isq.qstate, #map>
    affine.store %79#2, %71[0] : memref<1x!isq.qstate, #map>
    affine.store %79#3, %72[0] : memref<1x!isq.qstate, #map>
    %80 = memref.get_global @q : memref<3x!isq.qstate>
    %c0_18 = arith.constant 0 : index
    %81 = memref.subview %80[%c0_18] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %82 = memref.get_global @q : memref<3x!isq.qstate>
    %c2_19 = arith.constant 2 : index
    %83 = memref.subview %82[%c2_19] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %84 = memref.get_global @p : memref<1x!isq.qstate, #map>
    %85 = memref.get_global @q : memref<3x!isq.qstate>
    %c1_20 = arith.constant 1 : index
    %86 = memref.subview %85[%c1_20] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %87 = isq.use @Rt2 : !isq.gate<2>
    %88 = isq.decorate(%87 : !isq.gate<2>) {adjoint = false, ctrl = [false, true]} : !isq.gate<4>
    %89 = affine.load %81[0] : memref<1x!isq.qstate, #map>
    %90 = affine.load %83[0] : memref<1x!isq.qstate, #map>
    %91 = affine.load %84[0] : memref<1x!isq.qstate, #map>
    %92 = affine.load %86[0] : memref<1x!isq.qstate, #map>
    %93:4 = isq.apply %88(%89, %90, %91, %92) : !isq.gate<4>
    affine.store %93#0, %81[0] : memref<1x!isq.qstate, #map>
    affine.store %93#1, %83[0] : memref<1x!isq.qstate, #map>
    affine.store %93#2, %84[0] : memref<1x!isq.qstate, #map>
    affine.store %93#3, %86[0] : memref<1x!isq.qstate, #map>
    %false_21 = arith.constant false
    %c0_22 = arith.constant 0 : index
    %94 = memref.alloc()[%c0_22] : memref<1xi1, #map>
    affine.store %false_21, %94[0] : memref<1xi1, #map>
    %false_23 = arith.constant false
    %c0_24 = arith.constant 0 : index
    %95 = memref.alloc()[%c0_24] : memref<1xi1, #map>
    affine.store %false_23, %95[0] : memref<1xi1, #map>
    scf.while : () -> () {
      %105 = scf.execute_region -> i1 {
        %106 = affine.load %94[0] : memref<1xi1, #map>
        cond_br %106, ^bb2, ^bb1
      ^bb1:  // pred: ^bb0
        %107 = memref.get_global @a : memref<1xindex, #map>
        %108 = affine.load %107[0] : memref<1xindex, #map>
        %c2_26 = arith.constant 2 : index
        %109 = arith.cmpi slt, %108, %c2_26 : index
        scf.yield %109 : i1
      ^bb2:  // pred: ^bb0
        %false_27 = arith.constant false
        scf.yield %false_27 : i1
      }
      scf.condition(%105)
    } do {
      scf.execute_region {
        %105 = memref.get_global @a : memref<1xindex, #map>
        %106 = memref.get_global @a : memref<1xindex, #map>
        %107 = affine.load %106[0] : memref<1xindex, #map>
        %c1_26 = arith.constant 1 : index
        %108 = arith.addi %107, %c1_26 : index
        affine.store %108, %105[0] : memref<1xindex, #map>
        br ^bb1
      ^bb1:  // pred: ^bb0
        scf.yield
      }
      scf.yield
    }
    %96 = memref.get_global @a : memref<1xindex, #map>
    %97 = affine.load %96[0] : memref<1xindex, #map>
    isq.call_qop@isq_builtin::@print_int(%97) : [0] (index) -> ()
    %98 = memref.get_global @p : memref<1x!isq.qstate, #map>
    %99 = affine.load %98[0] : memref<1x!isq.qstate, #map>
    %100 = isq.call_qop@isq_builtin::@reset(%99) : [1] () -> ()
    affine.store %100, %98[0] : memref<1x!isq.qstate, #map>
    %101 = memref.get_global @q : memref<3x!isq.qstate>
    %c1_25 = arith.constant 1 : index
    %102 = memref.subview %101[%c1_25] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map>
    %103 = affine.load %102[0] : memref<1x!isq.qstate, #map>
    %104 = isq.call_qop@isq_builtin::@reset(%103) : [1] () -> ()
    affine.store %104, %102[0] : memref<1x!isq.qstate, #map>
    br ^bb3
  ^bb3:  // pred: ^bb2
    memref.dealloc %95 : memref<1xi1, #map>
    br ^bb4
  ^bb4:  // pred: ^bb3
    memref.dealloc %94 : memref<1xi1, #map>
    br ^bb5
  ^bb5:  // pred: ^bb4
    memref.dealloc %10 : memref<5xindex>
    br ^bb6
  ^bb6:  // 2 preds: ^bb1, ^bb5
    memref.dealloc %6 : memref<1xi1, #map>
    br ^bb7
  ^bb7:  // 2 preds: ^bb0, ^bb6
    memref.dealloc %3 : memref<1xi1, #map>
    br ^bb8
  ^bb8:  // pred: ^bb7
    memref.dealloc %2 : memref<1xindex, #map>
    br ^bb9
  ^bb9:  // pred: ^bb8
    memref.dealloc %1 : memref<1xindex, #map>
    br ^bb10
  ^bb10:  // pred: ^bb9
    memref.dealloc %0 : memref<1xi1, #map>
    br ^bb11
  ^bb11:  // pred: ^bb10
    return
  }
  func @__isq__global_initialize() {
    return
  }
}

