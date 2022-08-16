#map0 = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0 + 2)>
module  {
  module @isq_builtin  {
    isq.declare_qop@measure : [1] () -> i1
    isq.declare_qop@reset : [1] () -> ()
  }
  func @printInt(%a: index)->(){
      return
  }
  isq.defgate@Rs {definition = [{type = "unitary", value = [[#isq.complex<5.000000e-01, 0.86602539999999995>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>]]}]} : !isq.gate<2>
  isq.defgate@Rs2 {definition = [{type = "unitary", value = [[#isq.complex<5.000000e-01, -0.86602539999999995>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>]]}]} : !isq.gate<2>
  isq.defgate@Rt {definition = [{type = "unitary", value = [[#isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<5.000000e-01, 0.86602539999999995>]]}]} : !isq.gate<2>
  isq.defgate@Rt2 {definition = [{type = "unitary", value = [[#isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<5.000000e-01, -0.86602539999999995>]]}]} : !isq.gate<2>
  isq.defgate@CNOT {definition = [{type = "unitary", value = [[#isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>], [#isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>, #isq.complex<1.000000e+00, 0.000000e+00>, #isq.complex<0.000000e+00, 0.000000e+00>]]}]} : !isq.gate<2>
  isq.defgate@H {definition = [{type = "unitary", value = [[#isq.complex<0.70710678118654757, 0.000000e+00>, #isq.complex<0.70710678118654757, 0.000000e+00>], [#isq.complex<0.70710678118654757, 0.000000e+00>, #isq.complex<-0.70710678118654757, -0.000000e+00>]]}]} : !isq.gate<1>
  memref.global @a : memref<1xindex> = uninitialized
  memref.global @b : memref<1xindex> = uninitialized
  memref.global @c : memref<1xindex> = uninitialized
  memref.global @q : memref<3x!isq.qstate> = uninitialized
  memref.global @p : memref<1x!isq.qstate> = uninitialized
  func @test(%arg0: memref<1x!isq.qstate, #map0>, %arg1: memref<1x!isq.qstate, #map0>, %arg2: index) -> index {
    %c2 = arith.constant 2 : index
    %0 = memref.alloca() : memref<1xindex>
    affine.store %arg2, %0[0] : memref<1xindex>
    %1 = isq.use @H : !isq.gate<1>
    %2 = isq.decorate(%1 : !isq.gate<1>) {adjoint = false, ctrl = []} : !isq.gate<1>
    %3 = affine.load %arg0[0] : memref<1x!isq.qstate, #map0>
    %4 = isq.apply %2(%3) : !isq.gate<1>
    affine.store %4, %arg0[0] : memref<1x!isq.qstate, #map0>
    %5 = memref.alloca() : memref<1x!isq.qstate>
    %6 = isq.use @CNOT : !isq.gate<2>
    %7 = isq.decorate(%6 : !isq.gate<2>) {adjoint = false, ctrl = []} : !isq.gate<2>
    %8 = affine.load %5[0] : memref<1x!isq.qstate>
    %9 = affine.load %arg0[0] : memref<1x!isq.qstate, #map0>
    %10:2 = isq.apply %7(%8, %9) : !isq.gate<2>
    affine.store %10#0, %5[0] : memref<1x!isq.qstate>
    affine.store %10#1, %arg0[0] : memref<1x!isq.qstate, #map0>
    %11 = isq.use @H : !isq.gate<1>
    %12 = isq.decorate(%11 : !isq.gate<1>) {adjoint = false, ctrl = []} : !isq.gate<1>
    %13 = affine.load %arg0[0] : memref<1x!isq.qstate, #map0>
    %14 = isq.apply %12(%13) : !isq.gate<1>
    affine.store %14, %arg0[0] : memref<1x!isq.qstate, #map0>
    return %c2 : index
  }
  func @test2(%arg0: memref<?x!isq.qstate, #map0>, %arg1: index) {
    %0 = memref.alloca() : memref<1xindex>
    affine.store %arg1, %0[0] : memref<1xindex>
    %1 = affine.load %0[0] : memref<1xindex>
    %2 = memref.subview %arg0[%1] [1] [1] : memref<?x!isq.qstate, #map0> to memref<1x!isq.qstate, #map0>
    %3 = isq.use @H : !isq.gate<1>
    %4 = isq.decorate(%3 : !isq.gate<1>) {adjoint = false, ctrl = []} : !isq.gate<1>
    %5 = affine.load %2[0] : memref<1x!isq.qstate, #map0>
    %6 = isq.apply %4(%5) : !isq.gate<1>
    affine.store %6, %2[0] : memref<1x!isq.qstate, #map0>
    return
  }
  func @main() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %0 = memref.alloca() : memref<1xindex>
    %1 = memref.get_global @a : memref<1xindex>
    %2 = affine.load %1[0] : memref<1xindex>
    %3 = arith.addi %2, %c6 : index
    %4 = memref.get_global @b : memref<1xindex>
    %5 = affine.load %4[0] : memref<1xindex>
    %6 = memref.get_global @c : memref<1xindex>
    %7 = affine.load %6[0] : memref<1xindex>
    %8 = arith.addi %5, %7 : index
    %9 = arith.muli %3, %8 : index
    affine.store %9, %0[0] : memref<1xindex>
    %10 = memref.get_global @b : memref<1xindex>
    %11 = affine.load %10[0] : memref<1xindex>
    affine.for %arg0 = 1 to %11 {
      %114 = memref.alloca() : memref<1xindex>
      affine.store %arg0, %114[0] : memref<1xindex>
      isq.pass
    }
    %12 = memref.alloca() : memref<5xindex>
    %13 = memref.get_global @c : memref<1xindex>
    %14 = affine.load %13[0] : memref<1xindex>
    %15 = memref.subview %12[%14] [1] [1] : memref<5xindex> to memref<1xindex, #map0>
    %16 = affine.load %15[0] : memref<1xindex, #map0>
    %17 = arith.addi %16, %c2 : index
    %18 = memref.get_global @a : memref<1xindex>
    affine.store %17, %18[0] : memref<1xindex>
    %19 = affine.load %0[0] : memref<1xindex>
    %20 = memref.get_global @p : memref<1x!isq.qstate>
    %21 = memref.cast %20 : memref<1x!isq.qstate> to memref<1x!isq.qstate, #map0>
    %22 = memref.get_global @p : memref<1x!isq.qstate>
    %23 = memref.cast %22 : memref<1x!isq.qstate> to memref<1x!isq.qstate, #map0>
    %24 = call @test(%21, %23, %19) : (memref<1x!isq.qstate, #map0>, memref<1x!isq.qstate, #map0>, index) -> index
    %25 = memref.get_global @b : memref<1xindex>
    affine.store %24, %25[0] : memref<1xindex>
    %26 = memref.get_global @q : memref<3x!isq.qstate>
    %27 = memref.subview %26[0] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate>
    %28 = affine.load %27[0] : memref<1x!isq.qstate>
    %29:2 = isq.call_qop@isq_builtin::@measure(%28) : [1] () -> i1
    affine.store %29#0, %27[0] : memref<1x!isq.qstate>
    %30 = arith.index_cast %29#1 : i1 to index
    %31 = memref.get_global @a : memref<1xindex>
    affine.store %30, %31[0] : memref<1xindex>
    %32 = memref.get_global @q : memref<3x!isq.qstate>
    %33 = memref.subview %32[0] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate>
    %34 = isq.use @CNOT : !isq.gate<2>
    %35 = isq.decorate(%34 : !isq.gate<2>) {adjoint = false, ctrl = []} : !isq.gate<2>
    %36 = memref.get_global @p : memref<1x!isq.qstate>
    %37 = affine.load %36[0] : memref<1x!isq.qstate>
    %38 = affine.load %33[0] : memref<1x!isq.qstate>
    %39:2 = isq.apply %35(%37, %38) : !isq.gate<2>
    %40 = memref.get_global @p : memref<1x!isq.qstate>
    affine.store %39#0, %40[0] : memref<1x!isq.qstate>
    affine.store %39#1, %33[0] : memref<1x!isq.qstate>
    %41 = memref.get_global @q : memref<3x!isq.qstate>
    %42 = memref.subview %41[0] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate>
    %43 = memref.get_global @q : memref<3x!isq.qstate>
    %44 = memref.subview %43[1] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map1>
    %45 = isq.use @H : !isq.gate<1>
    %46 = isq.decorate(%45 : !isq.gate<1>) {adjoint = true, ctrl = [true, true]} : !isq.gate<3>
    %47 = affine.load %42[0] : memref<1x!isq.qstate>
    %48 = affine.load %44[0] : memref<1x!isq.qstate, #map1>
    %49 = memref.get_global @p : memref<1x!isq.qstate>
    %50 = affine.load %49[0] : memref<1x!isq.qstate>
    %51:3 = isq.apply %46(%47, %48, %50) : !isq.gate<3>
    affine.store %51#0, %42[0] : memref<1x!isq.qstate>
    affine.store %51#1, %44[0] : memref<1x!isq.qstate, #map1>
    %52 = memref.get_global @p : memref<1x!isq.qstate>
    affine.store %51#2, %52[0] : memref<1x!isq.qstate>
    %53 = memref.get_global @q : memref<3x!isq.qstate>
    %54 = memref.subview %53[1] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map1>
    %55 = memref.get_global @q : memref<3x!isq.qstate>
    %56 = memref.subview %55[0] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate>
    %57 = isq.use @H : !isq.gate<1>
    %58 = isq.decorate(%57 : !isq.gate<1>) {adjoint = false, ctrl = [true, true]} : !isq.gate<3>
    %59 = affine.load %54[0] : memref<1x!isq.qstate, #map1>
    %60 = affine.load %56[0] : memref<1x!isq.qstate>
    %61 = memref.get_global @p : memref<1x!isq.qstate>
    %62 = affine.load %61[0] : memref<1x!isq.qstate>
    %63:3 = isq.apply %58(%59, %60, %62) : !isq.gate<3>
    affine.store %63#0, %54[0] : memref<1x!isq.qstate, #map1>
    affine.store %63#1, %56[0] : memref<1x!isq.qstate>
    %64 = memref.get_global @p : memref<1x!isq.qstate>
    affine.store %63#2, %64[0] : memref<1x!isq.qstate>
    %65 = memref.get_global @q : memref<3x!isq.qstate>
    %66 = memref.subview %65[0] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate>
    %67 = isq.use @H : !isq.gate<1>
    %68 = isq.decorate(%67 : !isq.gate<1>) {adjoint = true, ctrl = [false]} : !isq.gate<2>
    %69 = affine.load %66[0] : memref<1x!isq.qstate>
    %70 = memref.get_global @p : memref<1x!isq.qstate>
    %71 = affine.load %70[0] : memref<1x!isq.qstate>
    %72:2 = isq.apply %68(%69, %71) : !isq.gate<2>
    affine.store %72#0, %66[0] : memref<1x!isq.qstate>
    %73 = memref.get_global @p : memref<1x!isq.qstate>
    affine.store %72#1, %73[0] : memref<1x!isq.qstate>
    %74 = memref.get_global @q : memref<3x!isq.qstate>
    %75 = memref.subview %74[0] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate>
    %76 = memref.get_global @q : memref<3x!isq.qstate>
    %77 = memref.subview %76[1] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map1>
    %78 = memref.get_global @q : memref<3x!isq.qstate>
    %79 = memref.subview %78[2] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map2>
    %80 = isq.use @H : !isq.gate<1>
    %81 = isq.decorate(%80 : !isq.gate<1>) {adjoint = true, ctrl = [true, false, false]} : !isq.gate<4>
    %82 = affine.load %75[0] : memref<1x!isq.qstate>
    %83 = affine.load %77[0] : memref<1x!isq.qstate, #map1>
    %84 = affine.load %79[0] : memref<1x!isq.qstate, #map2>
    %85 = memref.get_global @p : memref<1x!isq.qstate>
    %86 = affine.load %85[0] : memref<1x!isq.qstate>
    %87:4 = isq.apply %81(%82, %83, %84, %86) : !isq.gate<4>
    affine.store %87#0, %75[0] : memref<1x!isq.qstate>
    affine.store %87#1, %77[0] : memref<1x!isq.qstate, #map1>
    affine.store %87#2, %79[0] : memref<1x!isq.qstate, #map2>
    %88 = memref.get_global @p : memref<1x!isq.qstate>
    affine.store %87#3, %88[0] : memref<1x!isq.qstate>
    %89 = memref.get_global @q : memref<3x!isq.qstate>
    %90 = memref.subview %89[0] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate>
    %91 = memref.get_global @q : memref<3x!isq.qstate>
    %92 = memref.subview %91[2] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map2>
    %93 = memref.get_global @q : memref<3x!isq.qstate>
    %94 = memref.subview %93[1] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map1>
    %95 = isq.use @Rt2 : !isq.gate<2>
    %96 = isq.decorate(%95 : !isq.gate<2>) {adjoint = false, ctrl = [false, true]} : !isq.gate<4>
    %97 = affine.load %90[0] : memref<1x!isq.qstate>
    %98 = affine.load %92[0] : memref<1x!isq.qstate, #map2>
    %99 = memref.get_global @p : memref<1x!isq.qstate>
    %100 = affine.load %99[0] : memref<1x!isq.qstate>
    %101 = affine.load %94[0] : memref<1x!isq.qstate, #map1>
    %102:4 = isq.apply %96(%97, %98, %100, %101) : !isq.gate<4>
    affine.store %102#0, %90[0] : memref<1x!isq.qstate>
    affine.store %102#1, %92[0] : memref<1x!isq.qstate, #map2>
    %103 = memref.get_global @p : memref<1x!isq.qstate>
    affine.store %102#2, %103[0] : memref<1x!isq.qstate>
    affine.store %102#3, %94[0] : memref<1x!isq.qstate, #map1>
    scf.while : () -> () {
      %114 = memref.get_global @a : memref<1xindex>
      %115 = affine.load %114[0] : memref<1xindex>
      %116 = arith.cmpi slt, %115, %c2 : index
      scf.condition(%116)
    } do {
      %114 = memref.get_global @a : memref<1xindex>
      %115 = affine.load %114[0] : memref<1xindex>
      %116 = arith.addi %115, %c1 : index
      %117 = memref.get_global @a : memref<1xindex>
      affine.store %116, %117[0] : memref<1xindex>
      scf.yield
    }
    %104 = memref.get_global @a : memref<1xindex>
    %105 = affine.load %104[0] : memref<1xindex>
    call @printInt(%105) : (index) -> ()
    %106 = memref.get_global @p : memref<1x!isq.qstate>
    %107 = affine.load %106[0] : memref<1x!isq.qstate>
    %108 = isq.call_qop@isq_builtin::@reset(%107) : [1] () -> ()
    %109 = memref.get_global @p : memref<1x!isq.qstate>
    affine.store %108, %109[0] : memref<1x!isq.qstate>
    %110 = memref.get_global @q : memref<3x!isq.qstate>
    %111 = memref.subview %110[1] [1] [1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, #map1>
    %112 = affine.load %111[0] : memref<1x!isq.qstate, #map1>
    %113 = isq.call_qop@isq_builtin::@reset(%112) : [1] () -> ()
    affine.store %113, %111[0] : memref<1x!isq.qstate, #map1>
    return
  }
}

