// This file is generated by the isQ Experimental compiler
// Source file name: main.isq
module @isq_builtin {
    isq.declare_qop @measure : [1]()->i1
    isq.declare_qop @reset : [1]()->()
}
func private @printInt(index)->()
isq.defgate @Rs {definition = [{type="unitary", value = [[#isq.complex<0.5, 0.8660254>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>]] }]}: !isq.gate<2> loc("main.isq":1:1)
isq.defgate @Rs2 {definition = [{type="unitary", value = [[#isq.complex<0.5, -0.8660254>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>]] }]}: !isq.gate<2> loc("main.isq":5:1)
isq.defgate @Rt {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.5, 0.8660254>]] }]}: !isq.gate<2> loc("main.isq":10:1)
isq.defgate @Rt2 {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.5, -0.8660254>]] }]}: !isq.gate<2> loc("main.isq":14:1)
isq.defgate @CNOT {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>], [#isq.complex<0.0, 0.0>, #isq.complex<0.0, 0.0>, #isq.complex<1.0, 0.0>, #isq.complex<0.0, 0.0>]] }]}: !isq.gate<2> loc("main.isq":18:1)
isq.defgate @H {definition = [{type="unitary", value = [[#isq.complex<0.7071067811865476, 0.0>, #isq.complex<0.7071067811865476, 0.0>], [#isq.complex<0.7071067811865476, 0.0>, #isq.complex<-0.7071067811865476, -0.0>]] }]}: !isq.gate<1> loc("main.isq":22:1)
memref.global @a : memref<1xindex> = uninitialized loc("main.isq":31:5)
memref.global @b : memref<1xindex> = uninitialized loc("main.isq":31:8)
memref.global @c : memref<1xindex> = uninitialized loc("main.isq":31:11)
memref.global @q : memref<3x!isq.qstate> = uninitialized loc("main.isq":32:6)
memref.global @p : memref<1x!isq.qstate> = uninitialized loc("main.isq":32:12)
func @test(%x5: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %x6: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %t16: index)->index{
    %t17 = memref.alloca() : memref<1xindex> loc("main.isq":34:1)
    affine.store %t16, %t17[0] : memref<1xindex> loc("main.isq":34:1)
    %t18 = arith.constant 0 : index loc("main.isq":34:1)
    %x7 = memref.subview %t17[%t18][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":34:1)
    %t0 = isq.use @H : !isq.gate<1> loc("main.isq":35:9)
    %t1 = isq.decorate(%t0: !isq.gate<1>) {ctrl = [], adjoint = false} :!isq.gate<1> loc("main.isq":35:9)
    %t2 = affine.load %x5[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":35:9)
    %t3 = isq.apply %t1(%t2) : !isq.gate<1> loc("main.isq":35:9)
    affine.store %t3, %x5[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":35:9)
    %t4 = memref.alloca() : memref<1x!isq.qstate> loc("main.isq":36:14)
    %t5 = arith.constant 0 : index loc("main.isq":36:14)
    %x8 = memref.subview %t4[%t5][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":36:14)
    %t6 = isq.use @CNOT : !isq.gate<2> loc("main.isq":37:9)
    %t7 = isq.decorate(%t6: !isq.gate<2>) {ctrl = [], adjoint = false} :!isq.gate<2> loc("main.isq":37:9)
    %t8 = affine.load %x8[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":37:9)
    %t9 = affine.load %x5[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":37:9)
    %t10,%t11 = isq.apply %t7(%t8,%t9) : !isq.gate<2> loc("main.isq":37:9)
    affine.store %t10, %x8[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":37:9)
    affine.store %t11, %x5[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":37:9)
    %t12 = isq.use @H : !isq.gate<1> loc("main.isq":38:9)
    %t13 = isq.decorate(%t12: !isq.gate<1>) {ctrl = [], adjoint = false} :!isq.gate<1> loc("main.isq":38:9)
    %t14 = affine.load %x5[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":38:9)
    %t15 = isq.apply %t13(%t14) : !isq.gate<1> loc("main.isq":38:9)
    affine.store %t15, %x5[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":38:9)
    %x9 = arith.constant 2 : index loc("main.isq":39:16)
    return %x9 : index loc("main.isq":39:9)
}
func @test2(%x10: memref<?x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %t23: index)->(){
    %t24 = memref.alloca() : memref<1xindex> loc("main.isq":42:1)
    affine.store %t23, %t24[0] : memref<1xindex> loc("main.isq":42:1)
    %t25 = arith.constant 0 : index loc("main.isq":42:1)
    %x11 = memref.subview %t24[%t25][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":42:1)
    %x12 = affine.load %x11[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":43:13)
    %x13 = memref.subview %x10[%x12][1][1] : memref<?x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":43:11)
    %t19 = isq.use @H : !isq.gate<1> loc("main.isq":43:9)
    %t20 = isq.decorate(%t19: !isq.gate<1>) {ctrl = [], adjoint = false} :!isq.gate<1> loc("main.isq":43:9)
    %t21 = affine.load %x13[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":43:9)
    %t22 = isq.apply %t20(%t21) : !isq.gate<1> loc("main.isq":43:9)
    affine.store %t22, %x13[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":43:9)
    return
}
func @main(%x16: index, %x17: index)->(){
    %t26 = memref.alloca() : memref<1xindex> loc("main.isq":48:13)
    %t27 = arith.constant 0 : index loc("main.isq":48:13)
    %x14 = memref.subview %t26[%t27][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":48:13)
    %t28 = memref.alloca() : memref<1xindex> loc("main.isq":49:13)
    %t29 = arith.constant 0 : index loc("main.isq":49:13)
    %x15 = memref.subview %t28[%t29][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":49:13)
    //%x16 = arith.constant 1 : index loc("main.isq":50:13)
    //%x17 = arith.constant 2 : index loc("main.isq":50:17)
    %t30 = arith.cmpi "slt", %x16, %x17 : index loc("main.isq":50:12)
    %x18 = arith.index_cast %t30: i1 to index loc("main.isq":50:12)
    affine.if affine_set<()[d0]: (d0+1==0)>()[%x18] {
        %t31 = memref.get_global @a : memref<1xindex> loc("main.isq":51:22)
        %t32 = arith.constant 0 : index loc("main.isq":51:22)
        %t33 = memref.subview %t31[%t32][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":51:22)
        %x19 = affine.load %t33[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":51:22)
        %x20 = arith.constant 3 : index loc("main.isq":51:24)
        %x21 = arith.constant 2 : index loc("main.isq":51:26)
        %x22 = arith.muli %x20, %x21 : index loc("main.isq":51:24)
        %x23 = arith.addi %x19, %x22 : index loc("main.isq":51:21)
        %t34 = memref.get_global @b : memref<1xindex> loc("main.isq":51:30)
        %t35 = arith.constant 0 : index loc("main.isq":51:30)
        %t36 = memref.subview %t34[%t35][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":51:30)
        %x24 = affine.load %t36[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":51:30)
        %t37 = memref.get_global @c : memref<1xindex> loc("main.isq":51:32)
        %t38 = arith.constant 0 : index loc("main.isq":51:32)
        %t39 = memref.subview %t37[%t38][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":51:32)
        %x25 = affine.load %t39[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":51:32)
        %x26 = arith.addi %x24, %x25 : index loc("main.isq":51:29)
        %x27 = arith.muli %x23, %x26 : index loc("main.isq":51:21)
        affine.store %x27, %x14[0]: memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":51:17)
    } else {
        %t40 = memref.alloca() : memref<1xindex> loc("main.isq":53:21)
        %t41 = arith.constant 0 : index loc("main.isq":53:21)
        %x28 = memref.subview %t40[%t41][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":53:21)
        %t42 = memref.get_global @c : memref<1xindex> loc("main.isq":54:21)
        %t43 = arith.constant 0 : index loc("main.isq":54:21)
        %t44 = memref.subview %t42[%t43][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":54:21)
        %x29 = affine.load %t44[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":54:21)
        %x30 = arith.constant 1 : index loc("main.isq":54:23)
        %x31 = arith.addi %x29, %x30 : index loc("main.isq":54:21)
        affine.store %x31, %x28[0]: memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":54:17)
    } loc("main.isq":50:9)
    %x32 = arith.constant 1 : index loc("main.isq":56:18)
    %t45 = memref.get_global @b : memref<1xindex> loc("main.isq":56:23)
    %t46 = arith.constant 0 : index loc("main.isq":56:23)
    %t47 = memref.subview %t45[%t46][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":56:23)
    %x33 = affine.load %t47[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":56:23)
    affine.for %t48 = %x32 to %x33 step 1 {
    %t49 = memref.alloca() : memref<1xindex> loc("main.isq":56:9)
    %t50 = arith.constant 0 : index loc("main.isq":56:9)
    %x34 = memref.subview %t49[%t50][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":56:9)
    affine.store %t48, %x34[0]: memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":56:9)
        isq.pass loc("main.isq":57:17)
    } loc("main.isq":56:9)
    %t51 = memref.alloca() : memref<5xindex> loc("main.isq":59:13)
    %t52 = arith.constant 0 : index loc("main.isq":59:13)
    %x35 = memref.subview %t51[%t52][5][1] : memref<5xindex> to memref<5xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":59:13)
    %t53 = memref.get_global @c : memref<1xindex> loc("main.isq":60:15)
    %t54 = arith.constant 0 : index loc("main.isq":60:15)
    %t55 = memref.subview %t53[%t54][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":60:15)
    %x36 = affine.load %t55[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":60:15)
    %x37 = memref.subview %x35[%x36][1][1] : memref<5xindex, affine_map<(d0)[s0]->(d0+s0)>> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":60:13)
    %x38 = affine.load %x37[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":60:13)
    %x39 = arith.constant 2 : index loc("main.isq":60:18)
    %x40 = arith.addi %x38, %x39 : index loc("main.isq":60:13)
    %t56 = memref.get_global @a : memref<1xindex> loc("main.isq":60:9)
    %t57 = arith.constant 0 : index loc("main.isq":60:9)
    %t58 = memref.subview %t56[%t57][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":60:9)
    affine.store %x40, %t58[0]: memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":60:9)
    %x41 = affine.load %x14[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":61:20)
    %t59 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":61:9)
    %t60 = arith.constant 0 : index loc("main.isq":61:9)
    %t61 = memref.subview %t59[%t60][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":61:9)
    %t62 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":61:9)
    %t63 = arith.constant 0 : index loc("main.isq":61:9)
    %t64 = memref.subview %t62[%t63][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":61:9)
    %x42 = call @test(%t61, %t64, %x41) : (memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index)->index loc("main.isq":61:9)
    %t65 = memref.get_global @b : memref<1xindex> loc("main.isq":61:5)
    %t66 = arith.constant 0 : index loc("main.isq":61:5)
    %t67 = memref.subview %t65[%t66][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":61:5)
    affine.store %x42, %t67[0]: memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":61:5)
    %x43 = arith.constant 0 : index loc("main.isq":62:17)
    %t68 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":62:15)
    %t69 = arith.constant 0 : index loc("main.isq":62:15)
    %t70 = memref.subview %t68[%t69][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":62:15)
    %x44 = memref.subview %t70[%x43][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":62:15)
    %t71 = affine.load %x44[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":62:13)
    %t72, %t73 = isq.call_qop @isq_builtin::@measure(%t71): [1]()->i1 loc("main.isq":62:13)
    affine.store %t72, %x44[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":62:13)
    %x45 = arith.index_cast %t73 : i1 to index loc("main.isq":62:13)
    %t74 = memref.get_global @a : memref<1xindex> loc("main.isq":62:9)
    %t75 = arith.constant 0 : index loc("main.isq":62:9)
    %t76 = memref.subview %t74[%t75][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":62:9)
    affine.store %x45, %t76[0]: memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":62:9)
    %x46 = arith.constant 0 : index loc("main.isq":63:19)
    %t77 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":63:17)
    %t78 = arith.constant 0 : index loc("main.isq":63:17)
    %t79 = memref.subview %t77[%t78][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":63:17)
    %x47 = memref.subview %t79[%x46][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":63:17)
    %t80 = isq.use @CNOT : !isq.gate<2> loc("main.isq":63:9)
    %t81 = isq.decorate(%t80: !isq.gate<2>) {ctrl = [], adjoint = false} :!isq.gate<2> loc("main.isq":63:9)
    %t82 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":63:9)
    %t83 = arith.constant 0 : index loc("main.isq":63:9)
    %t84 = memref.subview %t82[%t83][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":63:9)
    %t85 = affine.load %t84[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":63:9)
    %t86 = affine.load %x47[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":63:9)
    %t87,%t88 = isq.apply %t81(%t85,%t86) : !isq.gate<2> loc("main.isq":63:9)
    %t89 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":63:9)
    %t90 = arith.constant 0 : index loc("main.isq":63:9)
    %t91 = memref.subview %t89[%t90][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":63:9)
    affine.store %t87, %t91[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":63:9)
    affine.store %t88, %x47[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":63:9)
    %x48 = arith.constant 0 : index loc("main.isq":65:25)
    %t92 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":65:23)
    %t93 = arith.constant 0 : index loc("main.isq":65:23)
    %t94 = memref.subview %t92[%t93][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:23)
    %x49 = memref.subview %t94[%x48][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:23)
    %x50 = arith.constant 1 : index loc("main.isq":65:31)
    %t95 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":65:29)
    %t96 = arith.constant 0 : index loc("main.isq":65:29)
    %t97 = memref.subview %t95[%t96][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:29)
    %x51 = memref.subview %t97[%x50][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:29)
    %t98 = isq.use @H : !isq.gate<1> loc("main.isq":65:9)
    %t99 = isq.decorate(%t98: !isq.gate<1>) {ctrl = [true, true], adjoint = true} :!isq.gate<3> loc("main.isq":65:9)
    %t100 = affine.load %x49[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:9)
    %t101 = affine.load %x51[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:9)
    %t102 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":65:9)
    %t103 = arith.constant 0 : index loc("main.isq":65:9)
    %t104 = memref.subview %t102[%t103][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:9)
    %t105 = affine.load %t104[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:9)
    %t106,%t107,%t108 = isq.apply %t99(%t100,%t101,%t105) : !isq.gate<3> loc("main.isq":65:9)
    affine.store %t106, %x49[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:9)
    affine.store %t107, %x51[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:9)
    %t109 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":65:9)
    %t110 = arith.constant 0 : index loc("main.isq":65:9)
    %t111 = memref.subview %t109[%t110][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:9)
    affine.store %t108, %t111[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":65:9)
    %x52 = arith.constant 1 : index loc("main.isq":66:21)
    %t112 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":66:19)
    %t113 = arith.constant 0 : index loc("main.isq":66:19)
    %t114 = memref.subview %t112[%t113][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:19)
    %x53 = memref.subview %t114[%x52][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:19)
    %x54 = arith.constant 0 : index loc("main.isq":66:27)
    %t115 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":66:25)
    %t116 = arith.constant 0 : index loc("main.isq":66:25)
    %t117 = memref.subview %t115[%t116][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:25)
    %x55 = memref.subview %t117[%x54][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:25)
    %t118 = isq.use @H : !isq.gate<1> loc("main.isq":66:9)
    %t119 = isq.decorate(%t118: !isq.gate<1>) {ctrl = [true, true], adjoint = false} :!isq.gate<3> loc("main.isq":66:9)
    %t120 = affine.load %x53[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:9)
    %t121 = affine.load %x55[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:9)
    %t122 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":66:9)
    %t123 = arith.constant 0 : index loc("main.isq":66:9)
    %t124 = memref.subview %t122[%t123][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:9)
    %t125 = affine.load %t124[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:9)
    %t126,%t127,%t128 = isq.apply %t119(%t120,%t121,%t125) : !isq.gate<3> loc("main.isq":66:9)
    affine.store %t126, %x53[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:9)
    affine.store %t127, %x55[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:9)
    %t129 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":66:9)
    %t130 = arith.constant 0 : index loc("main.isq":66:9)
    %t131 = memref.subview %t129[%t130][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:9)
    affine.store %t128, %t131[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":66:9)
    %x56 = arith.constant 0 : index loc("main.isq":67:23)
    %t132 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":67:21)
    %t133 = arith.constant 0 : index loc("main.isq":67:21)
    %t134 = memref.subview %t132[%t133][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":67:21)
    %x57 = memref.subview %t134[%x56][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":67:21)
    %t135 = isq.use @H : !isq.gate<1> loc("main.isq":67:9)
    %t136 = isq.decorate(%t135: !isq.gate<1>) {ctrl = [false], adjoint = true} :!isq.gate<2> loc("main.isq":67:9)
    %t137 = affine.load %x57[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":67:9)
    %t138 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":67:9)
    %t139 = arith.constant 0 : index loc("main.isq":67:9)
    %t140 = memref.subview %t138[%t139][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":67:9)
    %t141 = affine.load %t140[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":67:9)
    %t142,%t143 = isq.apply %t136(%t137,%t141) : !isq.gate<2> loc("main.isq":67:9)
    affine.store %t142, %x57[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":67:9)
    %t144 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":67:9)
    %t145 = arith.constant 0 : index loc("main.isq":67:9)
    %t146 = memref.subview %t144[%t145][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":67:9)
    affine.store %t143, %t146[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":67:9)
    %x58 = arith.constant 0 : index loc("main.isq":68:31)
    %t147 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":68:29)
    %t148 = arith.constant 0 : index loc("main.isq":68:29)
    %t149 = memref.subview %t147[%t148][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:29)
    %x59 = memref.subview %t149[%x58][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:29)
    %x60 = arith.constant 1 : index loc("main.isq":68:37)
    %t150 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":68:35)
    %t151 = arith.constant 0 : index loc("main.isq":68:35)
    %t152 = memref.subview %t150[%t151][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:35)
    %x61 = memref.subview %t152[%x60][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:35)
    %x62 = arith.constant 2 : index loc("main.isq":68:43)
    %t153 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":68:41)
    %t154 = arith.constant 0 : index loc("main.isq":68:41)
    %t155 = memref.subview %t153[%t154][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:41)
    %x63 = memref.subview %t155[%x62][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:41)
    %t156 = isq.use @H : !isq.gate<1> loc("main.isq":68:9)
    %t157 = isq.decorate(%t156: !isq.gate<1>) {ctrl = [true, false, false], adjoint = true} :!isq.gate<4> loc("main.isq":68:9)
    %t158 = affine.load %x59[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:9)
    %t159 = affine.load %x61[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:9)
    %t160 = affine.load %x63[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:9)
    %t161 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":68:9)
    %t162 = arith.constant 0 : index loc("main.isq":68:9)
    %t163 = memref.subview %t161[%t162][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:9)
    %t164 = affine.load %t163[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:9)
    %t165,%t166,%t167,%t168 = isq.apply %t157(%t158,%t159,%t160,%t164) : !isq.gate<4> loc("main.isq":68:9)
    affine.store %t165, %x59[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:9)
    affine.store %t166, %x61[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:9)
    affine.store %t167, %x63[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:9)
    %t169 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":68:9)
    %t170 = arith.constant 0 : index loc("main.isq":68:9)
    %t171 = memref.subview %t169[%t170][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:9)
    affine.store %t168, %t171[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":68:9)
    %x64 = arith.constant 0 : index loc("main.isq":69:26)
    %t172 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":69:24)
    %t173 = arith.constant 0 : index loc("main.isq":69:24)
    %t174 = memref.subview %t172[%t173][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:24)
    %x65 = memref.subview %t174[%x64][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:24)
    %x66 = arith.constant 2 : index loc("main.isq":69:32)
    %t175 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":69:30)
    %t176 = arith.constant 0 : index loc("main.isq":69:30)
    %t177 = memref.subview %t175[%t176][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:30)
    %x67 = memref.subview %t177[%x66][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:30)
    %x68 = arith.constant 1 : index loc("main.isq":69:41)
    %t178 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":69:39)
    %t179 = arith.constant 0 : index loc("main.isq":69:39)
    %t180 = memref.subview %t178[%t179][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:39)
    %x69 = memref.subview %t180[%x68][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:39)
    %t181 = isq.use @Rt2 : !isq.gate<2> loc("main.isq":69:9)
    %t182 = isq.decorate(%t181: !isq.gate<2>) {ctrl = [false, true], adjoint = false} :!isq.gate<4> loc("main.isq":69:9)
    %t183 = affine.load %x65[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:9)
    %t184 = affine.load %x67[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:9)
    %t185 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":69:9)
    %t186 = arith.constant 0 : index loc("main.isq":69:9)
    %t187 = memref.subview %t185[%t186][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:9)
    %t188 = affine.load %t187[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:9)
    %t189 = affine.load %x69[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:9)
    %t190,%t191,%t192,%t193 = isq.apply %t182(%t183,%t184,%t188,%t189) : !isq.gate<4> loc("main.isq":69:9)
    affine.store %t190, %x65[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:9)
    affine.store %t191, %x67[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:9)
    %t194 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":69:9)
    %t195 = arith.constant 0 : index loc("main.isq":69:9)
    %t196 = memref.subview %t194[%t195][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:9)
    affine.store %t192, %t196[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:9)
    affine.store %t193, %x69[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":69:9)
    scf.while () : ()->(){
        %t197 = memref.get_global @a : memref<1xindex> loc("main.isq":71:16)
        %t198 = arith.constant 0 : index loc("main.isq":71:16)
        %t199 = memref.subview %t197[%t198][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":71:16)
        %x70 = affine.load %t199[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":71:16)
        %x71 = arith.constant 2 : index loc("main.isq":71:20)
        %t200 = arith.cmpi "slt", %x70, %x71 : index loc("main.isq":71:16)
        %x72 = arith.index_cast %t200: i1 to index loc("main.isq":71:16)
        %t207 = arith.index_cast %x72 : index to i1 loc("main.isq":71:9)
        scf.condition(%t207) loc("main.isq":71:9)
    } do {
        %t201 = memref.get_global @a : memref<1xindex> loc("main.isq":72:21)
        %t202 = arith.constant 0 : index loc("main.isq":72:21)
        %t203 = memref.subview %t201[%t202][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":72:21)
        %x73 = affine.load %t203[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":72:21)
        %x74 = arith.constant 1 : index loc("main.isq":72:25)
        %x75 = arith.addi %x73, %x74 : index loc("main.isq":72:21)
        %t204 = memref.get_global @a : memref<1xindex> loc("main.isq":72:17)
        %t205 = arith.constant 0 : index loc("main.isq":72:17)
        %t206 = memref.subview %t204[%t205][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":72:17)
        affine.store %x75, %t206[0]: memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":72:17)
        scf.yield
    } loc("main.isq":71:9)
    %t208 = memref.get_global @a : memref<1xindex> loc("main.isq":75:15)
    %t209 = arith.constant 0 : index loc("main.isq":75:15)
    %t210 = memref.subview %t208[%t209][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":75:15)
    %x76 = affine.load %t210[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":75:15)
    call @printInt(%x76): (index)->() loc("main.isq":75:9)
    %t211 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":76:9)
    %t212 = arith.constant 0 : index loc("main.isq":76:9)
    %t213 = memref.subview %t211[%t212][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":76:9)
    %t214 = affine.load %t213[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":76:9)
    %t215 = isq.call_qop @isq_builtin::@reset(%t214): [1]()->() loc("main.isq":76:9)
    %t216 = memref.get_global @p : memref<1x!isq.qstate> loc("main.isq":76:9)
    %t217 = arith.constant 0 : index loc("main.isq":76:9)
    %t218 = memref.subview %t216[%t217][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":76:9)
    affine.store %t215, %t218[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":76:9)
    %x77 = arith.constant 1 : index loc("main.isq":77:11)
    %t219 = memref.get_global @q : memref<3x!isq.qstate> loc("main.isq":77:9)
    %t220 = arith.constant 0 : index loc("main.isq":77:9)
    %t221 = memref.subview %t219[%t220][3][1] : memref<3x!isq.qstate> to memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":77:9)
    %x78 = memref.subview %t221[%x77][1][1] : memref<3x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":77:9)
    %t222 = affine.load %x78[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":77:9)
    %t223 = isq.call_qop @isq_builtin::@reset(%t222): [1]()->() loc("main.isq":77:9)
    affine.store %t223, %x78[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":77:9)
    return
}
