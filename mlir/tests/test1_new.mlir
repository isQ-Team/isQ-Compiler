module{
    module @isq_builtin {
        isq.declare_qop @measure : [1]()->i1
        isq.declare_qop @reset : [1]()->()
        isq.declare_qop @print_int : [0](index)->()
        isq.declare_qop @print_double : [0](f64)->()
    }
    isq.defgate @Rs {definition = [{type="unitary", value = [[#isq.complex<0.5, 0.8660254>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>]]}]}: !isq.gate<2> loc("tests/test1.isq":1:1)
    isq.defgate @Rs2 {definition = [{type="unitary", value = [[#isq.complex<0.5, -0.8660254>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>]]}]}: !isq.gate<2> loc("tests/test1.isq":5:1)
    isq.defgate @Rt {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.5, 0.8660254>]]}]}: !isq.gate<2> loc("tests/test1.isq":10:1)
    isq.defgate @Rt2 {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.5, -0.8660254>]]}]}: !isq.gate<2> loc("tests/test1.isq":14:1)
    isq.defgate @CNOT {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>]]}]}: !isq.gate<2> loc("tests/test1.isq":18:1)
    isq.defgate @H {definition = [{type="unitary", value = [[#isq.complex<0.7071067811865476, 0.0>,#isq.complex<0.7071067811865476, 0.0>],[#isq.complex<0.7071067811865476, 0.0>,#isq.complex<-0.7071067811865476, -0.0>]]}]}: !isq.gate<1> loc("tests/test1.isq":22:1)
    memref.global @a : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> = uninitialized loc("tests/test1.isq":31:1)
    memref.global @b : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> = uninitialized loc("tests/test1.isq":31:1)
    memref.global @c : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> = uninitialized loc("tests/test1.isq":31:1)
    memref.global @q : memref<3x!isq.qstate> = uninitialized loc("tests/test1.isq":32:1)
    memref.global @p : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> = uninitialized loc("tests/test1.isq":32:1)
    func @test(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index)->index 
    {
    ^entry(%ssa_15: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_16: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_18: index):
        %ssa_14_zero = arith.constant 0 : index
        %ssa_14 = memref.alloc()[%ssa_14_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":34:1)
        %ssa_20_zero = arith.constant 0 : index
        %ssa_20 = memref.alloc()[%ssa_20_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":34:1)
        affine.store %ssa_18, %ssa_20[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":34:1)
        %ssa_21 = arith.constant 0 : i1 loc("tests/test1.isq":34:1)
        %ssa_22_zero = arith.constant 0 : index
        %ssa_22 = memref.alloc()[%ssa_22_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":34:1)
        affine.store %ssa_21, %ssa_22[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":34:1)
        %ssa_23 = isq.use @H : !isq.gate<1> loc("tests/test1.isq":35:9) 
        %ssa_23_in_1 = affine.load %ssa_15[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":35:9)
        %ssa_23_out_1 = isq.apply %ssa_23(%ssa_23_in_1) : !isq.gate<1> loc("tests/test1.isq":35:9)
        affine.store %ssa_23_out_1, %ssa_15[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":35:9)
        %ssa_25_zero = arith.constant 0 : index
        %ssa_25 = memref.alloc()[%ssa_25_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":36:9)
        %ssa_26 = isq.use @CNOT : !isq.gate<2> loc("tests/test1.isq":37:9) 
        %ssa_26_in_1 = affine.load %ssa_25[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":37:9)
        %ssa_26_in_2 = affine.load %ssa_15[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":37:9)
        %ssa_26_out_1, %ssa_26_out_2 = isq.apply %ssa_26(%ssa_26_in_1, %ssa_26_in_2) : !isq.gate<2> loc("tests/test1.isq":37:9)
        affine.store %ssa_26_out_1, %ssa_25[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":37:9)
        affine.store %ssa_26_out_2, %ssa_15[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":37:9)
        %ssa_29 = isq.use @H : !isq.gate<1> loc("tests/test1.isq":38:9) 
        %ssa_29_in_1 = affine.load %ssa_15[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":38:9)
        %ssa_29_out_1 = isq.apply %ssa_29(%ssa_29_in_1) : !isq.gate<1> loc("tests/test1.isq":38:9)
        affine.store %ssa_29_out_1, %ssa_15[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":38:9)
        %ssa_32 = arith.constant 2 : index loc("tests/test1.isq":39:16)
        affine.store %ssa_32, %ssa_14[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":39:9)
        br ^exit_3 loc("tests/test1.isq":39:9)
    ^block1:
        br ^exit_3 
    ^exit_3:
        memref.dealloc %ssa_25 : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":36:9)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_22 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":34:1)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_20 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":34:1)
        br ^exit 
    ^exit:
        %ssa_34 = affine.load %ssa_14[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":34:1)
        memref.dealloc %ssa_14 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":34:1)
        return %ssa_34 : index loc("tests/test1.isq":34:1)
    } loc("tests/test1.isq":34:1)
    func @test2(memref<?x!isq.qstate>, index) 
    {
    ^entry(%ssa_35: memref<?x!isq.qstate>, %ssa_37: index):
        %ssa_39_zero = arith.constant 0 : index
        %ssa_39 = memref.alloc()[%ssa_39_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":42:1)
        affine.store %ssa_37, %ssa_39[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":42:1)
        %ssa_40 = arith.constant 0 : i1 loc("tests/test1.isq":42:1)
        %ssa_41_zero = arith.constant 0 : index
        %ssa_41 = memref.alloc()[%ssa_41_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":42:1)
        affine.store %ssa_40, %ssa_41[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":42:1)
        %ssa_45 = affine.load %ssa_39[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":43:13)
        %ssa_46 = memref.subview %ssa_35[%ssa_45][1][1] : memref<?x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":43:12)
        %ssa_42 = isq.use @H : !isq.gate<1> loc("tests/test1.isq":43:9) 
        %ssa_42_in_1 = affine.load %ssa_46[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":43:9)
        %ssa_42_out_1 = isq.apply %ssa_42(%ssa_42_in_1) : !isq.gate<1> loc("tests/test1.isq":43:9)
        affine.store %ssa_42_out_1, %ssa_46[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":43:9)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_41 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":42:1)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_39 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":42:1)
        br ^exit 
    ^exit:
        return loc("tests/test1.isq":42:1)
    } loc("tests/test1.isq":42:1)
    func @main() 
    {
    ^entry:
        %ssa_47 = arith.constant 0 : i1 loc("tests/test1.isq":46:1)
        %ssa_48_zero = arith.constant 0 : index
        %ssa_48 = memref.alloc()[%ssa_48_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":46:1)
        affine.store %ssa_47, %ssa_48[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":46:1)
        %ssa_49_zero = arith.constant 0 : index
        %ssa_49 = memref.alloc()[%ssa_49_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":48:9)
        %ssa_50_zero = arith.constant 0 : index
        %ssa_50 = memref.alloc()[%ssa_50_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":49:9)
        %ssa_51 = arith.constant 0 : i1 loc("tests/test1.isq":50:9)
        %ssa_52_zero = arith.constant 0 : index
        %ssa_52 = memref.alloc()[%ssa_52_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":50:9)
        affine.store %ssa_51, %ssa_52[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":50:9)
        %ssa_53 = arith.constant 1 : index loc("tests/test1.isq":50:13)
        %ssa_54 = arith.constant 2 : index loc("tests/test1.isq":50:17)
        %ssa_55 = arith.cmpi "slt", %ssa_53, %ssa_54 : index loc("tests/test1.isq":50:15)
        scf.if %ssa_55 {
            scf.execute_region {
            ^entry:
                %ssa_57 = memref.get_global @a : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":51:22)
                %ssa_61 = affine.load %ssa_57[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":51:22)
                %ssa_58 = arith.constant 3 : index loc("tests/test1.isq":51:24)
                %ssa_59 = arith.constant 2 : index loc("tests/test1.isq":51:26)
                %ssa_60 = arith.muli %ssa_58, %ssa_59 : index loc("tests/test1.isq":51:25)
                %ssa_62 = arith.addi %ssa_61, %ssa_60 : index loc("tests/test1.isq":51:23)
                %ssa_63 = memref.get_global @b : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":51:30)
                %ssa_65 = affine.load %ssa_63[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":51:30)
                %ssa_64 = memref.get_global @c : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":51:32)
                %ssa_66 = affine.load %ssa_64[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":51:32)
                %ssa_67 = arith.addi %ssa_65, %ssa_66 : index loc("tests/test1.isq":51:31)
                %ssa_68 = arith.muli %ssa_62, %ssa_67 : index loc("tests/test1.isq":51:28)
                affine.store %ssa_68, %ssa_49[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":51:19)
                br ^exit 
            ^exit:
                scf.yield loc("tests/test1.isq":50:9)
            } loc("tests/test1.isq":50:9)
        } else {
            scf.execute_region {
            ^entry:
                %ssa_69_zero = arith.constant 0 : index
                %ssa_69 = memref.alloc()[%ssa_69_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":53:17)
                %ssa_71 = memref.get_global @c : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":54:21)
                %ssa_73 = affine.load %ssa_71[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":54:21)
                %ssa_72 = arith.constant 1 : index loc("tests/test1.isq":54:23)
                %ssa_74 = arith.addi %ssa_73, %ssa_72 : index loc("tests/test1.isq":54:22)
                affine.store %ssa_74, %ssa_69[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":54:19)
                br ^exit_1 
            ^exit_1:
                memref.dealloc %ssa_69 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":53:17)
                br ^exit 
            ^exit:
                scf.yield loc("tests/test1.isq":50:9)
            } loc("tests/test1.isq":50:9)
        }
        %ssa_76 = affine.load %ssa_48[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":50:9)
        cond_br %ssa_76, ^exit_4, ^block1 loc("tests/test1.isq":50:9)
    ^block1:
        %ssa_77 = arith.constant 0 : i1 loc("tests/test1.isq":56:9)
        %ssa_78_zero = arith.constant 0 : index
        %ssa_78 = memref.alloc()[%ssa_78_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":56:9)
        affine.store %ssa_77, %ssa_78[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":56:9)
        %ssa_79 = arith.constant 1 : index loc("tests/test1.isq":56:18)
        %ssa_80 = memref.get_global @b : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":56:20)
        %ssa_81 = affine.load %ssa_80[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":56:20)
        affine.for %ssa_84 = %ssa_79 to %ssa_81 step 1 {
            scf.execute_region {
            ^entry:
                br ^exit 
            ^exit:
                scf.yield loc("tests/test1.isq":56:9)
            } loc("tests/test1.isq":56:9)
        } loc("tests/test1.isq":56:9)
        %ssa_86 = affine.load %ssa_48[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":56:9)
        cond_br %ssa_86, ^exit_5, ^block2 loc("tests/test1.isq":56:9)
    ^block2:
        %ssa_87 = memref.alloc() : memref<5xindex> loc("tests/test1.isq":59:9)
        %ssa_88 = memref.get_global @a : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":60:9)
        %ssa_90 = memref.get_global @c : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":60:15)
        %ssa_91 = affine.load %ssa_90[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":60:15)
        %ssa_92 = memref.subview %ssa_87[%ssa_91][1][1] : memref<5xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":60:14)
        %ssa_94 = affine.load %ssa_92[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":60:14)
        %ssa_93 = arith.constant 2 : index loc("tests/test1.isq":60:18)
        %ssa_95 = arith.addi %ssa_94, %ssa_93 : index loc("tests/test1.isq":60:17)
        affine.store %ssa_95, %ssa_88[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":60:11)
        %ssa_96 = memref.get_global @b : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":61:5)
        %ssa_98 = memref.get_global @p : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":61:14)
        %ssa_99 = memref.get_global @p : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":61:17)
        %ssa_101 = affine.load %ssa_49[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":61:20)
        %ssa_102 = call @test(%ssa_98, %ssa_99, %ssa_101) : (memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index)->index loc("tests/test1.isq":61:9)
        affine.store %ssa_102, %ssa_96[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":61:7)
        %ssa_103 = memref.get_global @a : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":62:5)
        %ssa_104 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":62:11)
        %ssa_105 = arith.constant 0 : index loc("tests/test1.isq":62:13)
        %ssa_106 = memref.subview %ssa_104[%ssa_105][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":62:12)
        %ssa_107_in = affine.load %ssa_106[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":62:9)
        %ssa_107_out, %ssa_107 = isq.call_qop @isq_builtin::@measure(%ssa_107_in): [1]()->i1 loc("tests/test1.isq":62:9)
        affine.store %ssa_107_out, %ssa_106[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":62:9)
        %ssa_108_i2 = arith.extui %ssa_107 : i1 to i2 loc("tests/test1.isq":62:9)
        %ssa_108 = arith.index_cast %ssa_108_i2 : i2 to index loc("tests/test1.isq":62:9)
        affine.store %ssa_108, %ssa_103[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":62:7)
        %ssa_110 = memref.get_global @p : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":63:14)
        %ssa_111 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":63:17)
        %ssa_112 = arith.constant 0 : index loc("tests/test1.isq":63:19)
        %ssa_113 = memref.subview %ssa_111[%ssa_112][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":63:18)
        %ssa_109 = isq.use @CNOT : !isq.gate<2> loc("tests/test1.isq":63:9) 
        %ssa_109_in_1 = affine.load %ssa_110[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":63:9)
        %ssa_109_in_2 = affine.load %ssa_113[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":63:9)
        %ssa_109_out_1, %ssa_109_out_2 = isq.apply %ssa_109(%ssa_109_in_1, %ssa_109_in_2) : !isq.gate<2> loc("tests/test1.isq":63:9)
        affine.store %ssa_109_out_1, %ssa_110[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":63:9)
        affine.store %ssa_109_out_2, %ssa_113[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":63:9)
        %ssa_115 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":65:23)
        %ssa_116 = arith.constant 0 : index loc("tests/test1.isq":65:25)
        %ssa_117 = memref.subview %ssa_115[%ssa_116][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":65:24)
        %ssa_118 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":65:29)
        %ssa_119 = arith.constant 1 : index loc("tests/test1.isq":65:31)
        %ssa_120 = memref.subview %ssa_118[%ssa_119][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":65:30)
        %ssa_121 = memref.get_global @p : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":65:35)
        %ssa_114 = isq.use @H : !isq.gate<1> loc("tests/test1.isq":65:21) 
        %ssa_114_decorated = isq.decorate(%ssa_114: !isq.gate<1>) {ctrl = [true, true], adjoint = true} : !isq.gate<3> loc("tests/test1.isq":65:21)
        %ssa_114_in_1 = affine.load %ssa_117[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":65:21)
        %ssa_114_in_2 = affine.load %ssa_120[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":65:21)
        %ssa_114_in_3 = affine.load %ssa_121[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":65:21)
        %ssa_114_out_1, %ssa_114_out_2, %ssa_114_out_3 = isq.apply %ssa_114_decorated(%ssa_114_in_1, %ssa_114_in_2, %ssa_114_in_3) : !isq.gate<3> loc("tests/test1.isq":65:21)
        affine.store %ssa_114_out_1, %ssa_117[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":65:21)
        affine.store %ssa_114_out_2, %ssa_120[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":65:21)
        affine.store %ssa_114_out_3, %ssa_121[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":65:21)
        %ssa_123 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":66:19)
        %ssa_124 = arith.constant 1 : index loc("tests/test1.isq":66:21)
        %ssa_125 = memref.subview %ssa_123[%ssa_124][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":66:20)
        %ssa_126 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":66:25)
        %ssa_127 = arith.constant 0 : index loc("tests/test1.isq":66:27)
        %ssa_128 = memref.subview %ssa_126[%ssa_127][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":66:26)
        %ssa_129 = memref.get_global @p : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":66:31)
        %ssa_122 = isq.use @H : !isq.gate<1> loc("tests/test1.isq":66:17) 
        %ssa_122_decorated = isq.decorate(%ssa_122: !isq.gate<1>) {ctrl = [true, true], adjoint = false} : !isq.gate<3> loc("tests/test1.isq":66:17)
        %ssa_122_in_1 = affine.load %ssa_125[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":66:17)
        %ssa_122_in_2 = affine.load %ssa_128[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":66:17)
        %ssa_122_in_3 = affine.load %ssa_129[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":66:17)
        %ssa_122_out_1, %ssa_122_out_2, %ssa_122_out_3 = isq.apply %ssa_122_decorated(%ssa_122_in_1, %ssa_122_in_2, %ssa_122_in_3) : !isq.gate<3> loc("tests/test1.isq":66:17)
        affine.store %ssa_122_out_1, %ssa_125[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":66:17)
        affine.store %ssa_122_out_2, %ssa_128[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":66:17)
        affine.store %ssa_122_out_3, %ssa_129[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":66:17)
        %ssa_131 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":67:21)
        %ssa_132 = arith.constant 0 : index loc("tests/test1.isq":67:23)
        %ssa_133 = memref.subview %ssa_131[%ssa_132][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":67:22)
        %ssa_134 = memref.get_global @p : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":67:27)
        %ssa_130 = isq.use @H : !isq.gate<1> loc("tests/test1.isq":67:19) 
        %ssa_130_decorated = isq.decorate(%ssa_130: !isq.gate<1>) {ctrl = [false], adjoint = true} : !isq.gate<2> loc("tests/test1.isq":67:19)
        %ssa_130_in_1 = affine.load %ssa_133[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":67:19)
        %ssa_130_in_2 = affine.load %ssa_134[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":67:19)
        %ssa_130_out_1, %ssa_130_out_2 = isq.apply %ssa_130_decorated(%ssa_130_in_1, %ssa_130_in_2) : !isq.gate<2> loc("tests/test1.isq":67:19)
        affine.store %ssa_130_out_1, %ssa_133[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":67:19)
        affine.store %ssa_130_out_2, %ssa_134[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":67:19)
        %ssa_136 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":68:29)
        %ssa_137 = arith.constant 0 : index loc("tests/test1.isq":68:31)
        %ssa_138 = memref.subview %ssa_136[%ssa_137][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:30)
        %ssa_139 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":68:35)
        %ssa_140 = arith.constant 1 : index loc("tests/test1.isq":68:37)
        %ssa_141 = memref.subview %ssa_139[%ssa_140][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:36)
        %ssa_142 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":68:41)
        %ssa_143 = arith.constant 2 : index loc("tests/test1.isq":68:43)
        %ssa_144 = memref.subview %ssa_142[%ssa_143][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:42)
        %ssa_145 = memref.get_global @p : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:47)
        %ssa_135 = isq.use @H : !isq.gate<1> loc("tests/test1.isq":68:27) 
        %ssa_135_decorated = isq.decorate(%ssa_135: !isq.gate<1>) {ctrl = [true, false, false], adjoint = true} : !isq.gate<4> loc("tests/test1.isq":68:27)
        %ssa_135_in_1 = affine.load %ssa_138[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:27)
        %ssa_135_in_2 = affine.load %ssa_141[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:27)
        %ssa_135_in_3 = affine.load %ssa_144[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:27)
        %ssa_135_in_4 = affine.load %ssa_145[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:27)
        %ssa_135_out_1, %ssa_135_out_2, %ssa_135_out_3, %ssa_135_out_4 = isq.apply %ssa_135_decorated(%ssa_135_in_1, %ssa_135_in_2, %ssa_135_in_3, %ssa_135_in_4) : !isq.gate<4> loc("tests/test1.isq":68:27)
        affine.store %ssa_135_out_1, %ssa_138[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:27)
        affine.store %ssa_135_out_2, %ssa_141[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:27)
        affine.store %ssa_135_out_3, %ssa_144[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:27)
        affine.store %ssa_135_out_4, %ssa_145[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":68:27)
        %ssa_147 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":69:24)
        %ssa_148 = arith.constant 0 : index loc("tests/test1.isq":69:26)
        %ssa_149 = memref.subview %ssa_147[%ssa_148][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:25)
        %ssa_150 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":69:30)
        %ssa_151 = arith.constant 2 : index loc("tests/test1.isq":69:32)
        %ssa_152 = memref.subview %ssa_150[%ssa_151][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:31)
        %ssa_153 = memref.get_global @p : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:36)
        %ssa_154 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":69:39)
        %ssa_155 = arith.constant 1 : index loc("tests/test1.isq":69:41)
        %ssa_156 = memref.subview %ssa_154[%ssa_155][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:40)
        %ssa_146 = isq.use @Rt2 : !isq.gate<2> loc("tests/test1.isq":69:20) 
        %ssa_146_decorated = isq.decorate(%ssa_146: !isq.gate<2>) {ctrl = [false, true], adjoint = false} : !isq.gate<4> loc("tests/test1.isq":69:20)
        %ssa_146_in_1 = affine.load %ssa_149[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:20)
        %ssa_146_in_2 = affine.load %ssa_152[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:20)
        %ssa_146_in_3 = affine.load %ssa_153[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:20)
        %ssa_146_in_4 = affine.load %ssa_156[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:20)
        %ssa_146_out_1, %ssa_146_out_2, %ssa_146_out_3, %ssa_146_out_4 = isq.apply %ssa_146_decorated(%ssa_146_in_1, %ssa_146_in_2, %ssa_146_in_3, %ssa_146_in_4) : !isq.gate<4> loc("tests/test1.isq":69:20)
        affine.store %ssa_146_out_1, %ssa_149[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:20)
        affine.store %ssa_146_out_2, %ssa_152[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:20)
        affine.store %ssa_146_out_3, %ssa_153[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:20)
        affine.store %ssa_146_out_4, %ssa_156[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":69:20)
        %ssa_157 = arith.constant 0 : i1 loc("tests/test1.isq":71:9)
        %ssa_158_zero = arith.constant 0 : index
        %ssa_158 = memref.alloc()[%ssa_158_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":71:9)
        affine.store %ssa_157, %ssa_158[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":71:9)
        %ssa_159 = arith.constant 0 : i1 loc("tests/test1.isq":71:9)
        %ssa_160_zero = arith.constant 0 : index
        %ssa_160 = memref.alloc()[%ssa_160_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":71:9)
        affine.store %ssa_159, %ssa_160[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":71:9)
        scf.while : ()->() {
            %cond = scf.execute_region->i1 {
                ^break_check:
                    %ssa_166 = affine.load %ssa_158[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":71:9)
                    cond_br %ssa_166, ^break, ^while_cond
                ^while_cond:
                    %ssa_161 = memref.get_global @a : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":71:16)
                    %ssa_163 = affine.load %ssa_161[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":71:16)
                    %ssa_162 = arith.constant 2 : index loc("tests/test1.isq":71:20)
                    %ssa_164 = arith.cmpi "slt", %ssa_163, %ssa_162 : index loc("tests/test1.isq":71:18)
                    scf.yield %ssa_164: i1
                ^break:
                    %zero=arith.constant 0: i1
                    scf.yield %zero: i1
            }
            scf.condition(%cond)
        } do {
            scf.execute_region {
            ^entry:
                %ssa_167 = memref.get_global @a : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":72:17)
                %ssa_168 = memref.get_global @a : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":72:21)
                %ssa_170 = affine.load %ssa_168[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":72:21)
                %ssa_169 = arith.constant 1 : index loc("tests/test1.isq":72:25)
                %ssa_171 = arith.addi %ssa_170, %ssa_169 : index loc("tests/test1.isq":72:23)
                affine.store %ssa_171, %ssa_167[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":72:19)
                br ^exit 
            ^exit:
                scf.yield loc("tests/test1.isq":71:9)
            } loc("tests/test1.isq":71:9)
        scf.yield
        } loc("tests/test1.isq":71:9)
        %ssa_172 = memref.get_global @a : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":75:15)
        %ssa_173 = affine.load %ssa_172[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":75:15)
        isq.call_qop @isq_builtin::@print_int(%ssa_173): [0](index)->() loc("tests/test1.isq":75:9)
        %ssa_174 = memref.get_global @p : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":76:9)
        %ssa_174_in = affine.load %ssa_174[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":76:9)
        %ssa_174_out = isq.call_qop @isq_builtin::@reset(%ssa_174_in): [1]()->() loc("tests/test1.isq":76:9)
        affine.store %ssa_174_out, %ssa_174[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":76:9)
        %ssa_175 = memref.get_global @q : memref<3x!isq.qstate> loc("tests/test1.isq":77:9)
        %ssa_176 = arith.constant 1 : index loc("tests/test1.isq":77:11)
        %ssa_177 = memref.subview %ssa_175[%ssa_176][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":77:10)
        %ssa_177_in = affine.load %ssa_177[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":77:10)
        %ssa_177_out = isq.call_qop @isq_builtin::@reset(%ssa_177_in): [1]()->() loc("tests/test1.isq":77:10)
        affine.store %ssa_177_out, %ssa_177[0] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":77:10)
        br ^exit_8 
    ^exit_8:
        memref.dealloc %ssa_160 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":71:9)
        br ^exit_7 
    ^exit_7:
        memref.dealloc %ssa_158 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":71:9)
        br ^exit_6 
    ^exit_6:
        memref.dealloc %ssa_87 : memref<5xindex> loc("tests/test1.isq":59:9)
        br ^exit_5 
    ^exit_5:
        memref.dealloc %ssa_78 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":56:9)
        br ^exit_4 
    ^exit_4:
        memref.dealloc %ssa_52 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":50:9)
        br ^exit_3 
    ^exit_3:
        memref.dealloc %ssa_50 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":49:9)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_49 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":48:9)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_48 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/test1.isq":46:1)
        br ^exit 
    ^exit:
        return loc("tests/test1.isq":46:1)
    } loc("tests/test1.isq":46:1)
    func @__isq__global_initialize() 
    {
    ^block1:
        return 
    } 
}