module{
isq.declare_qop @__isq__builtin__measure : [1]()->i1
isq.declare_qop @__isq__builtin__reset : [1]()->()
isq.declare_qop @__isq__builtin__print_int : [0](index)->()
isq.declare_qop @__isq__builtin__print_double : [0](f64)->()
isq.defgate @__isq__builtin__u3(f64, f64, f64) {definition = [{type = "qir", value = "__quantum__qis__u3"}]} : !isq.gate<1>
isq.defgate @__isq__builtin__cnot {definition = [{type = "qir", value = "__quantum__qis__cnot"}]} : !isq.gate<2>
func private @__quantum__qis__u3(f64, f64, f64, !isq.qir.qubit)
func private @__quantum__qis__cnot(!isq.qir.qubit, !isq.qir.qubit)
    isq.defgate @Rs {definition = [{type="unitary", value = [[#isq.complex<0.5, 0.8660254>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>]]}]}: !isq.gate<2> loc("<stdin>":1:1)
    isq.defgate @Rs2 {definition = [{type="unitary", value = [[#isq.complex<0.5, -0.8660254>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>]]}]}: !isq.gate<2> loc("<stdin>":5:1)
    isq.defgate @Rt {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.5, 0.8660254>]]}]}: !isq.gate<2> loc("<stdin>":10:1)
    isq.defgate @Rt2 {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.5, -0.8660254>]]}]}: !isq.gate<2> loc("<stdin>":14:1)
    isq.defgate @CNOT {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>]]}]}: !isq.gate<2> loc("<stdin>":18:1)
    isq.defgate @H {definition = [{type="unitary", value = [[#isq.complex<0.7071067811865476, 0.0>,#isq.complex<0.7071067811865476, 0.0>],[#isq.complex<0.7071067811865476, 0.0>,#isq.complex<-0.7071067811865476, -0.0>]]}]}: !isq.gate<1> loc("<stdin>":22:1)
    memref.global @a : memref<1xindex> = uninitialized loc("<stdin>":31:1)
    memref.global @b : memref<1xindex> = uninitialized loc("<stdin>":31:1)
    memref.global @c : memref<1xindex> = uninitialized loc("<stdin>":31:1)
    memref.global @q : memref<3x!isq.qstate> = uninitialized loc("<stdin>":32:1)
    memref.global @p : memref<1x!isq.qstate> = uninitialized loc("<stdin>":32:1)
    func @test(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index)->index 
    {
    ^entry(%ssa_15: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_16: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_18: index):
        %ssa_14_real = memref.alloc() : memref<1xindex> loc("<stdin>":34:1)
        %ssa_14_zero = arith.constant 0 : index
        %ssa_14 = memref.subview %ssa_14_real[%ssa_14_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":34:1)
        %ssa_20_real = memref.alloc() : memref<1xindex> loc("<stdin>":34:1)
        %ssa_20_zero = arith.constant 0 : index
        %ssa_20 = memref.subview %ssa_20_real[%ssa_20_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":34:1)
        %ssa_18_zero = arith.constant 0: index loc("<stdin>":34:1)
        memref.store %ssa_18, %ssa_20[%ssa_18_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":34:1)
        %ssa_21 = arith.constant 0 : i1 loc("<stdin>":34:1)
        %ssa_22_real = memref.alloc() : memref<1xi1> loc("<stdin>":34:1)
        %ssa_22_zero = arith.constant 0 : index
        %ssa_22 = memref.subview %ssa_22_real[%ssa_22_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":34:1)
        %ssa_21_zero = arith.constant 0: index loc("<stdin>":34:1)
        memref.store %ssa_21, %ssa_22[%ssa_21_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":34:1)
        %ssa_23 = isq.use @H : !isq.gate<1> loc("<stdin>":35:9) 
        %ssa_23_in_1_zero = arith.constant 0: index loc("<stdin>":35:9)
        %ssa_23_in_1 = memref.load %ssa_15[%ssa_23_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":35:9)
        %ssa_23_out_1 = isq.apply %ssa_23(%ssa_23_in_1) : !isq.gate<1> loc("<stdin>":35:9)
        %ssa_23_out_1_zero = arith.constant 0: index loc("<stdin>":35:9)
        memref.store %ssa_23_out_1, %ssa_15[%ssa_23_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":35:9)
        %ssa_25_real = memref.alloc() : memref<1x!isq.qstate> loc("<stdin>":36:9)
        %ssa_25_zero = arith.constant 0 : index
        %ssa_25 = memref.subview %ssa_25_real[%ssa_25_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":36:9)
        %ssa_26 = isq.use @CNOT : !isq.gate<2> loc("<stdin>":37:9) 
        %ssa_26_in_1_zero = arith.constant 0: index loc("<stdin>":37:9)
        %ssa_26_in_1 = memref.load %ssa_25[%ssa_26_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":37:9)
        %ssa_26_in_2_zero = arith.constant 0: index loc("<stdin>":37:9)
        %ssa_26_in_2 = memref.load %ssa_15[%ssa_26_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":37:9)
        %ssa_26_out_1, %ssa_26_out_2 = isq.apply %ssa_26(%ssa_26_in_1, %ssa_26_in_2) : !isq.gate<2> loc("<stdin>":37:9)
        %ssa_26_out_1_zero = arith.constant 0: index loc("<stdin>":37:9)
        memref.store %ssa_26_out_1, %ssa_25[%ssa_26_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":37:9)
        %ssa_26_out_2_zero = arith.constant 0: index loc("<stdin>":37:9)
        memref.store %ssa_26_out_2, %ssa_15[%ssa_26_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":37:9)
        %ssa_29 = isq.use @H : !isq.gate<1> loc("<stdin>":38:9) 
        %ssa_29_in_1_zero = arith.constant 0: index loc("<stdin>":38:9)
        %ssa_29_in_1 = memref.load %ssa_15[%ssa_29_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":38:9)
        %ssa_29_out_1 = isq.apply %ssa_29(%ssa_29_in_1) : !isq.gate<1> loc("<stdin>":38:9)
        %ssa_29_out_1_zero = arith.constant 0: index loc("<stdin>":38:9)
        memref.store %ssa_29_out_1, %ssa_15[%ssa_29_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":38:9)
        %ssa_32 = arith.constant 2 : index loc("<stdin>":39:16)
        %ssa_32_zero = arith.constant 0: index loc("<stdin>":39:9)
        memref.store %ssa_32, %ssa_14[%ssa_32_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":39:9)
        br ^exit_3 loc("<stdin>":39:9)
    ^block1:
        br ^exit_3 
    ^exit_3:
        memref.dealloc %ssa_25_real : memref<1x!isq.qstate> loc("<stdin>":36:9)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_22_real : memref<1xi1> loc("<stdin>":34:1)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_20_real : memref<1xindex> loc("<stdin>":34:1)
        br ^exit 
    ^exit:
        %ssa_34_zero = arith.constant 0: index loc("<stdin>":34:1)
        %ssa_34 = memref.load %ssa_14[%ssa_34_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":34:1)
        memref.dealloc %ssa_14_real : memref<1xindex> loc("<stdin>":34:1)
        return %ssa_34 : index loc("<stdin>":34:1)
    } loc("<stdin>":34:1)
    func @test2(memref<?x!isq.qstate>, index) 
    {
    ^entry(%ssa_35: memref<?x!isq.qstate>, %ssa_37: index):
        %ssa_39_real = memref.alloc() : memref<1xindex> loc("<stdin>":42:1)
        %ssa_39_zero = arith.constant 0 : index
        %ssa_39 = memref.subview %ssa_39_real[%ssa_39_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":42:1)
        %ssa_37_zero = arith.constant 0: index loc("<stdin>":42:1)
        memref.store %ssa_37, %ssa_39[%ssa_37_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":42:1)
        %ssa_40 = arith.constant 0 : i1 loc("<stdin>":42:1)
        %ssa_41_real = memref.alloc() : memref<1xi1> loc("<stdin>":42:1)
        %ssa_41_zero = arith.constant 0 : index
        %ssa_41 = memref.subview %ssa_41_real[%ssa_41_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":42:1)
        %ssa_40_zero = arith.constant 0: index loc("<stdin>":42:1)
        memref.store %ssa_40, %ssa_41[%ssa_40_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":42:1)
        %ssa_45_zero = arith.constant 0: index loc("<stdin>":43:13)
        %ssa_45 = memref.load %ssa_39[%ssa_45_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":43:13)
        %ssa_46 = memref.subview %ssa_35[%ssa_45][1][1] : memref<?x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":43:12)
        %ssa_42 = isq.use @H : !isq.gate<1> loc("<stdin>":43:9) 
        %ssa_42_in_1_zero = arith.constant 0: index loc("<stdin>":43:9)
        %ssa_42_in_1 = memref.load %ssa_46[%ssa_42_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":43:9)
        %ssa_42_out_1 = isq.apply %ssa_42(%ssa_42_in_1) : !isq.gate<1> loc("<stdin>":43:9)
        %ssa_42_out_1_zero = arith.constant 0: index loc("<stdin>":43:9)
        memref.store %ssa_42_out_1, %ssa_46[%ssa_42_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":43:9)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_41_real : memref<1xi1> loc("<stdin>":42:1)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_39_real : memref<1xindex> loc("<stdin>":42:1)
        br ^exit 
    ^exit:
        return loc("<stdin>":42:1)
    } loc("<stdin>":42:1)
    func @__isq__main() 
    {
    ^entry:
        %ssa_47 = arith.constant 0 : i1 loc("<stdin>":46:1)
        %ssa_48_real = memref.alloc() : memref<1xi1> loc("<stdin>":46:1)
        %ssa_48_zero = arith.constant 0 : index
        %ssa_48 = memref.subview %ssa_48_real[%ssa_48_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":46:1)
        %ssa_47_zero = arith.constant 0: index loc("<stdin>":46:1)
        memref.store %ssa_47, %ssa_48[%ssa_47_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":46:1)
        %ssa_49_real = memref.alloc() : memref<1xindex> loc("<stdin>":48:9)
        %ssa_49_zero = arith.constant 0 : index
        %ssa_49 = memref.subview %ssa_49_real[%ssa_49_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":48:9)
        %ssa_50_real = memref.alloc() : memref<1xindex> loc("<stdin>":49:9)
        %ssa_50_zero = arith.constant 0 : index
        %ssa_50 = memref.subview %ssa_50_real[%ssa_50_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":49:9)
        %ssa_51 = arith.constant 0 : i1 loc("<stdin>":50:9)
        %ssa_52_real = memref.alloc() : memref<1xi1> loc("<stdin>":50:9)
        %ssa_52_zero = arith.constant 0 : index
        %ssa_52 = memref.subview %ssa_52_real[%ssa_52_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":50:9)
        %ssa_51_zero = arith.constant 0: index loc("<stdin>":50:9)
        memref.store %ssa_51, %ssa_52[%ssa_51_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":50:9)
        %ssa_53 = arith.constant 1 : index loc("<stdin>":50:13)
        %ssa_54 = arith.constant 2 : index loc("<stdin>":50:17)
        %ssa_55 = arith.cmpi "slt", %ssa_53, %ssa_54 : index loc("<stdin>":50:15)
        scf.if %ssa_55 {
            scf.execute_region {
            ^entry:
                %ssa_57_uncast = memref.get_global @a : memref<1xindex> loc("<stdin>":51:22)
                %ssa_57_zero = arith.constant 0 : index
                %ssa_57 = memref.subview %ssa_57_uncast[%ssa_57_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":51:22)
                %ssa_61_zero = arith.constant 0: index loc("<stdin>":51:22)
                %ssa_61 = memref.load %ssa_57[%ssa_61_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":51:22)
                %ssa_58 = arith.constant 3 : index loc("<stdin>":51:24)
                %ssa_59 = arith.constant 2 : index loc("<stdin>":51:26)
                %ssa_60 = arith.muli %ssa_58, %ssa_59 : index loc("<stdin>":51:25)
                %ssa_62 = arith.addi %ssa_61, %ssa_60 : index loc("<stdin>":51:23)
                %ssa_63_uncast = memref.get_global @b : memref<1xindex> loc("<stdin>":51:30)
                %ssa_63_zero = arith.constant 0 : index
                %ssa_63 = memref.subview %ssa_63_uncast[%ssa_63_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":51:30)
                %ssa_65_zero = arith.constant 0: index loc("<stdin>":51:30)
                %ssa_65 = memref.load %ssa_63[%ssa_65_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":51:30)
                %ssa_64_uncast = memref.get_global @c : memref<1xindex> loc("<stdin>":51:32)
                %ssa_64_zero = arith.constant 0 : index
                %ssa_64 = memref.subview %ssa_64_uncast[%ssa_64_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":51:32)
                %ssa_66_zero = arith.constant 0: index loc("<stdin>":51:32)
                %ssa_66 = memref.load %ssa_64[%ssa_66_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":51:32)
                %ssa_67 = arith.addi %ssa_65, %ssa_66 : index loc("<stdin>":51:31)
                %ssa_68 = arith.muli %ssa_62, %ssa_67 : index loc("<stdin>":51:28)
                %ssa_68_zero = arith.constant 0: index loc("<stdin>":51:19)
                memref.store %ssa_68, %ssa_49[%ssa_68_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":51:19)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":50:9)
            } loc("<stdin>":50:9)
        } else {
            scf.execute_region {
            ^entry:
                %ssa_69_real = memref.alloc() : memref<1xindex> loc("<stdin>":53:17)
                %ssa_69_zero = arith.constant 0 : index
                %ssa_69 = memref.subview %ssa_69_real[%ssa_69_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":53:17)
                %ssa_71_uncast = memref.get_global @c : memref<1xindex> loc("<stdin>":54:21)
                %ssa_71_zero = arith.constant 0 : index
                %ssa_71 = memref.subview %ssa_71_uncast[%ssa_71_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":54:21)
                %ssa_73_zero = arith.constant 0: index loc("<stdin>":54:21)
                %ssa_73 = memref.load %ssa_71[%ssa_73_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":54:21)
                %ssa_72 = arith.constant 1 : index loc("<stdin>":54:23)
                %ssa_74 = arith.addi %ssa_73, %ssa_72 : index loc("<stdin>":54:22)
                %ssa_74_zero = arith.constant 0: index loc("<stdin>":54:19)
                memref.store %ssa_74, %ssa_69[%ssa_74_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":54:19)
                br ^exit_1 
            ^exit_1:
                memref.dealloc %ssa_69_real : memref<1xindex> loc("<stdin>":53:17)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":50:9)
            } loc("<stdin>":50:9)
        }
        %ssa_76_zero = arith.constant 0: index loc("<stdin>":50:9)
        %ssa_76 = memref.load %ssa_48[%ssa_76_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":50:9)
        cond_br %ssa_76, ^exit_4, ^block1 loc("<stdin>":50:9)
    ^block1:
        %ssa_77 = arith.constant 0 : i1 loc("<stdin>":56:9)
        %ssa_78_real = memref.alloc() : memref<1xi1> loc("<stdin>":56:9)
        %ssa_78_zero = arith.constant 0 : index
        %ssa_78 = memref.subview %ssa_78_real[%ssa_78_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":56:9)
        %ssa_77_zero = arith.constant 0: index loc("<stdin>":56:9)
        memref.store %ssa_77, %ssa_78[%ssa_77_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":56:9)
        %ssa_79 = arith.constant 1 : index loc("<stdin>":56:18)
        %ssa_80_uncast = memref.get_global @b : memref<1xindex> loc("<stdin>":56:20)
        %ssa_80_zero = arith.constant 0 : index
        %ssa_80 = memref.subview %ssa_80_uncast[%ssa_80_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":56:20)
        %ssa_81_zero = arith.constant 0: index loc("<stdin>":56:20)
        %ssa_81 = memref.load %ssa_80[%ssa_81_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":56:20)
        affine.for %ssa_84 = %ssa_79 to %ssa_81 step 1 {
            scf.execute_region {
            ^entry:
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":56:9)
            } loc("<stdin>":56:9)
        } loc("<stdin>":56:9)
        %ssa_86_zero = arith.constant 0: index loc("<stdin>":56:9)
        %ssa_86 = memref.load %ssa_48[%ssa_86_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":56:9)
        cond_br %ssa_86, ^exit_5, ^block2 loc("<stdin>":56:9)
    ^block2:
        %ssa_87 = memref.alloc() : memref<5xindex> loc("<stdin>":59:9)
        %ssa_88_uncast = memref.get_global @a : memref<1xindex> loc("<stdin>":60:9)
        %ssa_88_zero = arith.constant 0 : index
        %ssa_88 = memref.subview %ssa_88_uncast[%ssa_88_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":60:9)
        %ssa_90_uncast = memref.get_global @c : memref<1xindex> loc("<stdin>":60:15)
        %ssa_90_zero = arith.constant 0 : index
        %ssa_90 = memref.subview %ssa_90_uncast[%ssa_90_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":60:15)
        %ssa_91_zero = arith.constant 0: index loc("<stdin>":60:15)
        %ssa_91 = memref.load %ssa_90[%ssa_91_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":60:15)
        %ssa_92 = memref.subview %ssa_87[%ssa_91][1][1] : memref<5xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":60:14)
        %ssa_94_zero = arith.constant 0: index loc("<stdin>":60:14)
        %ssa_94 = memref.load %ssa_92[%ssa_94_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":60:14)
        %ssa_93 = arith.constant 2 : index loc("<stdin>":60:18)
        %ssa_95 = arith.addi %ssa_94, %ssa_93 : index loc("<stdin>":60:17)
        %ssa_95_zero = arith.constant 0: index loc("<stdin>":60:11)
        memref.store %ssa_95, %ssa_88[%ssa_95_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":60:11)
        %ssa_96_uncast = memref.get_global @b : memref<1xindex> loc("<stdin>":61:5)
        %ssa_96_zero = arith.constant 0 : index
        %ssa_96 = memref.subview %ssa_96_uncast[%ssa_96_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":61:5)
        %ssa_98_uncast = memref.get_global @p : memref<1x!isq.qstate> loc("<stdin>":61:14)
        %ssa_98_zero = arith.constant 0 : index
        %ssa_98 = memref.subview %ssa_98_uncast[%ssa_98_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":61:14)
        %ssa_99_uncast = memref.get_global @p : memref<1x!isq.qstate> loc("<stdin>":61:17)
        %ssa_99_zero = arith.constant 0 : index
        %ssa_99 = memref.subview %ssa_99_uncast[%ssa_99_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":61:17)
        %ssa_101_zero = arith.constant 0: index loc("<stdin>":61:20)
        %ssa_101 = memref.load %ssa_49[%ssa_101_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":61:20)
        %ssa_102 = call @test(%ssa_98, %ssa_99, %ssa_101) : (memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index)->index loc("<stdin>":61:9)
        %ssa_102_zero = arith.constant 0: index loc("<stdin>":61:7)
        memref.store %ssa_102, %ssa_96[%ssa_102_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":61:7)
        %ssa_103_uncast = memref.get_global @a : memref<1xindex> loc("<stdin>":62:5)
        %ssa_103_zero = arith.constant 0 : index
        %ssa_103 = memref.subview %ssa_103_uncast[%ssa_103_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":62:5)
        %ssa_104 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":62:11)
        %ssa_105 = arith.constant 0 : index loc("<stdin>":62:13)
        %ssa_106 = memref.subview %ssa_104[%ssa_105][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":62:12)
        %ssa_107_in_zero = arith.constant 0: index loc("<stdin>":62:9)
        %ssa_107_in = memref.load %ssa_106[%ssa_107_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":62:9)
        %ssa_107_out, %ssa_107 = isq.call_qop @__isq__builtin__measure(%ssa_107_in): [1]()->i1 loc("<stdin>":62:9)
        %ssa_107_out_zero = arith.constant 0: index loc("<stdin>":62:9)
        memref.store %ssa_107_out, %ssa_106[%ssa_107_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":62:9)
        %ssa_108_i2 = arith.extui %ssa_107 : i1 to i2 loc("<stdin>":62:9)
        %ssa_108 = arith.index_cast %ssa_108_i2 : i2 to index loc("<stdin>":62:9)
        %ssa_108_zero = arith.constant 0: index loc("<stdin>":62:7)
        memref.store %ssa_108, %ssa_103[%ssa_108_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":62:7)
        %ssa_110_uncast = memref.get_global @p : memref<1x!isq.qstate> loc("<stdin>":63:14)
        %ssa_110_zero = arith.constant 0 : index
        %ssa_110 = memref.subview %ssa_110_uncast[%ssa_110_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":63:14)
        %ssa_111 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":63:17)
        %ssa_112 = arith.constant 0 : index loc("<stdin>":63:19)
        %ssa_113 = memref.subview %ssa_111[%ssa_112][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":63:18)
        %ssa_109 = isq.use @CNOT : !isq.gate<2> loc("<stdin>":63:9) 
        %ssa_109_in_1_zero = arith.constant 0: index loc("<stdin>":63:9)
        %ssa_109_in_1 = memref.load %ssa_110[%ssa_109_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":63:9)
        %ssa_109_in_2_zero = arith.constant 0: index loc("<stdin>":63:9)
        %ssa_109_in_2 = memref.load %ssa_113[%ssa_109_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":63:9)
        %ssa_109_out_1, %ssa_109_out_2 = isq.apply %ssa_109(%ssa_109_in_1, %ssa_109_in_2) : !isq.gate<2> loc("<stdin>":63:9)
        %ssa_109_out_1_zero = arith.constant 0: index loc("<stdin>":63:9)
        memref.store %ssa_109_out_1, %ssa_110[%ssa_109_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":63:9)
        %ssa_109_out_2_zero = arith.constant 0: index loc("<stdin>":63:9)
        memref.store %ssa_109_out_2, %ssa_113[%ssa_109_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":63:9)
        %ssa_115 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":65:23)
        %ssa_116 = arith.constant 0 : index loc("<stdin>":65:25)
        %ssa_117 = memref.subview %ssa_115[%ssa_116][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":65:24)
        %ssa_118 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":65:29)
        %ssa_119 = arith.constant 1 : index loc("<stdin>":65:31)
        %ssa_120 = memref.subview %ssa_118[%ssa_119][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":65:30)
        %ssa_121_uncast = memref.get_global @p : memref<1x!isq.qstate> loc("<stdin>":65:35)
        %ssa_121_zero = arith.constant 0 : index
        %ssa_121 = memref.subview %ssa_121_uncast[%ssa_121_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":65:35)
        %ssa_114 = isq.use @H : !isq.gate<1> loc("<stdin>":65:21) 
        %ssa_114_decorated = isq.decorate(%ssa_114: !isq.gate<1>) {ctrl = [true, true], adjoint = true} : !isq.gate<3> loc("<stdin>":65:21)
        %ssa_114_in_1_zero = arith.constant 0: index loc("<stdin>":65:21)
        %ssa_114_in_1 = memref.load %ssa_117[%ssa_114_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":65:21)
        %ssa_114_in_2_zero = arith.constant 0: index loc("<stdin>":65:21)
        %ssa_114_in_2 = memref.load %ssa_120[%ssa_114_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":65:21)
        %ssa_114_in_3_zero = arith.constant 0: index loc("<stdin>":65:21)
        %ssa_114_in_3 = memref.load %ssa_121[%ssa_114_in_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":65:21)
        %ssa_114_out_1, %ssa_114_out_2, %ssa_114_out_3 = isq.apply %ssa_114_decorated(%ssa_114_in_1, %ssa_114_in_2, %ssa_114_in_3) : !isq.gate<3> loc("<stdin>":65:21)
        %ssa_114_out_1_zero = arith.constant 0: index loc("<stdin>":65:21)
        memref.store %ssa_114_out_1, %ssa_117[%ssa_114_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":65:21)
        %ssa_114_out_2_zero = arith.constant 0: index loc("<stdin>":65:21)
        memref.store %ssa_114_out_2, %ssa_120[%ssa_114_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":65:21)
        %ssa_114_out_3_zero = arith.constant 0: index loc("<stdin>":65:21)
        memref.store %ssa_114_out_3, %ssa_121[%ssa_114_out_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":65:21)
        %ssa_123 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":66:19)
        %ssa_124 = arith.constant 1 : index loc("<stdin>":66:21)
        %ssa_125 = memref.subview %ssa_123[%ssa_124][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:20)
        %ssa_126 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":66:25)
        %ssa_127 = arith.constant 0 : index loc("<stdin>":66:27)
        %ssa_128 = memref.subview %ssa_126[%ssa_127][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:26)
        %ssa_129_uncast = memref.get_global @p : memref<1x!isq.qstate> loc("<stdin>":66:31)
        %ssa_129_zero = arith.constant 0 : index
        %ssa_129 = memref.subview %ssa_129_uncast[%ssa_129_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:31)
        %ssa_122 = isq.use @H : !isq.gate<1> loc("<stdin>":66:17) 
        %ssa_122_decorated = isq.decorate(%ssa_122: !isq.gate<1>) {ctrl = [true, true], adjoint = false} : !isq.gate<3> loc("<stdin>":66:17)
        %ssa_122_in_1_zero = arith.constant 0: index loc("<stdin>":66:17)
        %ssa_122_in_1 = memref.load %ssa_125[%ssa_122_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:17)
        %ssa_122_in_2_zero = arith.constant 0: index loc("<stdin>":66:17)
        %ssa_122_in_2 = memref.load %ssa_128[%ssa_122_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:17)
        %ssa_122_in_3_zero = arith.constant 0: index loc("<stdin>":66:17)
        %ssa_122_in_3 = memref.load %ssa_129[%ssa_122_in_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:17)
        %ssa_122_out_1, %ssa_122_out_2, %ssa_122_out_3 = isq.apply %ssa_122_decorated(%ssa_122_in_1, %ssa_122_in_2, %ssa_122_in_3) : !isq.gate<3> loc("<stdin>":66:17)
        %ssa_122_out_1_zero = arith.constant 0: index loc("<stdin>":66:17)
        memref.store %ssa_122_out_1, %ssa_125[%ssa_122_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:17)
        %ssa_122_out_2_zero = arith.constant 0: index loc("<stdin>":66:17)
        memref.store %ssa_122_out_2, %ssa_128[%ssa_122_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:17)
        %ssa_122_out_3_zero = arith.constant 0: index loc("<stdin>":66:17)
        memref.store %ssa_122_out_3, %ssa_129[%ssa_122_out_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:17)
        %ssa_131 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":67:21)
        %ssa_132 = arith.constant 0 : index loc("<stdin>":67:23)
        %ssa_133 = memref.subview %ssa_131[%ssa_132][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:22)
        %ssa_134_uncast = memref.get_global @p : memref<1x!isq.qstate> loc("<stdin>":67:27)
        %ssa_134_zero = arith.constant 0 : index
        %ssa_134 = memref.subview %ssa_134_uncast[%ssa_134_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:27)
        %ssa_130 = isq.use @H : !isq.gate<1> loc("<stdin>":67:19) 
        %ssa_130_decorated = isq.decorate(%ssa_130: !isq.gate<1>) {ctrl = [false], adjoint = true} : !isq.gate<2> loc("<stdin>":67:19)
        %ssa_130_in_1_zero = arith.constant 0: index loc("<stdin>":67:19)
        %ssa_130_in_1 = memref.load %ssa_133[%ssa_130_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:19)
        %ssa_130_in_2_zero = arith.constant 0: index loc("<stdin>":67:19)
        %ssa_130_in_2 = memref.load %ssa_134[%ssa_130_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:19)
        %ssa_130_out_1, %ssa_130_out_2 = isq.apply %ssa_130_decorated(%ssa_130_in_1, %ssa_130_in_2) : !isq.gate<2> loc("<stdin>":67:19)
        %ssa_130_out_1_zero = arith.constant 0: index loc("<stdin>":67:19)
        memref.store %ssa_130_out_1, %ssa_133[%ssa_130_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:19)
        %ssa_130_out_2_zero = arith.constant 0: index loc("<stdin>":67:19)
        memref.store %ssa_130_out_2, %ssa_134[%ssa_130_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:19)
        %ssa_136 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":68:29)
        %ssa_137 = arith.constant 0 : index loc("<stdin>":68:31)
        %ssa_138 = memref.subview %ssa_136[%ssa_137][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:30)
        %ssa_139 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":68:35)
        %ssa_140 = arith.constant 1 : index loc("<stdin>":68:37)
        %ssa_141 = memref.subview %ssa_139[%ssa_140][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:36)
        %ssa_142 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":68:41)
        %ssa_143 = arith.constant 2 : index loc("<stdin>":68:43)
        %ssa_144 = memref.subview %ssa_142[%ssa_143][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:42)
        %ssa_145_uncast = memref.get_global @p : memref<1x!isq.qstate> loc("<stdin>":68:47)
        %ssa_145_zero = arith.constant 0 : index
        %ssa_145 = memref.subview %ssa_145_uncast[%ssa_145_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:47)
        %ssa_135 = isq.use @H : !isq.gate<1> loc("<stdin>":68:27) 
        %ssa_135_decorated = isq.decorate(%ssa_135: !isq.gate<1>) {ctrl = [true, false, false], adjoint = true} : !isq.gate<4> loc("<stdin>":68:27)
        %ssa_135_in_1_zero = arith.constant 0: index loc("<stdin>":68:27)
        %ssa_135_in_1 = memref.load %ssa_138[%ssa_135_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:27)
        %ssa_135_in_2_zero = arith.constant 0: index loc("<stdin>":68:27)
        %ssa_135_in_2 = memref.load %ssa_141[%ssa_135_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:27)
        %ssa_135_in_3_zero = arith.constant 0: index loc("<stdin>":68:27)
        %ssa_135_in_3 = memref.load %ssa_144[%ssa_135_in_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:27)
        %ssa_135_in_4_zero = arith.constant 0: index loc("<stdin>":68:27)
        %ssa_135_in_4 = memref.load %ssa_145[%ssa_135_in_4_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:27)
        %ssa_135_out_1, %ssa_135_out_2, %ssa_135_out_3, %ssa_135_out_4 = isq.apply %ssa_135_decorated(%ssa_135_in_1, %ssa_135_in_2, %ssa_135_in_3, %ssa_135_in_4) : !isq.gate<4> loc("<stdin>":68:27)
        %ssa_135_out_1_zero = arith.constant 0: index loc("<stdin>":68:27)
        memref.store %ssa_135_out_1, %ssa_138[%ssa_135_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:27)
        %ssa_135_out_2_zero = arith.constant 0: index loc("<stdin>":68:27)
        memref.store %ssa_135_out_2, %ssa_141[%ssa_135_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:27)
        %ssa_135_out_3_zero = arith.constant 0: index loc("<stdin>":68:27)
        memref.store %ssa_135_out_3, %ssa_144[%ssa_135_out_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:27)
        %ssa_135_out_4_zero = arith.constant 0: index loc("<stdin>":68:27)
        memref.store %ssa_135_out_4, %ssa_145[%ssa_135_out_4_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:27)
        %ssa_147 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":69:24)
        %ssa_148 = arith.constant 0 : index loc("<stdin>":69:26)
        %ssa_149 = memref.subview %ssa_147[%ssa_148][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:25)
        %ssa_150 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":69:30)
        %ssa_151 = arith.constant 2 : index loc("<stdin>":69:32)
        %ssa_152 = memref.subview %ssa_150[%ssa_151][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:31)
        %ssa_153_uncast = memref.get_global @p : memref<1x!isq.qstate> loc("<stdin>":69:36)
        %ssa_153_zero = arith.constant 0 : index
        %ssa_153 = memref.subview %ssa_153_uncast[%ssa_153_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:36)
        %ssa_154 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":69:39)
        %ssa_155 = arith.constant 1 : index loc("<stdin>":69:41)
        %ssa_156 = memref.subview %ssa_154[%ssa_155][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:40)
        %ssa_146 = isq.use @Rt2 : !isq.gate<2> loc("<stdin>":69:20) 
        %ssa_146_decorated = isq.decorate(%ssa_146: !isq.gate<2>) {ctrl = [false, true], adjoint = false} : !isq.gate<4> loc("<stdin>":69:20)
        %ssa_146_in_1_zero = arith.constant 0: index loc("<stdin>":69:20)
        %ssa_146_in_1 = memref.load %ssa_149[%ssa_146_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:20)
        %ssa_146_in_2_zero = arith.constant 0: index loc("<stdin>":69:20)
        %ssa_146_in_2 = memref.load %ssa_152[%ssa_146_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:20)
        %ssa_146_in_3_zero = arith.constant 0: index loc("<stdin>":69:20)
        %ssa_146_in_3 = memref.load %ssa_153[%ssa_146_in_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:20)
        %ssa_146_in_4_zero = arith.constant 0: index loc("<stdin>":69:20)
        %ssa_146_in_4 = memref.load %ssa_156[%ssa_146_in_4_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:20)
        %ssa_146_out_1, %ssa_146_out_2, %ssa_146_out_3, %ssa_146_out_4 = isq.apply %ssa_146_decorated(%ssa_146_in_1, %ssa_146_in_2, %ssa_146_in_3, %ssa_146_in_4) : !isq.gate<4> loc("<stdin>":69:20)
        %ssa_146_out_1_zero = arith.constant 0: index loc("<stdin>":69:20)
        memref.store %ssa_146_out_1, %ssa_149[%ssa_146_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:20)
        %ssa_146_out_2_zero = arith.constant 0: index loc("<stdin>":69:20)
        memref.store %ssa_146_out_2, %ssa_152[%ssa_146_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:20)
        %ssa_146_out_3_zero = arith.constant 0: index loc("<stdin>":69:20)
        memref.store %ssa_146_out_3, %ssa_153[%ssa_146_out_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:20)
        %ssa_146_out_4_zero = arith.constant 0: index loc("<stdin>":69:20)
        memref.store %ssa_146_out_4, %ssa_156[%ssa_146_out_4_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:20)
        %ssa_157 = arith.constant 0 : i1 loc("<stdin>":71:9)
        %ssa_158_real = memref.alloc() : memref<1xi1> loc("<stdin>":71:9)
        %ssa_158_zero = arith.constant 0 : index
        %ssa_158 = memref.subview %ssa_158_real[%ssa_158_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":71:9)
        %ssa_157_zero = arith.constant 0: index loc("<stdin>":71:9)
        memref.store %ssa_157, %ssa_158[%ssa_157_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":71:9)
        %ssa_159 = arith.constant 0 : i1 loc("<stdin>":71:9)
        %ssa_160_real = memref.alloc() : memref<1xi1> loc("<stdin>":71:9)
        %ssa_160_zero = arith.constant 0 : index
        %ssa_160 = memref.subview %ssa_160_real[%ssa_160_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":71:9)
        %ssa_159_zero = arith.constant 0: index loc("<stdin>":71:9)
        memref.store %ssa_159, %ssa_160[%ssa_159_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":71:9)
        scf.while : ()->() {
            %cond = scf.execute_region->i1 {
                ^break_check:
                    %ssa_166_zero = arith.constant 0: index loc("<stdin>":71:9)
                    %ssa_166 = memref.load %ssa_158[%ssa_166_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":71:9)
                    cond_br %ssa_166, ^break, ^while_cond
                ^while_cond:
                    %ssa_161_uncast = memref.get_global @a : memref<1xindex> loc("<stdin>":71:16)
                    %ssa_161_zero = arith.constant 0 : index
                    %ssa_161 = memref.subview %ssa_161_uncast[%ssa_161_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":71:16)
                    %ssa_163_zero = arith.constant 0: index loc("<stdin>":71:16)
                    %ssa_163 = memref.load %ssa_161[%ssa_163_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":71:16)
                    %ssa_162 = arith.constant 2 : index loc("<stdin>":71:20)
                    %ssa_164 = arith.cmpi "slt", %ssa_163, %ssa_162 : index loc("<stdin>":71:18)
                    scf.yield %ssa_164: i1
                ^break:
                    %zero=arith.constant 0: i1
                    scf.yield %zero: i1
            }
            scf.condition(%cond)
        } do {
            scf.execute_region {
            ^entry:
                %ssa_167_uncast = memref.get_global @a : memref<1xindex> loc("<stdin>":72:17)
                %ssa_167_zero = arith.constant 0 : index
                %ssa_167 = memref.subview %ssa_167_uncast[%ssa_167_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":72:17)
                %ssa_168_uncast = memref.get_global @a : memref<1xindex> loc("<stdin>":72:21)
                %ssa_168_zero = arith.constant 0 : index
                %ssa_168 = memref.subview %ssa_168_uncast[%ssa_168_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":72:21)
                %ssa_170_zero = arith.constant 0: index loc("<stdin>":72:21)
                %ssa_170 = memref.load %ssa_168[%ssa_170_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":72:21)
                %ssa_169 = arith.constant 1 : index loc("<stdin>":72:25)
                %ssa_171 = arith.addi %ssa_170, %ssa_169 : index loc("<stdin>":72:23)
                %ssa_171_zero = arith.constant 0: index loc("<stdin>":72:19)
                memref.store %ssa_171, %ssa_167[%ssa_171_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":72:19)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":71:9)
            } loc("<stdin>":71:9)
        scf.yield
        } loc("<stdin>":71:9)
        %ssa_172_uncast = memref.get_global @a : memref<1xindex> loc("<stdin>":75:15)
        %ssa_172_zero = arith.constant 0 : index
        %ssa_172 = memref.subview %ssa_172_uncast[%ssa_172_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":75:15)
        %ssa_173_zero = arith.constant 0: index loc("<stdin>":75:15)
        %ssa_173 = memref.load %ssa_172[%ssa_173_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":75:15)
        isq.call_qop @__isq__builtin__print_int(%ssa_173): [0](index)->() loc("<stdin>":75:9)
        %ssa_174_uncast = memref.get_global @p : memref<1x!isq.qstate> loc("<stdin>":76:9)
        %ssa_174_zero = arith.constant 0 : index
        %ssa_174 = memref.subview %ssa_174_uncast[%ssa_174_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:9)
        %ssa_174_in_zero = arith.constant 0: index loc("<stdin>":76:9)
        %ssa_174_in = memref.load %ssa_174[%ssa_174_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:9)
        %ssa_174_out = isq.call_qop @__isq__builtin__reset(%ssa_174_in): [1]()->() loc("<stdin>":76:9)
        %ssa_174_out_zero = arith.constant 0: index loc("<stdin>":76:9)
        memref.store %ssa_174_out, %ssa_174[%ssa_174_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:9)
        %ssa_175 = memref.get_global @q : memref<3x!isq.qstate> loc("<stdin>":77:9)
        %ssa_176 = arith.constant 1 : index loc("<stdin>":77:11)
        %ssa_177 = memref.subview %ssa_175[%ssa_176][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":77:10)
        %ssa_177_in_zero = arith.constant 0: index loc("<stdin>":77:10)
        %ssa_177_in = memref.load %ssa_177[%ssa_177_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":77:10)
        %ssa_177_out = isq.call_qop @__isq__builtin__reset(%ssa_177_in): [1]()->() loc("<stdin>":77:10)
        %ssa_177_out_zero = arith.constant 0: index loc("<stdin>":77:10)
        memref.store %ssa_177_out, %ssa_177[%ssa_177_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":77:10)
        br ^exit_8 
    ^exit_8:
        memref.dealloc %ssa_160_real : memref<1xi1> loc("<stdin>":71:9)
        br ^exit_7 
    ^exit_7:
        memref.dealloc %ssa_158_real : memref<1xi1> loc("<stdin>":71:9)
        br ^exit_6 
    ^exit_6:
        memref.dealloc %ssa_87 : memref<5xindex> loc("<stdin>":59:9)
        br ^exit_5 
    ^exit_5:
        memref.dealloc %ssa_78_real : memref<1xi1> loc("<stdin>":56:9)
        br ^exit_4 
    ^exit_4:
        memref.dealloc %ssa_52_real : memref<1xi1> loc("<stdin>":50:9)
        br ^exit_3 
    ^exit_3:
        memref.dealloc %ssa_50_real : memref<1xindex> loc("<stdin>":49:9)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_49_real : memref<1xindex> loc("<stdin>":48:9)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_48_real : memref<1xi1> loc("<stdin>":46:1)
        br ^exit 
    ^exit:
        return loc("<stdin>":46:1)
    } loc("<stdin>":46:1)
    func @__isq__global_initialize() 
    {
    ^block1:
        return 
    } 
    func @__isq__global_finalize() 
    {
    ^block1:
        return 
    } 
    func @__isq__entry() 
    {
    ^block1:
        call @__isq__global_initialize() : ()->() 
        call @__isq__main() : ()->() 
        call @__isq__global_finalize() : ()->() 
        return 
    } 
}
