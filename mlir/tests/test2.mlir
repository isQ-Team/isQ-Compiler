module{
    isq.declare_qop @__isq__builtin__measure : [1]()->i1
    isq.declare_qop @__isq__builtin__reset : [1]()->()
    isq.declare_qop @__isq__builtin__print_int : [0](index)->()
    isq.declare_qop @__isq__builtin__print_double : [0](f64)->()
    isq.defgate @"Rs" {definition = [{type="unitary", value = [[#isq.complex<0.5, 0.8660254>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>]]}]}: !isq.gate<2> loc("<stdin>":1:1)
    isq.defgate @"Rs2" {definition = [{type="unitary", value = [[#isq.complex<0.5, -0.8660254>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>]]}]}: !isq.gate<2> loc("<stdin>":6:1)
    isq.defgate @"Rt" {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.5, 0.8660254>]]}]}: !isq.gate<2> loc("<stdin>":11:1)
    isq.defgate @"Rt2" {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.5, -0.8660254>]]}]}: !isq.gate<2> loc("<stdin>":15:1)
    func private @"__quantum__qis__rz__body"(f64, !isq.qir.qubit) loc("<stdin>":20:1)
    isq.defgate @"Rz"(f64) {definition = [{type = "qir", value = @"__quantum__qis__rz__body"}]}: !isq.gate<1> loc("<stdin>":20:1)
    func private @"__quantum__qis__rx__body"(f64, !isq.qir.qubit) loc("<stdin>":21:1)
    isq.defgate @"Rx"(f64) {definition = [{type = "qir", value = @"__quantum__qis__rx__body"}]}: !isq.gate<1> loc("<stdin>":21:1)
    func private @"__quantum__qis__ry__body"(f64, !isq.qir.qubit) loc("<stdin>":22:1)
    isq.defgate @"Ry"(f64) {definition = [{type = "qir", value = @"__quantum__qis__ry__body"}]}: !isq.gate<1> loc("<stdin>":22:1)
    func private @"__quantum__qis__h__body"(!isq.qir.qubit) loc("<stdin>":23:1)
    isq.defgate @"H" {definition = [{type = "qir", value = @"__quantum__qis__h__body"}]}: !isq.gate<1> loc("<stdin>":23:1)
    func private @"__quantum__qis__s__body"(!isq.qir.qubit) loc("<stdin>":24:1)
    isq.defgate @"S" {definition = [{type = "qir", value = @"__quantum__qis__s__body"}]}: !isq.gate<1> loc("<stdin>":24:1)
    func private @"__quantum__qis__t__body"(!isq.qir.qubit) loc("<stdin>":25:1)
    isq.defgate @"T" {definition = [{type = "qir", value = @"__quantum__qis__t__body"}]}: !isq.gate<1> loc("<stdin>":25:1)
    func private @"__quantum__qis__x__body"(!isq.qir.qubit) loc("<stdin>":26:1)
    isq.defgate @"X" {definition = [{type = "qir", value = @"__quantum__qis__x__body"}]}: !isq.gate<1> loc("<stdin>":26:1)
    func private @"__quantum__qis__y__body"(!isq.qir.qubit) loc("<stdin>":27:1)
    isq.defgate @"Y" {definition = [{type = "qir", value = @"__quantum__qis__y__body"}]}: !isq.gate<1> loc("<stdin>":27:1)
    func private @"__quantum__qis__z__b.0233.ody"(!isq.qir.qubit) loc("<stdin>":28:1)
    isq.defgate @"Z';kh'ljhggjhgjhhhhgjjjhjhjjhgjhjhgfd'hggjhgjhgjhghgfd'hhj';" {definition = [{type = "qir", value = @"__quantum__qis__z__body"}]}: !isq.gate<1> loc("<stdin>":28:1)
    func private @"__quantum__qis__cnot"(!isq.qir.qubit, !isq.qir.qubit) loc("<stdin>":29:1)
    isq.defgate @"CNOT" {definition = [{type = "qir", value = @"__quantum__qis__cnot"}]}: !isq.gate<2> loc("<stdin>":29:1)
    func private @"__quantum__qis__u3"(f64, f64, f64, !isq.qir.qubit) loc("<stdin>":30:1)
    isq.defgate @"u3"(f64, f64, f64) {definition = [{type = "qir", value = @"__quantum__qis__u3"}]}: !isq.gate<1> loc("<stdin>":30:1)
    memref.global @"a" : memref<1xindex> = uninitialized loc("<stdin>":38:1)
    memref.global @"b" : memref<1xindex> = uninitialized loc("<stdin>":38:1)
    memref.global @"c" : memref<1xindex> = uninitialized loc("<stdin>":38:1)
    memref.global @"q" : memref<3x!isq.qstate> = uninitialized loc("<stdin>":39:1)
    memref.global @"p" : memref<1x!isq.qstate> = uninitialized loc("<stdin>":39:1)
    memref.global @"dd" : memref<1xf64> = uninitialized loc("<stdin>":40:1)
    func @"$_ISQ_GATEDEF_test"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> {memref.restricted}, memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> {memref.restricted}) 
    {
    ^entry(%ssa_26: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_27: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>):
        %ssa_28 = arith.constant 0 : i1 loc("<stdin>":42:1)
        %ssa_29_real = memref.alloc() : memref<1xi1> loc("<stdin>":42:1)
        %ssa_29_zero = arith.constant 0 : index
        %ssa_29 = memref.subview %ssa_29_real[%ssa_29_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":42:1)
        %ssa_28_zero = arith.constant 0: index loc("<stdin>":42:1)
        affine.store %ssa_28, %ssa_29[%ssa_28_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":42:1)
        %ssa_30 = isq.use @"H" : !isq.gate<1> loc("<stdin>":43:9) 
        %ssa_30_in_1_zero = arith.constant 0: index loc("<stdin>":43:9)
        %ssa_30_in_1 = affine.load %ssa_27[%ssa_30_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":43:9)
        %ssa_30_out_1 = isq.apply %ssa_30(%ssa_30_in_1) : !isq.gate<1> loc("<stdin>":43:9)
        %ssa_30_out_1_zero = arith.constant 0: index loc("<stdin>":43:9)
        affine.store %ssa_30_out_1, %ssa_27[%ssa_30_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":43:9)
        %ssa_32 = isq.use @"CNOT" : !isq.gate<2> loc("<stdin>":44:9) 
        %ssa_32_in_1_zero = arith.constant 0: index loc("<stdin>":44:9)
        %ssa_32_in_1 = affine.load %ssa_26[%ssa_32_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":44:9)
        %ssa_32_in_2_zero = arith.constant 0: index loc("<stdin>":44:9)
        %ssa_32_in_2 = affine.load %ssa_27[%ssa_32_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":44:9)
        %ssa_32_out_1, %ssa_32_out_2 = isq.apply %ssa_32(%ssa_32_in_1, %ssa_32_in_2) : !isq.gate<2> loc("<stdin>":44:9)
        %ssa_32_out_1_zero = arith.constant 0: index loc("<stdin>":44:9)
        affine.store %ssa_32_out_1, %ssa_26[%ssa_32_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":44:9)
        %ssa_32_out_2_zero = arith.constant 0: index loc("<stdin>":44:9)
        affine.store %ssa_32_out_2, %ssa_27[%ssa_32_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":44:9)
        %ssa_35 = isq.use @"H" : !isq.gate<1> loc("<stdin>":45:9) 
        %ssa_35_in_1_zero = arith.constant 0: index loc("<stdin>":45:9)
        %ssa_35_in_1 = affine.load %ssa_27[%ssa_35_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":45:9)
        %ssa_35_out_1 = isq.apply %ssa_35(%ssa_35_in_1) : !isq.gate<1> loc("<stdin>":45:9)
        %ssa_35_out_1_zero = arith.constant 0: index loc("<stdin>":45:9)
        affine.store %ssa_35_out_1, %ssa_27[%ssa_35_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":45:9)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_29_real : memref<1xi1> loc("<stdin>":42:1)
        br ^exit 
    ^exit:
        return loc("<stdin>":42:1)
    } loc("<stdin>":42:1)
    isq.defgate @"test" {definition = [{type = "decomposition_raw", value = @"$_ISQ_GATEDEF_test"}]}: !isq.gate<2> loc("<stdin>":42:1)
    func @"test2"(memref<?x!isq.qstate>, f64) 
    {
    ^entry(%ssa_38: memref<?x!isq.qstate>, %ssa_39: f64):
        %ssa_40 = arith.constant 0 : i1 loc("<stdin>":48:1)
        %ssa_41_real = memref.alloc() : memref<1xi1> loc("<stdin>":48:1)
        %ssa_41_zero = arith.constant 0 : index
        %ssa_41 = memref.subview %ssa_41_real[%ssa_41_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":48:1)
        %ssa_40_zero = arith.constant 0: index loc("<stdin>":48:1)
        affine.store %ssa_40, %ssa_41[%ssa_40_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":48:1)
        %ssa_44 = arith.constant 1 : index loc("<stdin>":49:13)
        %ssa_45 = memref.subview %ssa_38[%ssa_44][1][1] : memref<?x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":49:12)
        %ssa_42 = isq.use @"H" : !isq.gate<1> loc("<stdin>":49:9) 
        %ssa_42_in_1_zero = arith.constant 0: index loc("<stdin>":49:9)
        %ssa_42_in_1 = affine.load %ssa_45[%ssa_42_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":49:9)
        %ssa_42_out_1 = isq.apply %ssa_42(%ssa_42_in_1) : !isq.gate<1> loc("<stdin>":49:9)
        %ssa_42_out_1_zero = arith.constant 0: index loc("<stdin>":49:9)
        affine.store %ssa_42_out_1, %ssa_45[%ssa_42_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":49:9)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_41_real : memref<1xi1> loc("<stdin>":48:1)
        br ^exit 
    ^exit:
        return loc("<stdin>":48:1)
    } loc("<stdin>":48:1)
    func @"__isq__main"() 
    {
    ^entry:
        %ssa_47 = arith.constant 0 : i1 loc("<stdin>":52:1)
        %ssa_48_real = memref.alloc() : memref<1xi1> loc("<stdin>":52:1)
        %ssa_48_zero = arith.constant 0 : index
        %ssa_48 = memref.subview %ssa_48_real[%ssa_48_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":52:1)
        %ssa_47_zero = arith.constant 0: index loc("<stdin>":52:1)
        affine.store %ssa_47, %ssa_48[%ssa_47_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":52:1)
        %ssa_49_real = memref.alloc() : memref<1xindex> loc("<stdin>":54:9)
        %ssa_49_zero = arith.constant 0 : index
        %ssa_49 = memref.subview %ssa_49_real[%ssa_49_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":54:9)
        %ssa_50_real = memref.alloc() : memref<1xindex> loc("<stdin>":55:9)
        %ssa_50_zero = arith.constant 0 : index
        %ssa_50 = memref.subview %ssa_50_real[%ssa_50_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":55:9)
        %ssa_51 = arith.constant 0 : i1 loc("<stdin>":56:9)
        %ssa_52_real = memref.alloc() : memref<1xi1> loc("<stdin>":56:9)
        %ssa_52_zero = arith.constant 0 : index
        %ssa_52 = memref.subview %ssa_52_real[%ssa_52_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":56:9)
        %ssa_51_zero = arith.constant 0: index loc("<stdin>":56:9)
        affine.store %ssa_51, %ssa_52[%ssa_51_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":56:9)
        %ssa_53 = arith.constant 1 : index loc("<stdin>":56:13)
        %ssa_54 = arith.constant 2 : index loc("<stdin>":56:17)
        %ssa_55 = arith.cmpi "slt", %ssa_53, %ssa_54 : index loc("<stdin>":56:15)
        scf.if %ssa_55 {
            scf.execute_region {
            ^entry:
                %ssa_56_uncast = memref.get_global @"dd" : memref<1xf64> loc("<stdin>":57:17)
                %ssa_56_zero = arith.constant 0 : index
                %ssa_56 = memref.subview %ssa_56_uncast[%ssa_56_zero][1][1] : memref<1xf64> to memref<1xf64, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":57:17)
                %ssa_57_uncast = memref.get_global @"a" : memref<1xindex> loc("<stdin>":57:23)
                %ssa_57_zero = arith.constant 0 : index
                %ssa_57 = memref.subview %ssa_57_uncast[%ssa_57_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":57:23)
                %ssa_63_zero = arith.constant 0: index loc("<stdin>":57:23)
                %ssa_63 = affine.load %ssa_57[%ssa_63_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":57:23)
                %ssa_64_i64 = arith.index_cast %ssa_63 : index to i64 loc("<stdin>":57:23)
                %ssa_64 = arith.sitofp %ssa_64_i64 : i64 to f64 loc("<stdin>":57:23)
                %ssa_58 = arith.constant 3 : index loc("<stdin>":57:25)
                %ssa_61_i64 = arith.index_cast %ssa_58 : index to i64 loc("<stdin>":57:25)
                %ssa_61 = arith.sitofp %ssa_61_i64 : i64 to f64 loc("<stdin>":57:25)
                %ssa_59 = arith.constant 2.2 : f64 loc("<stdin>":57:27)
                %ssa_60 = arith.mulf %ssa_61, %ssa_59 : f64 loc("<stdin>":57:26)
                %ssa_62 = arith.addf %ssa_64, %ssa_60 : f64 loc("<stdin>":57:24)
                %ssa_65_uncast = memref.get_global @"b" : memref<1xindex> loc("<stdin>":57:33)
                %ssa_65_zero = arith.constant 0 : index
                %ssa_65 = memref.subview %ssa_65_uncast[%ssa_65_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":57:33)
                %ssa_68_zero = arith.constant 0: index loc("<stdin>":57:33)
                %ssa_68 = affine.load %ssa_65[%ssa_68_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":57:33)
                %ssa_66_uncast = memref.get_global @"c" : memref<1xindex> loc("<stdin>":57:35)
                %ssa_66_zero = arith.constant 0 : index
                %ssa_66 = memref.subview %ssa_66_uncast[%ssa_66_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":57:35)
                %ssa_69_zero = arith.constant 0: index loc("<stdin>":57:35)
                %ssa_69 = affine.load %ssa_66[%ssa_69_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":57:35)
                %ssa_67 = arith.addi %ssa_68, %ssa_69 : index loc("<stdin>":57:34)
                %ssa_71_i64 = arith.index_cast %ssa_67 : index to i64 loc("<stdin>":57:34)
                %ssa_71 = arith.sitofp %ssa_71_i64 : i64 to f64 loc("<stdin>":57:34)
                %ssa_70 = arith.mulf %ssa_62, %ssa_71 : f64 loc("<stdin>":57:31)
                %ssa_70_zero = arith.constant 0: index loc("<stdin>":57:20)
                affine.store %ssa_70, %ssa_56[%ssa_70_zero] : memref<1xf64, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":57:20)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":56:9)
            } loc("<stdin>":56:9)
        } else {
            scf.execute_region {
            ^entry:
                %ssa_72_real = memref.alloc() : memref<1xindex> loc("<stdin>":59:17)
                %ssa_72_zero = arith.constant 0 : index
                %ssa_72 = memref.subview %ssa_72_real[%ssa_72_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":59:17)
                %ssa_74_uncast = memref.get_global @"c" : memref<1xindex> loc("<stdin>":60:21)
                %ssa_74_zero = arith.constant 0 : index
                %ssa_74 = memref.subview %ssa_74_uncast[%ssa_74_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":60:21)
                %ssa_77_zero = arith.constant 0: index loc("<stdin>":60:21)
                %ssa_77 = affine.load %ssa_74[%ssa_77_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":60:21)
                %ssa_75 = arith.constant 1 : index loc("<stdin>":60:23)
                %ssa_76 = arith.addi %ssa_77, %ssa_75 : index loc("<stdin>":60:22)
                %ssa_76_zero = arith.constant 0: index loc("<stdin>":60:19)
                affine.store %ssa_76, %ssa_72[%ssa_76_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":60:19)
                br ^exit_1 
            ^exit_1:
                memref.dealloc %ssa_72_real : memref<1xindex> loc("<stdin>":59:17)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":56:9)
            } loc("<stdin>":56:9)
        }
        %ssa_79_zero = arith.constant 0: index loc("<stdin>":56:9)
        %ssa_79 = affine.load %ssa_48[%ssa_79_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":56:9)
        cond_br %ssa_79, ^exit_4, ^block1 loc("<stdin>":56:9)
    ^block1:
        %ssa_80 = arith.constant 0 : i1 loc("<stdin>":62:9)
        %ssa_81_real = memref.alloc() : memref<1xi1> loc("<stdin>":62:9)
        %ssa_81_zero = arith.constant 0 : index
        %ssa_81 = memref.subview %ssa_81_real[%ssa_81_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":62:9)
        %ssa_80_zero = arith.constant 0: index loc("<stdin>":62:9)
        affine.store %ssa_80, %ssa_81[%ssa_80_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":62:9)
        %ssa_82 = arith.constant 1 : index loc("<stdin>":62:18)
        %ssa_83_uncast = memref.get_global @"b" : memref<1xindex> loc("<stdin>":62:20)
        %ssa_83_zero = arith.constant 0 : index
        %ssa_83 = memref.subview %ssa_83_uncast[%ssa_83_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":62:20)
        %ssa_84_zero = arith.constant 0: index loc("<stdin>":62:20)
        %ssa_84 = affine.load %ssa_83[%ssa_84_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":62:20)
        affine.for %ssa_87 = %ssa_82 to %ssa_84 step 1 {
            scf.execute_region {
            ^entry:
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":62:9)
            } loc("<stdin>":62:9)
        } loc("<stdin>":62:9)
        %ssa_89_zero = arith.constant 0: index loc("<stdin>":62:9)
        %ssa_89 = affine.load %ssa_48[%ssa_89_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":62:9)
        cond_br %ssa_89, ^exit_5, ^block2 loc("<stdin>":62:9)
    ^block2:
        %ssa_90 = memref.alloc() : memref<5xindex> loc("<stdin>":65:9)
        %ssa_91_uncast = memref.get_global @"a" : memref<1xindex> loc("<stdin>":66:9)
        %ssa_91_zero = arith.constant 0 : index
        %ssa_91 = memref.subview %ssa_91_uncast[%ssa_91_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:9)
        %ssa_93_uncast = memref.get_global @"c" : memref<1xindex> loc("<stdin>":66:15)
        %ssa_93_zero = arith.constant 0 : index
        %ssa_93 = memref.subview %ssa_93_uncast[%ssa_93_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:15)
        %ssa_94_zero = arith.constant 0: index loc("<stdin>":66:15)
        %ssa_94 = affine.load %ssa_93[%ssa_94_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:15)
        %ssa_95 = memref.subview %ssa_90[%ssa_94][1][1] : memref<5xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:14)
        %ssa_98_zero = arith.constant 0: index loc("<stdin>":66:14)
        %ssa_98 = affine.load %ssa_95[%ssa_98_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:14)
        %ssa_96 = arith.constant 2 : index loc("<stdin>":66:18)
        %ssa_97 = arith.addi %ssa_98, %ssa_96 : index loc("<stdin>":66:17)
        %ssa_97_zero = arith.constant 0: index loc("<stdin>":66:11)
        affine.store %ssa_97, %ssa_91[%ssa_97_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":66:11)
        %ssa_101_uncast = memref.get_global @"p" : memref<1x!isq.qstate> loc("<stdin>":67:10)
        %ssa_101_zero = arith.constant 0 : index
        %ssa_101 = memref.subview %ssa_101_uncast[%ssa_101_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:10)
        %ssa_102 = memref.get_global @"q" : memref<3x!isq.qstate> loc("<stdin>":67:13)
        %ssa_103 = arith.constant 0 : index loc("<stdin>":67:15)
        %ssa_104 = memref.subview %ssa_102[%ssa_103][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:14)
        %ssa_100 = isq.use @"test" : !isq.gate<2> loc("<stdin>":67:5) 
        %ssa_100_in_1_zero = arith.constant 0: index loc("<stdin>":67:5)
        %ssa_100_in_1 = affine.load %ssa_101[%ssa_100_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:5)
        %ssa_100_in_2_zero = arith.constant 0: index loc("<stdin>":67:5)
        %ssa_100_in_2 = affine.load %ssa_104[%ssa_100_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:5)
        %ssa_100_out_1, %ssa_100_out_2 = isq.apply %ssa_100(%ssa_100_in_1, %ssa_100_in_2) : !isq.gate<2> loc("<stdin>":67:5)
        %ssa_100_out_1_zero = arith.constant 0: index loc("<stdin>":67:5)
        affine.store %ssa_100_out_1, %ssa_101[%ssa_100_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:5)
        %ssa_100_out_2_zero = arith.constant 0: index loc("<stdin>":67:5)
        affine.store %ssa_100_out_2, %ssa_104[%ssa_100_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":67:5)
        %ssa_107 = memref.get_global @"q" : memref<3x!isq.qstate> loc("<stdin>":68:15)
        %ssa_109 = memref.cast %ssa_107 : memref<3x!isq.qstate> to memref<?x!isq.qstate> loc("<stdin>":68:15)
        %ssa_108_uncast = memref.get_global @"a" : memref<1xindex> loc("<stdin>":68:18)
        %ssa_108_zero = arith.constant 0 : index
        %ssa_108 = memref.subview %ssa_108_uncast[%ssa_108_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:18)
        %ssa_110_zero = arith.constant 0: index loc("<stdin>":68:18)
        %ssa_110 = affine.load %ssa_108[%ssa_110_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":68:18)
        %ssa_111_i64 = arith.index_cast %ssa_110 : index to i64 loc("<stdin>":68:18)
        %ssa_111 = arith.sitofp %ssa_111_i64 : i64 to f64 loc("<stdin>":68:18)
        call @"test2"(%ssa_109, %ssa_111) : (memref<?x!isq.qstate>, f64)->() loc("<stdin>":68:9)
        %ssa_113_uncast = memref.get_global @"a" : memref<1xindex> loc("<stdin>":69:5)
        %ssa_113_zero = arith.constant 0 : index
        %ssa_113 = memref.subview %ssa_113_uncast[%ssa_113_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:5)
        %ssa_114 = memref.get_global @"q" : memref<3x!isq.qstate> loc("<stdin>":69:11)
        %ssa_115 = arith.constant 0 : index loc("<stdin>":69:13)
        %ssa_116 = memref.subview %ssa_114[%ssa_115][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:12)
        %ssa_117_in_zero = arith.constant 0: index loc("<stdin>":69:9)
        %ssa_117_in = affine.load %ssa_116[%ssa_117_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:9)
        %ssa_117_out, %ssa_117 = isq.call_qop @__isq__builtin__measure(%ssa_117_in): [1]()->i1 loc("<stdin>":69:9)
        %ssa_117_out_zero = arith.constant 0: index loc("<stdin>":69:9)
        affine.store %ssa_117_out, %ssa_116[%ssa_117_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:9)
        %ssa_118_i2 = arith.extui %ssa_117 : i1 to i2 loc("<stdin>":69:9)
        %ssa_118 = arith.index_cast %ssa_118_i2 : i2 to index loc("<stdin>":69:9)
        %ssa_118_zero = arith.constant 0: index loc("<stdin>":69:7)
        affine.store %ssa_118, %ssa_113[%ssa_118_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":69:7)
        %ssa_121_uncast = memref.get_global @"p" : memref<1x!isq.qstate> loc("<stdin>":70:14)
        %ssa_121_zero = arith.constant 0 : index
        %ssa_121 = memref.subview %ssa_121_uncast[%ssa_121_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":70:14)
        %ssa_122 = memref.get_global @"q" : memref<3x!isq.qstate> loc("<stdin>":70:17)
        %ssa_123 = arith.constant 0 : index loc("<stdin>":70:19)
        %ssa_124 = memref.subview %ssa_122[%ssa_123][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":70:18)
        %ssa_120 = isq.use @"CNOT" : !isq.gate<2> loc("<stdin>":70:9) 
        %ssa_120_in_1_zero = arith.constant 0: index loc("<stdin>":70:9)
        %ssa_120_in_1 = affine.load %ssa_121[%ssa_120_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":70:9)
        %ssa_120_in_2_zero = arith.constant 0: index loc("<stdin>":70:9)
        %ssa_120_in_2 = affine.load %ssa_124[%ssa_120_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":70:9)
        %ssa_120_out_1, %ssa_120_out_2 = isq.apply %ssa_120(%ssa_120_in_1, %ssa_120_in_2) : !isq.gate<2> loc("<stdin>":70:9)
        %ssa_120_out_1_zero = arith.constant 0: index loc("<stdin>":70:9)
        affine.store %ssa_120_out_1, %ssa_121[%ssa_120_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":70:9)
        %ssa_120_out_2_zero = arith.constant 0: index loc("<stdin>":70:9)
        affine.store %ssa_120_out_2, %ssa_124[%ssa_120_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":70:9)
        %ssa_126 = memref.get_global @"q" : memref<3x!isq.qstate> loc("<stdin>":72:21)
        %ssa_127 = arith.constant 0 : index loc("<stdin>":72:23)
        %ssa_128 = memref.subview %ssa_126[%ssa_127][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":72:22)
        %ssa_129_uncast = memref.get_global @"p" : memref<1x!isq.qstate> loc("<stdin>":72:27)
        %ssa_129_zero = arith.constant 0 : index
        %ssa_129 = memref.subview %ssa_129_uncast[%ssa_129_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":72:27)
        %ssa_125 = isq.use @"H" : !isq.gate<1> loc("<stdin>":72:19) 
        %ssa_125_decorated = isq.decorate(%ssa_125: !isq.gate<1>) {ctrl = [false], adjoint = true} : !isq.gate<2> loc("<stdin>":72:19)
        %ssa_125_in_1_zero = arith.constant 0: index loc("<stdin>":72:19)
        %ssa_125_in_1 = affine.load %ssa_128[%ssa_125_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":72:19)
        %ssa_125_in_2_zero = arith.constant 0: index loc("<stdin>":72:19)
        %ssa_125_in_2 = affine.load %ssa_129[%ssa_125_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":72:19)
        %ssa_125_out_1, %ssa_125_out_2 = isq.apply %ssa_125_decorated(%ssa_125_in_1, %ssa_125_in_2) : !isq.gate<2> loc("<stdin>":72:19)
        %ssa_125_out_1_zero = arith.constant 0: index loc("<stdin>":72:19)
        affine.store %ssa_125_out_1, %ssa_128[%ssa_125_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":72:19)
        %ssa_125_out_2_zero = arith.constant 0: index loc("<stdin>":72:19)
        affine.store %ssa_125_out_2, %ssa_129[%ssa_125_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":72:19)
        %ssa_131 = memref.get_global @"q" : memref<3x!isq.qstate> loc("<stdin>":73:24)
        %ssa_132 = arith.constant 0 : index loc("<stdin>":73:26)
        %ssa_133 = memref.subview %ssa_131[%ssa_132][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:25)
        %ssa_134 = memref.get_global @"q" : memref<3x!isq.qstate> loc("<stdin>":73:30)
        %ssa_135 = arith.constant 2 : index loc("<stdin>":73:32)
        %ssa_136 = memref.subview %ssa_134[%ssa_135][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:31)
        %ssa_137_uncast = memref.get_global @"p" : memref<1x!isq.qstate> loc("<stdin>":73:36)
        %ssa_137_zero = arith.constant 0 : index
        %ssa_137 = memref.subview %ssa_137_uncast[%ssa_137_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:36)
        %ssa_138 = memref.get_global @"q" : memref<3x!isq.qstate> loc("<stdin>":73:39)
        %ssa_139 = arith.constant 1 : index loc("<stdin>":73:41)
        %ssa_140 = memref.subview %ssa_138[%ssa_139][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:40)
        %ssa_130 = isq.use @"Rt2" : !isq.gate<2> loc("<stdin>":73:20) 
        %ssa_130_decorated = isq.decorate(%ssa_130: !isq.gate<2>) {ctrl = [false, true], adjoint = false} : !isq.gate<4> loc("<stdin>":73:20)
        %ssa_130_in_1_zero = arith.constant 0: index loc("<stdin>":73:20)
        %ssa_130_in_1 = affine.load %ssa_133[%ssa_130_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:20)
        %ssa_130_in_2_zero = arith.constant 0: index loc("<stdin>":73:20)
        %ssa_130_in_2 = affine.load %ssa_136[%ssa_130_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:20)
        %ssa_130_in_3_zero = arith.constant 0: index loc("<stdin>":73:20)
        %ssa_130_in_3 = affine.load %ssa_137[%ssa_130_in_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:20)
        %ssa_130_in_4_zero = arith.constant 0: index loc("<stdin>":73:20)
        %ssa_130_in_4 = affine.load %ssa_140[%ssa_130_in_4_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:20)
        %ssa_130_out_1, %ssa_130_out_2, %ssa_130_out_3, %ssa_130_out_4 = isq.apply %ssa_130_decorated(%ssa_130_in_1, %ssa_130_in_2, %ssa_130_in_3, %ssa_130_in_4) : !isq.gate<4> loc("<stdin>":73:20)
        %ssa_130_out_1_zero = arith.constant 0: index loc("<stdin>":73:20)
        affine.store %ssa_130_out_1, %ssa_133[%ssa_130_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:20)
        %ssa_130_out_2_zero = arith.constant 0: index loc("<stdin>":73:20)
        affine.store %ssa_130_out_2, %ssa_136[%ssa_130_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:20)
        %ssa_130_out_3_zero = arith.constant 0: index loc("<stdin>":73:20)
        affine.store %ssa_130_out_3, %ssa_137[%ssa_130_out_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:20)
        %ssa_130_out_4_zero = arith.constant 0: index loc("<stdin>":73:20)
        affine.store %ssa_130_out_4, %ssa_140[%ssa_130_out_4_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":73:20)
        %ssa_141_uncast = memref.get_global @"dd" : memref<1xf64> loc("<stdin>":74:9)
        %ssa_141_zero = arith.constant 0 : index
        %ssa_141 = memref.subview %ssa_141_uncast[%ssa_141_zero][1][1] : memref<1xf64> to memref<1xf64, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":74:9)
        %ssa_142 = arith.constant 0.7 : f64 loc("<stdin>":74:14)
        %ssa_142_zero = arith.constant 0: index loc("<stdin>":74:12)
        affine.store %ssa_142, %ssa_141[%ssa_142_zero] : memref<1xf64, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":74:12)
        %ssa_145_uncast = memref.get_global @"a" : memref<1xindex> loc("<stdin>":75:12)
        %ssa_145_zero = arith.constant 0 : index
        %ssa_145 = memref.subview %ssa_145_uncast[%ssa_145_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":75:12)
        %ssa_148_zero = arith.constant 0: index loc("<stdin>":75:12)
        %ssa_148 = affine.load %ssa_145[%ssa_148_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":75:12)
        %ssa_149_i64 = arith.index_cast %ssa_148 : index to i64 loc("<stdin>":75:12)
        %ssa_149 = arith.sitofp %ssa_149_i64 : i64 to f64 loc("<stdin>":75:12)
        %ssa_146 = arith.constant 0.3 : f64 loc("<stdin>":75:14)
        %ssa_147 = arith.addf %ssa_149, %ssa_146 : f64 loc("<stdin>":75:13)
        %ssa_150_uncast = memref.get_global @"p" : memref<1x!isq.qstate> loc("<stdin>":75:19)
        %ssa_150_zero = arith.constant 0 : index
        %ssa_150 = memref.subview %ssa_150_uncast[%ssa_150_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":75:19)
        %ssa_144 = isq.use @"Rx"(%ssa_147) : (f64) -> !isq.gate<1> loc("<stdin>":75:9) 
        %ssa_144_in_1_zero = arith.constant 0: index loc("<stdin>":75:9)
        %ssa_144_in_1 = affine.load %ssa_150[%ssa_144_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":75:9)
        %ssa_144_out_1 = isq.apply %ssa_144(%ssa_144_in_1) : !isq.gate<1> loc("<stdin>":75:9)
        %ssa_144_out_1_zero = arith.constant 0: index loc("<stdin>":75:9)
        affine.store %ssa_144_out_1, %ssa_150[%ssa_144_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":75:9)
        %ssa_152_uncast = memref.get_global @"dd" : memref<1xf64> loc("<stdin>":76:20)
        %ssa_152_zero = arith.constant 0 : index
        %ssa_152 = memref.subview %ssa_152_uncast[%ssa_152_zero][1][1] : memref<1xf64> to memref<1xf64, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:20)
        %ssa_160_zero = arith.constant 0: index loc("<stdin>":76:20)
        %ssa_160 = affine.load %ssa_152[%ssa_160_zero] : memref<1xf64, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:20)
        %ssa_153_uncast = memref.get_global @"p" : memref<1x!isq.qstate> loc("<stdin>":76:24)
        %ssa_153_zero = arith.constant 0 : index
        %ssa_153 = memref.subview %ssa_153_uncast[%ssa_153_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:24)
        %ssa_154 = memref.get_global @"q" : memref<3x!isq.qstate> loc("<stdin>":76:27)
        %ssa_155 = arith.constant 0 : index loc("<stdin>":76:29)
        %ssa_156 = memref.subview %ssa_154[%ssa_155][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:28)
        %ssa_157 = memref.get_global @"q" : memref<3x!isq.qstate> loc("<stdin>":76:33)
        %ssa_158 = arith.constant 1 : index loc("<stdin>":76:35)
        %ssa_159 = memref.subview %ssa_157[%ssa_158][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:34)
        %ssa_151 = isq.use @"Rz"(%ssa_160) : (f64) -> !isq.gate<1> loc("<stdin>":76:17) 
        %ssa_151_decorated = isq.decorate(%ssa_151: !isq.gate<1>) {ctrl = [true, true], adjoint = false} : !isq.gate<3> loc("<stdin>":76:17)
        %ssa_151_in_1_zero = arith.constant 0: index loc("<stdin>":76:17)
        %ssa_151_in_1 = affine.load %ssa_153[%ssa_151_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:17)
        %ssa_151_in_2_zero = arith.constant 0: index loc("<stdin>":76:17)
        %ssa_151_in_2 = affine.load %ssa_156[%ssa_151_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:17)
        %ssa_151_in_3_zero = arith.constant 0: index loc("<stdin>":76:17)
        %ssa_151_in_3 = affine.load %ssa_159[%ssa_151_in_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:17)
        %ssa_151_out_1, %ssa_151_out_2, %ssa_151_out_3 = isq.apply %ssa_151_decorated(%ssa_151_in_1, %ssa_151_in_2, %ssa_151_in_3) : !isq.gate<3> loc("<stdin>":76:17)
        %ssa_151_out_1_zero = arith.constant 0: index loc("<stdin>":76:17)
        affine.store %ssa_151_out_1, %ssa_153[%ssa_151_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:17)
        %ssa_151_out_2_zero = arith.constant 0: index loc("<stdin>":76:17)
        affine.store %ssa_151_out_2, %ssa_156[%ssa_151_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:17)
        %ssa_151_out_3_zero = arith.constant 0: index loc("<stdin>":76:17)
        affine.store %ssa_151_out_3, %ssa_159[%ssa_151_out_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":76:17)
        %ssa_163 = arith.constant 0.1 : f64 loc("<stdin>":77:12)
        %ssa_164 = arith.constant 0.3 : f64 loc("<stdin>":77:17)
        %ssa_165_uncast = memref.get_global @"dd" : memref<1xf64> loc("<stdin>":77:22)
        %ssa_165_zero = arith.constant 0 : index
        %ssa_165 = memref.subview %ssa_165_uncast[%ssa_165_zero][1][1] : memref<1xf64> to memref<1xf64, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":77:22)
        %ssa_167_zero = arith.constant 0: index loc("<stdin>":77:22)
        %ssa_167 = affine.load %ssa_165[%ssa_167_zero] : memref<1xf64, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":77:22)
        %ssa_166_uncast = memref.get_global @"p" : memref<1x!isq.qstate> loc("<stdin>":77:26)
        %ssa_166_zero = arith.constant 0 : index
        %ssa_166 = memref.subview %ssa_166_uncast[%ssa_166_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":77:26)
        %ssa_162 = isq.use @"u3"(%ssa_163, %ssa_164, %ssa_167) : (f64, f64, f64) -> !isq.gate<1> loc("<stdin>":77:9) 
        %ssa_162_in_1_zero = arith.constant 0: index loc("<stdin>":77:9)
        %ssa_162_in_1 = affine.load %ssa_166[%ssa_162_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":77:9)
        %ssa_162_out_1 = isq.apply %ssa_162(%ssa_162_in_1) : !isq.gate<1> loc("<stdin>":77:9)
        %ssa_162_out_1_zero = arith.constant 0: index loc("<stdin>":77:9)
        affine.store %ssa_162_out_1, %ssa_166[%ssa_162_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":77:9)
        %ssa_168 = arith.constant 0 : i1 loc("<stdin>":78:9)
        %ssa_169_real = memref.alloc() : memref<1xi1> loc("<stdin>":78:9)
        %ssa_169_zero = arith.constant 0 : index
        %ssa_169 = memref.subview %ssa_169_real[%ssa_169_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":78:9)
        %ssa_168_zero = arith.constant 0: index loc("<stdin>":78:9)
        affine.store %ssa_168, %ssa_169[%ssa_168_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":78:9)
        %ssa_170 = arith.constant 0 : i1 loc("<stdin>":78:9)
        %ssa_171_real = memref.alloc() : memref<1xi1> loc("<stdin>":78:9)
        %ssa_171_zero = arith.constant 0 : index
        %ssa_171 = memref.subview %ssa_171_real[%ssa_171_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":78:9)
        %ssa_170_zero = arith.constant 0: index loc("<stdin>":78:9)
        affine.store %ssa_170, %ssa_171[%ssa_170_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":78:9)
        scf.while : ()->() {
            %cond = scf.execute_region->i1 {
                ^break_check:
                    %ssa_177_zero = arith.constant 0: index loc("<stdin>":78:9)
                    %ssa_177 = affine.load %ssa_169[%ssa_177_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":78:9)
                    cond_br %ssa_177, ^break, ^while_cond
                ^while_cond:
                    %ssa_172_uncast = memref.get_global @"a" : memref<1xindex> loc("<stdin>":78:16)
                    %ssa_172_zero = arith.constant 0 : index
                    %ssa_172 = memref.subview %ssa_172_uncast[%ssa_172_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":78:16)
                    %ssa_175_zero = arith.constant 0: index loc("<stdin>":78:16)
                    %ssa_175 = affine.load %ssa_172[%ssa_175_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":78:16)
                    %ssa_173 = arith.constant 2 : index loc("<stdin>":78:20)
                    %ssa_174 = arith.cmpi "slt", %ssa_175, %ssa_173 : index loc("<stdin>":78:18)
                    scf.yield %ssa_174: i1
                ^break:
                    %zero=arith.constant 0: i1
                    scf.yield %zero: i1
            }
            scf.condition(%cond)
        } do {
            scf.execute_region {
            ^entry:
                %ssa_178_uncast = memref.get_global @"a" : memref<1xindex> loc("<stdin>":79:17)
                %ssa_178_zero = arith.constant 0 : index
                %ssa_178 = memref.subview %ssa_178_uncast[%ssa_178_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":79:17)
                %ssa_179_uncast = memref.get_global @"a" : memref<1xindex> loc("<stdin>":79:21)
                %ssa_179_zero = arith.constant 0 : index
                %ssa_179 = memref.subview %ssa_179_uncast[%ssa_179_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":79:21)
                %ssa_182_zero = arith.constant 0: index loc("<stdin>":79:21)
                %ssa_182 = affine.load %ssa_179[%ssa_182_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":79:21)
                %ssa_180 = arith.constant 1 : index loc("<stdin>":79:25)
                %ssa_181 = arith.addi %ssa_182, %ssa_180 : index loc("<stdin>":79:23)
                %ssa_181_zero = arith.constant 0: index loc("<stdin>":79:19)
                affine.store %ssa_181, %ssa_178[%ssa_181_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":79:19)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":78:9)
            } loc("<stdin>":78:9)
        scf.yield
        } loc("<stdin>":78:9)
        %ssa_183_uncast = memref.get_global @"a" : memref<1xindex> loc("<stdin>":82:15)
        %ssa_183_zero = arith.constant 0 : index
        %ssa_183 = memref.subview %ssa_183_uncast[%ssa_183_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":82:15)
        %ssa_184_zero = arith.constant 0: index loc("<stdin>":82:15)
        %ssa_184 = affine.load %ssa_183[%ssa_184_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":82:15)
        isq.call_qop @__isq__builtin__print_int(%ssa_184): [0](index)->() loc("<stdin>":82:9)
        %ssa_185_uncast = memref.get_global @"p" : memref<1x!isq.qstate> loc("<stdin>":83:9)
        %ssa_185_zero = arith.constant 0 : index
        %ssa_185 = memref.subview %ssa_185_uncast[%ssa_185_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":83:9)
        %ssa_185_in_zero = arith.constant 0: index loc("<stdin>":83:9)
        %ssa_185_in = affine.load %ssa_185[%ssa_185_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":83:9)
        %ssa_185_out = isq.call_qop @__isq__builtin__reset(%ssa_185_in): [1]()->() loc("<stdin>":83:9)
        %ssa_185_out_zero = arith.constant 0: index loc("<stdin>":83:9)
        affine.store %ssa_185_out, %ssa_185[%ssa_185_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":83:9)
        %ssa_186 = memref.get_global @"q" : memref<3x!isq.qstate> loc("<stdin>":84:9)
        %ssa_187 = arith.constant 1 : index loc("<stdin>":84:11)
        %ssa_188 = memref.subview %ssa_186[%ssa_187][1][1] : memref<3x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":84:10)
        %ssa_188_in_zero = arith.constant 0: index loc("<stdin>":84:10)
        %ssa_188_in = affine.load %ssa_188[%ssa_188_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":84:10)
        %ssa_188_out = isq.call_qop @__isq__builtin__reset(%ssa_188_in): [1]()->() loc("<stdin>":84:10)
        %ssa_188_out_zero = arith.constant 0: index loc("<stdin>":84:10)
        affine.store %ssa_188_out, %ssa_188[%ssa_188_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":84:10)
        br ^exit_8 
    ^exit_8:
        memref.dealloc %ssa_171_real : memref<1xi1> loc("<stdin>":78:9)
        br ^exit_7 
    ^exit_7:
        memref.dealloc %ssa_169_real : memref<1xi1> loc("<stdin>":78:9)
        br ^exit_6 
    ^exit_6:
        memref.dealloc %ssa_90 : memref<5xindex> loc("<stdin>":65:9)
        br ^exit_5 
    ^exit_5:
        memref.dealloc %ssa_81_real : memref<1xi1> loc("<stdin>":62:9)
        br ^exit_4 
    ^exit_4:
        memref.dealloc %ssa_52_real : memref<1xi1> loc("<stdin>":56:9)
        br ^exit_3 
    ^exit_3:
        memref.dealloc %ssa_50_real : memref<1xindex> loc("<stdin>":55:9)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_49_real : memref<1xindex> loc("<stdin>":54:9)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_48_real : memref<1xi1> loc("<stdin>":52:1)
        br ^exit 
    ^exit:
        return loc("<stdin>":52:1)
    } loc("<stdin>":52:1)
    func @"__isq__global_initialize"() 
    {
    ^block1:
        return 
    } 
    func @"__isq__global_finalize"() 
    {
    ^block1:
        return 
    } 
    func @"__isq__entry"() 
    {
    ^block1:
        call @"__isq__global_initialize"() : ()->() 
        call @"__isq__main"() : ()->() 
        call @"__isq__global_finalize"() : ()->() 
        return 
    } 
}
