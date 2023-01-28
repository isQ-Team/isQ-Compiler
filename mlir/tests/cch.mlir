module{
    isq.declare_qop @__isq__builtin__measure : [1]()->i1
    isq.declare_qop @__isq__builtin__reset : [1]()->()
    isq.declare_qop @__isq__builtin__print_int : [0](index)->()
    isq.declare_qop @__isq__builtin__print_double : [0](f64)->()
    func private @"__quantum__qis__rz__body"(f64, !isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"Rz"(f64) {definition = [{type = "qir", value = @"__quantum__qis__rz__body"}]}: !isq.gate<1> loc("<stdin>":0:0)
    func private @"__quantum__qis__rx__body"(f64, !isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"Rx"(f64) {definition = [{type = "qir", value = @"__quantum__qis__rx__body"}]}: !isq.gate<1> loc("<stdin>":0:0)
    func private @"__quantum__qis__ry__body"(f64, !isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"Ry"(f64) {definition = [{type = "qir", value = @"__quantum__qis__ry__body"}]}: !isq.gate<1> loc("<stdin>":0:0)
    func private @"__quantum__qis__u3"(f64, f64, f64, !isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"U3"(f64, f64, f64) {definition = [{type = "qir", value = @"__quantum__qis__u3"}]}: !isq.gate<1> loc("<stdin>":0:0)
    func private @"__quantum__qis__h__body"(!isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"H" {definition = [{type = "qir", value = @"__quantum__qis__h__body"}]}: !isq.gate<1> loc("<stdin>":0:0)
    func private @"__quantum__qis__s__body"(!isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"S" {definition = [{type = "qir", value = @"__quantum__qis__s__body"}]}: !isq.gate<1> loc("<stdin>":0:0)
    func private @"__quantum__qis__t__body"(!isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"T" {definition = [{type = "qir", value = @"__quantum__qis__t__body"}]}: !isq.gate<1> loc("<stdin>":0:0)
    func private @"__quantum__qis__x__body"(!isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"X" {definition = [{type = "qir", value = @"__quantum__qis__x__body"}]}: !isq.gate<1> loc("<stdin>":0:0)
    func private @"__quantum__qis__y__body"(!isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"Y" {definition = [{type = "qir", value = @"__quantum__qis__y__body"}]}: !isq.gate<1> loc("<stdin>":0:0)
    func private @"__quantum__qis__z__body"(!isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"Z" {definition = [{type = "qir", value = @"__quantum__qis__z__body"}]}: !isq.gate<1> loc("<stdin>":0:0)
    func private @"__quantum__qis__cnot"(!isq.qir.qubit, !isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"CNOT" {definition = [{type = "qir", value = @"__quantum__qis__cnot"}]}: !isq.gate<2> loc("<stdin>":0:0)
    func private @"__quantum__qis__toffoli"(!isq.qir.qubit, !isq.qir.qubit, !isq.qir.qubit) loc("<stdin>":0:0)
    isq.defgate @"Toffoli" {definition = [{type = "qir", value = @"__quantum__qis__toffoli"}]}: !isq.gate<3> loc("<stdin>":0:0)
    func @"$_ISQ_GATEDEF_coin"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>) 
    {
    ^entry(%ssa_16: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>):
        %ssa_17 = arith.constant 0 : i1 loc("<stdin>":1:1)
        %ssa_18_real = memref.alloc() : memref<1xi1> loc("<stdin>":1:1)
        %ssa_18_zero = arith.constant 0 : index
        %ssa_18 = memref.subview %ssa_18_real[%ssa_18_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":1:1)
        %ssa_17_zero = arith.constant 0: index loc("<stdin>":1:1)
        memref.store %ssa_17, %ssa_18[%ssa_17_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":1:1)
        %ssa_20 = isq.use @"H" : !isq.gate<1> loc("<stdin>":3:5) 
        %ssa_20_in_1_zero = arith.constant 0: index loc("<stdin>":3:5)
        %ssa_20_in_1 = memref.load %ssa_16[%ssa_20_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":3:5)
        %ssa_20_out_1 = isq.apply %ssa_20(%ssa_20_in_1) : !isq.gate<1> loc("<stdin>":3:5)
        %ssa_20_out_1_zero = arith.constant 0: index loc("<stdin>":3:5)
        memref.store %ssa_20_out_1, %ssa_16[%ssa_20_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":3:5)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_18_real : memref<1xi1> loc("<stdin>":1:1)
        br ^exit 
    ^exit:
        return loc("<stdin>":1:1)
    } loc("<stdin>":1:1)
    isq.defgate @"coin" {definition = [{type = "decomposition_raw", value = @"$_ISQ_GATEDEF_coin"}]}: !isq.gate<1> loc("<stdin>":1:1)
    func @"__isq__main"() 
    {
    ^entry:
        %ssa_23 = arith.constant 0 : i1 loc("<stdin>":6:1)
        %ssa_24_real = memref.alloc() : memref<1xi1> loc("<stdin>":6:1)
        %ssa_24_zero = arith.constant 0 : index
        %ssa_24 = memref.subview %ssa_24_real[%ssa_24_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":6:1)
        %ssa_23_zero = arith.constant 0: index loc("<stdin>":6:1)
        memref.store %ssa_23, %ssa_24[%ssa_23_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":6:1)
        %ssa_25 = memref.alloc() : memref<8xindex> loc("<stdin>":8:9)
        %ssa_26_real = memref.alloc() : memref<1x!isq.qstate> loc("<stdin>":9:9)
        %ssa_26_zero = arith.constant 0 : index
        %ssa_26 = memref.subview %ssa_26_real[%ssa_26_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":9:9)
        %ssa_27_real = memref.alloc() : memref<1x!isq.qstate> loc("<stdin>":9:9)
        %ssa_27_zero = arith.constant 0 : index
        %ssa_27 = memref.subview %ssa_27_real[%ssa_27_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":9:9)
        %ssa_28_real = memref.alloc() : memref<1x!isq.qstate> loc("<stdin>":9:9)
        %ssa_28_zero = arith.constant 0 : index
        %ssa_28 = memref.subview %ssa_28_real[%ssa_28_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":9:9)
        %ssa_29 = arith.constant 0 : i1 loc("<stdin>":10:9)
        %ssa_30_real = memref.alloc() : memref<1xi1> loc("<stdin>":10:9)
        %ssa_30_zero = arith.constant 0 : index
        %ssa_30 = memref.subview %ssa_30_real[%ssa_30_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":10:9)
        %ssa_29_zero = arith.constant 0: index loc("<stdin>":10:9)
        memref.store %ssa_29, %ssa_30[%ssa_29_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":10:9)
        %ssa_31 = arith.constant 0 : index loc("<stdin>":10:18)
        %ssa_32 = arith.constant 8 : index loc("<stdin>":10:20)
        affine.for %ssa_35 = %ssa_31 to %ssa_32 step 1 {
            scf.execute_region {
            ^entry:
                %ssa_38 = memref.subview %ssa_25[%ssa_35][1][1] : memref<8xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":11:18)
                %ssa_39 = arith.constant 0 : index loc("<stdin>":11:22)
                %ssa_39_zero = arith.constant 0: index loc("<stdin>":11:21)
                memref.store %ssa_39, %ssa_38[%ssa_39_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":11:21)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":10:9)
            } loc("<stdin>":10:9)
        } loc("<stdin>":10:9)
        %ssa_41_zero = arith.constant 0: index loc("<stdin>":10:9)
        %ssa_41 = memref.load %ssa_24[%ssa_41_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":10:9)
        cond_br %ssa_41, ^exit_6, ^block1 loc("<stdin>":10:9)
    ^block1:
        %ssa_42 = arith.constant 0 : i1 loc("<stdin>":14:9)
        %ssa_43_real = memref.alloc() : memref<1xi1> loc("<stdin>":14:9)
        %ssa_43_zero = arith.constant 0 : index
        %ssa_43 = memref.subview %ssa_43_real[%ssa_43_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":14:9)
        %ssa_42_zero = arith.constant 0: index loc("<stdin>":14:9)
        memref.store %ssa_42, %ssa_43[%ssa_42_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":14:9)
        %ssa_44 = arith.constant 0 : index loc("<stdin>":14:18)
        %ssa_45 = arith.constant 10000 : index loc("<stdin>":14:20)
        affine.for %ssa_48 = %ssa_44 to %ssa_45 step 1 {
            scf.execute_region {
            ^entry:
                %ssa_26_in_zero = arith.constant 0: index loc("<stdin>":15:17)
                %ssa_26_in = memref.load %ssa_26[%ssa_26_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:17)
                %ssa_26_out = isq.call_qop @__isq__builtin__reset(%ssa_26_in): [1]()->() loc("<stdin>":15:17)
                %ssa_26_out_zero = arith.constant 0: index loc("<stdin>":15:17)
                memref.store %ssa_26_out, %ssa_26[%ssa_26_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:17)
                %ssa_27_in_zero = arith.constant 0: index loc("<stdin>":15:24)
                %ssa_27_in = memref.load %ssa_27[%ssa_27_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:24)
                %ssa_27_out = isq.call_qop @__isq__builtin__reset(%ssa_27_in): [1]()->() loc("<stdin>":15:24)
                %ssa_27_out_zero = arith.constant 0: index loc("<stdin>":15:24)
                memref.store %ssa_27_out, %ssa_27[%ssa_27_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:24)
                %ssa_28_in_zero = arith.constant 0: index loc("<stdin>":15:31)
                %ssa_28_in = memref.load %ssa_28[%ssa_28_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:31)
                %ssa_28_out = isq.call_qop @__isq__builtin__reset(%ssa_28_in): [1]()->() loc("<stdin>":15:31)
                %ssa_28_out_zero = arith.constant 0: index loc("<stdin>":15:31)
                memref.store %ssa_28_out, %ssa_28[%ssa_28_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:31)
                %ssa_53 = isq.use @"H" : !isq.gate<1> loc("<stdin>":15:38) 
                %ssa_53_in_1_zero = arith.constant 0: index loc("<stdin>":15:38)
                %ssa_53_in_1 = memref.load %ssa_26[%ssa_53_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:38)
                %ssa_53_out_1 = isq.apply %ssa_53(%ssa_53_in_1) : !isq.gate<1> loc("<stdin>":15:38)
                %ssa_53_out_1_zero = arith.constant 0: index loc("<stdin>":15:38)
                memref.store %ssa_53_out_1, %ssa_26[%ssa_53_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:38)
                %ssa_56 = isq.use @"H" : !isq.gate<1> loc("<stdin>":15:44) 
                %ssa_56_in_1_zero = arith.constant 0: index loc("<stdin>":15:44)
                %ssa_56_in_1 = memref.load %ssa_27[%ssa_56_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:44)
                %ssa_56_out_1 = isq.apply %ssa_56(%ssa_56_in_1) : !isq.gate<1> loc("<stdin>":15:44)
                %ssa_56_out_1_zero = arith.constant 0: index loc("<stdin>":15:44)
                memref.store %ssa_56_out_1, %ssa_27[%ssa_56_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:44)
                %ssa_58 = isq.use @"coin" : !isq.gate<1> loc("<stdin>":16:25) 
                %ssa_58_decorated = isq.decorate(%ssa_58: !isq.gate<1>) {ctrl = [true, true], adjoint = false} : !isq.gate<3> loc("<stdin>":16:25)
                %ssa_58_in_1_zero = arith.constant 0: index loc("<stdin>":16:25)
                %ssa_58_in_1 = memref.load %ssa_26[%ssa_58_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_58_in_2_zero = arith.constant 0: index loc("<stdin>":16:25)
                %ssa_58_in_2 = memref.load %ssa_27[%ssa_58_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_58_in_3_zero = arith.constant 0: index loc("<stdin>":16:25)
                %ssa_58_in_3 = memref.load %ssa_28[%ssa_58_in_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_58_out_1, %ssa_58_out_2, %ssa_58_out_3 = isq.apply %ssa_58_decorated(%ssa_58_in_1, %ssa_58_in_2, %ssa_58_in_3) : !isq.gate<3> loc("<stdin>":16:25)
                %ssa_58_out_1_zero = arith.constant 0: index loc("<stdin>":16:25)
                memref.store %ssa_58_out_1, %ssa_26[%ssa_58_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_58_out_2_zero = arith.constant 0: index loc("<stdin>":16:25)
                memref.store %ssa_58_out_2, %ssa_27[%ssa_58_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_58_out_3_zero = arith.constant 0: index loc("<stdin>":16:25)
                memref.store %ssa_58_out_3, %ssa_28[%ssa_58_out_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_63_in_zero = arith.constant 0: index loc("<stdin>":17:24)
                %ssa_63_in = memref.load %ssa_26[%ssa_63_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:24)
                %ssa_63_out, %ssa_63 = isq.call_qop @__isq__builtin__measure(%ssa_63_in): [1]()->i1 loc("<stdin>":17:24)
                %ssa_63_out_zero = arith.constant 0: index loc("<stdin>":17:24)
                memref.store %ssa_63_out, %ssa_26[%ssa_63_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:24)
                %ssa_64_i2 = arith.extui %ssa_63 : i1 to i2 loc("<stdin>":17:24)
                %ssa_64 = arith.index_cast %ssa_64_i2 : i2 to index loc("<stdin>":17:24)
                %ssa_65_real = memref.alloc() : memref<1xindex> loc("<stdin>":17:17)
                %ssa_65_zero = arith.constant 0 : index
                %ssa_65 = memref.subview %ssa_65_real[%ssa_65_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:17)
                %ssa_64_zero = arith.constant 0: index loc("<stdin>":17:17)
                memref.store %ssa_64, %ssa_65[%ssa_64_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:17)
                %ssa_67_in_zero = arith.constant 0: index loc("<stdin>":17:37)
                %ssa_67_in = memref.load %ssa_27[%ssa_67_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:37)
                %ssa_67_out, %ssa_67 = isq.call_qop @__isq__builtin__measure(%ssa_67_in): [1]()->i1 loc("<stdin>":17:37)
                %ssa_67_out_zero = arith.constant 0: index loc("<stdin>":17:37)
                memref.store %ssa_67_out, %ssa_27[%ssa_67_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:37)
                %ssa_68_i2 = arith.extui %ssa_67 : i1 to i2 loc("<stdin>":17:37)
                %ssa_68 = arith.index_cast %ssa_68_i2 : i2 to index loc("<stdin>":17:37)
                %ssa_69_real = memref.alloc() : memref<1xindex> loc("<stdin>":17:30)
                %ssa_69_zero = arith.constant 0 : index
                %ssa_69 = memref.subview %ssa_69_real[%ssa_69_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:30)
                %ssa_68_zero = arith.constant 0: index loc("<stdin>":17:30)
                memref.store %ssa_68, %ssa_69[%ssa_68_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:30)
                %ssa_71_in_zero = arith.constant 0: index loc("<stdin>":18:26)
                %ssa_71_in = memref.load %ssa_28[%ssa_71_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":18:26)
                %ssa_71_out, %ssa_71 = isq.call_qop @__isq__builtin__measure(%ssa_71_in): [1]()->i1 loc("<stdin>":18:26)
                %ssa_71_out_zero = arith.constant 0: index loc("<stdin>":18:26)
                memref.store %ssa_71_out, %ssa_28[%ssa_71_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":18:26)
                %ssa_72_i2 = arith.extui %ssa_71 : i1 to i2 loc("<stdin>":18:26)
                %ssa_72 = arith.index_cast %ssa_72_i2 : i2 to index loc("<stdin>":18:26)
                %ssa_73_real = memref.alloc() : memref<1xindex> loc("<stdin>":18:17)
                %ssa_73_zero = arith.constant 0 : index
                %ssa_73 = memref.subview %ssa_73_real[%ssa_73_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":18:17)
                %ssa_72_zero = arith.constant 0: index loc("<stdin>":18:17)
                memref.store %ssa_72, %ssa_73[%ssa_72_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":18:17)
                %ssa_76_zero = arith.constant 0: index loc("<stdin>":19:28)
                %ssa_76 = memref.load %ssa_65[%ssa_76_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:28)
                %ssa_75 = arith.constant 4 : index loc("<stdin>":19:31)
                %ssa_77 = arith.muli %ssa_76, %ssa_75 : index loc("<stdin>":19:30)
                %ssa_80_zero = arith.constant 0: index loc("<stdin>":19:35)
                %ssa_80 = memref.load %ssa_69[%ssa_80_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:35)
                %ssa_79 = arith.constant 2 : index loc("<stdin>":19:38)
                %ssa_81 = arith.muli %ssa_80, %ssa_79 : index loc("<stdin>":19:37)
                %ssa_82 = arith.addi %ssa_77, %ssa_81 : index loc("<stdin>":19:33)
                %ssa_84_zero = arith.constant 0: index loc("<stdin>":19:42)
                %ssa_84 = memref.load %ssa_73[%ssa_84_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:42)
                %ssa_85 = arith.addi %ssa_82, %ssa_84 : index loc("<stdin>":19:40)
                %ssa_86_real = memref.alloc() : memref<1xindex> loc("<stdin>":19:17)
                %ssa_86_zero = arith.constant 0 : index
                %ssa_86 = memref.subview %ssa_86_real[%ssa_86_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:17)
                %ssa_85_zero = arith.constant 0: index loc("<stdin>":19:17)
                memref.store %ssa_85, %ssa_86[%ssa_85_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:17)
                %ssa_89_zero = arith.constant 0: index loc("<stdin>":20:25)
                %ssa_89 = memref.load %ssa_86[%ssa_89_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:25)
                %ssa_90 = memref.subview %ssa_25[%ssa_89][1][1] : memref<8xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:24)
                %ssa_93_zero = arith.constant 0: index loc("<stdin>":20:39)
                %ssa_93 = memref.load %ssa_86[%ssa_93_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:39)
                %ssa_94 = memref.subview %ssa_25[%ssa_93][1][1] : memref<8xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:38)
                %ssa_96_zero = arith.constant 0: index loc("<stdin>":20:38)
                %ssa_96 = memref.load %ssa_94[%ssa_96_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:38)
                %ssa_95 = arith.constant 1 : index loc("<stdin>":20:45)
                %ssa_97 = arith.addi %ssa_96, %ssa_95 : index loc("<stdin>":20:44)
                %ssa_97_zero = arith.constant 0: index loc("<stdin>":20:30)
                memref.store %ssa_97, %ssa_90[%ssa_97_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:30)
                br ^exit_4 
            ^exit_4:
                memref.dealloc %ssa_86_real : memref<1xindex> loc("<stdin>":19:17)
                br ^exit_3 
            ^exit_3:
                memref.dealloc %ssa_73_real : memref<1xindex> loc("<stdin>":18:17)
                br ^exit_2 
            ^exit_2:
                memref.dealloc %ssa_69_real : memref<1xindex> loc("<stdin>":17:30)
                br ^exit_1 
            ^exit_1:
                memref.dealloc %ssa_65_real : memref<1xindex> loc("<stdin>":17:17)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":14:9)
            } loc("<stdin>":14:9)
        } loc("<stdin>":14:9)
        %ssa_99_zero = arith.constant 0: index loc("<stdin>":14:9)
        %ssa_99 = memref.load %ssa_24[%ssa_99_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":14:9)
        cond_br %ssa_99, ^exit_7, ^block2 loc("<stdin>":14:9)
    ^block2:
        %ssa_100 = arith.constant 0 : i1 loc("<stdin>":22:9)
        %ssa_101_real = memref.alloc() : memref<1xi1> loc("<stdin>":22:9)
        %ssa_101_zero = arith.constant 0 : index
        %ssa_101 = memref.subview %ssa_101_real[%ssa_101_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":22:9)
        %ssa_100_zero = arith.constant 0: index loc("<stdin>":22:9)
        memref.store %ssa_100, %ssa_101[%ssa_100_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":22:9)
        %ssa_102 = arith.constant 0 : index loc("<stdin>":22:18)
        %ssa_103 = arith.constant 8 : index loc("<stdin>":22:20)
        affine.for %ssa_106 = %ssa_102 to %ssa_103 step 1 {
            scf.execute_region {
            ^entry:
                %ssa_109 = memref.subview %ssa_25[%ssa_106][1][1] : memref<8xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":23:24)
                %ssa_110_zero = arith.constant 0: index loc("<stdin>":23:24)
                %ssa_110 = memref.load %ssa_109[%ssa_110_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":23:24)
                isq.call_qop @__isq__builtin__print_int(%ssa_110): [0](index)->() loc("<stdin>":23:11)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":22:9)
            } loc("<stdin>":22:9)
        } loc("<stdin>":22:9)
        %ssa_112_zero = arith.constant 0: index loc("<stdin>":22:9)
        %ssa_112 = memref.load %ssa_24[%ssa_112_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":22:9)
        cond_br %ssa_112, ^exit_8, ^block3 loc("<stdin>":22:9)
    ^block3:
        %ssa_28_in_zero = arith.constant 0: index loc("<stdin>":25:9)
        %ssa_28_in = memref.load %ssa_28[%ssa_28_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":25:9)
        %ssa_28_out = isq.call_qop @__isq__builtin__reset(%ssa_28_in): [1]()->() loc("<stdin>":25:9)
        %ssa_28_out_zero = arith.constant 0: index loc("<stdin>":25:9)
        memref.store %ssa_28_out, %ssa_28[%ssa_28_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":25:9)
        %ssa_26_in_zero = arith.constant 0: index loc("<stdin>":26:9)
        %ssa_26_in = memref.load %ssa_26[%ssa_26_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":26:9)
        %ssa_26_out = isq.call_qop @__isq__builtin__reset(%ssa_26_in): [1]()->() loc("<stdin>":26:9)
        %ssa_26_out_zero = arith.constant 0: index loc("<stdin>":26:9)
        memref.store %ssa_26_out, %ssa_26[%ssa_26_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":26:9)
        %ssa_27_in_zero = arith.constant 0: index loc("<stdin>":26:16)
        %ssa_27_in = memref.load %ssa_27[%ssa_27_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":26:16)
        %ssa_27_out = isq.call_qop @__isq__builtin__reset(%ssa_27_in): [1]()->() loc("<stdin>":26:16)
        %ssa_27_out_zero = arith.constant 0: index loc("<stdin>":26:16)
        memref.store %ssa_27_out, %ssa_27[%ssa_27_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":26:16)
        br ^exit_8 
    ^exit_8:
        memref.dealloc %ssa_101_real : memref<1xi1> loc("<stdin>":22:9)
        br ^exit_7 
    ^exit_7:
        memref.dealloc %ssa_43_real : memref<1xi1> loc("<stdin>":14:9)
        br ^exit_6 
    ^exit_6:
        memref.dealloc %ssa_30_real : memref<1xi1> loc("<stdin>":10:9)
        br ^exit_5 
    ^exit_5:
        memref.dealloc %ssa_28_real : memref<1x!isq.qstate> loc("<stdin>":9:9)
        br ^exit_4 
    ^exit_4:
        memref.dealloc %ssa_27_real : memref<1x!isq.qstate> loc("<stdin>":9:9)
        br ^exit_3 
    ^exit_3:
        memref.dealloc %ssa_26_real : memref<1x!isq.qstate> loc("<stdin>":9:9)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_25 : memref<8xindex> loc("<stdin>":8:9)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_24_real : memref<1xi1> loc("<stdin>":6:1)
        br ^exit 
    ^exit:
        return loc("<stdin>":6:1)
    } loc("<stdin>":6:1)
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
