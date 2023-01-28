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
        %ssa_21 = arith.constant 1.0 : f64 loc("<stdin>":3:8)
        %ssa_20 = isq.use @"Rx"(%ssa_21) : (f64) -> !isq.gate<1> loc("<stdin>":3:5) 
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
        %ssa_24 = arith.constant 0 : i1 loc("<stdin>":6:1)
        %ssa_25_real = memref.alloc() : memref<1xi1> loc("<stdin>":6:1)
        %ssa_25_zero = arith.constant 0 : index
        %ssa_25 = memref.subview %ssa_25_real[%ssa_25_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":6:1)
        %ssa_24_zero = arith.constant 0: index loc("<stdin>":6:1)
        memref.store %ssa_24, %ssa_25[%ssa_24_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":6:1)
        %ssa_26 = memref.alloc() : memref<8xindex> loc("<stdin>":8:9)
        %ssa_27_real = memref.alloc() : memref<1x!isq.qstate> loc("<stdin>":9:9)
        %ssa_27_zero = arith.constant 0 : index
        %ssa_27 = memref.subview %ssa_27_real[%ssa_27_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":9:9)
        %ssa_28_real = memref.alloc() : memref<1x!isq.qstate> loc("<stdin>":9:9)
        %ssa_28_zero = arith.constant 0 : index
        %ssa_28 = memref.subview %ssa_28_real[%ssa_28_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":9:9)
        %ssa_29_real = memref.alloc() : memref<1x!isq.qstate> loc("<stdin>":9:9)
        %ssa_29_zero = arith.constant 0 : index
        %ssa_29 = memref.subview %ssa_29_real[%ssa_29_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":9:9)
        %ssa_30 = arith.constant 0 : i1 loc("<stdin>":10:9)
        %ssa_31_real = memref.alloc() : memref<1xi1> loc("<stdin>":10:9)
        %ssa_31_zero = arith.constant 0 : index
        %ssa_31 = memref.subview %ssa_31_real[%ssa_31_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":10:9)
        %ssa_30_zero = arith.constant 0: index loc("<stdin>":10:9)
        memref.store %ssa_30, %ssa_31[%ssa_30_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":10:9)
        %ssa_32 = arith.constant 0 : index loc("<stdin>":10:18)
        %ssa_33 = arith.constant 8 : index loc("<stdin>":10:20)
        affine.for %ssa_36 = %ssa_32 to %ssa_33 step 1 {
            scf.execute_region {
            ^entry:
                %ssa_39 = memref.subview %ssa_26[%ssa_36][1][1] : memref<8xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":11:18)
                %ssa_40 = arith.constant 0 : index loc("<stdin>":11:22)
                %ssa_40_zero = arith.constant 0: index loc("<stdin>":11:21)
                memref.store %ssa_40, %ssa_39[%ssa_40_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":11:21)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":10:9)
            } loc("<stdin>":10:9)
        } loc("<stdin>":10:9)
        %ssa_42_zero = arith.constant 0: index loc("<stdin>":10:9)
        %ssa_42 = memref.load %ssa_25[%ssa_42_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":10:9)
        cond_br %ssa_42, ^exit_6, ^block1 loc("<stdin>":10:9)
    ^block1:
        %ssa_43 = arith.constant 0 : i1 loc("<stdin>":14:9)
        %ssa_44_real = memref.alloc() : memref<1xi1> loc("<stdin>":14:9)
        %ssa_44_zero = arith.constant 0 : index
        %ssa_44 = memref.subview %ssa_44_real[%ssa_44_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":14:9)
        %ssa_43_zero = arith.constant 0: index loc("<stdin>":14:9)
        memref.store %ssa_43, %ssa_44[%ssa_43_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":14:9)
        %ssa_45 = arith.constant 0 : index loc("<stdin>":14:18)
        %ssa_46 = arith.constant 10000 : index loc("<stdin>":14:20)
        affine.for %ssa_49 = %ssa_45 to %ssa_46 step 1 {
            scf.execute_region {
            ^entry:
                %ssa_27_in_zero = arith.constant 0: index loc("<stdin>":15:17)
                %ssa_27_in = memref.load %ssa_27[%ssa_27_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:17)
                %ssa_27_out = isq.call_qop @__isq__builtin__reset(%ssa_27_in): [1]()->() loc("<stdin>":15:17)
                %ssa_27_out_zero = arith.constant 0: index loc("<stdin>":15:17)
                memref.store %ssa_27_out, %ssa_27[%ssa_27_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:17)
                %ssa_28_in_zero = arith.constant 0: index loc("<stdin>":15:24)
                %ssa_28_in = memref.load %ssa_28[%ssa_28_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:24)
                %ssa_28_out = isq.call_qop @__isq__builtin__reset(%ssa_28_in): [1]()->() loc("<stdin>":15:24)
                %ssa_28_out_zero = arith.constant 0: index loc("<stdin>":15:24)
                memref.store %ssa_28_out, %ssa_28[%ssa_28_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:24)
                %ssa_29_in_zero = arith.constant 0: index loc("<stdin>":15:31)
                %ssa_29_in = memref.load %ssa_29[%ssa_29_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:31)
                %ssa_29_out = isq.call_qop @__isq__builtin__reset(%ssa_29_in): [1]()->() loc("<stdin>":15:31)
                %ssa_29_out_zero = arith.constant 0: index loc("<stdin>":15:31)
                memref.store %ssa_29_out, %ssa_29[%ssa_29_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:31)
                %ssa_54 = isq.use @"H" : !isq.gate<1> loc("<stdin>":15:38) 
                %ssa_54_in_1_zero = arith.constant 0: index loc("<stdin>":15:38)
                %ssa_54_in_1 = memref.load %ssa_27[%ssa_54_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:38)
                %ssa_54_out_1 = isq.apply %ssa_54(%ssa_54_in_1) : !isq.gate<1> loc("<stdin>":15:38)
                %ssa_54_out_1_zero = arith.constant 0: index loc("<stdin>":15:38)
                memref.store %ssa_54_out_1, %ssa_27[%ssa_54_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:38)
                %ssa_57 = isq.use @"H" : !isq.gate<1> loc("<stdin>":15:44) 
                %ssa_57_in_1_zero = arith.constant 0: index loc("<stdin>":15:44)
                %ssa_57_in_1 = memref.load %ssa_28[%ssa_57_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:44)
                %ssa_57_out_1 = isq.apply %ssa_57(%ssa_57_in_1) : !isq.gate<1> loc("<stdin>":15:44)
                %ssa_57_out_1_zero = arith.constant 0: index loc("<stdin>":15:44)
                memref.store %ssa_57_out_1, %ssa_28[%ssa_57_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:44)
                %ssa_59 = isq.use @"coin" : !isq.gate<1> loc("<stdin>":16:25) 
                %ssa_59_decorated = isq.decorate(%ssa_59: !isq.gate<1>) {ctrl = [true, true], adjoint = false} : !isq.gate<3> loc("<stdin>":16:25)
                %ssa_59_in_1_zero = arith.constant 0: index loc("<stdin>":16:25)
                %ssa_59_in_1 = memref.load %ssa_27[%ssa_59_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_59_in_2_zero = arith.constant 0: index loc("<stdin>":16:25)
                %ssa_59_in_2 = memref.load %ssa_28[%ssa_59_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_59_in_3_zero = arith.constant 0: index loc("<stdin>":16:25)
                %ssa_59_in_3 = memref.load %ssa_29[%ssa_59_in_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_59_out_1, %ssa_59_out_2, %ssa_59_out_3 = isq.apply %ssa_59_decorated(%ssa_59_in_1, %ssa_59_in_2, %ssa_59_in_3) : !isq.gate<3> loc("<stdin>":16:25)
                %ssa_59_out_1_zero = arith.constant 0: index loc("<stdin>":16:25)
                memref.store %ssa_59_out_1, %ssa_27[%ssa_59_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_59_out_2_zero = arith.constant 0: index loc("<stdin>":16:25)
                memref.store %ssa_59_out_2, %ssa_28[%ssa_59_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_59_out_3_zero = arith.constant 0: index loc("<stdin>":16:25)
                memref.store %ssa_59_out_3, %ssa_29[%ssa_59_out_3_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:25)
                %ssa_64_in_zero = arith.constant 0: index loc("<stdin>":17:24)
                %ssa_64_in = memref.load %ssa_27[%ssa_64_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:24)
                %ssa_64_out, %ssa_64 = isq.call_qop @__isq__builtin__measure(%ssa_64_in): [1]()->i1 loc("<stdin>":17:24)
                %ssa_64_out_zero = arith.constant 0: index loc("<stdin>":17:24)
                memref.store %ssa_64_out, %ssa_27[%ssa_64_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:24)
                %ssa_65_i2 = arith.extui %ssa_64 : i1 to i2 loc("<stdin>":17:24)
                %ssa_65 = arith.index_cast %ssa_65_i2 : i2 to index loc("<stdin>":17:24)
                %ssa_66_real = memref.alloc() : memref<1xindex> loc("<stdin>":17:17)
                %ssa_66_zero = arith.constant 0 : index
                %ssa_66 = memref.subview %ssa_66_real[%ssa_66_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:17)
                %ssa_65_zero = arith.constant 0: index loc("<stdin>":17:17)
                memref.store %ssa_65, %ssa_66[%ssa_65_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:17)
                %ssa_68_in_zero = arith.constant 0: index loc("<stdin>":17:37)
                %ssa_68_in = memref.load %ssa_28[%ssa_68_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:37)
                %ssa_68_out, %ssa_68 = isq.call_qop @__isq__builtin__measure(%ssa_68_in): [1]()->i1 loc("<stdin>":17:37)
                %ssa_68_out_zero = arith.constant 0: index loc("<stdin>":17:37)
                memref.store %ssa_68_out, %ssa_28[%ssa_68_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:37)
                %ssa_69_i2 = arith.extui %ssa_68 : i1 to i2 loc("<stdin>":17:37)
                %ssa_69 = arith.index_cast %ssa_69_i2 : i2 to index loc("<stdin>":17:37)
                %ssa_70_real = memref.alloc() : memref<1xindex> loc("<stdin>":17:30)
                %ssa_70_zero = arith.constant 0 : index
                %ssa_70 = memref.subview %ssa_70_real[%ssa_70_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:30)
                %ssa_69_zero = arith.constant 0: index loc("<stdin>":17:30)
                memref.store %ssa_69, %ssa_70[%ssa_69_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:30)
                %ssa_72_in_zero = arith.constant 0: index loc("<stdin>":18:26)
                %ssa_72_in = memref.load %ssa_29[%ssa_72_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":18:26)
                %ssa_72_out, %ssa_72 = isq.call_qop @__isq__builtin__measure(%ssa_72_in): [1]()->i1 loc("<stdin>":18:26)
                %ssa_72_out_zero = arith.constant 0: index loc("<stdin>":18:26)
                memref.store %ssa_72_out, %ssa_29[%ssa_72_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":18:26)
                %ssa_73_i2 = arith.extui %ssa_72 : i1 to i2 loc("<stdin>":18:26)
                %ssa_73 = arith.index_cast %ssa_73_i2 : i2 to index loc("<stdin>":18:26)
                %ssa_74_real = memref.alloc() : memref<1xindex> loc("<stdin>":18:17)
                %ssa_74_zero = arith.constant 0 : index
                %ssa_74 = memref.subview %ssa_74_real[%ssa_74_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":18:17)
                %ssa_73_zero = arith.constant 0: index loc("<stdin>":18:17)
                memref.store %ssa_73, %ssa_74[%ssa_73_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":18:17)
                %ssa_77_zero = arith.constant 0: index loc("<stdin>":19:28)
                %ssa_77 = memref.load %ssa_66[%ssa_77_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:28)
                %ssa_76 = arith.constant 4 : index loc("<stdin>":19:31)
                %ssa_78 = arith.muli %ssa_77, %ssa_76 : index loc("<stdin>":19:30)
                %ssa_81_zero = arith.constant 0: index loc("<stdin>":19:35)
                %ssa_81 = memref.load %ssa_70[%ssa_81_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:35)
                %ssa_80 = arith.constant 2 : index loc("<stdin>":19:38)
                %ssa_82 = arith.muli %ssa_81, %ssa_80 : index loc("<stdin>":19:37)
                %ssa_83 = arith.addi %ssa_78, %ssa_82 : index loc("<stdin>":19:33)
                %ssa_85_zero = arith.constant 0: index loc("<stdin>":19:42)
                %ssa_85 = memref.load %ssa_74[%ssa_85_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:42)
                %ssa_86 = arith.addi %ssa_83, %ssa_85 : index loc("<stdin>":19:40)
                %ssa_87_real = memref.alloc() : memref<1xindex> loc("<stdin>":19:17)
                %ssa_87_zero = arith.constant 0 : index
                %ssa_87 = memref.subview %ssa_87_real[%ssa_87_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:17)
                %ssa_86_zero = arith.constant 0: index loc("<stdin>":19:17)
                memref.store %ssa_86, %ssa_87[%ssa_86_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:17)
                %ssa_90_zero = arith.constant 0: index loc("<stdin>":20:25)
                %ssa_90 = memref.load %ssa_87[%ssa_90_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:25)
                %ssa_91 = memref.subview %ssa_26[%ssa_90][1][1] : memref<8xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:24)
                %ssa_94_zero = arith.constant 0: index loc("<stdin>":20:39)
                %ssa_94 = memref.load %ssa_87[%ssa_94_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:39)
                %ssa_95 = memref.subview %ssa_26[%ssa_94][1][1] : memref<8xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:38)
                %ssa_97_zero = arith.constant 0: index loc("<stdin>":20:38)
                %ssa_97 = memref.load %ssa_95[%ssa_97_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:38)
                %ssa_96 = arith.constant 1 : index loc("<stdin>":20:45)
                %ssa_98 = arith.addi %ssa_97, %ssa_96 : index loc("<stdin>":20:44)
                %ssa_98_zero = arith.constant 0: index loc("<stdin>":20:30)
                memref.store %ssa_98, %ssa_91[%ssa_98_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:30)
                br ^exit_4 
            ^exit_4:
                memref.dealloc %ssa_87_real : memref<1xindex> loc("<stdin>":19:17)
                br ^exit_3 
            ^exit_3:
                memref.dealloc %ssa_74_real : memref<1xindex> loc("<stdin>":18:17)
                br ^exit_2 
            ^exit_2:
                memref.dealloc %ssa_70_real : memref<1xindex> loc("<stdin>":17:30)
                br ^exit_1 
            ^exit_1:
                memref.dealloc %ssa_66_real : memref<1xindex> loc("<stdin>":17:17)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":14:9)
            } loc("<stdin>":14:9)
        } loc("<stdin>":14:9)
        %ssa_100_zero = arith.constant 0: index loc("<stdin>":14:9)
        %ssa_100 = memref.load %ssa_25[%ssa_100_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":14:9)
        cond_br %ssa_100, ^exit_7, ^block2 loc("<stdin>":14:9)
    ^block2:
        %ssa_101 = arith.constant 0 : i1 loc("<stdin>":22:9)
        %ssa_102_real = memref.alloc() : memref<1xi1> loc("<stdin>":22:9)
        %ssa_102_zero = arith.constant 0 : index
        %ssa_102 = memref.subview %ssa_102_real[%ssa_102_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":22:9)
        %ssa_101_zero = arith.constant 0: index loc("<stdin>":22:9)
        memref.store %ssa_101, %ssa_102[%ssa_101_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":22:9)
        %ssa_103 = arith.constant 0 : index loc("<stdin>":22:18)
        %ssa_104 = arith.constant 8 : index loc("<stdin>":22:20)
        affine.for %ssa_107 = %ssa_103 to %ssa_104 step 1 {
            scf.execute_region {
            ^entry:
                %ssa_110 = memref.subview %ssa_26[%ssa_107][1][1] : memref<8xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":23:24)
                %ssa_111_zero = arith.constant 0: index loc("<stdin>":23:24)
                %ssa_111 = memref.load %ssa_110[%ssa_111_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":23:24)
                isq.call_qop @__isq__builtin__print_int(%ssa_111): [0](index)->() loc("<stdin>":23:11)
                br ^exit 
            ^exit:
                scf.yield loc("<stdin>":22:9)
            } loc("<stdin>":22:9)
        } loc("<stdin>":22:9)
        %ssa_113_zero = arith.constant 0: index loc("<stdin>":22:9)
        %ssa_113 = memref.load %ssa_25[%ssa_113_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":22:9)
        cond_br %ssa_113, ^exit_8, ^block3 loc("<stdin>":22:9)
    ^block3:
        %ssa_29_in_zero = arith.constant 0: index loc("<stdin>":25:9)
        %ssa_29_in = memref.load %ssa_29[%ssa_29_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":25:9)
        %ssa_29_out = isq.call_qop @__isq__builtin__reset(%ssa_29_in): [1]()->() loc("<stdin>":25:9)
        %ssa_29_out_zero = arith.constant 0: index loc("<stdin>":25:9)
        memref.store %ssa_29_out, %ssa_29[%ssa_29_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":25:9)
        %ssa_27_in_zero = arith.constant 0: index loc("<stdin>":26:9)
        %ssa_27_in = memref.load %ssa_27[%ssa_27_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":26:9)
        %ssa_27_out = isq.call_qop @__isq__builtin__reset(%ssa_27_in): [1]()->() loc("<stdin>":26:9)
        %ssa_27_out_zero = arith.constant 0: index loc("<stdin>":26:9)
        memref.store %ssa_27_out, %ssa_27[%ssa_27_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":26:9)
        %ssa_28_in_zero = arith.constant 0: index loc("<stdin>":26:16)
        %ssa_28_in = memref.load %ssa_28[%ssa_28_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":26:16)
        %ssa_28_out = isq.call_qop @__isq__builtin__reset(%ssa_28_in): [1]()->() loc("<stdin>":26:16)
        %ssa_28_out_zero = arith.constant 0: index loc("<stdin>":26:16)
        memref.store %ssa_28_out, %ssa_28[%ssa_28_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":26:16)
        br ^exit_8 
    ^exit_8:
        memref.dealloc %ssa_102_real : memref<1xi1> loc("<stdin>":22:9)
        br ^exit_7 
    ^exit_7:
        memref.dealloc %ssa_44_real : memref<1xi1> loc("<stdin>":14:9)
        br ^exit_6 
    ^exit_6:
        memref.dealloc %ssa_31_real : memref<1xi1> loc("<stdin>":10:9)
        br ^exit_5 
    ^exit_5:
        memref.dealloc %ssa_29_real : memref<1x!isq.qstate> loc("<stdin>":9:9)
        br ^exit_4 
    ^exit_4:
        memref.dealloc %ssa_28_real : memref<1x!isq.qstate> loc("<stdin>":9:9)
        br ^exit_3 
    ^exit_3:
        memref.dealloc %ssa_27_real : memref<1x!isq.qstate> loc("<stdin>":9:9)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_26 : memref<8xindex> loc("<stdin>":8:9)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_25_real : memref<1xi1> loc("<stdin>":6:1)
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
