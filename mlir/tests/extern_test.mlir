module{
    isq.declare_qop @__isq__builtin__measure : [1]()->i1
    isq.declare_qop @__isq__builtin__reset : [1]()->()
    isq.declare_qop @__isq__builtin__print_int : [0](index)->()
    isq.declare_qop @__isq__builtin__print_double : [0](f64)->()
    isq.defgate @__isq__builtin__u3(f64, f64, f64) {definition = [{type = "qir", value = "__quantum__qis__u3"}]} : !isq.gate<1>
    isq.defgate @__isq__builtin__cnot {definition = [{type = "qir", value = "__quantum__qis__cnot"}]} : !isq.gate<2>
func private @__quantum__qis__u3(f64, f64, f64, !isq.qir.qubit)
func private @__quantum__qis__cnot(!isq.qir.qubit, !isq.qir.qubit)
    func private @"__quantum__qis__rz__body"(f64, !isq.qir.qubit) loc("<stdin>":1:1)
    isq.defgate @"Rz"(f64) {definition = [{type = "qir", value = @"__quantum__qis__rz__body"}]}: !isq.gate<1> loc("<stdin>":1:1)
    func private @"__quantum__qis__rx__body"(f64, !isq.qir.qubit) loc("<stdin>":2:1)
    isq.defgate @"Rx"(f64) {definition = [{type = "qir", value = @"__quantum__qis__rx__body"}]}: !isq.gate<1> loc("<stdin>":2:1)
    func private @"__quantum__qis__ry__body"(f64, !isq.qir.qubit) loc("<stdin>":3:1)
    isq.defgate @"Ry"(f64) {definition = [{type = "qir", value = @"__quantum__qis__ry__body"}]}: !isq.gate<1> loc("<stdin>":3:1)
    func private @"__quantum__qis__h__body"(!isq.qir.qubit) loc("<stdin>":4:1)
    isq.defgate @"H" {definition = [{type = "qir", value = @"__quantum__qis__h__body"}]}: !isq.gate<1> loc("<stdin>":4:1)
    func private @"__quantum__qis__x__body"(!isq.qir.qubit) loc("<stdin>":5:1)
    isq.defgate @"X" {definition = [{type = "qir", value = @"__quantum__qis__x__body"}]}: !isq.gate<1> loc("<stdin>":5:1)
    func private @"__quantum__qis__y__body"(!isq.qir.qubit) loc("<stdin>":6:1)
    isq.defgate @"Y" {definition = [{type = "qir", value = @"__quantum__qis__y__body"}]}: !isq.gate<1> loc("<stdin>":6:1)
    func private @"__quantum__qis__z__body"(!isq.qir.qubit) loc("<stdin>":7:1)
    isq.defgate @"Z" {definition = [{type = "qir", value = @"__quantum__qis__z__body"}]}: !isq.gate<1> loc("<stdin>":7:1)
    isq.defgate @"CNOT" {definition = [{type="unitary", value = [[#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>],[#isq.complex<0.0, 0.0>,#isq.complex<0.0, 0.0>,#isq.complex<1.0, 0.0>,#isq.complex<0.0, 0.0>]]}, {type = "qir", value = @"__quantum__qis__cnot"}]}: !isq.gate<2> loc("<stdin>":8:1)
    func @"$_ISQ_GATEDEF_CZ"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>) 
    {
    ^entry(%ssa_12: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_13: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>):
        %ssa_14 = arith.constant 0 : i1 loc("<stdin>":9:1)
        %ssa_15_real = memref.alloc() : memref<1xi1> loc("<stdin>":9:1)
        %ssa_15_zero = arith.constant 0 : index
        %ssa_15 = memref.subview %ssa_15_real[%ssa_15_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":9:1)
        %ssa_14_zero = arith.constant 0: index loc("<stdin>":9:1)
        memref.store %ssa_14, %ssa_15[%ssa_14_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":9:1)
        %ssa_17 = isq.use @"H" : !isq.gate<1> loc("<stdin>":10:5) 
        %ssa_17_in_1_zero = arith.constant 0: index loc("<stdin>":10:5)
        %ssa_17_in_1 = memref.load %ssa_13[%ssa_17_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":10:5)
        %ssa_17_out_1 = isq.apply %ssa_17(%ssa_17_in_1) : !isq.gate<1> loc("<stdin>":10:5)
        %ssa_17_out_1_zero = arith.constant 0: index loc("<stdin>":10:5)
        memref.store %ssa_17_out_1, %ssa_13[%ssa_17_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":10:5)
        %ssa_20 = isq.use @"CNOT" : !isq.gate<2> loc("<stdin>":11:5) 
        %ssa_20_in_1_zero = arith.constant 0: index loc("<stdin>":11:5)
        %ssa_20_in_1 = memref.load %ssa_12[%ssa_20_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":11:5)
        %ssa_20_in_2_zero = arith.constant 0: index loc("<stdin>":11:5)
        %ssa_20_in_2 = memref.load %ssa_13[%ssa_20_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":11:5)
        %ssa_20_out_1, %ssa_20_out_2 = isq.apply %ssa_20(%ssa_20_in_1, %ssa_20_in_2) : !isq.gate<2> loc("<stdin>":11:5)
        %ssa_20_out_1_zero = arith.constant 0: index loc("<stdin>":11:5)
        memref.store %ssa_20_out_1, %ssa_12[%ssa_20_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":11:5)
        %ssa_20_out_2_zero = arith.constant 0: index loc("<stdin>":11:5)
        memref.store %ssa_20_out_2, %ssa_13[%ssa_20_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":11:5)
        %ssa_24 = isq.use @"H" : !isq.gate<1> loc("<stdin>":12:5) 
        %ssa_24_in_1_zero = arith.constant 0: index loc("<stdin>":12:5)
        %ssa_24_in_1 = memref.load %ssa_13[%ssa_24_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":12:5)
        %ssa_24_out_1 = isq.apply %ssa_24(%ssa_24_in_1) : !isq.gate<1> loc("<stdin>":12:5)
        %ssa_24_out_1_zero = arith.constant 0: index loc("<stdin>":12:5)
        memref.store %ssa_24_out_1, %ssa_13[%ssa_24_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":12:5)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_15_real : memref<1xi1> loc("<stdin>":9:1)
        br ^exit 
    ^exit:
        return loc("<stdin>":9:1)
    } loc("<stdin>":9:1)
    isq.defgate @"CZ" {definition = [{type = "decomposition_raw", value = @"$_ISQ_GATEDEF_CZ"}]}: !isq.gate<2> loc("<stdin>":9:1)
    func @"__isq__main"() 
    {
    ^entry:
        %ssa_27 = arith.constant 0 : i1 loc("<stdin>":15:1)
        %ssa_28_real = memref.alloc() : memref<1xi1> loc("<stdin>":15:1)
        %ssa_28_zero = arith.constant 0 : index
        %ssa_28 = memref.subview %ssa_28_real[%ssa_28_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:1)
        %ssa_27_zero = arith.constant 0: index loc("<stdin>":15:1)
        memref.store %ssa_27, %ssa_28[%ssa_27_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":15:1)
        %ssa_29_real = memref.alloc() : memref<1x!isq.qstate> loc("<stdin>":16:5)
        %ssa_29_zero = arith.constant 0 : index
        %ssa_29 = memref.subview %ssa_29_real[%ssa_29_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:5)
        %ssa_30_real = memref.alloc() : memref<1x!isq.qstate> loc("<stdin>":16:5)
        %ssa_30_zero = arith.constant 0 : index
        %ssa_30 = memref.subview %ssa_30_real[%ssa_30_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":16:5)
        %ssa_32 = isq.use @"H" : !isq.gate<1> loc("<stdin>":17:5) 
        %ssa_32_in_1_zero = arith.constant 0: index loc("<stdin>":17:5)
        %ssa_32_in_1 = memref.load %ssa_29[%ssa_32_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:5)
        %ssa_32_out_1 = isq.apply %ssa_32(%ssa_32_in_1) : !isq.gate<1> loc("<stdin>":17:5)
        %ssa_32_out_1_zero = arith.constant 0: index loc("<stdin>":17:5)
        memref.store %ssa_32_out_1, %ssa_29[%ssa_32_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":17:5)
        %ssa_36 = arith.constant 1.57 : f64 loc("<stdin>":18:8)
        %ssa_35 = isq.use @"Rx"(%ssa_36) : (f64) -> !isq.gate<1> loc("<stdin>":18:5) 
        %ssa_35_in_1_zero = arith.constant 0: index loc("<stdin>":18:5)
        %ssa_35_in_1 = memref.load %ssa_30[%ssa_35_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":18:5)
        %ssa_35_out_1 = isq.apply %ssa_35(%ssa_35_in_1) : !isq.gate<1> loc("<stdin>":18:5)
        %ssa_35_out_1_zero = arith.constant 0: index loc("<stdin>":18:5)
        memref.store %ssa_35_out_1, %ssa_30[%ssa_35_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":18:5)
        %ssa_39 = isq.use @"CZ" : !isq.gate<2> loc("<stdin>":19:5) 
        %ssa_39_in_1_zero = arith.constant 0: index loc("<stdin>":19:5)
        %ssa_39_in_1 = memref.load %ssa_29[%ssa_39_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:5)
        %ssa_39_in_2_zero = arith.constant 0: index loc("<stdin>":19:5)
        %ssa_39_in_2 = memref.load %ssa_30[%ssa_39_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:5)
        %ssa_39_out_1, %ssa_39_out_2 = isq.apply %ssa_39(%ssa_39_in_1, %ssa_39_in_2) : !isq.gate<2> loc("<stdin>":19:5)
        %ssa_39_out_1_zero = arith.constant 0: index loc("<stdin>":19:5)
        memref.store %ssa_39_out_1, %ssa_29[%ssa_39_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:5)
        %ssa_39_out_2_zero = arith.constant 0: index loc("<stdin>":19:5)
        memref.store %ssa_39_out_2, %ssa_30[%ssa_39_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":19:5)
        %ssa_43 = arith.constant 0.5 : f64 loc("<stdin>":20:16)
        %ssa_42 = isq.use @"Rz"(%ssa_43) : (f64) -> !isq.gate<1> loc("<stdin>":20:13) 
        %ssa_42_decorated = isq.decorate(%ssa_42: !isq.gate<1>) {ctrl = [true], adjoint = false} : !isq.gate<2> loc("<stdin>":20:13)
        %ssa_42_in_1_zero = arith.constant 0: index loc("<stdin>":20:13)
        %ssa_42_in_1 = memref.load %ssa_29[%ssa_42_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:13)
        %ssa_42_in_2_zero = arith.constant 0: index loc("<stdin>":20:13)
        %ssa_42_in_2 = memref.load %ssa_30[%ssa_42_in_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:13)
        %ssa_42_out_1, %ssa_42_out_2 = isq.apply %ssa_42_decorated(%ssa_42_in_1, %ssa_42_in_2) : !isq.gate<2> loc("<stdin>":20:13)
        %ssa_42_out_1_zero = arith.constant 0: index loc("<stdin>":20:13)
        memref.store %ssa_42_out_1, %ssa_29[%ssa_42_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:13)
        %ssa_42_out_2_zero = arith.constant 0: index loc("<stdin>":20:13)
        memref.store %ssa_42_out_2, %ssa_30[%ssa_42_out_2_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":20:13)
        br ^exit_3 loc("<stdin>":21:5)
    ^block1:
        br ^exit_3 
    ^exit_3:
        memref.dealloc %ssa_30_real : memref<1x!isq.qstate> loc("<stdin>":16:5)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_29_real : memref<1x!isq.qstate> loc("<stdin>":16:5)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_28_real : memref<1xi1> loc("<stdin>":15:1)
        br ^exit 
    ^exit:
        return loc("<stdin>":15:1)
    } loc("<stdin>":15:1)
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
