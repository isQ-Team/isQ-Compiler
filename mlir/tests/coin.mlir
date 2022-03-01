module{
isq.declare_qop @__isq__builtin__measure : [1]()->i1
isq.declare_qop @__isq__builtin__reset : [1]()->()
isq.declare_qop @__isq__builtin__print_int : [0](index)->()
isq.declare_qop @__isq__builtin__print_double : [0](f64)->()
isq.defgate @__isq__builtin__u3(f64, f64, f64) {definition = [{type = "qir", value = "__quantum__qis__u3"}]} : !isq.gate<1>
isq.defgate @__isq__builtin__cnot {definition = [{type = "qir", value = "__quantum__qis__cnot"}]} : !isq.gate<2>
func private @__quantum__qis__u3(f64, f64, f64, !isq.qir.qubit)
func private @__quantum__qis__cnot(!isq.qir.qubit, !isq.qir.qubit)
    isq.defgate @H {definition = [{type="unitary", value = [[#isq.complex<0.7071067811865476, 0.0>,#isq.complex<0.7071067811865476, 0.0>],[#isq.complex<0.7071067811865476, 0.0>,#isq.complex<-0.7071067811865476, -0.0>]]}]}: !isq.gate<1> loc("<stdin>":1:1)
    func @__isq__main() 
    {
    ^entry:
        %ssa_2 = arith.constant 0 : i1 loc("<stdin>":4:1)
        %ssa_3_real = memref.alloc() : memref<1xi1> loc("<stdin>":4:1)
        %ssa_3_zero = arith.constant 0 : index
        %ssa_3 = memref.subview %ssa_3_real[%ssa_3_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":4:1)
        %ssa_2_zero = arith.constant 0: index loc("<stdin>":4:1)
        memref.store %ssa_2, %ssa_3[%ssa_2_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":4:1)
        %ssa_4 = memref.alloc() : memref<1x!isq.qstate> loc("<stdin>":5:9)
        %ssa_8 = arith.constant 0 : index loc("<stdin>":6:13)
        %ssa_9 = memref.subview %ssa_4[%ssa_8][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":6:12)
        %ssa_6 = isq.use @H : !isq.gate<1> loc("<stdin>":6:9) 
        %ssa_6_in_1_zero = arith.constant 0: index loc("<stdin>":6:9)
        %ssa_6_in_1 = memref.load %ssa_9[%ssa_6_in_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":6:9)
        %ssa_6_out_1 = isq.apply %ssa_6(%ssa_6_in_1) : !isq.gate<1> loc("<stdin>":6:9)
        %ssa_6_out_1_zero = arith.constant 0: index loc("<stdin>":6:9)
        memref.store %ssa_6_out_1, %ssa_9[%ssa_6_out_1_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":6:9)
        %ssa_11 = arith.constant 0 : index loc("<stdin>":7:19)
        %ssa_12 = memref.subview %ssa_4[%ssa_11][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":7:18)
        %ssa_13_in_zero = arith.constant 0: index loc("<stdin>":7:15)
        %ssa_13_in = memref.load %ssa_12[%ssa_13_in_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":7:15)
        %ssa_13_out, %ssa_13 = isq.call_qop @__isq__builtin__measure(%ssa_13_in): [1]()->i1 loc("<stdin>":7:15)
        %ssa_13_out_zero = arith.constant 0: index loc("<stdin>":7:15)
        memref.store %ssa_13_out, %ssa_12[%ssa_13_out_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":7:15)
        %ssa_14_i2 = arith.extui %ssa_13 : i1 to i2 loc("<stdin>":7:15)
        %ssa_14 = arith.index_cast %ssa_14_i2 : i2 to index loc("<stdin>":7:15)
        %ssa_15_real = memref.alloc() : memref<1xindex> loc("<stdin>":7:9)
        %ssa_15_zero = arith.constant 0 : index
        %ssa_15 = memref.subview %ssa_15_real[%ssa_15_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":7:9)
        %ssa_14_zero = arith.constant 0: index loc("<stdin>":7:9)
        memref.store %ssa_14, %ssa_15[%ssa_14_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":7:9)
        %ssa_17_zero = arith.constant 0: index loc("<stdin>":8:15)
        %ssa_17 = memref.load %ssa_15[%ssa_17_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("<stdin>":8:15)
        isq.call_qop @__isq__builtin__print_int(%ssa_17): [0](index)->() loc("<stdin>":8:9)
        br ^exit_3 
    ^exit_3:
        memref.dealloc %ssa_15_real : memref<1xindex> loc("<stdin>":7:9)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_4 : memref<1x!isq.qstate> loc("<stdin>":5:9)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_3_real : memref<1xi1> loc("<stdin>":4:1)
        br ^exit 
    ^exit:
        return loc("<stdin>":4:1)
    } loc("<stdin>":4:1)
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
        call @__isq__global_initialize() : ()->() 
        return 
    } 
}
