module{
    func @for_test()->index 
    {
    ^entry:
        %ssa_1_zero = arith.constant 0 : index
        %ssa_1 = memref.alloc()[%ssa_1_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":1:1)
        %ssa_2 = arith.constant 0 : i1 loc("tests/for_loop_raii.isq":1:1)
        %ssa_3_zero = arith.constant 0 : index
        %ssa_3 = memref.alloc()[%ssa_3_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":1:1)
        affine.store %ssa_2, %ssa_3[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":1:1)
        %ssa_4 = arith.constant 0 : i1 loc("tests/for_loop_raii.isq":2:5)
        %ssa_5_zero = arith.constant 0 : index
        %ssa_5 = memref.alloc()[%ssa_5_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":2:5)
        affine.store %ssa_4, %ssa_5[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":2:5)
        %ssa_6 = arith.constant 0 : index loc("tests/for_loop_raii.isq":2:14)
        %ssa_7 = arith.constant 10 : index loc("tests/for_loop_raii.isq":2:16)
        affine.for %ssa_10 = %ssa_6 to %ssa_7 step 1 {
            scf.execute_region {
            ^entry:
                %ssa_11 = arith.constant 0 : i1 loc("tests/for_loop_raii.isq":3:9)
                %ssa_12_zero = arith.constant 0 : index
                %ssa_12 = memref.alloc()[%ssa_12_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":3:9)
                affine.store %ssa_11, %ssa_12[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":3:9)
                %ssa_13 = arith.constant 0 : index loc("tests/for_loop_raii.isq":3:18)
                %ssa_14 = arith.constant 10 : index loc("tests/for_loop_raii.isq":3:20)
                affine.for %ssa_17 = %ssa_13 to %ssa_14 step 2 {
                    scf.execute_region {
                    ^entry:
                        %ssa_18 = arith.constant 0 : index loc("tests/for_loop_raii.isq":4:19)
                        %ssa_19_zero = arith.constant 0 : index
                        %ssa_19 = memref.alloc()[%ssa_19_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":4:13)
                        affine.store %ssa_18, %ssa_19[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":4:13)
                        %ssa_20 = arith.constant 0 : i1 loc("tests/for_loop_raii.isq":5:13)
                        %ssa_21_zero = arith.constant 0 : index
                        %ssa_21 = memref.alloc()[%ssa_21_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":5:13)
                        affine.store %ssa_20, %ssa_21[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":5:13)
                        %ssa_24 = arith.addi %ssa_10, %ssa_17 : index loc("tests/for_loop_raii.isq":5:17)
                        %ssa_25 = arith.constant 10 : index loc("tests/for_loop_raii.isq":5:21)
                        %ssa_26 = arith.cmpi "eq", %ssa_24, %ssa_25 : index loc("tests/for_loop_raii.isq":5:19)
                        scf.if %ssa_26 {
                            scf.execute_region {
                            ^entry:
                                %ssa_28 = arith.constant 1 : i1 loc("tests/for_loop_raii.isq":6:17)
                                affine.store %ssa_28, %ssa_12[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":6:17)
                                br ^exit loc("tests/for_loop_raii.isq":6:17)
                            ^block1:
                                br ^exit 
                            ^exit:
                                scf.yield loc("tests/for_loop_raii.isq":5:13)
                            } loc("tests/for_loop_raii.isq":5:13)
                        } else {
                            scf.execute_region {
                            ^entry:
                                br ^exit 
                            ^exit:
                                scf.yield loc("tests/for_loop_raii.isq":5:13)
                            } loc("tests/for_loop_raii.isq":5:13)
                        }
                        %ssa_30 = affine.load %ssa_12[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":5:13)
                        cond_br %ssa_30, ^exit_2, ^block1 loc("tests/for_loop_raii.isq":5:13)
                    ^block1:
                        br ^exit_2 
                    ^exit_2:
                        memref.dealloc %ssa_21 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":5:13)
                        br ^exit_1 
                    ^exit_1:
                        memref.dealloc %ssa_19 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":4:13)
                        br ^exit 
                    ^exit:
                        scf.yield loc("tests/for_loop_raii.isq":3:9)
                    } loc("tests/for_loop_raii.isq":3:9)
                } loc("tests/for_loop_raii.isq":3:9)
                %ssa_32 = affine.load %ssa_5[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":3:9)
                cond_br %ssa_32, ^exit_1, ^block1 loc("tests/for_loop_raii.isq":3:9)
            ^block1:
                br ^exit_1 
            ^exit_1:
                memref.dealloc %ssa_12 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":3:9)
                br ^exit 
            ^exit:
                scf.yield loc("tests/for_loop_raii.isq":2:5)
            } loc("tests/for_loop_raii.isq":2:5)
        } loc("tests/for_loop_raii.isq":2:5)
        %ssa_34 = affine.load %ssa_3[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":2:5)
        cond_br %ssa_34, ^exit_2, ^block1 loc("tests/for_loop_raii.isq":2:5)
    ^block1:
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_5 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":2:5)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_3 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":1:1)
        br ^exit 
    ^exit:
        %ssa_36 = affine.load %ssa_1[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":1:1)
        memref.dealloc %ssa_1 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("tests/for_loop_raii.isq":1:1)
        return %ssa_36 : index loc("tests/for_loop_raii.isq":1:1)
    } loc("tests/for_loop_raii.isq":1:1)
    func @__isq__global_initialize() 
    {
    ^block1:
        return 
    } 
}