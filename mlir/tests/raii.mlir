module{
    func @for_test()->index 
    {
    ^entry:
        %ssa_1_zero = arith.constant 0 : index
        %ssa_1 = memref.alloc()[%ssa_1_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":1:1)
        %ssa_2 = arith.constant 0 : i1 loc("main.isq":1:1)
        %ssa_3_zero = arith.constant 0 : index
        %ssa_3 = memref.alloc()[%ssa_3_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":1:1)
        affine.store %ssa_2, %ssa_3[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":1:1)
        %ssa_4 = arith.constant 0 : index loc("main.isq":2:14)
        %ssa_5_zero = arith.constant 0 : index
        %ssa_5 = memref.alloc()[%ssa_5_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        affine.store %ssa_4, %ssa_5[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        %ssa_6 = arith.constant 10 : index loc("main.isq":2:16)
        %ssa_7_zero = arith.constant 0 : index
        %ssa_7 = memref.alloc()[%ssa_7_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        affine.store %ssa_6, %ssa_7[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        %ssa_8 = arith.constant 1 : index loc("main.isq":2:5)
        %ssa_9_zero = arith.constant 0 : index
        %ssa_9 = memref.alloc()[%ssa_9_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        affine.store %ssa_8, %ssa_9[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        %ssa_11 = affine.load %ssa_5[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:15)
        %ssa_12_zero = arith.constant 0 : index
        %ssa_12 = memref.alloc()[%ssa_12_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        affine.store %ssa_11, %ssa_12[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        %ssa_13 = arith.constant 0 : i1 loc("main.isq":2:5)
        %ssa_14_zero = arith.constant 0 : index
        %ssa_14 = memref.alloc()[%ssa_14_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        affine.store %ssa_13, %ssa_14[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        %ssa_15 = arith.constant 0 : i1 loc("main.isq":2:5)
        %ssa_16_zero = arith.constant 0 : index
        %ssa_16 = memref.alloc()[%ssa_16_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        affine.store %ssa_15, %ssa_16[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        scf.while : ()->() {
            %cond = scf.execute_region->i1 {
                ^break_check:
                    %ssa_23 = affine.load %ssa_14[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
                    cond_br %ssa_23, ^break, ^while_cond
                ^while_cond:
                    %ssa_19 = affine.load %ssa_12[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
                    %ssa_20 = affine.load %ssa_7[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:15)
                    %ssa_21 = arith.cmpi "slt", %ssa_19, %ssa_20 : index loc("main.isq":2:15)
                    scf.yield %ssa_21: i1
                ^break:
                    %zero=arith.constant 0: i1
                    scf.yield %zero: i1
            }
            scf.condition(%cond)
        } do {
            scf.execute_region {
            ^entry:
                %ssa_24 = arith.constant 0 : index loc("main.isq":3:18)
                %ssa_25_zero = arith.constant 0 : index
                %ssa_25 = memref.alloc()[%ssa_25_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                affine.store %ssa_24, %ssa_25[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                %ssa_26 = arith.constant 10 : index loc("main.isq":3:20)
                %ssa_27_zero = arith.constant 0 : index
                %ssa_27 = memref.alloc()[%ssa_27_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                affine.store %ssa_26, %ssa_27[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                %ssa_28 = arith.constant 2 : index loc("main.isq":3:23)
                %ssa_29_zero = arith.constant 0 : index
                %ssa_29 = memref.alloc()[%ssa_29_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                affine.store %ssa_28, %ssa_29[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                %ssa_31 = affine.load %ssa_25[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:19)
                %ssa_32_zero = arith.constant 0 : index
                %ssa_32 = memref.alloc()[%ssa_32_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                affine.store %ssa_31, %ssa_32[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                %ssa_33 = arith.constant 0 : i1 loc("main.isq":3:9)
                %ssa_34_zero = arith.constant 0 : index
                %ssa_34 = memref.alloc()[%ssa_34_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                affine.store %ssa_33, %ssa_34[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                %ssa_35 = arith.constant 0 : i1 loc("main.isq":3:9)
                %ssa_36_zero = arith.constant 0 : index
                %ssa_36 = memref.alloc()[%ssa_36_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                affine.store %ssa_35, %ssa_36[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                scf.while : ()->() {
                    %cond = scf.execute_region->i1 {
                        ^break_check:
                            %ssa_43 = affine.load %ssa_34[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                            cond_br %ssa_43, ^break, ^while_cond
                        ^while_cond:
                            %ssa_39 = affine.load %ssa_32[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                            %ssa_40 = affine.load %ssa_27[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:19)
                            %ssa_41 = arith.cmpi "slt", %ssa_39, %ssa_40 : index loc("main.isq":3:19)
                            scf.yield %ssa_41: i1
                        ^break:
                            %zero=arith.constant 0: i1
                            scf.yield %zero: i1
                    }
                    scf.condition(%cond)
                } do {
                    scf.execute_region {
                    ^entry:
                        %ssa_44 = arith.constant 0 : i1 loc("main.isq":4:13)
                        %ssa_45_zero = arith.constant 0 : index
                        %ssa_45 = memref.alloc()[%ssa_45_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":4:13)
                        affine.store %ssa_44, %ssa_45[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":4:13)
                        %ssa_48 = affine.load %ssa_12[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":4:16)
                        %ssa_49 = affine.load %ssa_32[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":4:18)
                        %ssa_50 = arith.addi %ssa_48, %ssa_49 : index loc("main.isq":4:17)
                        %ssa_51 = arith.constant 10 : index loc("main.isq":4:21)
                        %ssa_52 = arith.cmpi "eq", %ssa_50, %ssa_51 : index loc("main.isq":4:19)
                        scf.if %ssa_52 {
                            scf.execute_region {
                            ^entry:
                                %ssa_54 = arith.constant 1 : i1 loc("main.isq":5:17)
                                affine.store %ssa_54, %ssa_34[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":5:17)
                                %ssa_56 = arith.constant 1 : i1 loc("main.isq":5:17)
                                affine.store %ssa_56, %ssa_14[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":5:17)
                                %ssa_58 = arith.constant 1 : i1 loc("main.isq":5:17)
                                affine.store %ssa_58, %ssa_36[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":5:17)
                                %ssa_60 = arith.constant 1 : i1 loc("main.isq":5:17)
                                affine.store %ssa_60, %ssa_16[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":5:17)
                                %ssa_62 = arith.constant 1 : i1 loc("main.isq":5:17)
                                affine.store %ssa_62, %ssa_3[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":5:17)
                                %ssa_64 = arith.constant 1 : index loc("main.isq":5:24)
                                affine.store %ssa_64, %ssa_1[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":5:17)
                                br ^exit loc("main.isq":5:17)
                            ^block1:
                                br ^exit 
                            ^exit:
                                scf.yield loc("main.isq":4:13)
                            } loc("main.isq":4:13)
                        } else {
                            scf.execute_region {
                            ^entry:
                                br ^exit 
                            ^exit:
                                scf.yield loc("main.isq":4:13)
                            } loc("main.isq":4:13)
                        }
                        %ssa_66 = affine.load %ssa_36[0] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":4:13)
                        cond_br %ssa_66, ^exit_1, ^block1 loc("main.isq":4:13)
                    ^block1:
                        %ssa_70 = affine.load %ssa_32[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                        %ssa_71 = affine.load %ssa_29[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:19)
                        %ssa_72 = arith.addi %ssa_70, %ssa_71 : index loc("main.isq":3:19)
                        affine.store %ssa_72, %ssa_32[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                        br ^exit_1 
                    ^exit_1:
                        memref.dealloc %ssa_45 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":4:13)
                        br ^exit 
                    ^exit:
                        scf.yield loc("main.isq":3:9)
                    } loc("main.isq":3:9)
                scf.yield
                } loc("main.isq":3:9)
                %ssa_76 = affine.load %ssa_12[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
                %ssa_77 = affine.load %ssa_9[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:15)
                %ssa_78 = arith.addi %ssa_76, %ssa_77 : index loc("main.isq":2:15)
                affine.store %ssa_78, %ssa_12[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
                br ^exit_6 
            ^exit_6:
                memref.dealloc %ssa_36 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                br ^exit_5 
            ^exit_5:
                memref.dealloc %ssa_34 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                br ^exit_4 
            ^exit_4:
                memref.dealloc %ssa_32 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                br ^exit_3 
            ^exit_3:
                memref.dealloc %ssa_29 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                br ^exit_2 
            ^exit_2:
                memref.dealloc %ssa_27 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                br ^exit_1 
            ^exit_1:
                memref.dealloc %ssa_25 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":3:9)
                br ^exit 
            ^exit:
                scf.yield loc("main.isq":2:5)
            } loc("main.isq":2:5)
        scf.yield
        } loc("main.isq":2:5)
        br ^exit_7 
    ^exit_7:
        memref.dealloc %ssa_16 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        br ^exit_6 
    ^exit_6:
        memref.dealloc %ssa_14 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        br ^exit_5 
    ^exit_5:
        memref.dealloc %ssa_12 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        br ^exit_4 
    ^exit_4:
        memref.dealloc %ssa_9 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        br ^exit_3 
    ^exit_3:
        memref.dealloc %ssa_7 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        br ^exit_2 
    ^exit_2:
        memref.dealloc %ssa_5 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":2:5)
        br ^exit_1 
    ^exit_1:
        memref.dealloc %ssa_3 : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":1:1)
        br ^exit 
    ^exit:
        %ssa_80 = affine.load %ssa_1[0] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":1:1)
        memref.dealloc %ssa_1 : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("main.isq":1:1)
        return %ssa_80 : index loc("main.isq":1:1)
    } loc("main.isq":1:1)
    func @__isq__global_initialize() 
    {
    ^block1:
        return 
    } 
}