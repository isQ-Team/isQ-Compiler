module{
    isq.declare_qop @__isq__builtin__measure : [1]()->i1
    isq.declare_qop @__isq__builtin__reset : [1]()->()
    isq.declare_qop @__isq__builtin__bp : [0](index)->()
    isq.declare_qop @__isq__builtin__print_int : [0](index)->()
    isq.declare_qop @__isq__builtin__print_double : [0](f64)->()
    isq.declare_qop @__isq__qmpiprim__me : [0]()->index
    isq.declare_qop @__isq__qmpiprim__size : [0]()->index
    isq.declare_qop @__isq__qmpiprim__epr : [1](index)->()
    isq.declare_qop @__isq__qmpiprim__csend : [0](i1, index)->()
    isq.declare_qop @__isq__qmpiprim__crecv : [0](index)->i1
    func.func @"random_walk.$_ISQ_GATEDEF_increment"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>) 
    {
    ^entry(%ssa_22: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_23: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>):
        %ssa_24 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":2:1)
        %ssa_25_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":2:1)
        %ssa_25_zero = arith.constant 0 : index
        %ssa_25 = memref.subview %ssa_25_real[%ssa_25_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":2:1)
        %ssa_24_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":2:1)
        affine.store %ssa_24, %ssa_25[%ssa_24_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":2:1)
        %ssa_27 = isq.use @"std.CNOT" : !isq.gate<2> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":3:5) 
        %ssa_27_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":3:5)
        %ssa_27_in_1 = affine.load %ssa_22[%ssa_27_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":3:5)
        %ssa_27_in_2_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":3:5)
        %ssa_27_in_2 = affine.load %ssa_23[%ssa_27_in_2_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":3:5)
        %ssa_27_out_1, %ssa_27_out_2 = isq.apply %ssa_27(%ssa_27_in_1, %ssa_27_in_2) : !isq.gate<2> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":3:5)
        %ssa_27_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":3:5)
        affine.store %ssa_27_out_1, %ssa_22[%ssa_27_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":3:5)
        %ssa_27_out_2_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":3:5)
        affine.store %ssa_27_out_2, %ssa_23[%ssa_27_out_2_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":3:5)
        %ssa_31 = isq.use @"std.X" : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":4:5) 
        %ssa_31_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":4:5)
        %ssa_31_in_1 = affine.load %ssa_22[%ssa_31_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":4:5)
        %ssa_31_out_1 = isq.apply %ssa_31(%ssa_31_in_1) : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":4:5)
        %ssa_31_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":4:5)
        affine.store %ssa_31_out_1, %ssa_22[%ssa_31_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":4:5)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_25_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":2:1)
        memref.dealloc %ssa_25_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":2:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":2:1)
    } loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":2:1)
    isq.defgate @"random_walk.increment" {definition = [#isq.gatedef<type = "decomposition_raw", value = @"random_walk.$_ISQ_GATEDEF_increment">]}: !isq.gate<2> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":2:1)
    func.func @"__isq__main"(memref<?xindex>, memref<?xf64>) 
    {
    ^entry(%ssa_34: memref<?xindex>, %ssa_35: memref<?xf64>):
        %ssa_36 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":6:1)
        %ssa_37_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":6:1)
        %ssa_37_zero = arith.constant 0 : index
        %ssa_37 = memref.subview %ssa_37_real[%ssa_37_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":6:1)
        %ssa_36_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":6:1)
        affine.store %ssa_36, %ssa_37[%ssa_36_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":6:1)
        %ssa_38_real = memref.alloc() : memref<1x!isq.qstate> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":7:5)
        %ssa_38_zero = arith.constant 0 : index
        %ssa_38 = memref.subview %ssa_38_real[%ssa_38_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":7:5)
        %ssa_39 = memref.alloc() : memref<2x!isq.qstate> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":8:5)
        %ssa_40 = arith.constant 10 : index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":9:13)
        %ssa_41_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":9:5)
        %ssa_41_zero = arith.constant 0 : index
        %ssa_41 = memref.subview %ssa_41_real[%ssa_41_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":9:5)
        %ssa_40_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":9:5)
        affine.store %ssa_40, %ssa_41[%ssa_40_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":9:5)
        %ssa_42 = arith.constant 0 : index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:14)
        %ssa_44_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:16)
        %ssa_44 = affine.load %ssa_41[%ssa_44_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:16)
        affine.for %ssa_47 = %ssa_42 to %ssa_44 step 1 {
            scf.execute_region {
            ^entry:
                %ssa_48 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
                %ssa_49_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
                %ssa_49_zero = arith.constant 0 : index
                %ssa_49 = memref.subview %ssa_49_real[%ssa_49_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
                %ssa_48_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
                affine.store %ssa_48, %ssa_49[%ssa_48_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
                %ssa_50 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
                %ssa_51_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
                %ssa_51_zero = arith.constant 0 : index
                %ssa_51 = memref.subview %ssa_51_real[%ssa_51_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
                %ssa_50_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
                affine.store %ssa_50, %ssa_51[%ssa_50_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
                scf.execute_region {
                ^entry:
                    %ssa_53 = isq.use @"std.H" : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":11:9) 
                    %ssa_53_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":11:9)
                    %ssa_53_in_1 = affine.load %ssa_38[%ssa_53_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":11:9)
                    %ssa_53_out_1 = isq.apply %ssa_53(%ssa_53_in_1) : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":11:9)
                    %ssa_53_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":11:9)
                    affine.store %ssa_53_out_1, %ssa_38[%ssa_53_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":11:9)
                    %ssa_58 = arith.constant 0 : index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:35)
                    %ssa_59 = memref.subview %ssa_39[%ssa_58][1][1] : memref<2x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:34)
                    %ssa_61 = arith.constant 1 : index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:44)
                    %ssa_62 = memref.subview %ssa_39[%ssa_61][1][1] : memref<2x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:43)
                    %ssa_55 = isq.use @"random_walk.increment" : !isq.gate<2> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14) 
                    %ssa_55_decorated = isq.decorate(%ssa_55: !isq.gate<2>) {ctrl = [true], adjoint = false} : !isq.gate<3> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    %ssa_55_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    %ssa_55_in_1 = affine.load %ssa_38[%ssa_55_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    %ssa_55_in_2_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    %ssa_55_in_2 = affine.load %ssa_59[%ssa_55_in_2_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    %ssa_55_in_3_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    %ssa_55_in_3 = affine.load %ssa_62[%ssa_55_in_3_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    %ssa_55_out_1, %ssa_55_out_2, %ssa_55_out_3 = isq.apply %ssa_55_decorated(%ssa_55_in_1, %ssa_55_in_2, %ssa_55_in_3) : !isq.gate<3> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    %ssa_55_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    affine.store %ssa_55_out_1, %ssa_38[%ssa_55_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    %ssa_55_out_2_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    affine.store %ssa_55_out_2, %ssa_59[%ssa_55_out_2_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    %ssa_55_out_3_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    affine.store %ssa_55_out_3, %ssa_62[%ssa_55_out_3_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":12:14)
                    %ssa_66 = arith.constant 0 : index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:40)
                    %ssa_67 = memref.subview %ssa_39[%ssa_66][1][1] : memref<2x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:39)
                    %ssa_69 = arith.constant 1 : index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:49)
                    %ssa_70 = memref.subview %ssa_39[%ssa_69][1][1] : memref<2x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:48)
                    %ssa_63 = isq.use @"random_walk.increment" : !isq.gate<2> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19) 
                    %ssa_63_decorated = isq.decorate(%ssa_63: !isq.gate<2>) {ctrl = [false], adjoint = true} : !isq.gate<3> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    %ssa_63_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    %ssa_63_in_1 = affine.load %ssa_38[%ssa_63_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    %ssa_63_in_2_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    %ssa_63_in_2 = affine.load %ssa_67[%ssa_63_in_2_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    %ssa_63_in_3_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    %ssa_63_in_3 = affine.load %ssa_70[%ssa_63_in_3_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    %ssa_63_out_1, %ssa_63_out_2, %ssa_63_out_3 = isq.apply %ssa_63_decorated(%ssa_63_in_1, %ssa_63_in_2, %ssa_63_in_3) : !isq.gate<3> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    %ssa_63_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    affine.store %ssa_63_out_1, %ssa_38[%ssa_63_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    %ssa_63_out_2_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    affine.store %ssa_63_out_2, %ssa_67[%ssa_63_out_2_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    %ssa_63_out_3_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    affine.store %ssa_63_out_3, %ssa_70[%ssa_63_out_3_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":13:19)
                    cf.br ^exit 
                ^exit:
                    scf.yield loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
                } loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
                %ssa_72_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
                %ssa_72 = affine.load %ssa_49[%ssa_72_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
                cf.cond_br %ssa_72, ^exit_2, ^block1 loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
            ^block1:
                cf.br ^exit_2 
            ^exit_2:
                isq.accumulate_gphase %ssa_51_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
                memref.dealloc %ssa_51_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:17)
                cf.br ^exit_1 
            ^exit_1:
                isq.accumulate_gphase %ssa_49_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
                memref.dealloc %ssa_49_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
                cf.br ^exit 
            ^exit:
                scf.yield loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
            } loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
        } loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
        %ssa_74_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
        %ssa_74 = affine.load %ssa_37[%ssa_74_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
        cf.cond_br %ssa_74, ^exit_4, ^block1 loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":10:5)
    ^block1:
        cf.br ^exit_4 
    ^exit_4:
        isq.accumulate_gphase %ssa_41_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":9:5)
        memref.dealloc %ssa_41_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":9:5)
        cf.br ^exit_3 
    ^exit_3:
        isq.accumulate_gphase %ssa_39 : memref<2x!isq.qstate> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":8:5)
        memref.dealloc %ssa_39 : memref<2x!isq.qstate> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":8:5)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_38_real : memref<1x!isq.qstate> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":7:5)
        memref.dealloc %ssa_38_real : memref<1x!isq.qstate> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":7:5)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_37_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":6:1)
        memref.dealloc %ssa_37_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":6:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":6:1)
    } loc("/home/gjz010/isQ-Compiler/examples/random_walk.isq":6:1)
    func.func private @"__quantum__qis__rz__body"(f64, !isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":0:1)
    isq.defgate @"Rz"(f64) {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__rz__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":0:1)
    isq.defgate @"std.Rz"(f64) {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__rz__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":0:1)
    func.func private @"__quantum__qis__rx__body"(f64, !isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":1:1)
    isq.defgate @"Rx"(f64) {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__rx__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":1:1)
    isq.defgate @"std.Rx"(f64) {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__rx__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":1:1)
    func.func private @"__quantum__qis__ry__body"(f64, !isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":2:1)
    isq.defgate @"Ry"(f64) {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__ry__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":2:1)
    isq.defgate @"std.Ry"(f64) {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__ry__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":2:1)
    func.func private @"__quantum__qis__u3"(f64, f64, f64, !isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":3:1)
    isq.defgate @"U3"(f64, f64, f64) {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__u3">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":3:1)
    isq.defgate @"std.U3"(f64, f64, f64) {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__u3">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":3:1)
    func.func private @"__quantum__qis__h__body"(!isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":4:1)
    isq.defgate @"H" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__h__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":4:1)
    isq.defgate @"std.H" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__h__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":4:1)
    func.func private @"__quantum__qis__s__body"(!isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":5:1)
    isq.defgate @"S" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__s__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":5:1)
    isq.defgate @"std.S" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__s__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":5:1)
    func.func private @"__quantum__qis__t__body"(!isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":6:1)
    isq.defgate @"T" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__t__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":6:1)
    isq.defgate @"std.T" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__t__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":6:1)
    func.func private @"__quantum__qis__x__body"(!isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":7:1)
    isq.defgate @"X" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__x__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":7:1)
    isq.defgate @"std.X" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__x__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":7:1)
    func.func private @"__quantum__qis__y__body"(!isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":8:1)
    isq.defgate @"Y" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__y__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":8:1)
    isq.defgate @"std.Y" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__y__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":8:1)
    func.func private @"__quantum__qis__z__body"(!isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":9:1)
    isq.defgate @"Z" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__z__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":9:1)
    isq.defgate @"std.Z" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__z__body">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":9:1)
    func.func private @"__quantum__qis__cnot"(!isq.qir.qubit, !isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":10:1)
    isq.defgate @"CNOT" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__cnot">]}: !isq.gate<2> loc("/home/gjz010/isQ-Compiler/lib/std.isq":10:1)
    isq.defgate @"std.CNOT" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__cnot">]}: !isq.gate<2> loc("/home/gjz010/isQ-Compiler/lib/std.isq":10:1)
    func.func private @"__quantum__qis__toffoli"(!isq.qir.qubit, !isq.qir.qubit, !isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":11:1)
    isq.defgate @"Toffoli" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__toffoli">]}: !isq.gate<3> loc("/home/gjz010/isQ-Compiler/lib/std.isq":11:1)
    isq.defgate @"std.Toffoli" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__toffoli">]}: !isq.gate<3> loc("/home/gjz010/isQ-Compiler/lib/std.isq":11:1)
    func.func private @"__quantum__qis__x2m"(!isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":12:1)
    isq.defgate @"X2M" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__x2m">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":12:1)
    isq.defgate @"std.X2M" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__x2m">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":12:1)
    func.func private @"__quantum__qis__x2p"(!isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":13:1)
    isq.defgate @"X2P" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__x2p">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":13:1)
    isq.defgate @"std.X2P" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__x2p">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":13:1)
    func.func private @"__quantum__qis__y2m"(!isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":14:1)
    isq.defgate @"Y2M" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__y2m">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":14:1)
    isq.defgate @"std.Y2M" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__y2m">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":14:1)
    func.func private @"__quantum__qis__y2p"(!isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":15:1)
    isq.defgate @"Y2P" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__y2p">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":15:1)
    isq.defgate @"std.Y2P" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__y2p">]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/lib/std.isq":15:1)
    func.func private @"__quantum__qis__cz"(!isq.qir.qubit, !isq.qir.qubit) loc("/home/gjz010/isQ-Compiler/lib/std.isq":16:1)
    isq.defgate @"CZ" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__cz">]}: !isq.gate<2> loc("/home/gjz010/isQ-Compiler/lib/std.isq":16:1)
    isq.defgate @"std.CZ" {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__cz">]}: !isq.gate<2> loc("/home/gjz010/isQ-Compiler/lib/std.isq":16:1)
    func.func private @"__quantum__qis__gphase"(f64) loc("/home/gjz010/isQ-Compiler/lib/std.isq":17:1)
    isq.defgate @"GPhase"(f64) {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__gphase">]}: !isq.gate<0> loc("/home/gjz010/isQ-Compiler/lib/std.isq":17:1)
    isq.defgate @"std.GPhase"(f64) {definition = [#isq.gatedef<type = "qir", value = @"__quantum__qis__gphase">]}: !isq.gate<0> loc("/home/gjz010/isQ-Compiler/lib/std.isq":17:1)
    func.func @"__isq__global_initialize"() 
    {
    ^block1:
        return 
    } 
    func.func @"__isq__global_finalize"() 
    {
    ^block1:
        return 
    } 
    func.func @"__isq__entry"(memref<?xindex>, memref<?xf64>) 
    {
    ^block1(%ssa_1: memref<?xindex>, %ssa_2: memref<?xf64>):
        func.call @"__isq__global_initialize"() : ()->() 
        func.call @"__isq__main"(%ssa_1, %ssa_2) : (memref<?xindex>, memref<?xf64>)->() 
        func.call @"__isq__global_finalize"() : ()->() 
        return 
    } 
}
