module{
    isq.declare_qop @__isq__builtin__measure : [1]()->i1
    isq.declare_qop @__isq__builtin__reset : [1]()->()
    isq.declare_qop @__isq__builtin__bp : [0](index)->()
    isq.declare_qop @__isq__builtin__print_int : [0](index)->()
    isq.declare_qop @__isq__builtin__print_double : [0](f64)->()
    isq.declare_qop @__isq__qmpiprim__me : [0]()->index
    isq.declare_qop @__isq__qmpiprim__size : [0]()->index
    isq.declare_qop @__isq__qmpiprim__epr : [1](index)->()
    isq.declare_qop @__isq__qmpiprim__csend : [0](index, i1)->()
    isq.declare_qop @__isq__qmpiprim__crecv : [0](index)->i1
    isq.defgate @"qmpi.H" {definition = [#isq.gatedef<type="unitary", value = #isq.matrix<dense<[[(0.7071067811865475,0.0),(0.7071067811865475,0.0)],[(0.7071067811865475,0.0),(-0.7071067811865475,-0.0)]]>: tensor<2x2xcomplex<f64>>>>]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":0:1)
    isq.defgate @"qmpi.X" {definition = [#isq.gatedef<type="unitary", value = #isq.matrix<dense<[[(0.0,0.0),(1.0,0.0)],[(1.0,0.0),(0.0,0.0)]]>: tensor<2x2xcomplex<f64>>>>]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":4:1)
    isq.defgate @"qmpi.Z" {definition = [#isq.gatedef<type="unitary", value = #isq.matrix<dense<[[(1.0,0.0),(0.0,0.0)],[(0.0,0.0),(-1.0,-0.0)]]>: tensor<2x2xcomplex<f64>>>>]}: !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":5:1)
    func.func @"qmpi.qmpi_me"()->index 
    {
    ^entry:
        %ssa_20_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        %ssa_20_zero = arith.constant 0 : index
        %ssa_20 = memref.subview %ssa_20_real[%ssa_20_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        %ssa_21 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        %ssa_22_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        %ssa_22_zero = arith.constant 0 : index
        %ssa_22 = memref.subview %ssa_22_real[%ssa_22_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        %ssa_21_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        affine.store %ssa_21, %ssa_22[%ssa_21_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_22_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        memref.dealloc %ssa_22_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        cf.br ^exit 
    ^exit:
        %ssa_24_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        %ssa_24 = affine.load %ssa_20[%ssa_24_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        isq.accumulate_gphase %ssa_20_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        memref.dealloc %ssa_20_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
        return %ssa_24 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":6:1)
    func.func @"qmpi.qmpi_size"()->index 
    {
    ^entry:
        %ssa_25_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        %ssa_25_zero = arith.constant 0 : index
        %ssa_25 = memref.subview %ssa_25_real[%ssa_25_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        %ssa_26 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        %ssa_27_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        %ssa_27_zero = arith.constant 0 : index
        %ssa_27 = memref.subview %ssa_27_real[%ssa_27_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        %ssa_26_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        affine.store %ssa_26, %ssa_27[%ssa_26_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_27_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        memref.dealloc %ssa_27_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        cf.br ^exit 
    ^exit:
        %ssa_29_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        %ssa_29 = affine.load %ssa_25[%ssa_29_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        isq.accumulate_gphase %ssa_25_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        memref.dealloc %ssa_25_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
        return %ssa_29 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":9:1)
    func.func @"qmpi.qmpi_epr"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index) 
    {
    ^entry(%ssa_31: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_33: index):
        %ssa_35_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        %ssa_35_zero = arith.constant 0 : index
        %ssa_35 = memref.subview %ssa_35_real[%ssa_35_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        %ssa_33_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        affine.store %ssa_33, %ssa_35[%ssa_33_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        %ssa_36 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        %ssa_37_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        %ssa_37_zero = arith.constant 0 : index
        %ssa_37 = memref.subview %ssa_37_real[%ssa_37_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        %ssa_36_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        affine.store %ssa_36, %ssa_37[%ssa_36_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_37_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        memref.dealloc %ssa_37_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_35_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        memref.dealloc %ssa_35_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":12:1)
    func.func @"qmpi.qmpi_csend"(i1, index) 
    {
    ^entry(%ssa_39: i1, %ssa_41: index):
        %ssa_43_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        %ssa_43_zero = arith.constant 0 : index
        %ssa_43 = memref.subview %ssa_43_real[%ssa_43_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        %ssa_41_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        affine.store %ssa_41, %ssa_43[%ssa_41_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        %ssa_44 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        %ssa_45_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        %ssa_45_zero = arith.constant 0 : index
        %ssa_45 = memref.subview %ssa_45_real[%ssa_45_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        %ssa_44_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        affine.store %ssa_44, %ssa_45[%ssa_44_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_45_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        memref.dealloc %ssa_45_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_43_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        memref.dealloc %ssa_43_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":15:1)
    func.func @"qmpi.qmpi_crecv"(index)->i1 
    {
    ^entry(%ssa_48: index):
        %ssa_46_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        %ssa_46_zero = arith.constant 0 : index
        %ssa_46 = memref.subview %ssa_46_real[%ssa_46_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        %ssa_50_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        %ssa_50_zero = arith.constant 0 : index
        %ssa_50 = memref.subview %ssa_50_real[%ssa_50_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        %ssa_48_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        affine.store %ssa_48, %ssa_50[%ssa_48_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        %ssa_51 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        %ssa_52_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        %ssa_52_zero = arith.constant 0 : index
        %ssa_52 = memref.subview %ssa_52_real[%ssa_52_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        %ssa_51_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        affine.store %ssa_51, %ssa_52[%ssa_51_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_52_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        memref.dealloc %ssa_52_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_50_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        memref.dealloc %ssa_50_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        cf.br ^exit 
    ^exit:
        %ssa_54_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        %ssa_54 = affine.load %ssa_46[%ssa_54_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        isq.accumulate_gphase %ssa_46_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        memref.dealloc %ssa_46_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
        return %ssa_54 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":18:1)
    func.func @"qmpi.qmpi_send"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index) 
    {
    ^entry(%ssa_56: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_58: index):
        %ssa_60_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        %ssa_60_zero = arith.constant 0 : index
        %ssa_60 = memref.subview %ssa_60_real[%ssa_60_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        %ssa_58_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        affine.store %ssa_58, %ssa_60[%ssa_58_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        %ssa_61 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        %ssa_62_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        %ssa_62_zero = arith.constant 0 : index
        %ssa_62 = memref.subview %ssa_62_real[%ssa_62_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        %ssa_61_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        affine.store %ssa_61, %ssa_62[%ssa_61_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        %ssa_63_real = memref.alloc() : memref<1x!isq.qstate> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":26:5)
        %ssa_63_zero = arith.constant 0 : index
        %ssa_63 = memref.subview %ssa_63_real[%ssa_63_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":26:5)
        %ssa_68_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":27:17)
        %ssa_68 = affine.load %ssa_60[%ssa_68_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":27:17)
        call @"qmpi.qmpi_epr"(%ssa_63, %ssa_68) : (memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index)->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":27:5)
        %ssa_70 = isq.use @"qmpi.X" : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":28:10) 
        %ssa_70_decorated = isq.decorate(%ssa_70: !isq.gate<1>) {ctrl = [true], adjoint = false} : !isq.gate<2> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":28:10)
        %ssa_70_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":28:10)
        %ssa_70_in_1 = affine.load %ssa_56[%ssa_70_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":28:10)
        %ssa_70_in_2_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":28:10)
        %ssa_70_in_2 = affine.load %ssa_63[%ssa_70_in_2_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":28:10)
        %ssa_70_out_1, %ssa_70_out_2 = isq.apply %ssa_70_decorated(%ssa_70_in_1, %ssa_70_in_2) : !isq.gate<2> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":28:10)
        %ssa_70_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":28:10)
        affine.store %ssa_70_out_1, %ssa_56[%ssa_70_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":28:10)
        %ssa_70_out_2_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":28:10)
        affine.store %ssa_70_out_2, %ssa_63[%ssa_70_out_2_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":28:10)
        %ssa_76_in_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":29:16)
        %ssa_76_in = affine.load %ssa_63[%ssa_76_in_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":29:16)
        %ssa_76_out, %ssa_76 = isq.call_qop @__isq__builtin__measure(%ssa_76_in): [1]()->i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":29:16)
        %ssa_76_out_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":29:16)
        affine.store %ssa_76_out, %ssa_63[%ssa_76_out_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":29:16)
        %ssa_78_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":29:22)
        %ssa_78 = affine.load %ssa_60[%ssa_78_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":29:22)
        call @"qmpi.qmpi_csend"(%ssa_76, %ssa_78) : (i1, index)->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":29:5)
        cf.br ^exit_3 
    ^exit_3:
        isq.accumulate_gphase %ssa_63_real : memref<1x!isq.qstate> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":26:5)
        memref.dealloc %ssa_63_real : memref<1x!isq.qstate> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":26:5)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_62_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        memref.dealloc %ssa_62_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_60_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        memref.dealloc %ssa_60_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":25:1)
    func.func @"qmpi.qmpi_recv"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index) 
    {
    ^entry(%ssa_81: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_83: index):
        %ssa_85_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        %ssa_85_zero = arith.constant 0 : index
        %ssa_85 = memref.subview %ssa_85_real[%ssa_85_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        %ssa_83_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        affine.store %ssa_83, %ssa_85[%ssa_83_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        %ssa_86 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        %ssa_87_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        %ssa_87_zero = arith.constant 0 : index
        %ssa_87 = memref.subview %ssa_87_real[%ssa_87_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        %ssa_86_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        affine.store %ssa_86, %ssa_87[%ssa_86_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        %ssa_89_in_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":33:5)
        %ssa_89_in = affine.load %ssa_81[%ssa_89_in_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":33:5)
        %ssa_89_out = isq.call_qop @__isq__builtin__reset(%ssa_89_in): [1]()->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":33:5)
        %ssa_89_out_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":33:5)
        affine.store %ssa_89_out, %ssa_81[%ssa_89_out_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":33:5)
        %ssa_94_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":34:17)
        %ssa_94 = affine.load %ssa_85[%ssa_94_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":34:17)
        call @"qmpi.qmpi_epr"(%ssa_81, %ssa_94) : (memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index)->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":34:5)
        %ssa_96 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
        %ssa_97_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
        %ssa_97_zero = arith.constant 0 : index
        %ssa_97 = memref.subview %ssa_97_real[%ssa_97_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
        %ssa_96_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
        affine.store %ssa_96, %ssa_97[%ssa_96_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
        %ssa_100_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:19)
        %ssa_100 = affine.load %ssa_85[%ssa_100_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:19)
        %ssa_101 = call @"qmpi.qmpi_crecv"(%ssa_100) : (index)->i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:8)
        scf.if %ssa_101 {
            scf.execute_region {
            ^entry:
                %ssa_102 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
                %ssa_103_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
                %ssa_103_zero = arith.constant 0 : index
                %ssa_103 = memref.subview %ssa_103_real[%ssa_103_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
                %ssa_102_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
                affine.store %ssa_102, %ssa_103[%ssa_102_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
                scf.execute_region {
                ^entry:
                    %ssa_105 = isq.use @"qmpi.X" : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":36:9) 
                    %ssa_105_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":36:9)
                    %ssa_105_in_1 = affine.load %ssa_81[%ssa_105_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":36:9)
                    %ssa_105_out_1 = isq.apply %ssa_105(%ssa_105_in_1) : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":36:9)
                    %ssa_105_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":36:9)
                    affine.store %ssa_105_out_1, %ssa_81[%ssa_105_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":36:9)
                    cf.br ^exit 
                ^exit:
                    scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
                } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
                %ssa_108_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
                %ssa_108 = affine.load %ssa_97[%ssa_108_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
                cf.cond_br %ssa_108, ^exit_1, ^block1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
            ^block1:
                cf.br ^exit_1 
            ^exit_1:
                isq.accumulate_gphase %ssa_103_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
                memref.dealloc %ssa_103_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:24)
                cf.br ^exit 
            ^exit:
                scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
            } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
        } else {
            scf.execute_region {
            ^entry:
                cf.br ^exit 
            ^exit:
                scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
            } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
        }
        %ssa_110_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
        %ssa_110 = affine.load %ssa_87[%ssa_110_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
        cf.cond_br %ssa_110, ^exit_3, ^block1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
    ^block1:
        cf.br ^exit_3 
    ^exit_3:
        isq.accumulate_gphase %ssa_97_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
        memref.dealloc %ssa_97_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":35:5)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_87_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        memref.dealloc %ssa_87_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_85_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        memref.dealloc %ssa_85_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":32:1)
    func.func @"qmpi.qmpi_unsend"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index) 
    {
    ^entry(%ssa_112: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_114: index):
        %ssa_116_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        %ssa_116_zero = arith.constant 0 : index
        %ssa_116 = memref.subview %ssa_116_real[%ssa_116_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        %ssa_114_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        affine.store %ssa_114, %ssa_116[%ssa_114_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        %ssa_117 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        %ssa_118_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        %ssa_118_zero = arith.constant 0 : index
        %ssa_118 = memref.subview %ssa_118_real[%ssa_118_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        %ssa_117_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        affine.store %ssa_117, %ssa_118[%ssa_117_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        %ssa_119 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
        %ssa_120_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
        %ssa_120_zero = arith.constant 0 : index
        %ssa_120 = memref.subview %ssa_120_real[%ssa_120_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
        %ssa_119_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
        affine.store %ssa_119, %ssa_120[%ssa_119_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
        %ssa_123_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:19)
        %ssa_123 = affine.load %ssa_116[%ssa_123_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:19)
        %ssa_124 = call @"qmpi.qmpi_crecv"(%ssa_123) : (index)->i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:8)
        scf.if %ssa_124 {
            scf.execute_region {
            ^entry:
                %ssa_125 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
                %ssa_126_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
                %ssa_126_zero = arith.constant 0 : index
                %ssa_126 = memref.subview %ssa_126_real[%ssa_126_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
                %ssa_125_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
                affine.store %ssa_125, %ssa_126[%ssa_125_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
                scf.execute_region {
                ^entry:
                    %ssa_128 = isq.use @"qmpi.Z" : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":42:9) 
                    %ssa_128_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":42:9)
                    %ssa_128_in_1 = affine.load %ssa_112[%ssa_128_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":42:9)
                    %ssa_128_out_1 = isq.apply %ssa_128(%ssa_128_in_1) : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":42:9)
                    %ssa_128_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":42:9)
                    affine.store %ssa_128_out_1, %ssa_112[%ssa_128_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":42:9)
                    cf.br ^exit 
                ^exit:
                    scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
                } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
                %ssa_131_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
                %ssa_131 = affine.load %ssa_120[%ssa_131_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
                cf.cond_br %ssa_131, ^exit_1, ^block1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
            ^block1:
                cf.br ^exit_1 
            ^exit_1:
                isq.accumulate_gphase %ssa_126_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
                memref.dealloc %ssa_126_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:24)
                cf.br ^exit 
            ^exit:
                scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
            } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
        } else {
            scf.execute_region {
            ^entry:
                cf.br ^exit 
            ^exit:
                scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
            } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
        }
        %ssa_133_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
        %ssa_133 = affine.load %ssa_118[%ssa_133_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
        cf.cond_br %ssa_133, ^exit_3, ^block1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
    ^block1:
        cf.br ^exit_3 
    ^exit_3:
        isq.accumulate_gphase %ssa_120_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
        memref.dealloc %ssa_120_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":41:5)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_118_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        memref.dealloc %ssa_118_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_116_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        memref.dealloc %ssa_116_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":40:1)
    func.func @"qmpi.qmpi_unrecv"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index) 
    {
    ^entry(%ssa_135: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_137: index):
        %ssa_139_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        %ssa_139_zero = arith.constant 0 : index
        %ssa_139 = memref.subview %ssa_139_real[%ssa_139_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        %ssa_137_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        affine.store %ssa_137, %ssa_139[%ssa_137_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        %ssa_140 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        %ssa_141_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        %ssa_141_zero = arith.constant 0 : index
        %ssa_141 = memref.subview %ssa_141_real[%ssa_141_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        %ssa_140_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        affine.store %ssa_140, %ssa_141[%ssa_140_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        %ssa_143 = isq.use @"qmpi.H" : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":46:5) 
        %ssa_143_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":46:5)
        %ssa_143_in_1 = affine.load %ssa_135[%ssa_143_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":46:5)
        %ssa_143_out_1 = isq.apply %ssa_143(%ssa_143_in_1) : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":46:5)
        %ssa_143_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":46:5)
        affine.store %ssa_143_out_1, %ssa_135[%ssa_143_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":46:5)
        %ssa_148_in_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":47:16)
        %ssa_148_in = affine.load %ssa_135[%ssa_148_in_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":47:16)
        %ssa_148_out, %ssa_148 = isq.call_qop @__isq__builtin__measure(%ssa_148_in): [1]()->i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":47:16)
        %ssa_148_out_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":47:16)
        affine.store %ssa_148_out, %ssa_135[%ssa_148_out_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":47:16)
        %ssa_150_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":47:22)
        %ssa_150 = affine.load %ssa_139[%ssa_150_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":47:22)
        call @"qmpi.qmpi_csend"(%ssa_148, %ssa_150) : (i1, index)->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":47:5)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_141_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        memref.dealloc %ssa_141_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_139_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        memref.dealloc %ssa_139_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":45:1)
    func.func @"qmpi.qmpi_send_move"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index) 
    {
    ^entry(%ssa_153: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_155: index):
        %ssa_157_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        %ssa_157_zero = arith.constant 0 : index
        %ssa_157 = memref.subview %ssa_157_real[%ssa_157_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        %ssa_155_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        affine.store %ssa_155, %ssa_157[%ssa_155_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        %ssa_158 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        %ssa_159_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        %ssa_159_zero = arith.constant 0 : index
        %ssa_159 = memref.subview %ssa_159_real[%ssa_159_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        %ssa_158_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        affine.store %ssa_158, %ssa_159[%ssa_158_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        %ssa_160_real = memref.alloc() : memref<1x!isq.qstate> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":51:5)
        %ssa_160_zero = arith.constant 0 : index
        %ssa_160 = memref.subview %ssa_160_real[%ssa_160_zero][1][1] : memref<1x!isq.qstate> to memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":51:5)
        %ssa_165_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":52:17)
        %ssa_165 = affine.load %ssa_157[%ssa_165_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":52:17)
        call @"qmpi.qmpi_epr"(%ssa_160, %ssa_165) : (memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index)->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":52:5)
        %ssa_167 = isq.use @"qmpi.X" : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":53:10) 
        %ssa_167_decorated = isq.decorate(%ssa_167: !isq.gate<1>) {ctrl = [true], adjoint = false} : !isq.gate<2> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":53:10)
        %ssa_167_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":53:10)
        %ssa_167_in_1 = affine.load %ssa_153[%ssa_167_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":53:10)
        %ssa_167_in_2_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":53:10)
        %ssa_167_in_2 = affine.load %ssa_160[%ssa_167_in_2_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":53:10)
        %ssa_167_out_1, %ssa_167_out_2 = isq.apply %ssa_167_decorated(%ssa_167_in_1, %ssa_167_in_2) : !isq.gate<2> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":53:10)
        %ssa_167_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":53:10)
        affine.store %ssa_167_out_1, %ssa_153[%ssa_167_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":53:10)
        %ssa_167_out_2_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":53:10)
        affine.store %ssa_167_out_2, %ssa_160[%ssa_167_out_2_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":53:10)
        %ssa_171 = isq.use @"qmpi.H" : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":54:5) 
        %ssa_171_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":54:5)
        %ssa_171_in_1 = affine.load %ssa_153[%ssa_171_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":54:5)
        %ssa_171_out_1 = isq.apply %ssa_171(%ssa_171_in_1) : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":54:5)
        %ssa_171_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":54:5)
        affine.store %ssa_171_out_1, %ssa_153[%ssa_171_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":54:5)
        %ssa_176_in_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":55:16)
        %ssa_176_in = affine.load %ssa_160[%ssa_176_in_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":55:16)
        %ssa_176_out, %ssa_176 = isq.call_qop @__isq__builtin__measure(%ssa_176_in): [1]()->i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":55:16)
        %ssa_176_out_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":55:16)
        affine.store %ssa_176_out, %ssa_160[%ssa_176_out_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":55:16)
        %ssa_178_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":55:22)
        %ssa_178 = affine.load %ssa_157[%ssa_178_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":55:22)
        call @"qmpi.qmpi_csend"(%ssa_176, %ssa_178) : (i1, index)->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":55:5)
        %ssa_183_in_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":56:16)
        %ssa_183_in = affine.load %ssa_153[%ssa_183_in_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":56:16)
        %ssa_183_out, %ssa_183 = isq.call_qop @__isq__builtin__measure(%ssa_183_in): [1]()->i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":56:16)
        %ssa_183_out_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":56:16)
        affine.store %ssa_183_out, %ssa_153[%ssa_183_out_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":56:16)
        %ssa_185_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":56:22)
        %ssa_185 = affine.load %ssa_157[%ssa_185_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":56:22)
        call @"qmpi.qmpi_csend"(%ssa_183, %ssa_185) : (i1, index)->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":56:5)
        %ssa_188_in_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":57:5)
        %ssa_188_in = affine.load %ssa_153[%ssa_188_in_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":57:5)
        %ssa_188_out = isq.call_qop @__isq__builtin__reset(%ssa_188_in): [1]()->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":57:5)
        %ssa_188_out_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":57:5)
        affine.store %ssa_188_out, %ssa_153[%ssa_188_out_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":57:5)
        %ssa_190_in_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":58:5)
        %ssa_190_in = affine.load %ssa_160[%ssa_190_in_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":58:5)
        %ssa_190_out = isq.call_qop @__isq__builtin__reset(%ssa_190_in): [1]()->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":58:5)
        %ssa_190_out_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":58:5)
        affine.store %ssa_190_out, %ssa_160[%ssa_190_out_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":58:5)
        cf.br ^exit_3 
    ^exit_3:
        isq.accumulate_gphase %ssa_160_real : memref<1x!isq.qstate> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":51:5)
        memref.dealloc %ssa_160_real : memref<1x!isq.qstate> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":51:5)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_159_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        memref.dealloc %ssa_159_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_157_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        memref.dealloc %ssa_157_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":50:1)
    func.func @"qmpi.qmpi_recv_move"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index) 
    {
    ^entry(%ssa_192: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_194: index):
        %ssa_196_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        %ssa_196_zero = arith.constant 0 : index
        %ssa_196 = memref.subview %ssa_196_real[%ssa_196_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        %ssa_194_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        affine.store %ssa_194, %ssa_196[%ssa_194_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        %ssa_197 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        %ssa_198_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        %ssa_198_zero = arith.constant 0 : index
        %ssa_198 = memref.subview %ssa_198_real[%ssa_198_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        %ssa_197_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        affine.store %ssa_197, %ssa_198[%ssa_197_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        %ssa_200_in_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":61:5)
        %ssa_200_in = affine.load %ssa_192[%ssa_200_in_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":61:5)
        %ssa_200_out = isq.call_qop @__isq__builtin__reset(%ssa_200_in): [1]()->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":61:5)
        %ssa_200_out_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":61:5)
        affine.store %ssa_200_out, %ssa_192[%ssa_200_out_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":61:5)
        %ssa_205_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":62:17)
        %ssa_205 = affine.load %ssa_196[%ssa_205_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":62:17)
        call @"qmpi.qmpi_epr"(%ssa_192, %ssa_205) : (memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index)->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":62:5)
        %ssa_207 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
        %ssa_208_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
        %ssa_208_zero = arith.constant 0 : index
        %ssa_208 = memref.subview %ssa_208_real[%ssa_208_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
        %ssa_207_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
        affine.store %ssa_207, %ssa_208[%ssa_207_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
        %ssa_211_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:19)
        %ssa_211 = affine.load %ssa_196[%ssa_211_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:19)
        %ssa_212 = call @"qmpi.qmpi_crecv"(%ssa_211) : (index)->i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:8)
        scf.if %ssa_212 {
            scf.execute_region {
            ^entry:
                %ssa_213 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
                %ssa_214_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
                %ssa_214_zero = arith.constant 0 : index
                %ssa_214 = memref.subview %ssa_214_real[%ssa_214_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
                %ssa_213_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
                affine.store %ssa_213, %ssa_214[%ssa_213_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
                scf.execute_region {
                ^entry:
                    %ssa_216 = isq.use @"qmpi.Z" : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":64:9) 
                    %ssa_216_in_1_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":64:9)
                    %ssa_216_in_1 = affine.load %ssa_192[%ssa_216_in_1_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":64:9)
                    %ssa_216_out_1 = isq.apply %ssa_216(%ssa_216_in_1) : !isq.gate<1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":64:9)
                    %ssa_216_out_1_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":64:9)
                    affine.store %ssa_216_out_1, %ssa_192[%ssa_216_out_1_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":64:9)
                    cf.br ^exit 
                ^exit:
                    scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
                } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
                %ssa_219_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
                %ssa_219 = affine.load %ssa_208[%ssa_219_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
                cf.cond_br %ssa_219, ^exit_1, ^block1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
            ^block1:
                cf.br ^exit_1 
            ^exit_1:
                isq.accumulate_gphase %ssa_214_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
                memref.dealloc %ssa_214_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:24)
                cf.br ^exit 
            ^exit:
                scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
            } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
        } else {
            scf.execute_region {
            ^entry:
                cf.br ^exit 
            ^exit:
                scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
            } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
        }
        %ssa_221_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
        %ssa_221 = affine.load %ssa_198[%ssa_221_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
        cf.cond_br %ssa_221, ^exit_3, ^block1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
    ^block1:
        cf.br ^exit_3 
    ^exit_3:
        isq.accumulate_gphase %ssa_208_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
        memref.dealloc %ssa_208_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":63:5)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_198_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        memref.dealloc %ssa_198_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_196_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        memref.dealloc %ssa_196_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":60:1)
    func.func @"qmpi.qmpi_unsend_move"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index) 
    {
    ^entry(%ssa_223: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_225: index):
        %ssa_227_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        %ssa_227_zero = arith.constant 0 : index
        %ssa_227 = memref.subview %ssa_227_real[%ssa_227_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        %ssa_225_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        affine.store %ssa_225, %ssa_227[%ssa_225_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        %ssa_228 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        %ssa_229_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        %ssa_229_zero = arith.constant 0 : index
        %ssa_229 = memref.subview %ssa_229_real[%ssa_229_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        %ssa_228_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        affine.store %ssa_228, %ssa_229[%ssa_228_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        %ssa_234_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":68:23)
        %ssa_234 = affine.load %ssa_227[%ssa_234_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":68:23)
        call @"qmpi.qmpi_recv_move"(%ssa_223, %ssa_234) : (memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index)->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":68:5)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_229_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        memref.dealloc %ssa_229_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_227_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        memref.dealloc %ssa_227_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":67:1)
    func.func @"qmpi.qmpi_unrecv_move"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index) 
    {
    ^entry(%ssa_237: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_239: index):
        %ssa_241_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        %ssa_241_zero = arith.constant 0 : index
        %ssa_241 = memref.subview %ssa_241_real[%ssa_241_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        %ssa_239_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        affine.store %ssa_239, %ssa_241[%ssa_239_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        %ssa_242 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        %ssa_243_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        %ssa_243_zero = arith.constant 0 : index
        %ssa_243 = memref.subview %ssa_243_real[%ssa_243_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        %ssa_242_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        affine.store %ssa_242, %ssa_243[%ssa_242_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        %ssa_248_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":72:23)
        %ssa_248 = affine.load %ssa_241[%ssa_248_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":72:23)
        call @"qmpi.qmpi_send_move"(%ssa_237, %ssa_248) : (memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index)->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":72:5)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_243_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        memref.dealloc %ssa_243_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_241_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        memref.dealloc %ssa_241_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":71:1)
    func.func @"qmpi.log2"(index)->index 
    {
    ^entry(%ssa_252: index):
        %ssa_250_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        %ssa_250_zero = arith.constant 0 : index
        %ssa_250 = memref.subview %ssa_250_real[%ssa_250_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        %ssa_254_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        %ssa_254_zero = arith.constant 0 : index
        %ssa_254 = memref.subview %ssa_254_real[%ssa_254_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        %ssa_252_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        affine.store %ssa_252, %ssa_254[%ssa_252_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        %ssa_255 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        %ssa_256_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        %ssa_256_zero = arith.constant 0 : index
        %ssa_256 = memref.subview %ssa_256_real[%ssa_256_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        %ssa_255_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        affine.store %ssa_255, %ssa_256[%ssa_255_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        %ssa_257 = arith.constant 1 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":75:15)
        %ssa_258_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":75:5)
        %ssa_258_zero = arith.constant 0 : index
        %ssa_258 = memref.subview %ssa_258_real[%ssa_258_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":75:5)
        %ssa_257_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":75:5)
        affine.store %ssa_257, %ssa_258[%ssa_257_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":75:5)
        %ssa_259 = arith.constant 0 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":76:15)
        %ssa_260_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":76:5)
        %ssa_260_zero = arith.constant 0 : index
        %ssa_260 = memref.subview %ssa_260_real[%ssa_260_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":76:5)
        %ssa_259_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":76:5)
        affine.store %ssa_259, %ssa_260[%ssa_259_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":76:5)
        %ssa_261 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
        %ssa_262_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
        %ssa_262_zero = arith.constant 0 : index
        %ssa_262 = memref.subview %ssa_262_real[%ssa_262_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
        %ssa_261_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
        affine.store %ssa_261, %ssa_262[%ssa_261_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
        scf.while : ()->() {
            %cond = scf.execute_region->i1 {
                ^break_check:
                    %ssa_269_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
                    %ssa_269 = affine.load %ssa_262[%ssa_269_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
                    cf.cond_br %ssa_269, ^break, ^while_cond
                ^while_cond:
                    %ssa_265_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:11)
                    %ssa_265 = affine.load %ssa_258[%ssa_265_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:11)
                    %ssa_266_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:15)
                    %ssa_266 = affine.load %ssa_254[%ssa_266_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:15)
                    %ssa_267 = arith.cmpi "slt", %ssa_265, %ssa_266 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:14)
                    scf.yield %ssa_267: i1
                ^break:
                    %zero=arith.constant 0: i1
                    scf.yield %zero: i1
            }
            scf.condition(%cond)
        } do {
            scf.execute_region {
            ^entry:
                %ssa_270 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
                %ssa_271_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
                %ssa_271_zero = arith.constant 0 : index
                %ssa_271 = memref.subview %ssa_271_real[%ssa_271_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
                %ssa_270_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
                affine.store %ssa_270, %ssa_271[%ssa_270_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
                %ssa_272 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
                %ssa_273_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
                %ssa_273_zero = arith.constant 0 : index
                %ssa_273 = memref.subview %ssa_273_real[%ssa_273_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
                %ssa_272_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
                affine.store %ssa_272, %ssa_273[%ssa_272_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
                scf.execute_region {
                ^entry:
                    %ssa_277_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":78:15)
                    %ssa_277 = affine.load %ssa_258[%ssa_277_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":78:15)
                    %ssa_276 = arith.constant 2 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":78:19)
                    %ssa_278 = arith.muli %ssa_277, %ssa_276 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":78:18)
                    %ssa_278_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":78:13)
                    affine.store %ssa_278, %ssa_258[%ssa_278_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":78:13)
                    %ssa_282_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":79:15)
                    %ssa_282 = affine.load %ssa_260[%ssa_282_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":79:15)
                    %ssa_281 = arith.constant 1 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":79:19)
                    %ssa_283 = arith.addi %ssa_282, %ssa_281 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":79:18)
                    %ssa_283_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":79:13)
                    affine.store %ssa_283, %ssa_260[%ssa_283_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":79:13)
                    cf.br ^exit 
                ^exit:
                    scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
                } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
                %ssa_285_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
                %ssa_285 = affine.load %ssa_271[%ssa_285_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
                cf.cond_br %ssa_285, ^exit_2, ^block1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
            ^block1:
                cf.br ^exit_2 
            ^exit_2:
                isq.accumulate_gphase %ssa_273_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
                memref.dealloc %ssa_273_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:17)
                cf.br ^exit_1 
            ^exit_1:
                isq.accumulate_gphase %ssa_271_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
                memref.dealloc %ssa_271_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
                cf.br ^exit 
            ^exit:
                scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
            } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
        scf.yield
        } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
        %ssa_288_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":81:12)
        %ssa_288 = affine.load %ssa_260[%ssa_288_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":81:12)
        %ssa_288_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":81:5)
        affine.store %ssa_288, %ssa_250[%ssa_288_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":81:5)
        cf.br ^exit_5 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":81:5)
    ^block1:
        cf.br ^exit_5 
    ^exit_5:
        isq.accumulate_gphase %ssa_262_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
        memref.dealloc %ssa_262_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":77:5)
        cf.br ^exit_4 
    ^exit_4:
        isq.accumulate_gphase %ssa_260_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":76:5)
        memref.dealloc %ssa_260_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":76:5)
        cf.br ^exit_3 
    ^exit_3:
        isq.accumulate_gphase %ssa_258_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":75:5)
        memref.dealloc %ssa_258_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":75:5)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_256_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        memref.dealloc %ssa_256_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_254_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        memref.dealloc %ssa_254_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        cf.br ^exit 
    ^exit:
        %ssa_290_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        %ssa_290 = affine.load %ssa_250[%ssa_290_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        isq.accumulate_gphase %ssa_250_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        memref.dealloc %ssa_250_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
        return %ssa_290 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":74:1)
    func.func @"qmpi.qmpi_poorman_cat"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index, index, index) 
    {
    ^entry(%ssa_292: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_294: index, %ssa_296: index, %ssa_298: index):
        %ssa_300_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_300_zero = arith.constant 0 : index
        %ssa_300 = memref.subview %ssa_300_real[%ssa_300_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_294_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        affine.store %ssa_294, %ssa_300[%ssa_294_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_302_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_302_zero = arith.constant 0 : index
        %ssa_302 = memref.subview %ssa_302_real[%ssa_302_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_296_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        affine.store %ssa_296, %ssa_302[%ssa_296_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_304_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_304_zero = arith.constant 0 : index
        %ssa_304 = memref.subview %ssa_304_real[%ssa_304_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_298_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        affine.store %ssa_298, %ssa_304[%ssa_298_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_305 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_306_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_306_zero = arith.constant 0 : index
        %ssa_306 = memref.subview %ssa_306_real[%ssa_306_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        %ssa_305_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        affine.store %ssa_305, %ssa_306[%ssa_305_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        cf.br ^exit_4 
    ^exit_4:
        isq.accumulate_gphase %ssa_306_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        memref.dealloc %ssa_306_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        cf.br ^exit_3 
    ^exit_3:
        isq.accumulate_gphase %ssa_304_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        memref.dealloc %ssa_304_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_302_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        memref.dealloc %ssa_302_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_300_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        memref.dealloc %ssa_300_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":83:1)
    func.func @"qmpi.qmpi_bcast"(memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, index) 
    {
    ^entry(%ssa_308: memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>>, %ssa_310: index):
        %ssa_312_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        %ssa_312_zero = arith.constant 0 : index
        %ssa_312 = memref.subview %ssa_312_real[%ssa_312_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        %ssa_310_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        affine.store %ssa_310, %ssa_312[%ssa_310_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        %ssa_313 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        %ssa_314_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        %ssa_314_zero = arith.constant 0 : index
        %ssa_314 = memref.subview %ssa_314_real[%ssa_314_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        %ssa_313_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        affine.store %ssa_313, %ssa_314[%ssa_313_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        %ssa_316 = call @"qmpi.qmpi_me"() : ()->index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":87:14)
        %ssa_317_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":87:5)
        %ssa_317_zero = arith.constant 0 : index
        %ssa_317 = memref.subview %ssa_317_real[%ssa_317_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":87:5)
        %ssa_316_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":87:5)
        affine.store %ssa_316, %ssa_317[%ssa_316_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":87:5)
        %ssa_318 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
        %ssa_319_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
        %ssa_319_zero = arith.constant 0 : index
        %ssa_319 = memref.subview %ssa_319_real[%ssa_319_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
        %ssa_318_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
        affine.store %ssa_318, %ssa_319[%ssa_318_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
        %ssa_322_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:8)
        %ssa_322 = affine.load %ssa_317[%ssa_322_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:8)
        %ssa_323_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:12)
        %ssa_323 = affine.load %ssa_312[%ssa_323_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:12)
        %ssa_324 = arith.cmpi "ne", %ssa_322, %ssa_323 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:10)
        scf.if %ssa_324 {
            scf.execute_region {
            ^entry:
                %ssa_325 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
                %ssa_326_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
                %ssa_326_zero = arith.constant 0 : index
                %ssa_326 = memref.subview %ssa_326_real[%ssa_326_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
                %ssa_325_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
                affine.store %ssa_325, %ssa_326[%ssa_325_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
                scf.execute_region {
                ^entry:
                    %ssa_328_in_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":89:9)
                    %ssa_328_in = affine.load %ssa_308[%ssa_328_in_load_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":89:9)
                    %ssa_328_out = isq.call_qop @__isq__builtin__reset(%ssa_328_in): [1]()->() loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":89:9)
                    %ssa_328_out_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":89:9)
                    affine.store %ssa_328_out, %ssa_308[%ssa_328_out_store_zero] : memref<1x!isq.qstate, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":89:9)
                    cf.br ^exit 
                ^exit:
                    scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
                } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
                %ssa_330_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
                %ssa_330 = affine.load %ssa_319[%ssa_330_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
                cf.cond_br %ssa_330, ^exit_1, ^block1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
            ^block1:
                cf.br ^exit_1 
            ^exit_1:
                isq.accumulate_gphase %ssa_326_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
                memref.dealloc %ssa_326_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:17)
                cf.br ^exit 
            ^exit:
                scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
            } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
        } else {
            scf.execute_region {
            ^entry:
                cf.br ^exit 
            ^exit:
                scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
            } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
        }
        %ssa_332_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
        %ssa_332 = affine.load %ssa_314[%ssa_332_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
        cf.cond_br %ssa_332, ^exit_4, ^block1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
    ^block1:
        %ssa_334 = call @"qmpi.qmpi_size"() : ()->index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":91:16)
        %ssa_335_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":91:5)
        %ssa_335_zero = arith.constant 0 : index
        %ssa_335 = memref.subview %ssa_335_real[%ssa_335_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":91:5)
        %ssa_334_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":91:5)
        affine.store %ssa_334, %ssa_335[%ssa_334_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":91:5)
        %ssa_338_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:16)
        %ssa_338 = affine.load %ssa_317[%ssa_338_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:16)
        %ssa_339_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:21)
        %ssa_339 = affine.load %ssa_335[%ssa_339_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:21)
        %ssa_340 = arith.addi %ssa_338, %ssa_339 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:19)
        %ssa_342_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:28)
        %ssa_342 = affine.load %ssa_312[%ssa_342_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:28)
        %ssa_343 = arith.subi %ssa_340, %ssa_342 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:26)
        %ssa_345_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:34)
        %ssa_345 = affine.load %ssa_335[%ssa_345_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:34)
        %ssa_346 = arith.remsi %ssa_343, %ssa_345 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:33)
        %ssa_347_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:5)
        %ssa_347_zero = arith.constant 0 : index
        %ssa_347 = memref.subview %ssa_347_real[%ssa_347_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:5)
        %ssa_346_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:5)
        affine.store %ssa_346, %ssa_347[%ssa_346_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:5)
        %ssa_350_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":93:22)
        %ssa_350 = affine.load %ssa_335[%ssa_350_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":93:22)
        %ssa_351 = call @"qmpi.log2"(%ssa_350) : (index)->index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":93:17)
        %ssa_352_real = memref.alloc() : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":93:5)
        %ssa_352_zero = arith.constant 0 : index
        %ssa_352 = memref.subview %ssa_352_real[%ssa_352_zero][1][1] : memref<1xindex> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":93:5)
        %ssa_351_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":93:5)
        affine.store %ssa_351, %ssa_352[%ssa_351_store_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":93:5)
        %ssa_353 = arith.constant 0 : index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:14)
        %ssa_355_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:16)
        %ssa_355 = affine.load %ssa_352[%ssa_355_load_zero] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:16)
        affine.for %ssa_358 = %ssa_353 to %ssa_355 step 1 {
            scf.execute_region {
            ^entry:
                %ssa_359 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
                %ssa_360_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
                %ssa_360_zero = arith.constant 0 : index
                %ssa_360 = memref.subview %ssa_360_real[%ssa_360_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
                %ssa_359_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
                affine.store %ssa_359, %ssa_360[%ssa_359_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
                %ssa_361 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
                %ssa_362_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
                %ssa_362_zero = arith.constant 0 : index
                %ssa_362 = memref.subview %ssa_362_real[%ssa_362_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
                %ssa_361_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
                affine.store %ssa_361, %ssa_362[%ssa_361_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
                scf.execute_region {
                ^entry:
                    cf.br ^exit 
                ^exit:
                    scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
                } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
                %ssa_364_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
                %ssa_364 = affine.load %ssa_360[%ssa_364_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
                cf.cond_br %ssa_364, ^exit_2, ^block1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
            ^block1:
                cf.br ^exit_2 
            ^exit_2:
                isq.accumulate_gphase %ssa_362_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
                memref.dealloc %ssa_362_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:21)
                cf.br ^exit_1 
            ^exit_1:
                isq.accumulate_gphase %ssa_360_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
                memref.dealloc %ssa_360_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
                cf.br ^exit 
            ^exit:
                scf.yield loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
            } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
        } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
        %ssa_366_load_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
        %ssa_366 = affine.load %ssa_314[%ssa_366_load_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
        cf.cond_br %ssa_366, ^exit_7, ^block2 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":94:5)
    ^block2:
        cf.br ^exit_7 
    ^exit_7:
        isq.accumulate_gphase %ssa_352_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":93:5)
        memref.dealloc %ssa_352_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":93:5)
        cf.br ^exit_6 
    ^exit_6:
        isq.accumulate_gphase %ssa_347_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:5)
        memref.dealloc %ssa_347_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":92:5)
        cf.br ^exit_5 
    ^exit_5:
        isq.accumulate_gphase %ssa_335_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":91:5)
        memref.dealloc %ssa_335_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":91:5)
        cf.br ^exit_4 
    ^exit_4:
        isq.accumulate_gphase %ssa_319_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
        memref.dealloc %ssa_319_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":88:5)
        cf.br ^exit_3 
    ^exit_3:
        isq.accumulate_gphase %ssa_317_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":87:5)
        memref.dealloc %ssa_317_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":87:5)
        cf.br ^exit_2 
    ^exit_2:
        isq.accumulate_gphase %ssa_314_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        memref.dealloc %ssa_314_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_312_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        memref.dealloc %ssa_312_real : memref<1xindex> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":86:1)
    func.func @"__isq__main"(memref<?xindex>, memref<?xf64>) 
    {
    ^entry(%ssa_368: memref<?xindex>, %ssa_369: memref<?xf64>):
        %ssa_370 = arith.constant 0 : i1 loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":100:1)
        %ssa_371_real = memref.alloc() : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":100:1)
        %ssa_371_zero = arith.constant 0 : index
        %ssa_371 = memref.subview %ssa_371_real[%ssa_371_zero][1][1] : memref<1xi1> to memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":100:1)
        %ssa_370_store_zero = arith.constant 0: index loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":100:1)
        affine.store %ssa_370, %ssa_371[%ssa_370_store_zero] : memref<1xi1, affine_map<(d0)[s0]->(d0+s0)>> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":100:1)
        cf.br ^exit_1 
    ^exit_1:
        isq.accumulate_gphase %ssa_371_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":100:1)
        memref.dealloc %ssa_371_real : memref<1xi1> loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":100:1)
        cf.br ^exit 
    ^exit:
        return loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":100:1)
    } loc("/home/gjz010/isQ-Compiler/frontend/qmpi.isq":100:1)
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
        call @"__isq__global_initialize"() : ()->() 
        call @"__isq__main"(%ssa_1, %ssa_2) : (memref<?xindex>, memref<?xf64>)->() 
        call @"__isq__global_finalize"() : ()->() 
        return 
    } 
}
