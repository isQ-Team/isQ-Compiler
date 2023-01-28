#onedim = affine_map<(d0)[s0]->(d0+s0)>
memref.global @x : memref<2xindex>
func @main(){
    %g = memref.get_global @x: memref<2xindex>
    %zero = arith.constant 0: index
    %g1 = memref.subview %g[%zero][2][1]: memref<2xindex> to memref<2xindex, #onedim>
    %one = arith.constant 1: index
    %0 = memref.alloc()[%one] : memref<2048xindex, #onedim>
    %i = arith.constant 0: index
    %1 = memref.subview %0[%i][1][1]: memref<2048xindex, affine_map<(d0)[s0]->(d0+s0)>> to memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>>
    %2 = affine.load %1[%i] : memref<1xindex, affine_map<(d0)[s0]->(d0+s0)>>
    return
} loc ("1.c":1:2)

func @val(%a: index)->(index) {
    return %a : index
}

func @foo(){
    return
}
func @test()->(){
    %a = "toy.foo"() : ()->!isq.gate<1>
    %b = "isq.upgrade" (%a) {control_states = [true, false, true], adjoint = false} : (!isq.gate<1>)->!isq.gate<2>
    return
}

memref.global "private" @y : memref<4x!isq.qstate> = uninitialized