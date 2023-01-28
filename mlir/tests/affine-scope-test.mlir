func @test(){
    %a = memref.alloc() : memref<1xi1>
    %b = affine.load %a[0] : memref<1xi1>
    %zero = arith.constant false
    scf.if %zero{
        %c = affine.load %a[0] : memref<1xi1>
        scf.yield
    }
    
    scf.execute_region {
        ^bb0:
        %c = affine.load %a[0] : memref<1xi1>
        scf.yield
    }
    return
}