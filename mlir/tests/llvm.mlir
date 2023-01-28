func @main(){
    %a = memref.alloc() : memref<1x!isq.qir.qubit>
    return
}