func @main(){
    scf.while : ()->(){
        %cond = scf.execute_region -> i1{
            ^break_check:
            %a=arith.constant 1: i1
            cond_br %a, ^break, ^while_cond 
            ^while_cond:
            %b=arith.constant 1: i1
            scf.yield %b: i1
            ^break:
            %zero=arith.constant 0: i1
            scf.yield %zero: i1
        }
        scf.condition(%cond)
    }do{
        scf.yield
    }
    return
}