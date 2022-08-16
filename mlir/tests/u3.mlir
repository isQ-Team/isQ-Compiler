isq.defgate @u3(f64, f64, f64) {definition = [{type = "qir", value = "__quantum__qis__u3"}]} : !isq.gate<1>

func @main(){
    %zero = arith.constant 0.0 : f64
    %a = isq.use @u3(%zero, %zero, %zero) : (f64, f64, f64)->!isq.gate<1>
    return
}