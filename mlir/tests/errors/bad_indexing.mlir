module{
	func @main()->(){
		%a = constant 0: i64
		%b = memref.alloca() : memref<10xi32>
		%c = memref.load %b[%a] : memref<10xi32>
		return 
	}
}
