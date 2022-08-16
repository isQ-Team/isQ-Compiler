builtin.func @main(%N: index){
	affine.for %i = 1 to %N step 1 {
                %ii = affine.apply affine_map<(i) -> (i)> (%i)
		affine.for %j = %ii to %N step 1{
			affine.yield
		}
		affine.yield
	}
	return
}
