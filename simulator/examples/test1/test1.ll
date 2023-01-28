; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%Qubit = type opaque
%Result = type opaque

@a = global [1 x i64] undef
@b = global [1 x i64] undef
@c = global [1 x i64] undef
@q = global [3 x %Qubit*] undef
@p = global [1 x %Qubit*] undef

declare i8* @malloc(i64)

declare void @free(i8*)

declare void @__quantum__rt__qubit_release(%Qubit*)

declare %Qubit* @__quantum__rt__qubit_allocate()

declare i1 @__quantum__rt__result_equal(%Result*, %Result*)

declare %Result* @__quantum__rt__result_get_one()

declare %Result* @__quantum__qis__measure(%Qubit*)

declare void @__quantum__qis__isq_print_i64(i64)

declare void @__quantum__qis__reset(%Qubit*)

define { %Qubit*, %Qubit*, %Qubit* } @H_ctrl_11_adj__qsd__decomposition(%Qubit* %0, %Qubit* %1, %Qubit* %2) {
  call void @__quantum__qis__u3(double 0x3FE0F4F6B200FC57, double 0x40088B7AAA2F489F, double 0x4002D97C7F3321D2, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FEB7BFDFC97BEFE, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0x3FE0F4F6B200FC57, double 0x3FE921FB54442D18, double 0xBFE921FB54442D18, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FE921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FE921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0xBFF2D97C7F3321D3, double 0x3FF921FB54442D18, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFD921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0x3FF921FB54442D18, double 0xBFF921FB54442D18, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  %4 = insertvalue { %Qubit*, %Qubit*, %Qubit* } undef, %Qubit* %0, 0
  %5 = insertvalue { %Qubit*, %Qubit*, %Qubit* } %4, %Qubit* %1, 1
  %6 = insertvalue { %Qubit*, %Qubit*, %Qubit* } %5, %Qubit* %2, 2
  ret { %Qubit*, %Qubit*, %Qubit* } %6
}

define { %Qubit*, %Qubit*, %Qubit* } @H_ctrl_11__qsd__decomposition(%Qubit* %0, %Qubit* %1, %Qubit* %2) {
  call void @__quantum__qis__u3(double 0x3FE0F4F6B200FC57, double 0x40088B7AAA2F489F, double 0x4002D97C7F3321D2, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FEB7BFDFC97BEFE, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0x3FE0F4F6B200FC57, double 0x3FE921FB54442D18, double 0xBFE921FB54442D18, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FE921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FE921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0xBFF2D97C7F3321D3, double 0x3FF921FB54442D18, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFD921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0x3FF921FB54442D18, double 0xBFF921FB54442D18, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  %4 = insertvalue { %Qubit*, %Qubit*, %Qubit* } undef, %Qubit* %0, 0
  %5 = insertvalue { %Qubit*, %Qubit*, %Qubit* } %4, %Qubit* %1, 1
  %6 = insertvalue { %Qubit*, %Qubit*, %Qubit* } %5, %Qubit* %2, 2
  ret { %Qubit*, %Qubit*, %Qubit* } %6
}

define { %Qubit*, %Qubit* } @H_ctrl_0_adj__qsd__decomposition(%Qubit* %0, %Qubit* %1) {
  call void @__quantum__qis__u3(double 0x3FE921FB54442D17, double 0xBFF921FB54442D18, double 0x400921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FF921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0x3FE921FB54442D18, double 0.000000e+00, double 0.000000e+00, %Qubit* %1)
  %3 = insertvalue { %Qubit*, %Qubit* } undef, %Qubit* %0, 0
  %4 = insertvalue { %Qubit*, %Qubit* } %3, %Qubit* %1, 1
  ret { %Qubit*, %Qubit* } %4
}

define { %Qubit*, %Qubit*, %Qubit*, %Qubit* } @H_ctrl_100_adj__qsd__decomposition(%Qubit* %0, %Qubit* %1, %Qubit* %2, %Qubit* %3) {
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0x3FE921FB54442D18, double 0.000000e+00, %Qubit* %3)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FE921FB54442D18, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FE921FB54442D18, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0x3FE27B405369DA9F, double 0x400845C16E9CFCF2, double 0xBFD6C7F8ABF09B32, %Qubit* %3)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF2D97C7F3321D2, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE001E540BED725, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0x3FE27B405369DAA0, double 0xC00648FC3EC619B2, double 0xBFD921FB54442D18, %Qubit* %3)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FD921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FD921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFDB7BFDFC97BEFE, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFDB7BFDFC97BEFE, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0x3FD921FB54442D19, double 0.000000e+00, %Qubit* %3)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FD921FB54442D19, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FD921FB54442D19, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0x3FD24827B8781B3C, double 0xC002F70034F45B77, double 0x4005FDBBE9BBA776, %Qubit* %3)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FD921FB54442D18, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF31483EAB5951B, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0x3FD24827B8781B3B, double 0x3FD921FB54442D17, double 0xBFFF6A7A2955385E, %Qubit* %3)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFD921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFD921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFD921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFD921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FD921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FD921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FD921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FD921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__u3(double 0x3FF60CCDE34BB5A9, double 0xC00300782B4EE9D2, double 0xC0078FDB9EFFEA47, %Qubit* %3)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE9BDEA04B34D1A, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0x3FF60CCDE34BB5A9, double 0xBFC921FB54442D11, double 0xBFF921FB54442D19, %Qubit* %3)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFC921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFC921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0xC002D97C7F3321D2, double 0.000000e+00, %Qubit* %3)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE921FB54442D18, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0xC00921FB54442D18, double 0xBFF921FB54442D18, %Qubit* %3)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  %5 = insertvalue { %Qubit*, %Qubit*, %Qubit*, %Qubit* } undef, %Qubit* %0, 0
  %6 = insertvalue { %Qubit*, %Qubit*, %Qubit*, %Qubit* } %5, %Qubit* %1, 1
  %7 = insertvalue { %Qubit*, %Qubit*, %Qubit*, %Qubit* } %6, %Qubit* %2, 2
  %8 = insertvalue { %Qubit*, %Qubit*, %Qubit*, %Qubit* } %7, %Qubit* %3, 3
  ret { %Qubit*, %Qubit*, %Qubit*, %Qubit* } %8
}

define { %Qubit*, %Qubit*, %Qubit*, %Qubit* } @Rt2_ctrl_01__qsd__decomposition(%Qubit* %0, %Qubit* %1, %Qubit* %2, %Qubit* %3) {
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFC0C15237AB6B20, %Qubit* %3)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFC0C15237AB6B20, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FC0C15237AB6B20, %Qubit* %2)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %2)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFC0C15237AB6B1F, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FC0C15237AB6B1F, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFC0C15237AB6B1F, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FC0C15237AB6B1F, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FC0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFC0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFC0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FC0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FC0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFC0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFC0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FC0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %2, %Qubit* %1)
  call void @__quantum__qis__cnot(%Qubit* %3, %Qubit* %1)
  %5 = insertvalue { %Qubit*, %Qubit*, %Qubit*, %Qubit* } undef, %Qubit* %0, 0
  %6 = insertvalue { %Qubit*, %Qubit*, %Qubit*, %Qubit* } %5, %Qubit* %1, 1
  %7 = insertvalue { %Qubit*, %Qubit*, %Qubit*, %Qubit* } %6, %Qubit* %2, 2
  %8 = insertvalue { %Qubit*, %Qubit*, %Qubit*, %Qubit* } %7, %Qubit* %3, 3
  ret { %Qubit*, %Qubit*, %Qubit*, %Qubit* } %8
}

define { %Qubit*, %Qubit* } @Rs__qsd__decomposition(%Qubit* %0, %Qubit* %1) {
  call void @__quantum__qis__u3(double 0x3FF921FB54B4C996, double 0xBFF921FB54442D18, double 0x400921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FF921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0x3FF921FB54B4C996, double 0.000000e+00, double 0.000000e+00, %Qubit* %1)
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0x3FF0C152386E7788, double 0xC004F1A6C6595250, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FF0C152386E7788, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0xBFE0C15237AB6B1F, double 0x3FE0C15237AB6B20, %Qubit* %1)
  %3 = insertvalue { %Qubit*, %Qubit* } undef, %Qubit* %0, 0
  %4 = insertvalue { %Qubit*, %Qubit* } %3, %Qubit* %1, 1
  ret { %Qubit*, %Qubit* } %4
}

define { %Qubit*, %Qubit* } @Rs2__qsd__decomposition(%Qubit* %0, %Qubit* %1) {
  call void @__quantum__qis__u3(double 0x3FF921FB54B4C996, double 0xBFF921FB54442D18, double 0x400921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FF921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0x3FF921FB54B4C996, double 0.000000e+00, double 0.000000e+00, %Qubit* %1)
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0xBFF0C152386E7788, double 0x4004F1A6C6595250, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF0C152386E7788, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FF921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0x3FE0C15237AB6B1F, double 0xBFE0C15237AB6B20, %Qubit* %1)
  %3 = insertvalue { %Qubit*, %Qubit* } undef, %Qubit* %0, 0
  %4 = insertvalue { %Qubit*, %Qubit* } %3, %Qubit* %1, 1
  ret { %Qubit*, %Qubit* } %4
}

define { %Qubit*, %Qubit* } @Rt__qsd__decomposition(%Qubit* %0, %Qubit* %1) {
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FE0C15237AB6B20, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FE0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  %3 = insertvalue { %Qubit*, %Qubit* } undef, %Qubit* %0, 0
  %4 = insertvalue { %Qubit*, %Qubit* } %3, %Qubit* %1, 1
  ret { %Qubit*, %Qubit* } %4
}

define { %Qubit*, %Qubit* } @Rt2__qsd__decomposition(%Qubit* %0, %Qubit* %1) {
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE0C15237AB6B20, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFE0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FE0C15237AB6B1F, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  %3 = insertvalue { %Qubit*, %Qubit* } undef, %Qubit* %0, 0
  %4 = insertvalue { %Qubit*, %Qubit* } %3, %Qubit* %1, 1
  ret { %Qubit*, %Qubit* } %4
}

define { %Qubit*, %Qubit* } @CNOT__qsd__decomposition(%Qubit* %0, %Qubit* %1) {
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0x3FF921FB54442D18, double 0x400921FB54442D18, %Qubit* %1)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0xBFF921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0.000000e+00, double 0.000000e+00, double 0x3FF921FB54442D18, %Qubit* %0)
  call void @__quantum__qis__cnot(%Qubit* %1, %Qubit* %0)
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0.000000e+00, double 0.000000e+00, %Qubit* %1)
  %3 = insertvalue { %Qubit*, %Qubit* } undef, %Qubit* %0, 0
  %4 = insertvalue { %Qubit*, %Qubit* } %3, %Qubit* %1, 1
  ret { %Qubit*, %Qubit* } %4
}

define %Qubit* @H__qsd__decomposition(%Qubit* %0) {
  call void @__quantum__qis__u3(double 0x3FF921FB54442D18, double 0.000000e+00, double 0x400921FB54442D18, %Qubit* %0)
  ret %Qubit* %0
}

declare void @__quantum__qis__u3(double, double, double, %Qubit*)

declare void @__quantum__qis__cnot(%Qubit*, %Qubit*)

define i64 @test(%Qubit** %0, %Qubit** %1, i64 %2, i64 %3, i64 %4, %Qubit** %5, %Qubit** %6, i64 %7, i64 %8, i64 %9, i64 %10) !dbg !3 {
  %12 = call i8* @malloc(i64 ptrtoint (i64* getelementptr (i64, i64* null, i64 1) to i64)), !dbg !7
  %13 = bitcast i8* %12 to i64*, !dbg !7
  %14 = add i64 %2, 0, !dbg !9
  %15 = getelementptr %Qubit*, %Qubit** %1, i64 %14, !dbg !9
  %16 = load %Qubit*, %Qubit** %15, align 8, !dbg !9
  %17 = call %Qubit* @H__qsd__decomposition(%Qubit* %16), !dbg !9
  %18 = call i8* @malloc(i64 add (i64 ptrtoint (%Qubit** getelementptr (%Qubit*, %Qubit** null, i64 1) to i64), i64 ptrtoint (%Qubit** getelementptr (%Qubit*, %Qubit** null, i64 1) to i64))), !dbg !10
  %19 = bitcast i8* %18 to %Qubit**, !dbg !10
  %20 = ptrtoint %Qubit** %19 to i64, !dbg !10
  %21 = add i64 %20, sub (i64 ptrtoint (%Qubit** getelementptr (%Qubit*, %Qubit** null, i64 1) to i64), i64 1), !dbg !10
  %22 = urem i64 %21, ptrtoint (%Qubit** getelementptr (%Qubit*, %Qubit** null, i64 1) to i64), !dbg !10
  %23 = sub i64 %21, %22, !dbg !10
  %24 = inttoptr i64 %23 to %Qubit**, !dbg !10
  %25 = call %Qubit* @__quantum__rt__qubit_allocate(), !dbg !10
  store %Qubit* %25, %Qubit** %24, align 8, !dbg !10
  %26 = load %Qubit*, %Qubit** %24, align 8, !dbg !11
  %27 = load %Qubit*, %Qubit** %15, align 8, !dbg !11
  %28 = call { %Qubit*, %Qubit* } @CNOT__qsd__decomposition(%Qubit* %26, %Qubit* %27), !dbg !11
  %29 = load %Qubit*, %Qubit** %15, align 8, !dbg !12
  %30 = call %Qubit* @H__qsd__decomposition(%Qubit* %29), !dbg !12
  store i64 2, i64* %13, align 4, !dbg !13
  call void @free(i8* %18), !dbg !10
  %31 = load i64, i64* %13, align 4, !dbg !7
  call void @free(i8* %12), !dbg !7
  ret i64 %31, !dbg !7
}

define void @test2(%Qubit** %0, %Qubit** %1, i64 %2, i64 %3, i64 %4, i64 %5) !dbg !14 {
  %7 = call i8* @malloc(i64 ptrtoint (i64* getelementptr (i64, i64* null, i64 1) to i64)), !dbg !15
  %8 = bitcast i8* %7 to i64*, !dbg !15
  store i64 %5, i64* %8, align 4, !dbg !15
  %9 = load i64, i64* %8, align 4, !dbg !17
  %10 = mul i64 %9, %4, !dbg !18
  %11 = add i64 %2, %10, !dbg !18
  %12 = add i64 %11, 0, !dbg !19
  %13 = getelementptr %Qubit*, %Qubit** %1, i64 %12, !dbg !19
  %14 = load %Qubit*, %Qubit** %13, align 8, !dbg !19
  %15 = call %Qubit* @H__qsd__decomposition(%Qubit* %14), !dbg !19
  call void @free(i8* %7), !dbg !15
  ret void, !dbg !15
}

define void @__isq__main() !dbg !20 {
  %1 = call i8* @malloc(i64 ptrtoint (i1* getelementptr (i1, i1* null, i64 1) to i64)), !dbg !21
  %2 = bitcast i8* %1 to i1*, !dbg !21
  store i1 false, i1* %2, align 1, !dbg !21
  %3 = call i8* @malloc(i64 ptrtoint (i64* getelementptr (i64, i64* null, i64 1) to i64)), !dbg !23
  %4 = bitcast i8* %3 to i64*, !dbg !23
  %5 = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @a, i64 0, i64 0), align 4, !dbg !24
  %6 = add i64 %5, 6, !dbg !25
  %7 = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @b, i64 0, i64 0), align 4, !dbg !26
  %8 = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @c, i64 0, i64 0), align 4, !dbg !27
  %9 = add i64 %7, %8, !dbg !28
  %10 = mul i64 %6, %9, !dbg !29
  store i64 %10, i64* %4, align 4, !dbg !30
  %11 = load i1, i1* %2, align 1, !dbg !31
  br i1 %11, label %74, label %12, !dbg !31

12:                                               ; preds = %0
  %13 = load i1, i1* %2, align 1, !dbg !32
  br i1 %13, label %74, label %14, !dbg !32

14:                                               ; preds = %12
  %15 = call i8* @malloc(i64 ptrtoint (i64* getelementptr (i64, i64* null, i64 5) to i64)), !dbg !33
  %16 = bitcast i8* %15 to i64*, !dbg !33
  %17 = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @c, i64 0, i64 0), align 4, !dbg !34
  %18 = mul i64 %17, 1, !dbg !35
  %19 = add i64 %18, 0, !dbg !35
  %20 = add i64 %19, 0, !dbg !35
  %21 = getelementptr i64, i64* %16, i64 %20, !dbg !35
  %22 = load i64, i64* %21, align 4, !dbg !35
  %23 = add i64 %22, 2, !dbg !36
  store i64 %23, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @a, i64 0, i64 0), align 4, !dbg !37
  %24 = load i64, i64* %4, align 4, !dbg !38
  %25 = call i64 @test(%Qubit** inttoptr (i64 3735928559 to %Qubit**), %Qubit** getelementptr inbounds ([1 x %Qubit*], [1 x %Qubit*]* @p, i64 0, i64 0), i64 0, i64 1, i64 1, %Qubit** inttoptr (i64 3735928559 to %Qubit**), %Qubit** getelementptr inbounds ([1 x %Qubit*], [1 x %Qubit*]* @p, i64 0, i64 0), i64 0, i64 1, i64 1, i64 %24), !dbg !39
  store i64 %25, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @b, i64 0, i64 0), align 4, !dbg !40
  %26 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !41
  %27 = call %Result* @__quantum__qis__measure(%Qubit* %26), !dbg !41
  %28 = call %Result* @__quantum__rt__result_get_one(), !dbg !41
  %29 = call i1 @__quantum__rt__result_equal(%Result* %27, %Result* %28), !dbg !41
  %30 = zext i1 %29 to i2, !dbg !41
  %31 = sext i2 %30 to i64, !dbg !41
  store i64 %31, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @a, i64 0, i64 0), align 4, !dbg !42
  %32 = load %Qubit*, %Qubit** getelementptr inbounds ([1 x %Qubit*], [1 x %Qubit*]* @p, i64 0, i64 0), align 8, !dbg !43
  %33 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !43
  %34 = call { %Qubit*, %Qubit* } @CNOT__qsd__decomposition(%Qubit* %32, %Qubit* %33), !dbg !43
  %35 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !44
  %36 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !44
  %37 = load %Qubit*, %Qubit** getelementptr inbounds ([1 x %Qubit*], [1 x %Qubit*]* @p, i64 0, i64 0), align 8, !dbg !44
  %38 = call { %Qubit*, %Qubit*, %Qubit* } @H_ctrl_11_adj__qsd__decomposition(%Qubit* %35, %Qubit* %36, %Qubit* %37), !dbg !44
  %39 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !45
  %40 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !45
  %41 = load %Qubit*, %Qubit** getelementptr inbounds ([1 x %Qubit*], [1 x %Qubit*]* @p, i64 0, i64 0), align 8, !dbg !45
  %42 = call { %Qubit*, %Qubit*, %Qubit* } @H_ctrl_11__qsd__decomposition(%Qubit* %39, %Qubit* %40, %Qubit* %41), !dbg !45
  %43 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !46
  %44 = load %Qubit*, %Qubit** getelementptr inbounds ([1 x %Qubit*], [1 x %Qubit*]* @p, i64 0, i64 0), align 8, !dbg !46
  %45 = call { %Qubit*, %Qubit* } @H_ctrl_0_adj__qsd__decomposition(%Qubit* %43, %Qubit* %44), !dbg !46
  %46 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !47
  %47 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !47
  %48 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 2), align 8, !dbg !47
  %49 = load %Qubit*, %Qubit** getelementptr inbounds ([1 x %Qubit*], [1 x %Qubit*]* @p, i64 0, i64 0), align 8, !dbg !47
  %50 = call { %Qubit*, %Qubit*, %Qubit*, %Qubit* } @H_ctrl_100_adj__qsd__decomposition(%Qubit* %46, %Qubit* %47, %Qubit* %48, %Qubit* %49), !dbg !47
  %51 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), align 8, !dbg !48
  %52 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 2), align 8, !dbg !48
  %53 = load %Qubit*, %Qubit** getelementptr inbounds ([1 x %Qubit*], [1 x %Qubit*]* @p, i64 0, i64 0), align 8, !dbg !48
  %54 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !48
  %55 = call { %Qubit*, %Qubit*, %Qubit*, %Qubit* } @Rt2_ctrl_01__qsd__decomposition(%Qubit* %51, %Qubit* %52, %Qubit* %53, %Qubit* %54), !dbg !48
  %56 = call i8* @malloc(i64 ptrtoint (i1* getelementptr (i1, i1* null, i64 1) to i64)), !dbg !49
  %57 = bitcast i8* %56 to i1*, !dbg !49
  store i1 false, i1* %57, align 1, !dbg !49
  br label %58, !dbg !49

58:                                               ; preds = %67, %14
  br label %59, !dbg !50

59:                                               ; preds = %58
  %60 = load i1, i1* %57, align 1, !dbg !49
  br i1 %60, label %64, label %61, !dbg !53

61:                                               ; preds = %59
  %62 = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @a, i64 0, i64 0), align 4, !dbg !54
  %63 = icmp slt i64 %62, 2, !dbg !55
  br label %65, !dbg !50

64:                                               ; preds = %59
  br label %65, !dbg !50

65:                                               ; preds = %64, %61
  %66 = phi i1 [ false, %64 ], [ %63, %61 ]
  br i1 %66, label %67, label %70, !dbg !56

67:                                               ; preds = %65
  %68 = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @a, i64 0, i64 0), align 4, !dbg !57
  %69 = add i64 %68, 1, !dbg !58
  store i64 %69, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @a, i64 0, i64 0), align 4, !dbg !59
  br label %58, !dbg !60

70:                                               ; preds = %65
  %71 = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @a, i64 0, i64 0), align 4, !dbg !61
  call void @__quantum__qis__isq_print_i64(i64 %71), !dbg !62
  %72 = load %Qubit*, %Qubit** getelementptr inbounds ([1 x %Qubit*], [1 x %Qubit*]* @p, i64 0, i64 0), align 8, !dbg !63
  call void @__quantum__qis__reset(%Qubit* %72), !dbg !63
  %73 = load %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 1), align 8, !dbg !64
  call void @__quantum__qis__reset(%Qubit* %73), !dbg !64
  call void @free(i8* %56), !dbg !49
  call void @free(i8* %15), !dbg !33
  br label %74, !dbg !65

74:                                               ; preds = %70, %12, %0
  call void @free(i8* %3), !dbg !23
  call void @free(i8* %1), !dbg !21
  ret void, !dbg !21
}

define void @__isq__global_initialize() !dbg !66 {
  br label %1, !dbg !67

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %7, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 3, !dbg !67
  br i1 %3, label %4, label %8, !dbg !67

4:                                                ; preds = %1
  %5 = call %Qubit* @__quantum__rt__qubit_allocate(), !dbg !67
  %6 = getelementptr %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), i64 %2, !dbg !67
  store %Qubit* %5, %Qubit** %6, align 8, !dbg !67
  %7 = add i64 %2, 1, !dbg !67
  br label %1, !dbg !67

8:                                                ; preds = %1
  %9 = call %Qubit* @__quantum__rt__qubit_allocate(), !dbg !67
  store %Qubit* %9, %Qubit** getelementptr inbounds ([1 x %Qubit*], [1 x %Qubit*]* @p, i64 0, i64 0), align 8, !dbg !67
  ret void, !dbg !69
}

define void @__isq__global_finalize() !dbg !71 {
  br label %1, !dbg !72

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %7, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 3, !dbg !72
  br i1 %3, label %4, label %8, !dbg !72

4:                                                ; preds = %1
  %5 = getelementptr %Qubit*, %Qubit** getelementptr inbounds ([3 x %Qubit*], [3 x %Qubit*]* @q, i64 0, i64 0), i64 %2, !dbg !72
  %6 = load %Qubit*, %Qubit** %5, align 8, !dbg !72
  call void @__quantum__rt__qubit_release(%Qubit* %6), !dbg !72
  %7 = add i64 %2, 1, !dbg !72
  br label %1, !dbg !72

8:                                                ; preds = %1
  %9 = load %Qubit*, %Qubit** getelementptr inbounds ([1 x %Qubit*], [1 x %Qubit*]* @p, i64 0, i64 0), align 8, !dbg !72
  call void @__quantum__rt__qubit_release(%Qubit* %9), !dbg !72
  ret void, !dbg !74
}

define void @__isq__entry() !dbg !76 {
  call void @__isq__global_initialize(), !dbg !77
  call void @__isq__main(), !dbg !79
  call void @__isq__global_finalize(), !dbg !80
  ret void, !dbg !81
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test", linkageName: "test", scope: null, file: !4, line: 34, type: !5, scopeLine: 34, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "<stdin>", directory: "/home/gjz010/isqv2/mlir/build")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 34, column: 1, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 35, column: 9, scope: !8)
!10 = !DILocation(line: 36, column: 9, scope: !8)
!11 = !DILocation(line: 37, column: 9, scope: !8)
!12 = !DILocation(line: 38, column: 9, scope: !8)
!13 = !DILocation(line: 39, column: 9, scope: !8)
!14 = distinct !DISubprogram(name: "test2", linkageName: "test2", scope: null, file: !4, line: 42, type: !5, scopeLine: 42, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!15 = !DILocation(line: 42, column: 1, scope: !16)
!16 = !DILexicalBlockFile(scope: !14, file: !4, discriminator: 0)
!17 = !DILocation(line: 43, column: 13, scope: !16)
!18 = !DILocation(line: 43, column: 12, scope: !16)
!19 = !DILocation(line: 43, column: 9, scope: !16)
!20 = distinct !DISubprogram(name: "__isq__main", linkageName: "__isq__main", scope: null, file: !4, line: 46, type: !5, scopeLine: 46, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!21 = !DILocation(line: 46, column: 1, scope: !22)
!22 = !DILexicalBlockFile(scope: !20, file: !4, discriminator: 0)
!23 = !DILocation(line: 48, column: 9, scope: !22)
!24 = !DILocation(line: 51, column: 22, scope: !22)
!25 = !DILocation(line: 51, column: 23, scope: !22)
!26 = !DILocation(line: 51, column: 30, scope: !22)
!27 = !DILocation(line: 51, column: 32, scope: !22)
!28 = !DILocation(line: 51, column: 31, scope: !22)
!29 = !DILocation(line: 51, column: 28, scope: !22)
!30 = !DILocation(line: 51, column: 19, scope: !22)
!31 = !DILocation(line: 50, column: 9, scope: !22)
!32 = !DILocation(line: 56, column: 9, scope: !22)
!33 = !DILocation(line: 59, column: 9, scope: !22)
!34 = !DILocation(line: 60, column: 15, scope: !22)
!35 = !DILocation(line: 60, column: 14, scope: !22)
!36 = !DILocation(line: 60, column: 17, scope: !22)
!37 = !DILocation(line: 60, column: 11, scope: !22)
!38 = !DILocation(line: 61, column: 20, scope: !22)
!39 = !DILocation(line: 61, column: 9, scope: !22)
!40 = !DILocation(line: 61, column: 7, scope: !22)
!41 = !DILocation(line: 62, column: 9, scope: !22)
!42 = !DILocation(line: 62, column: 7, scope: !22)
!43 = !DILocation(line: 63, column: 9, scope: !22)
!44 = !DILocation(line: 65, column: 21, scope: !22)
!45 = !DILocation(line: 66, column: 17, scope: !22)
!46 = !DILocation(line: 67, column: 19, scope: !22)
!47 = !DILocation(line: 68, column: 27, scope: !22)
!48 = !DILocation(line: 69, column: 20, scope: !22)
!49 = !DILocation(line: 71, column: 9, scope: !22)
!50 = !DILocation(line: 423, column: 21, scope: !51)
!51 = !DILexicalBlockFile(scope: !20, file: !52, discriminator: 0)
!52 = !DIFile(filename: "../tests/test1.mlir", directory: "/home/gjz010/isqv2/mlir/build")
!53 = !DILocation(line: 427, column: 21, scope: !51)
!54 = !DILocation(line: 71, column: 16, scope: !22)
!55 = !DILocation(line: 71, column: 18, scope: !22)
!56 = !DILocation(line: 441, column: 13, scope: !51)
!57 = !DILocation(line: 72, column: 21, scope: !22)
!58 = !DILocation(line: 72, column: 23, scope: !22)
!59 = !DILocation(line: 72, column: 19, scope: !22)
!60 = !DILocation(line: 461, column: 9, scope: !51)
!61 = !DILocation(line: 75, column: 15, scope: !22)
!62 = !DILocation(line: 75, column: 9, scope: !22)
!63 = !DILocation(line: 76, column: 9, scope: !22)
!64 = !DILocation(line: 77, column: 10, scope: !22)
!65 = !DILocation(line: 497, column: 9, scope: !51)
!66 = distinct !DISubprogram(name: "__isq__global_initialize", linkageName: "__isq__global_initialize", scope: null, file: !52, line: 513, type: !5, scopeLine: 513, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!67 = !DILocation(line: 32, column: 1, scope: !68)
!68 = !DILexicalBlockFile(scope: !66, file: !4, discriminator: 0)
!69 = !DILocation(line: 516, column: 9, scope: !70)
!70 = !DILexicalBlockFile(scope: !66, file: !52, discriminator: 0)
!71 = distinct !DISubprogram(name: "__isq__global_finalize", linkageName: "__isq__global_finalize", scope: null, file: !52, line: 518, type: !5, scopeLine: 518, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!72 = !DILocation(line: 32, column: 1, scope: !73)
!73 = !DILexicalBlockFile(scope: !71, file: !4, discriminator: 0)
!74 = !DILocation(line: 521, column: 9, scope: !75)
!75 = !DILexicalBlockFile(scope: !71, file: !52, discriminator: 0)
!76 = distinct !DISubprogram(name: "__isq__entry", linkageName: "__isq__entry", scope: null, file: !52, line: 523, type: !5, scopeLine: 523, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!77 = !DILocation(line: 526, column: 9, scope: !78)
!78 = !DILexicalBlockFile(scope: !76, file: !52, discriminator: 0)
!79 = !DILocation(line: 527, column: 9, scope: !78)
!80 = !DILocation(line: 528, column: 9, scope: !78)
!81 = !DILocation(line: 529, column: 9, scope: !78)
