// Generates both LLVM and Rust shims simultaneously.
// Since there are many kinds of signatures, we use a DSL to describe each intrinsic function.

// There will be 3 shims:
// One written in LLVM IR, providing QIR interfaces by callling primitive interfaces.
// One written in Rust, providing primitive interfaces by calling Rust interfaces.
// One written in Rust, providing Rust interfaces.
extern crate std;
type StructExpander = fn(ssa: &str, &mut QIRInterface) -> (Vec<String>, Vec<(QIRType, String)>);
#[derive(Copy, Clone)]
enum QIRType {
    Qubit,
    Array,
    BigInt,
    QString,
    Tuple,
    Callable,
    Int,
    Double,
    Pauli,
    Result,
    QVoid,
    QI1,
    QI32,
    QI64,
    QI8P,
    QI64P,
    MeasurementProbabilityArgs,
    Other(&'static str, StructExpander),
}
use QIRType::*;
const QIRPauli: QIRType = Other("%Pauli", |ssa, qir| {
    let new_ssa = qir.next_ssa();
    (
        vec![format!("{} = zext %Pauli {} to i8", new_ssa, ssa)],
        vec![(Pauli, new_ssa)],
    )
});
const Range: QIRType = Other("%Range", |ssa, qir|{
    let mut code = Vec::new();
    let struct_memory = qir.next_ssa();
    let (start_ptr, step_ptr, end_ptr) = (qir.next_ssa(), qir.next_ssa(), qir.next_ssa());
    let (start, step, end) = (qir.next_ssa(), qir.next_ssa(), qir.next_ssa());
    code.push(format!("{} = alloca %Range", struct_memory));
    code.push(format!("store %Range {}, %Range* {}", ssa, struct_memory));
    code.push(format!("{} = getelementptr inbounds %Range, %Range* {}, i64 0, i32 0", start_ptr, struct_memory));
    code.push(format!("{} = getelementptr inbounds %Range, %Range* {}, i64 0, i32 1", step_ptr, struct_memory));
    code.push(format!("{} = getelementptr inbounds %Range, %Range* {}, i64 0, i32 2", end_ptr, struct_memory));
    code.push(format!("{} = load i64, i64* {}", start, start_ptr));
    code.push(format!("{} = load i64, i64* {}", step, step_ptr));
    code.push(format!("{} = load i64, i64* {}", end, end_ptr));
    (code, [start, step, end].into_iter().map(|x| (QI64, x)).collect())
});
const CallableTable: QIRType = Other("[4 x void (%Tuple*, %Tuple*, %Tuple*)*]*", |ssa, qir| {
    let mut code = Vec::new();
    let address_ssa = vec![
        qir.next_ssa(),
        qir.next_ssa(),
        qir.next_ssa(),
        qir.next_ssa(),
    ];
    let f_ssa = vec![
        qir.next_ssa(),
        qir.next_ssa(),
        qir.next_ssa(),
        qir.next_ssa(),
    ];
    let new_ssa = vec![
        qir.next_ssa(),
        qir.next_ssa(),
        qir.next_ssa(),
        qir.next_ssa(),
    ];

    for i in 0..4 {
        code.push(format!("{} = getelementptr inbounds [4 x void (%Tuple*, %Tuple*, %Tuple*)*],[4 x void (%Tuple*, %Tuple*, %Tuple*)*]* {}, i32 0, i32 {}", address_ssa[i], ssa, i));
        code.push(format!(
            "{} = load void (%Tuple*, %Tuple*, %Tuple*)*, void (%Tuple*, %Tuple*, %Tuple*)** {}",
            f_ssa[i], address_ssa[i]
        ));
        code.push(format!(
            "{} = bitcast void (%Tuple*, %Tuple*, %Tuple*)* {} to i8*",
            new_ssa[i], f_ssa[i]
        ));
    }

    (code, new_ssa.into_iter().map(|x| (QI8P, x)).collect())
});
const MemManTable: QIRType = Other("[2 x void(%Tuple*, i32)*]*", |ssa, qir| {
    let mut code = Vec::new();
    let flag = qir.next_ssa();
    let x = (0..8).map(|x| qir.next_ssa()).collect::<Vec<_>>();
    let mem_table_label = qir.next_label_name();
    let read_label = qir.next_label_name();
    let next_label = qir.next_label_name();
    code.push(format!("br label %{}
{}:
    {} = icmp eq [2 x void(%Tuple*, i32)*]* {}, null
    br i1 {}, label %{}, label %{}
{}:
    {} = getelementptr inbounds [2 x void(%Tuple*, i32)*], [2 x void(%Tuple*, i32)*]* {}, i64 0, i64 0
    {} = load void(%Tuple*, i32)*, void(%Tuple*, i32)**  {}
    {} = bitcast void(%Tuple*, i32)* {} to i8*
    {} = getelementptr inbounds [2 x void(%Tuple*, i32)*], [2 x void(%Tuple*, i32)*]* {}, i64 0, i64 1
    {} = load void(%Tuple*, i32)*, void(%Tuple*, i32)** {}
    {} = bitcast void(%Tuple*, i32)* {} to i8*
    br label %{}
{}:
    {} = phi i8* [null, %{}], [{}, %{}]
    {} = phi i8* [null, %{}], [{}, %{}]", mem_table_label,
mem_table_label, flag, ssa, flag, next_label, read_label,
read_label,
x[0], ssa, x[1], x[0], x[2], x[1],
x[3], ssa, x[4], x[3], x[5], x[4], next_label,
next_label,
x[6], mem_table_label, x[2], read_label,
x[7], mem_table_label, x[5], read_label
));
    (code, vec![(QI8P, x[6].clone()), (QI8P, x[7].clone())])
});

const RotTuple: QIRType = Other("{ i2, double, %Qubit* }*", |ssa, qir| {
    let mut code: Vec<String> = vec![];
    let addr_ssa = vec![qir.next_ssa(), qir.next_ssa(), qir.next_ssa()];
    let value_ssa = vec![qir.next_ssa(), qir.next_ssa()];
    let result_ssa = vec![
        (Pauli, qir.next_ssa()),
        (Double, qir.next_ssa()),
        (Qubit, qir.next_ssa()),
    ];
    code.push(format!("{} = getelementptr inbounds {{ i2, double, %Qubit* }}, {{ i2, double, %Qubit* }}* {}, i64 0, i32 0", addr_ssa[0], ssa));
    code.push(format!("{} = getelementptr inbounds {{ i2, double, %Qubit* }}, {{ i2, double, %Qubit* }}* {}, i64 0, i32 1", addr_ssa[1], ssa));
    code.push(format!("{} = getelementptr inbounds {{ i2, double, %Qubit* }}, {{ i2, double, %Qubit* }}* {}, i64 0, i32 2", addr_ssa[2], ssa));
    code.push(format!("{} = load i2, i2* {}", value_ssa[0], addr_ssa[0]));
    code.push(format!(
        "{} = load double, double* {}",
        result_ssa[1].1, addr_ssa[1]
    ));
    code.push(format!(
        "{} = load %Qubit*, %Qubit** {}",
        value_ssa[1], addr_ssa[2]
    ));
    code.push(format!(
        "{} = zext %Pauli {} to i8",
        result_ssa[0].1, value_ssa[0]
    ));
    code.push(format!(
        "{} = bitcast %Qubit* {} to i8*",
        result_ssa[2].1, value_ssa[1]
    ));

    (code, result_ssa)
});

impl QIRType {
    fn to_qir_type(&self) -> &'static str {
        match self {
            QIRType::Qubit => "%Qubit*",
            QIRType::Array => "%Array*",
            QIRType::BigInt => "%BigInt*",
            QIRType::QString => "%String*",
            QIRType::Tuple => "%Tuple*",
            QIRType::Callable => "%Callable*",
            QIRType::Int => "i64",
            QIRType::Double => "double",
            QIRType::Pauli => "i8",
            QIRType::Result => "%Result*",
            QIRType::QVoid => "void",
            QIRType::QI1 => "i1",
            QIRType::QI32 => "i32",
            QIRType::QI64 => "i64",
            QIRType::QI8P => "i8*",
            QIRType::QI64P => "i64*",
            QIRType::MeasurementProbabilityArgs => {
                "{ %Array, %Array, %Result, double, %String, double }*"
            }
            QIRType::Other(s, _) => s,
        }
    }
    fn to_primitive_type(&self) -> PrimitiveType {
        match self {
            QIRType::Qubit => PrimitiveType::Pointer,
            QIRType::Array => PrimitiveType::Pointer,
            QIRType::BigInt => PrimitiveType::Pointer,
            QIRType::QString => PrimitiveType::Pointer,
            QIRType::Tuple => PrimitiveType::Pointer,
            QIRType::Callable => PrimitiveType::Pointer,
            QIRType::Int => PrimitiveType::I64,
            QIRType::Double => PrimitiveType::F64,
            QIRType::Pauli => PrimitiveType::I8,
            QIRType::Result => PrimitiveType::Pointer,
            QIRType::QVoid => PrimitiveType::Void,
            QIRType::QI1 => PrimitiveType::I1,
            QIRType::QI32 => PrimitiveType::I32,
            QIRType::QI64 => PrimitiveType::I64,
            QIRType::QI8P => PrimitiveType::Pointer,
            QIRType::QI64P => PrimitiveType::Pointer,
            QIRType::MeasurementProbabilityArgs => PrimitiveType::Pointer,
            QIRType::Other(_, _) => panic!("Non primitive type"),
        }
    }
    fn to_rust_type(&self) -> &'static str {
        match self {
            QIRType::Qubit => "K<QIRQubit>",
            QIRType::Array => "K<QIRArray>",
            QIRType::BigInt => "K<QIRBigInt>",
            QIRType::QString => "K<QIRString>",
            QIRType::Tuple => "TupleBodyPtr",
            QIRType::Callable => "K<QIRCallable>",
            QIRType::Int => "i64",
            QIRType::Double => "f64",
            QIRType::Pauli => "QIRPauli",
            QIRType::Result => "QIRResult",
            QIRType::QVoid => "()",
            QIRType::QI1 => "bool",
            QIRType::QI32 => "i32",
            QIRType::QI64 => "i64",
            QIRType::QI8P => "*mut i8",
            QIRType::QI64P => "*mut i64",
            QIRType::MeasurementProbabilityArgs => "*const MeasurementProbabilityArgs",
            QIRType::Other(_, _) => panic!("Non primitive type"),
        }
    }
}
#[derive(Copy, Clone)]
enum PrimitiveType {
    Pointer,
    Void,
    I64,
    I8,
    I32,
    F64,
    I1,
}
impl PrimitiveType {
    fn to_llvm_type(&self) -> &'static str {
        match self {
            PrimitiveType::Pointer => "i8*",
            PrimitiveType::Void => "void",
            PrimitiveType::I64 => "i64",
            PrimitiveType::I8 => "i8",
            PrimitiveType::I32 => "i32",
            PrimitiveType::F64 => "double",
            PrimitiveType::I1 => "i1",
        }
    }
    fn to_rust_type(&self) -> &'static str {
        match self {
            PrimitiveType::Pointer => "*mut i8",
            PrimitiveType::Void => "()",
            PrimitiveType::I64 => "i64",
            PrimitiveType::I8 => "i8",
            PrimitiveType::I32 => "i32",
            PrimitiveType::F64 => "f64",
            PrimitiveType::I1 => "bool",
        }
    }
}

use itertools::Itertools;
struct QIRInterface {
    category: &'static str,
    name: &'static str,
    return_type: QIRType,
    args: &'static [QIRType],
    ssa: usize,
    label: usize,
    var_name: usize,
}

impl QIRInterface {
    fn new(
        category: &'static str,
        name: &'static str,
        return_type: QIRType,
        args: &'static [QIRType],
    ) -> Self {
        QIRInterface {
            category,
            name,
            return_type,
            args,
            ssa: 0,
            label: 0,
            var_name: 0,
        }
    }
    fn next_ssa(&mut self) -> String {
        let s = format!("%x{}", self.ssa);
        self.ssa += 1;
        s
    }
    fn next_label_name(&mut self) -> String {
        let s = format!("label_{}", self.label);
        self.label += 1;
        s
    }
    fn next_var_name(&mut self) -> String {
        let s = format!("x{}", self.var_name);
        self.var_name += 1;
        s
    }
    fn qir_name(&self) -> String {
        format!("__quantum__{}__{}", self.category, self.name)
    }
    fn interim_name(&self) -> String {
        format!("__isq__qir__shim__{}__{}", self.category, self.name)
    }
    fn rust_shim_name(&self) -> String {
        format!(
            "isq_qir_shim_{}_{}",
            self.category,
            self.name.replace("__", "_")
        )
    }
    fn codegen(&mut self) -> (String, String, String) {
        let mut qir_code: Vec<String> = Vec::new();
        let mut interim_code: Vec<String> = Vec::new();
        let mut rust_code: Vec<String> = Vec::new();
        let qir_args: Vec<(QIRType, String)> = self
            .args
            .iter()
            .map(|t| (t.clone(), self.next_ssa()))
            .collect();
        let mut interim_args: Vec<(QIRType, String)> = vec![];
        qir_code.push(format!(
            "define {} @{} ({}) alwaysinline {{\nentry:",
            self.return_type.to_qir_type(),
            self.qir_name(),
            qir_args
                .iter()
                .map(|(t, s)| format!("{} {}", t.to_qir_type(), s))
                .join(", ")
        ));
        for qir_arg in qir_args.iter() {
            if let Other(_, ssa_expand) = qir_arg.0 {
                let (conversion_code, new_interim_args) = ssa_expand(&qir_arg.1, self);
                qir_code.extend(conversion_code.into_iter().map(|x| format!("    {}", x)));
                interim_args.extend(new_interim_args.into_iter());
            } else {
                let new_ssa = self.next_ssa();
                let interim_type = qir_arg.0.to_primitive_type().to_llvm_type();
                qir_code.push(format!(
                    "    {} = bitcast {} {} to {}",
                    new_ssa,
                    qir_arg.0.to_qir_type(),
                    qir_arg.1,
                    interim_type
                ));
                interim_args.push((qir_arg.0, new_ssa))
            }
        }
        let call_stmt = format!(
            "call {} @{}({})",
            self.return_type.to_primitive_type().to_llvm_type(),
            self.interim_name(),
            interim_args
                .iter()
                .map(|x| format!("{} {}", x.0.to_primitive_type().to_llvm_type(), x.1))
                .join(", ")
        );
        if let QVoid = self.return_type {
            qir_code.push(format!("    {}", call_stmt));
            qir_code.push("    ret void".to_string());
        } else {
            let ssa = self.next_ssa();
            qir_code.push(format!("    {} = {}", ssa, call_stmt));
            let cast_ssa = self.next_ssa();
            qir_code.push(format!(
                "    {} = bitcast {} {} to {}",
                cast_ssa,
                self.return_type.to_primitive_type().to_llvm_type(),
                ssa,
                self.return_type.to_qir_type()
            ));
            qir_code.push(format!(
                "    ret {} {}",
                self.return_type.to_qir_type(),
                cast_ssa
            ));
        }

        qir_code.push("}".to_owned());
        qir_code.push(format!(
            "declare dllimport {} @{}({})",
            self.return_type.to_primitive_type().to_llvm_type(),
            self.interim_name(),
            interim_args
                .iter()
                .map(|(t, _)| format!("{}", t.to_primitive_type().to_llvm_type()))
                .collect::<Vec<String>>()
                .join(", ")
        ));

        // Interim codegen.
        let rust_interim_args = interim_args
            .iter()
            .map(|x| (x.0, self.next_var_name()))
            .collect_vec();
        interim_code.push(format!(
            "#[no_mangle]\npub extern \"C\" fn {}({})->{} {{",
            self.interim_name(),
            rust_interim_args
                .iter()
                .map(|(t, s)| format!("{}: {}", s, t.to_primitive_type().to_rust_type()))
                .join(", "),
            self.return_type.to_primitive_type().to_rust_type()
        ));
        // We assume that all arguments can be passed through one transmution.
        interim_code.push("    use core::mem::transmute as t;".to_owned());
        interim_code.push(format!(
            "    unsafe {{ t({}({}))}}",
            self.rust_shim_name(),
            rust_interim_args
                .iter()
                .map(|(t, s)| format!("t::<_, {}>({})", t.to_rust_type(), s))
                .join(", ")
        ));
        interim_code.push("}".to_owned());

        // rust codegen
        rust_code.push(format!(
            "pub fn {}({})->{} {{",
            self.rust_shim_name(),
            rust_interim_args
                .iter()
                .map(|(t, s)| format!("{}: {}", s, t.to_rust_type()))
                .join(", "),
            self.return_type.to_rust_type()
        ));
        rust_code.push("    todo!()".to_owned());
        rust_code.push("}".to_owned());
        (
            qir_code.join("\n"),
            interim_code.join("\n"),
            rust_code.join("\n"),
        )
    }
}

fn qir_builtin() -> Vec<QIRInterface> {
    let mut interfaces: Vec<QIRInterface> = Vec::new();
    interfaces.push(QIRInterface::new(
        "rt",
        "array_concatenate",
        Array,
        &[Array, Array],
    ));
    interfaces.push(QIRInterface::new("rt", "array_copy", Array, &[Array, QI1]));
    interfaces.push(QIRInterface::new(
        "rt",
        "array_create",
        Array,
        &[QI32, QI32, QI64P],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "array_create_1d",
        Array,
        &[QI32, QI64],
    ));
    interfaces.push(QIRInterface::new("rt", "array_get_dim", QI32, &[Array]));
    interfaces.push(QIRInterface::new(
        "rt",
        "array_get_element_ptr",
        QI8P,
        &[Array, QI64P],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "array_get_element_ptr_1d",
        QI8P,
        &[Array, QI64],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "array_get_size",
        QI64,
        &[Array, QI32],
    ));
    interfaces.push(QIRInterface::new("rt", "array_get_size_1d", QI64, &[Array]));
    interfaces.push(QIRInterface::new(
        "rt",
        "array_project",
        Array,
        &[Array, QI32, QI64, QI1],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "array_slice",
        Array,
        &[Array, QI32, Range, QI1],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "array_slice_1d",
        Array,
        &[Array, Range, QI1],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "array_update_alias_count",
        QVoid,
        &[Array, QI32],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "array_update_reference_count",
        QVoid,
        &[Array, QI32],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_add",
        BigInt,
        &[BigInt, BigInt],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_bitand",
        BigInt,
        &[BigInt, BigInt],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_bitor",
        BigInt,
        &[BigInt, BigInt],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_bitxor",
        BigInt,
        &[BigInt, BigInt],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_create_array",
        BigInt,
        &[QI32, QI8P],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_create_i64",
        BigInt,
        &[QI64],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_divide",
        BigInt,
        &[BigInt, BigInt],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_equal",
        QI1,
        &[BigInt, BigInt],
    ));
    interfaces.push(QIRInterface::new("rt", "bigint_get_data", QI8P, &[BigInt]));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_get_length",
        QI32,
        &[BigInt],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_greater",
        QI1,
        &[BigInt, BigInt],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_greater_eq",
        QI1,
        &[BigInt, BigInt],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_modulus",
        BigInt,
        &[BigInt, BigInt],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_multiply",
        BigInt,
        &[BigInt, BigInt],
    ));
    interfaces.push(QIRInterface::new("rt", "bigint_negate", BigInt, &[BigInt]));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_power",
        BigInt,
        &[BigInt, QI32],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_shiftleft",
        BigInt,
        &[BigInt, QI64],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_shiftright",
        BigInt,
        &[BigInt, QI64],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_subtract",
        BigInt,
        &[BigInt, BigInt],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_to_string",
        QString,
        &[BigInt],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "bigint_update_reference_count",
        QVoid,
        &[BigInt, QI32],
    ));
    interfaces.push(QIRInterface::new("rt", "bool_to_string", QString, &[QI1]));
    interfaces.push(QIRInterface::new(
        "rt",
        "callable_copy",
        Callable,
        &[Callable, QI1],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "callable_create",
        Callable,
        &[CallableTable, MemManTable, Tuple],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "callable_invoke",
        QVoid,
        &[Callable, Tuple, Tuple],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "callable_make_adjoint",
        QVoid,
        &[Callable],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "callable_make_controlled",
        QVoid,
        &[Callable],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "callable_update_alias_count",
        QVoid,
        &[Callable, QI32],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "callable_update_reference_count",
        QVoid,
        &[Callable, QI32],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "capture_update_alias_count",
        QVoid,
        &[Callable, QI32],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "capture_update_reference_count",
        QVoid,
        &[Callable, QI32],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "double_to_string",
        QString,
        &[Double],
    ));
    interfaces.push(QIRInterface::new("rt", "fail", QVoid, &[QString]));
    interfaces.push(QIRInterface::new("rt", "int_to_string", QString, &[QI64]));
    interfaces.push(QIRInterface::new("rt", "message", QVoid, &[QString]));
    interfaces.push(QIRInterface::new(
        "rt",
        "pauli_to_string",
        QString,
        &[QIRPauli],
    ));
    interfaces.push(QIRInterface::new("rt", "qubit_allocate", Qubit, &[]));
    interfaces.push(QIRInterface::new(
        "rt",
        "qubit_allocate_array",
        Array,
        &[QI32],
    ));
    interfaces.push(QIRInterface::new("rt", "qubit_release", QVoid, &[Qubit]));
    interfaces.push(QIRInterface::new(
        "rt",
        "qubit_release_array",
        QVoid,
        &[Array],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "qubit_to_string",
        QString,
        &[Qubit],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "range_to_string",
        QString,
        &[Range],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "result_equal",
        QI1,
        &[Result, Result],
    ));
    interfaces.push(QIRInterface::new("rt", "result_get_one", Result, &[]));
    interfaces.push(QIRInterface::new("rt", "result_get_zero", Result, &[]));
    interfaces.push(QIRInterface::new(
        "rt",
        "result_to_string",
        QString,
        &[Result],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "result_update_reference_count",
        QVoid,
        &[Result, QI32],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "string_concatenate",
        QString,
        &[QString, QString],
    ));
    interfaces.push(QIRInterface::new("rt", "string_create", QString, &[QI8P]));
    interfaces.push(QIRInterface::new(
        "rt",
        "string_equal",
        QI1,
        &[QString, QString],
    ));
    interfaces.push(QIRInterface::new("rt", "string_get_data", QI8P, &[QString]));
    interfaces.push(QIRInterface::new(
        "rt",
        "string_get_length",
        QI32,
        &[QString],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "string_update_reference_count",
        QVoid,
        &[QString, QI32],
    ));
    interfaces.push(QIRInterface::new("rt", "tuple_copy", Tuple, &[Tuple, QI1]));
    interfaces.push(QIRInterface::new("rt", "tuple_create", Tuple, &[QI64]));
    interfaces.push(QIRInterface::new(
        "rt",
        "tuple_update_alias_count",
        QVoid,
        &[Tuple, QI32],
    ));
    interfaces.push(QIRInterface::new(
        "rt",
        "tuple_update_reference_count",
        QVoid,
        &[Tuple, QI32],
    ));
    return interfaces;
}

// Extensions for Microsoft Q#.
// See: https://github.com/microsoft/qsharp-runtime/tree/main/src/Qir/Runtime/lib/QSharpCore
fn qir_microsoft_extension_core() -> Vec<QIRInterface> {
    let mut interfaces: Vec<QIRInterface> = Vec::new();
    interfaces.push(QIRInterface::new(
        "qis",
        "exp__body",
        QVoid,
        &[Array, Double, Array],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "exp__adj",
        QVoid,
        &[Array, Double, Array],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "exp__ctl",
        QVoid,
        &[Array, Array, Double, Array],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "exp__ctladj",
        QVoid,
        &[Array, Array, Double, Array],
    ));
    interfaces.push(QIRInterface::new("qis", "h__body", QVoid, &[Qubit]));
    interfaces.push(QIRInterface::new("qis", "h__ctl", QVoid, &[Array, Qubit]));
    interfaces.push(QIRInterface::new("qis", "measure__body", Result, &[Array, Array]));
    interfaces.push(QIRInterface::new(
        "qis",
        "r__body",
        QVoid,
        &[Pauli, Double, Qubit],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "r__adj",
        QVoid,
        &[Pauli, Double, Qubit],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "r__ctl",
        QVoid,
        &[Array, RotTuple],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "r__ctladj",
        QVoid,
        &[Array, RotTuple],
    ));
    interfaces.push(QIRInterface::new("qis", "s__body", QVoid, &[Qubit]));
    interfaces.push(QIRInterface::new("qis", "s__adj", QVoid, &[Qubit]));
    interfaces.push(QIRInterface::new("qis", "s__ctl", QVoid, &[Array, Qubit]));
    interfaces.push(QIRInterface::new(
        "qis",
        "s__ctladj",
        QVoid,
        &[Array, Qubit],
    ));
    interfaces.push(QIRInterface::new("qis", "t__body", QVoid, &[Qubit]));
    interfaces.push(QIRInterface::new("qis", "t__adj", QVoid, &[Qubit]));
    interfaces.push(QIRInterface::new("qis", "t__ctl", QVoid, &[Array, Qubit]));
    interfaces.push(QIRInterface::new(
        "qis",
        "t__ctladj",
        QVoid,
        &[Array, Qubit],
    ));
    interfaces.push(QIRInterface::new("qis", "x__body", QVoid, &[Qubit]));
    interfaces.push(QIRInterface::new("qis", "x__ctl", QVoid, &[Array, Qubit]));
    interfaces.push(QIRInterface::new("qis", "y__body", QVoid, &[Qubit]));
    interfaces.push(QIRInterface::new("qis", "y__ctl", QVoid, &[Array, Qubit]));
    interfaces.push(QIRInterface::new("qis", "z__body", QVoid, &[Qubit]));
    interfaces.push(QIRInterface::new("qis", "z__ctl", QVoid, &[Array, Qubit]));
    interfaces.push(QIRInterface::new(
        "qis",
        "dumpmachine__body",
        QVoid,
        &[QI8P],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "dumpregister__body",
        QVoid,
        &[QI8P, Array],
    ));
    return interfaces;
}

// See: https://github.com/microsoft/qsharp-runtime/tree/main/src/Qir/Runtime/lib/QSharpFoundation
fn qir_microsoft_extension_foundation() -> Vec<QIRInterface> {
    let mut interfaces: Vec<QIRInterface> = Vec::new();
    interfaces.push(QIRInterface::new("qis", "nan__body", Double, &[]));
    interfaces.push(QIRInterface::new("qis", "isnan__body", QI1, &[Double]));
    interfaces.push(QIRInterface::new("qis", "infinity__body", Double, &[]));
    interfaces.push(QIRInterface::new("qis", "isinf__body", QI1, &[Double]));
    interfaces.push(QIRInterface::new(
        "qis",
        "isnegativeinfinity__body",
        QI1,
        &[Double],
    ));
    interfaces.push(QIRInterface::new("qis", "sin__body", Double, &[Double]));
    interfaces.push(QIRInterface::new("qis", "cos__body", Double, &[Double]));
    interfaces.push(QIRInterface::new("qis", "tan__body", Double, &[Double]));
    interfaces.push(QIRInterface::new(
        "qis",
        "arctan2__body",
        Double,
        &[Double, Double],
    ));
    interfaces.push(QIRInterface::new("qis", "sinh__body", Double, &[Double]));
    interfaces.push(QIRInterface::new("qis", "cosh__body", Double, &[Double]));
    interfaces.push(QIRInterface::new("qis", "tanh__body", Double, &[Double]));
    interfaces.push(QIRInterface::new("qis", "arcsin__body", Double, &[Double]));
    interfaces.push(QIRInterface::new("qis", "arccos__body", Double, &[Double]));
    interfaces.push(QIRInterface::new("qis", "arctan__body", Double, &[Double]));
    interfaces.push(QIRInterface::new("qis", "sqrt__body", Double, &[Double]));
    interfaces.push(QIRInterface::new("qis", "log__body", Double, &[Double]));
    interfaces.push(QIRInterface::new(
        "qis",
        "ieeeremainder__body",
        Double,
        &[Double, Double],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "drawrandomint__body",
        Int,
        &[Int, Int],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "drawrandomdouble__body",
        Double,
        &[Double, Double],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "applyifelseintrinsic__body",
        QVoid,
        &[Result, Callable, Callable],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "applyconditionallyinstrinsic__body",
        QVoid,
        &[Array, Array, Callable, Callable],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "assertmeasurementprobability__body",
        QVoid,
        &[Array, Array, Result, Double, QString, Double],
    ));
    interfaces.push(QIRInterface::new(
        "qis",
        "assertmeasurementprobability__ctl",
        QVoid,
        &[Array, MeasurementProbabilityArgs],
    ));

    return interfaces;
}

// Extensions for isQ.
fn qir_isq() -> Vec<QIRInterface> {
    let mut interfaces: Vec<QIRInterface> = Vec::new();
    interfaces.push(QIRInterface::new("qis", "u3", QVoid, &[Qubit, Double, Double, Double]));
    interfaces.push(QIRInterface::new("qis", "gphase", QVoid, &[Double]));
    return interfaces;
}

fn main() {
    if std::env::args().len() < 3 {
        println!(
            "Usage: {} <preset> <type>",
            std::env::args().next().unwrap()
        );
        return;
    }
    let preset_name = std::env::args().nth(1).unwrap();
    let type_name = std::env::args().nth(2).unwrap();
    let mut interfaces = if preset_name == "qir" {
        qir_builtin()
    } else if preset_name == "qsharp-core" {
        qir_microsoft_extension_core()
    } else if preset_name == "qsharp-foundation" {
        qir_microsoft_extension_foundation()
    } else if preset_name == "isq"{
        qir_isq()
    } else {
        panic!("Unknown preset: {}", preset_name);
    };
    let mut qir_code: Vec<String> = Vec::new();
    let mut rust_code: Vec<String> = Vec::new();
    let mut interim_code: Vec<String> = Vec::new();
    for interface in interfaces.iter_mut() {
        let (qir_code_, interim_code_, rust_code_) = interface.codegen();
        qir_code.push(qir_code_);
        interim_code.push(interim_code_);
        rust_code.push(rust_code_);
    }
    if type_name == "qir" {
        println!(
            "%Range = type {{ i64, i64, i64 }}
%Tuple = type opaque
%Qubit = type opaque
%Result = type opaque
%Array = type opaque
%Callable = type opaque
%BigInt = type opaque
%Pauli = type i2
%String = type opaque"
        );
        println!("{}", qir_code.join("\n"));
    } else if type_name == "interim" {
        println!("use super::impls::*;");
        println!("use super::types::*;");
        println!("{}", interim_code.join("\n"));
    } else if type_name == "rust" {
        println!("use super::types::*;");
        println!("{}", rust_code.join("\n"));
    } else if type_name == "export" {
        for i in interfaces.iter(){
            println!("{}", i.interim_name());
        }
    } else {
        println!("Unknown type {}", type_name);
    }
}
