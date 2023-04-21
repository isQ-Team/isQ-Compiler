use miette::{Diagnostic, NamedSource, SourceSpan};
use thiserror::Error;


pub fn io_error_when(reason: &'static str)->impl FnOnce(std::io::Error)->IoErrorWhen{
    return move |e| IoErrorWhen(reason, e);
}

#[derive(Error, Diagnostic, Debug)]
#[error("IOError when `{0}`: {1:?}")]
pub struct IoErrorWhen (&'static str, pub std::io::Error);


#[derive(Error, Diagnostic, Debug)]
#[error(transparent)]
pub struct IoError (#[from] pub std::io::Error);


#[derive(Error, Debug, Diagnostic)]
#[error("ISQ_ROOT undefined.")]
#[diagnostic(
    code(isqv2::no_isqv2_root),
    help("This means something is wrong if you are calling from isqc entry.")
)]
pub struct NoISQv2RootError;

#[derive(Error, Debug, Diagnostic)]
#[error("QCIS mapping config undefined.")]
#[diagnostic(
    code(isqv2::no_qcis_config),
    help("Specify QCIS routing config by --qcis-config.")
)]
pub struct QCISConfigNotSpecified;

#[derive(Error, Debug, Diagnostic)]
#[error("Invalid output from frontend compiler.")]
#[diagnostic(
    code(isqv2::isqc1_invalid_output),
    help("This means something is wrong.")
)]
pub struct InvalidISQC1Json;

#[derive(Error, Debug, Diagnostic)]
#[error("{0}.")]
#[diagnostic(
    code(isqv2::isq_grammar_error),
    help("{1}")
)]
pub struct GeneralGrammarError(pub String, pub String);

#[derive(Error, Debug, Diagnostic)]
#[error("Frontend Error.")]
#[diagnostic(
    code(isqv2::isqc1_invalid_output),
    help("Unresolved error: {0}")
)]
pub struct GeneralISQC1Error(pub String);

#[derive(Error, Debug, Diagnostic)]
#[error("Bad source extension.")]
#[diagnostic(
    code(isqv2::bad_extension_error),
    help("Only sources with \".isq\" extension are accepted.")
)]
pub struct BadExtensionError;

#[derive(Error, Debug, Diagnostic)]
#[error("Syntax Error: {reason}")]
#[diagnostic(
    code(isqv2::frontend::syntax_error)
)]
pub struct SyntaxError {
    pub reason: String,
    #[source_code]
    pub src: NamedSource,
    #[label("Bad syntax here.")]
    pub pos: SourceSpan,
}

#[derive(Error, Debug, Diagnostic)]
#[error("Redefined symbol `{sym_name}`.")]
#[diagnostic(
    code(isqv2::frontend::redefined_symbol)
)]
pub struct RedefinedSymbolError {
    pub sym_name: String,
    #[source_code]
    pub src: NamedSource,
    #[label("Trying to redefine symbol here.")]
    pub pos: SourceSpan,
    #[related]
    pub related: Vec<FirstDefineError>
}

#[derive(Error, Debug, Diagnostic)]
#[error("Conflict symbol `{sym_name}`.")]
#[diagnostic(
    code(isqv2::frontend::conflict_symbol)
)]
pub struct ConflictSymbolError {
    pub sym_name: String,
    #[source_code]
    pub src: NamedSource,
    #[label("Also defined here.")]
    pub pos: SourceSpan,
    #[related]
    pub related: Vec<FirstDefineError>
}

#[derive(Error, Debug, Diagnostic)]
#[error("")]
#[diagnostic()]
pub struct FirstDefineError {
    #[source_code]
    pub src: NamedSource,
    #[label("First defined here.")]
    pub pos_first: SourceSpan
}


#[derive(Error, Debug, Diagnostic)]
#[error("Bad oracle, shape not match")]
#[diagnostic(code(isqv2::frontend::oracle_shape_error))]
pub struct OracleShapeError{
    #[source_code]
    pub src: NamedSource,
    #[label("list size can not match oracle result's shape")]
    pub pos: SourceSpan   
}


#[derive(Error, Debug, Diagnostic)]
#[error("Bad oracle, result value error")]
#[diagnostic(code(isqv2::frontend::oracle_value_error))]
pub struct OracleValueError{
    #[source_code]
    pub src: NamedSource,
    #[label("result value is not satisfied")]
    pub pos: SourceSpan   
}

#[derive(Error, Debug, Diagnostic)]
#[error("Bad Gate Signature")]
#[diagnostic(code(isqv2::frontend::derive_gate_error))]
pub struct DeriveGateError{
    #[source_code]
    pub src: NamedSource,
    #[label("derive gate here.")]
    pub pos: SourceSpan   
}

#[derive(Error, Debug, Diagnostic)]
#[error("Bad Oracle Signature")]
#[diagnostic(code(isqv2::frontend::derive_oracle_error))]
pub struct DeriveOracleError{
    #[source_code]
    pub src: NamedSource,
    #[label("derive oracle here.")]
    pub pos: SourceSpan   
}


#[derive(Error, Debug, Diagnostic)]
#[error("Undefined symbol `{sym_name}`.")]
#[diagnostic(
    code(isqv2::frontend::undefined_symbol)
)]
pub struct UndefinedSymbolError {
    pub sym_name: String,
    #[source_code]
    pub src: NamedSource,
    #[label("Trying to use symbol here.")]
    pub pos: SourceSpan,
}

#[derive(Error, Debug, Diagnostic)]
#[error("Type mismatch.\nexpected: {expected}\nactual: {actual}")]
#[diagnostic(
    code(isqv2::frontend::type_mismatch)
)]
pub struct TypeMismatchError {
    #[source_code]
    pub src: NamedSource,
    #[label("In this expression/statement.")]
    pub pos: SourceSpan,
    pub expected: String,
    pub actual: String,
}


#[derive(Error, Debug, Diagnostic)]
#[error("Trying to assign to a `qbit` variable.")]
#[diagnostic(
    code(isqv2::frontend::violate_non_cloning_theorem),
    help("You can only perform gates/measurements on qubits.\nAssigning one qubit to another violates non-cloning theorem.")
)]
pub struct AssignQbitError {
    #[source_code]
    pub src: NamedSource,
    #[label("Here.")]
    pub pos: SourceSpan,
}

#[derive(Error, Debug, Diagnostic)]
#[error("Argument number mismatch.\nexpected: {expected}\nactual: {actual}")]
#[diagnostic(
    code(isqv2::frontend::arg_number_mismatch)
)]
pub struct ArgNumberMismatchError {
    #[source_code]
    pub src: NamedSource,
    #[label("Calling expression here.")]
    pub pos: SourceSpan,
    pub expected: usize,
    pub actual: usize,
}

#[derive(Error, Debug, Diagnostic)]
#[error("Invalid argument type `{argtype}` for argument `{argname}`.")]
#[diagnostic(
    code(isqv2::frontend::bad_arg_type),
    help("This is considered a bug in compiler. Please contact the developers.")
)]
pub struct BadProcedureArgTypeError {
    #[source_code]
    pub src: NamedSource,
    #[label("This argument.")]
    pub pos: SourceSpan,
    pub argtype: String,
    pub argname: String
}

#[derive(Error, Debug, Diagnostic)]
#[error("Invalid return value type `{returntype}` for procedure `{procname}`.")]
#[diagnostic(
    code(isqv2::frontend::bad_return_type),
    help("Returning `qbit` violates non-cloning theorem. Try passing by reference.")
)]
pub struct BadProcedureRetTypeError {
    #[source_code]
    pub src: NamedSource,
    #[label("This return value.")]
    pub pos: SourceSpan,
    pub returntype: String,
    pub procname: String,
}

#[derive(Error, Debug, Diagnostic)]
#[error("Main function not defined.")]
#[diagnostic(
    code(isqv2::frontend::main_undefined),
    help("You need to define `procedure main()` as entry of your program.")
)]
pub struct MainUndefinedError;



#[derive(Error, Debug, Diagnostic)]
#[error("Main function defined with wrong signature `{0}`.")]
#[diagnostic(
    code(isqv2::frontend::bad_main_signature),
    help("The signature of main function `procedure main()`, no more, no less.")
)]
pub struct BadMainSigError(pub String);



#[derive(Error, Debug, Diagnostic)]
#[error("Unexpected statement outside a {scope_type}.")]
#[diagnostic(
    code(isqv2::frontend::unmatched_scope),
    help("You can only use `return` inside a function, and `break` or `continue` inside a loop.")
)]
pub struct UnmatchedScopeError{
    pub scope_type: String,
    #[label("Statement here.")]
    pub pos: SourceSpan,
}


#[derive(Error, Debug, Diagnostic)]
#[error("Internal compiler error: `{0}`.")]
#[diagnostic(
    code(isqv2::ice),
    help("Unhandled internal compiler error.")
)]
pub struct InternalCompilerError(pub String);

