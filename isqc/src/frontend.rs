use miette::*;
use serde_json::*;

use crate::error::*;
use serde_json::Value as V;

use std::{fs::File, io::{Read}};


pub fn parse_type(input: &Value)->String{
    let ty = input["ty"]["tag"].as_str().unwrap();
    let subtypes = input["subTypes"].as_array().unwrap();
    match ty{
        "Ref"=>format!("&{}", parse_type(&subtypes[0])),
        "Unit"=>format!("()"),
        "Qbit"=>format!("qbit"),
        "Int"=>format!("int"),
        "Bool"=>format!("bool"),
        "Double"=>format!("double"),
        "Complex"=>format!("complex"),
        "Param"=>format!("param"),
        "Array"=>{
            let length = input["ty"]["contents"].as_i64().unwrap();
            if (length == 0) {
                format!("{} array with dynamic length", parse_type(&subtypes[0]))
            } else {
                format!("[{};{}]", parse_type(&subtypes[0]), length)
            }
        },
        "UserType"=>format!("{}<{}>", input["ty"]["contents"].as_str().unwrap(), subtypes.iter().map(parse_type).collect::<Vec<_>>().join(", ")),
        "IntRange"=>format!("range"),
        "Gate"=>format!("gate<{}>", input["ty"]["contents"].as_i64().unwrap()),
        "FuncTy"=>format!("({})->{}", subtypes.iter().skip(1).map(parse_type).collect::<Vec<_>>().join(", "), parse_type(&subtypes[0])),
        _ => unreachable!()
    }
}

pub fn parse_pos(input: &Value)->miette::Result<(NamedSource, SourceSpan)>{

    let path = input["filename"].as_str().unwrap();
    let name: Vec<&str> = path.split("/").collect(); 
    let mut f = File::open(path).map_err(IoError)?;
    let mut buf = String::new();
    f.read_to_string(&mut buf).unwrap();

    let loc_line = input["line"].as_i64().unwrap() as usize;
    let loc_col = input["column"].as_i64().unwrap() as usize+1;
    let mut line = 0usize;
    let mut col = 0usize;
    let mut offset = 0usize;
    for char in buf.chars() {
        if char == '\t' {
            col += 8;
        } else {
            col += 1;
        }

        if line >= loc_line && col >= loc_col {
            break;
        }
        if char == '\n'{
            col = 0;
            line += 1;
        }
        offset += char.len_utf8();
    }
    let ss = SourceSpan::new(
        SourceOffset::from(offset),
        SourceOffset::from(0)
    );
    
    let src = NamedSource::new(name[name.len()-1], buf.to_owned());

    return Ok((src, ss));
}

fn get_first_define_err(source: NamedSource, lable: SourceSpan) -> core::result::Result<(), Vec<FirstDefineError>> {
    Err(vec![FirstDefineError{src: source, pos_first: lable}])
}

pub fn parse_symbol(symbol: &Value)->String{
    let ty = symbol["tag"].as_str().unwrap();
    match ty{
        "SymVar"=>symbol["contents"].as_str().unwrap().into(),
        "SymTempVar"=>format!("#tempvar{}", symbol["contents"].as_i64().unwrap()),
        "SymTempArg"=>format!("#temparg{}", symbol["contents"].as_i64().unwrap()),
        _=>unreachable!()

    }
}
pub fn parse_match_rule(match_rule: &Value)->String{
    let ty = match_rule["tag"].as_str().unwrap();
    match ty{
        "Exact"=>parse_type(&match_rule["contents"]),
        "AnyUnknownList"=>"[_]".into(),
        "AnyKnownList"=>format!("[_;{}]", match_rule["contents"].as_i64().unwrap()),
        "AnyList"=>"`array`".into(),
        "AnyFunc"=>"`function`".into(),
        "AnyGate"=>"gate<_>".into(),
        "AnyRef"=>"`left value`".into(),
        "ArrayType"=>format!("{}[]", parse_match_rule(&match_rule["contents"])),
        "FixedArray"=>format!("{} array with static length", parse_match_rule(&match_rule["contents"])),
        _=>unreachable!()
    }
}

pub fn resolve_isqc1_output(input: &str)->miette::Result<String>{
    //let src = NamedSource::new(name.to_owned(), source.to_owned());
    let json = serde_json::from_str(input).map_err(|_| InvalidISQC1Json)?;
    if let V::Object(kv) = json{
        if let Some(V::String(s)) = kv.get("Right"){
            return Ok(String::from(s));
        }else{
            if let Some(V::Object(err)) = kv.get("Left"){
                let tag = err["tag"].as_str().unwrap();
                let content=&err["contents"];
                if tag=="GrammarError"{
                    
                    match content["tag"].as_str().unwrap(){
                        "BadMatrixElement"=>{
                            let (src,matelem_pos) = parse_pos(&content["badExpr"]["annotationExpr"])?;
                            return Err(SyntaxError{reason: "bad matrix element.".into(), src, pos: matelem_pos})?;
                        }
                        "BadMatrixShape"=>{
                            let (src, mat_pos) = parse_pos(&content["badMatrix"]["annotationAST"])?;
                            return Err(SyntaxError{reason: "bad matrix shape.".into(), src, pos: mat_pos})?;
                        }
                        "BadGlobalVarSize"=>{
                            let (src,pos) = parse_pos(&content["badDefPos"])?;
                            return Err(SyntaxError{reason: format!("global array with non-integer length: `{}`.", content["badDefName"].as_str().unwrap()), src, pos: pos})?;
                        }
                        "UnexpectedToken"=>{
                            let (src,pos) = parse_pos(&content["token"]["annotationToken"])?;
                            return Err(SyntaxError{reason: "unexpected token.".into(), src, pos: pos})?;
                        }
                        "ImportNotFound"=>{
                            let import_string = content["missingImport"].as_str().unwrap();
                            let message = "Cannot find ".to_string() + import_string;
                            return Err(GeneralGrammarError("Cannot find import".to_string(), message))?;
                        }
                        "DuplicatedImport"=>{
                            let import_string = content["duplicatedImport"].as_str().unwrap();
                            let message = import_string.to_string() + " is imported multiple times.";
                            return Err(GeneralGrammarError("Duplicated import".to_string(), message))?;
                        }
                        "CyclicImport"=>{
                            let files = content["cyclicImport"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect::<Vec<_>>().join(", ");
                            let message = " cyclically import each other.";
                            Err(GeneralGrammarError("Cyclic import".to_string(), files + message))?
                        }
                        "AmbiguousImport"=>{
                            let amb_import = content["ambImport"].as_str().unwrap();
                            let cand1 = content["candidate1"].as_str().unwrap();
                            let cand2 = content["candidate2"].as_str().unwrap();
                            let message = amb_import.to_string() + " has at least two candidates: " + cand1 + " and " + cand2;
                            return Err(GeneralGrammarError("Ambiguous import".to_string(), message))?;
                        }
                        _=>{return Err(GeneralISQC1Error(V::Object(err.clone()).to_string()))?;}
                    }
                }else if tag=="TypeCheckError"{
                    match content["tag"].as_str().unwrap(){
                        "RedefinedSymbol"=>{
                            let (src,pos) = parse_pos(&content["pos"])?;
                            let symbol_name = parse_symbol(&content["symbolName"]);
                            let (src2, first_defined_at) = parse_pos(&content["firstDefinedAt"])?;
                            get_first_define_err(src2, first_defined_at).map_err(|err_list| RedefinedSymbolError{
                                sym_name: symbol_name.into(), 
                                src: src, 
                                pos: pos,
                                related: err_list
                            })?;
                        }
                        "UndefinedSymbol"=>{
                            let (src,pos) = parse_pos(&content["pos"])?;
                            let symbol_name = parse_symbol(&content["symbolName"]);
                            return Err(UndefinedSymbolError{sym_name: symbol_name.into(), src, pos})?;
                        }
                        "AmbiguousSymbol"=>{
                            let (src1, pos1) = parse_pos(&content["firstDefinedAt"])?;
                            let symbol_name = parse_symbol(&content["symbolName"]);
                            let (src2, pos2) = parse_pos(&content["secondDefinedAt"])?;
                            Err(vec![FirstDefineError{src: src1, pos_first: pos1}]).map_err(|err_list| ConflictSymbolError{
                                sym_name: symbol_name.into(),
                                src: src2,
                                pos: pos2,
                                related: err_list
                            })?
                        }
                        "TypeMismatch"=>{
                            let (src,pos) = parse_pos(&content["pos"])?;
                            let expected = content["expectedType"].as_array().unwrap().iter().map(parse_match_rule).collect::<Vec<_>>().join(" or ");
                            let actual = parse_type(&content["actualType"]);
                            return Err(TypeMismatchError{expected, actual, src, pos})?;
                        }
                        "UnsupportedType"=>{
                            let (src,pos) = parse_pos(&content["pos"])?;
                            let actual = parse_type(&content["actualType"]);
                            return Err(SyntaxError{reason: "unsupported type: ".to_string() + actual.as_str(), src, pos: pos})?;
                        }
                        "UnsupportedLeftSide"=>{
                            let (src,pos) = parse_pos(&content["pos"])?;
                            return Err(SyntaxError{reason: "unsupported left side".to_string(), src, pos: pos})?;
                        }
                        "ViolateNonCloningTheorem"=>{
                            let (src,pos) = parse_pos(&content["pos"])?;
                            return Err(AssignQbitError{src, pos})?;
                        }
                        "ArgNumberMismatch"=>{
                            let (src,pos) = parse_pos(&content["pos"])?;
                            let expected = content["expectedArgs"].as_i64().unwrap() as usize;
                            let actual = content["actualArgs"].as_i64().unwrap() as usize;
                            return Err(ArgNumberMismatchError{expected, actual, src, pos})?;
                        }
                        "BadProcedureArgType"=>{
                            let (src,pos) = parse_pos(&content["pos"])?;
                            let argtype = parse_type(&content["arg"][0]);
                            let argname = content["arg"][1].as_str().unwrap().into();
                            return Err(BadProcedureArgTypeError{argname, argtype, pos, src})?;
                        }
                        "BadProcedureReturnType"=>{
                            let (src,pos) = parse_pos(&content["pos"])?;
                            let returntype = parse_type(&content["ret"][0]);
                            let procname = content["ret"][1].as_str().unwrap().into();
                            return Err(BadProcedureRetTypeError{procname, returntype, pos, src})?;
                        }
                        "ICETypeCheckError"=>{
                            return Err(GeneralISQC1Error("Internal compiler error while typechecking.".into()))?;
                        }
                        "MainUndefined"=>{
                            return Err(MainUndefinedError)?;
                        }
                        "BadMainSignature"=>{
                            let sig = parse_type(&content["actualMainSignature"]);
                            return Err(BadMainSigError(sig))?;
                        }
                        _=>{return Err(GeneralISQC1Error(V::Object(err.clone()).to_string()))?;}
                    }
                }else if tag=="RAIIError"{
                    match content["tag"].as_str().unwrap(){
                        "UnmatchedScopeError"=>{
                            let (_src,pos) = parse_pos(&content["unmatchedPos"])?;
                            let rtype = content["wantedRegionType"].as_str().unwrap();
                            let r = match rtype{
                                "RFunc"=>"function",
                                "RLoop"=>"loop",
                                _=>unreachable!()
                            };
                            return Err(UnmatchedScopeError{pos, scope_type: r.into()})?;
                        }
                        _=>{return Err(GeneralISQC1Error(V::Object(err.clone()).to_string()))?;}
                    }
                }else if tag=="InternalCompilerError"{
                    let ice = content.as_str().unwrap().to_owned();
                    return Err(GeneralISQC1Error(format!("Internal compiler error: {}", ice)))?;
                }else if tag=="SyntaxError"{
                    let (src,pos) = parse_pos(content)?;
                    return Err(SyntaxError{reason: "tokenizing failed.".into(), src, pos: pos})?;
                }else if tag=="OracleError"{
                    match content["tag"].as_str().unwrap() {
                        "MultipleDefined"=>{
                            let (src,pos) = parse_pos(&content["sourcePos"])?;
                            let symbol_name = content["varName"].as_str().unwrap();
                            return Err(SyntaxError{reason: "Multiple defined symbol: ".to_string() + symbol_name, src, pos})?;
                        }
                        "NoReturnValue"=>{
                            let (src,pos) = parse_pos(&content["sourcePos"])?;
                            let value = content["varName"].as_i64().unwrap();
                            return Err(SyntaxError{reason: format!("No return value for input {}.", value), src, pos})?;
                        }
                        "UndefinedSymbol"=>{
                            let (src,pos) = parse_pos(&content["sourcePos"])?;
                            let symbol_name = content["varName"].as_str().unwrap();
                            return Err(SyntaxError{reason: "Undefined symbol: ".to_string() + symbol_name, src, pos})?;
                        }
                        _ => {
                            let (src, pos) = parse_pos(&content["contents"])?;
                            match content["tag"].as_str().unwrap() {
                                "BadOracleShape" => {
                                    return Err(OracleShapeError{src, pos})?;
                                }
                                "BadOracleValue" => {
                                    return Err(OracleValueError{src, pos})?;
                                }
                                "IllegalExpression"=>{
                                    return Err(SyntaxError{reason: "Illegal expression.".into(), src, pos})?;
                                }
                                "UnmatchedType"=>{
                                    return Err(SyntaxError{reason: "This type is not expected.".into(), src, pos})?;
                                }
                                "UnsupportedType"=>{
                                    return Err(SyntaxError{reason: "This type is not supported.".into(), src, pos})?;
                                }
                                _ =>{return Err(GeneralISQC1Error(V::Object(err.clone()).to_string()))?;}
                            };
                        }
                    }
                }
                else if tag=="DeriveError"{
                    let (src, pos) = parse_pos(&content["contents"])?;
                    match content["tag"].as_str().unwrap() {
                        "BadGateSignature" => {
                            return Err(DeriveGateError{src, pos})?;
                        }
                        "BadOracleSignature" => {
                            return Err(DeriveOracleError{src, pos})?;
                        }
                        _ =>{return Err(GeneralISQC1Error(V::Object(err.clone()).to_string()))?;}
                    };
                }
                return Err(GeneralISQC1Error(V::Object(err.clone()).to_string()))?;
            }else{
                return Err(InvalidISQC1Json)?;
            }
            
        }
    }else{
        return Err(InvalidISQC1Json)?;
    }
}