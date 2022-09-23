use miette::*;
use serde_json::*;

use crate::error::*;
use serde_json::Value as V;

use std::{fs::File, io::{Read, Write}, path::{Path, PathBuf}, ffi::OsStr};


pub fn parseType(input: &Value)->String{
    let ty = input["ty"]["tag"].as_str().unwrap();
    let subtypes = input["subTypes"].as_array().unwrap();
    match ty{
        "Ref"=>format!("&{}", parseType(&subtypes[0])),
        "Unit"=>format!("()"),
        "Qbit"=>format!("qbit"),
        "Int"=>format!("int"),
        "Bool"=>format!("bool"),
        "Double"=>format!("double"),
        "Complex"=>format!("complex"),
        "Array"=>format!("[{};{}]", parseType(&subtypes[0]), input["ty"]["contents"].as_i64().unwrap()),
        "UserType"=>format!("{}<{}>", input["ty"]["contents"].as_str().unwrap(), subtypes.iter().map(parseType).collect::<Vec<_>>().join(", ")),
        "IntRange"=>format!("range"),
        "Gate"=>format!("gate<{}>", input["ty"]["contents"].as_i64().unwrap()),
        "FuncTy"=>format!("({})->{}", subtypes.iter().skip(1).map(parseType).collect::<Vec<_>>().join(", "), parseType(&subtypes[0])),
        _ => unreachable!()
    }
}
/* 
pub fn parsePos(source: &str, input: &Value)->SourceSpan{
    let line = input["line"].as_i64().unwrap() as usize;
    let column = input["column"].as_i64().unwrap() as usize+1;
    SourceSpan::new(
        SourceOffset::from_location(source, line, column),
        SourceOffset::from(0)
    )
}*/
pub fn parsePos(input: &Value)->miette::Result<(NamedSource, SourceSpan)>{

    let path = input["filename"].as_str().unwrap();
    let name: Vec<&str> = path.split("/").collect(); 
    let mut f = File::open(path).map_err(IoError)?;
    let mut buf = String::new();
    f.read_to_string(&mut buf);

    let loc_line = input["line"].as_i64().unwrap() as usize;
    let loc_col = input["column"].as_i64().unwrap() as usize+1;


    let mut line = 0usize;
    let mut col = 0usize;
    let mut offset = 0usize;
    for char in buf.chars() {
        if char == '\n' {
            col = 0;
            line += 1;
        }else if char == '\t' {
            col += 8;
        } else {
            col += 1;
        }
        if line + 1 >= loc_line && col + 1 >= loc_col {
            break;
        }
        offset += 1;
    }
    let ss = SourceSpan::new(
        SourceOffset::from(offset),
        SourceOffset::from(0)
    );
    
    let src = NamedSource::new(name[name.len()-1], buf.to_owned());

    return Ok((src, ss));
}

fn getFirstDefineErr(source: NamedSource, lable: SourceSpan) -> core::result::Result<(), Vec<FirstDefineError>> {
    Err(vec![FirstDefineError{src: source, posFirst: lable}])
}

pub fn parseSymbol(symbol: &Value)->String{
    let ty = symbol["tag"].as_str().unwrap();
    match ty{
        "SymVar"=>symbol["contents"].as_str().unwrap().into(),
        "SymTempVar"=>format!("#tempvar{}", symbol["contents"].as_i64().unwrap()),
        "SymTempArg"=>format!("#temparg{}", symbol["contents"].as_i64().unwrap()),
        _=>unreachable!()

    }
}
pub fn parseMatchRule(matchRule: &Value)->String{
    let ty = matchRule["tag"].as_str().unwrap();
    match ty{
        "Exact"=>parseType(&matchRule["contents"]),
        "AnyUnknownList"=>"[_]".into(),
        "AnyKnownList"=>format!("[_;{}]", matchRule["contents"].as_i64().unwrap()),
        "AnyList"=>"`array`".into(),
        "AnyFunc"=>"`function`".into(),
        "AnyGate"=>"gate<_>".into(),
        "AnyRef"=>"`left value`".into(),
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
                            let (src,matelem_pos) = parsePos(&content["badExpr"]["annotationExpr"])?;
                            return Err(SyntaxError{reason: "bad matrix element.".into(), src, pos: matelem_pos})?;
                        }
                        "BadMatrixShape"=>{
                            let (src, mat_pos) = parsePos(&content["badMatrix"]["annotationAST"])?;
                            return Err(SyntaxError{reason: "bad matrix shape.".into(), src, pos: mat_pos})?;
                        }
                        "MissingGlobalVarSize"=>{
                            let (src,pos) = parsePos(&content["badDefPos"])?;
                            return Err(SyntaxError{reason: format!("missing size for global array `{}`.", content["badDefName"].as_str().unwrap()), src, pos: pos})?;
                        }
                        "UnexpectedToken"=>{
                            let (src,pos) = parsePos(&content["token"]["annotationToken"])?;
                            return Err(SyntaxError{reason: "unexpected token.".into(), src, pos: pos})?;
                        }
                        "InconsistentRoot"=>{
                            let filePath = content["importedFile"].as_str().unwrap();
                            let rootPath = content["rootPath"].as_str().unwrap();
                            let message = "The root path of file ".to_string() + filePath + " is " + rootPath + ", which does not consist with the project.";
                            Err(GeneralGrammarError("Inconsistent root path".to_string(), message))?
                        }
                        "ImportNotFound"=>{
                            let importString = content["missingImport"].as_str().unwrap();
                            let message = "Cannot find ".to_string() + importString;
                            return Err(GeneralGrammarError("Cannot find import".to_string(), message))?;
                        }
                        "DuplicatedImport"=>{
                            let importString = content["duplicatedImport"].as_str().unwrap();
                            let message = importString.to_string() + " is imported multiple times.";
                            return Err(GeneralGrammarError("Duplicated import".to_string(), message))?;
                        }
                        "CyclicImport"=>{
                            let files = content["cyclicImport"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect::<Vec<_>>().join(", ");
                            let message = " cyclically import each other.";
                            Err(GeneralGrammarError("Cyclic import".to_string(), files + message))?
                        }
                        "AmbiguousImport"=>{
                            let ambImport = content["ambImport"].as_str().unwrap();
                            let cand1 = content["candidate1"].as_str().unwrap();
                            let cand2 = content["candidate2"].as_str().unwrap();
                            let message = ambImport.to_string() + " has at least two candidates: " + cand1 + " and " + cand2;
                            return Err(GeneralGrammarError("Ambiguous import".to_string(), message))?;
                        }
                        _=>{return Err(GeneralISQC1Error(V::Object(err.clone()).to_string()))?;}
                    }
                }else if tag=="TypeCheckError"{
                    match content["tag"].as_str().unwrap(){
                        "RedefinedSymbol"=>{
                            let (src,pos) = parsePos(&content["pos"])?;
                            let symbolName = parseSymbol(&content["symbolName"]);
                            let (src2, firstDefinedAt) = parsePos(&content["firstDefinedAt"])?;
                            getFirstDefineErr(src2, firstDefinedAt).map_err(|err_list| RedefinedSymbolError{
                                symName: symbolName.into(), 
                                src: src, 
                                pos: pos,
                                related: err_list
                            })?;
                        }
                        "UndefinedSymbol"=>{
                            let (src,pos) = parsePos(&content["pos"])?;
                            let symbolName = parseSymbol(&content["symbolName"]);
                            return Err(UndefinedSymbolError{symName: symbolName.into(), src, pos})?;
                        }
                        "AmbiguousSymbol"=>{
                            let (src1, pos1) = parsePos(&content["firstDefinedAt"])?;
                            let symbolName = parseSymbol(&content["symbolName"]);
                            let (src2, pos2) = parsePos(&content["secondDefinedAt"])?;
                            Err(vec![FirstDefineError{src: src1, posFirst: pos1}]).map_err(|err_list| ConflictSymbolError{
                                symName: symbolName.into(),
                                src: src2,
                                pos: pos2,
                                related: err_list
                            })?
                        }
                        "TypeMismatch"=>{
                            let (src,pos) = parsePos(&content["pos"])?;
                            let expected = content["expectedType"].as_array().unwrap().iter().map(parseMatchRule).collect::<Vec<_>>().join(" or ");
                            let actual = parseType(&content["actualType"]);
                            return Err(TypeMismatchError{expected, actual, src, pos})?;
                        }
                        "UnsupportedType"=>{
                            let (src,pos) = parsePos(&content["pos"])?;
                            let actual = parseType(&content["actualType"]);
                            return Err(SyntaxError{reason: "UnsupportedType: ".to_string() + actual.as_str(), src, pos: pos})?;
                        }
                        "ViolateNonCloningTheorem"=>{
                            let (src,pos) = parsePos(&content["pos"])?;
                            return Err(AssignQbitError{src, pos})?;
                        }
                        "ArgNumberMismatch"=>{
                            let (src,pos) = parsePos(&content["pos"])?;
                            let expected = content["expectedArgs"].as_i64().unwrap() as usize;
                            let actual = content["actualArgs"].as_i64().unwrap() as usize;
                            return Err(ArgNumberMismatchError{expected, actual, src, pos})?;
                        }
                        "BadProcedureArgType"=>{
                            let (src,pos) = parsePos(&content["pos"])?;
                            let argtype = parseType(&content["arg"][0]);
                            let argname = content["arg"][1].as_str().unwrap().into();
                            return Err(BadProcedureArgTypeError{argname, argtype, pos, src})?;
                        }
                        "BadProcedureReturnType"=>{
                            let (src,pos) = parsePos(&content["pos"])?;
                            let returntype = parseType(&content["ret"][0]);
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
                            let sig = parseType(&content["actualMainSignature"]);
                            return Err(BadMainSigError(sig))?;
                        }
                        _=>{return Err(GeneralISQC1Error(V::Object(err.clone()).to_string()))?;}
                    }
                }else if tag=="RAIIError"{
                    match content["tag"].as_str().unwrap(){
                        "UnmatchedScopeError"=>{
                            let (src,pos) = parsePos(&content["unmatchedPos"])?;
                            let rtype = content["wantedRegionType"].as_str().unwrap();
                            let r = match rtype{
                                "RFunc"=>"function",
                                "RLoop"=>"loop",
                                _=>unreachable!()
                            };
                            return Err(UnmatchedScopeError{pos, scopeType: r.into()})?;
                        }
                        _=>{return Err(GeneralISQC1Error(V::Object(err.clone()).to_string()))?;}
                    }
                }else if tag=="InternalCompilerError"{
                    let ice = content.as_str().unwrap().to_owned();
                    return Err(GeneralISQC1Error(format!("Internal compiler error: {}", ice)))?;
                }else if tag=="SyntaxError"{
                    let (src,pos) = parsePos(content)?;
                    return Err(SyntaxError{reason: "tokenizing failed.".into(), src, pos: pos})?;
                }else if tag=="OracleError"{
                    match content["tag"].as_str().unwrap() {
                        "MultipleDefined"=>{
                            let (src,pos) = parsePos(&content["sourcePos"])?;
                            let symbolName = content["varName"].as_str().unwrap();
                            return Err(SyntaxError{reason: "Multiple defined symbol: ".to_string() + symbolName, src, pos})?;
                        }
                        "NoReturnValue"=>{
                            let (src,pos) = parsePos(&content["sourcePos"])?;
                            let value = content["varName"].as_i64().unwrap();
                            return Err(SyntaxError{reason: format!("No return value for input {}.", value), src, pos})?;
                        }
                        "UndefinedSymbol"=>{
                            let (src,pos) = parsePos(&content["sourcePos"])?;
                            let symbolName = content["varName"].as_str().unwrap();
                            return Err(SyntaxError{reason: "Undefined symbol: ".to_string() + symbolName, src, pos})?;
                        }
                        _ => {
                            let (src, pos) = parsePos(&content["contents"])?;
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
                    let (src, pos) = parsePos(&content["contents"])?;
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