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
        "FixedArray"=>format!("[{};{}]", parseType(&subtypes[0]), input["ty"]["contents"].as_i64().unwrap()),
        "UnknownArray"=>format!("[{}]", parseType(&subtypes[0])),
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
                        "TypeMismatch"=>{
                            let (src,pos) = parsePos(&content["pos"])?;
                            let expected = content["expectedType"].as_array().unwrap().iter().map(parseMatchRule).collect::<Vec<_>>().join(" or ");
                            let actual = parseType(&content["actualType"]);
                            return Err(TypeMismatchError{expected, actual, src, pos})?;
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
                            let rtype = content["wantedRegionType"]["tag"].as_str().unwrap();
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
                }else if tag=="IncFileError"{
                    let (src, pos) = parsePos(content)?;
                    let incFile = content["incpath"].as_str().unwrap().to_owned();
                    return Err(IncFileError{incFile, src, pos: pos})?;
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