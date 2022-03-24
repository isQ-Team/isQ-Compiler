use miette::*;
use serde_json::*;

use crate::error::*;
use serde_json::Value as V;

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
pub fn parsePos(source: &str, input: &Value)->SourceSpan{
    let line = input["line"].as_i64().unwrap() as usize;
    let column = input["column"].as_i64().unwrap() as usize+1;
    SourceSpan::new(
        SourceOffset::from_location(source, line, column),
        SourceOffset::from(0)
    )
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

pub fn resolve_isqc1_output(name: &str, source: &str, input: &str)->miette::Result<String>{
    let src = NamedSource::new(name.to_owned(), source.to_owned());
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
                            let matelem_pos = parsePos(source, &content["badExpr"]["annotationExpr"]);
                            return Err(SyntaxError{reason: "bad matrix element.".into(), src, pos: matelem_pos})?;
                        }
                        "BadMatrixShape"=>{
                            let mat_pos = parsePos(source, &content["badMatrix"]["annotationAST"]);
                            return Err(SyntaxError{reason: "bad matrix shape.".into(), src, pos: mat_pos})?;
                        }
                        "MissingGlobalVarSize"=>{
                            let pos = parsePos(source, &content["badDefPos"]);
                            return Err(SyntaxError{reason: format!("missing size for global array `{}`.", content["badDefName"].as_str().unwrap()), src, pos: pos})?;
                        }
                        "UnexpectedToken"=>{
                            let pos = parsePos(source, &content["token"]["annotationToken"]);
                            return Err(SyntaxError{reason: "unexpected token.".into(), src, pos: pos})?;
                        }
                        _=>{return Err(GeneralISQC1Error(V::Object(err.clone()).to_string()))?;}
                    }
                }else if tag=="TypeCheckError"{
                    match content["tag"].as_str().unwrap(){
                        "RedefinedSymbol"=>{
                            let pos = parsePos(source, &content["pos"]);
                            let symbolName = parseSymbol(&content["symbolName"]);
                            let firstDefinedAt = parsePos(source, &content["firstDefinedAt"]);
                            return Err(RedefinedSymbolError{symName: symbolName.into(), src, pos, posFirst: firstDefinedAt})?;
                        }
                        "UndefinedSymbol"=>{
                            let pos = parsePos(source, &content["pos"]);
                            let symbolName = parseSymbol(&content["symbolName"]);
                            return Err(UndefinedSymbolError{symName: symbolName.into(), src, pos})?;
                        }
                        "TypeMismatch"=>{
                            let pos = parsePos(source, &content["pos"]);
                            let expected = content["expectedType"].as_array().unwrap().iter().map(parseMatchRule).collect::<Vec<_>>().join(" or ");
                            let actual = parseType(&content["actualType"]);
                            return Err(TypeMismatchError{expected, actual, src, pos})?;
                        }
                        "ViolateNonCloningTheorem"=>{
                            let pos = parsePos(source, &content["pos"]);
                            return Err(AssignQbitError{src, pos})?;
                        }
                        "ArgNumberMismatch"=>{
                            let pos = parsePos(source, &content["pos"]);
                            let expected = content["expectedArgs"].as_i64().unwrap() as usize;
                            let actual = content["actualArgs"].as_i64().unwrap() as usize;
                            return Err(ArgNumberMismatchError{expected, actual, src, pos})?;
                        }
                        "BadProcedureArgType"=>{
                            let pos = parsePos(source, &content["pos"]);
                            let argtype = parseType(&content["arg"][0]);
                            let argname = content["arg"][1].as_str().unwrap().into();
                            return Err(BadProcedureArgTypeError{argname, argtype, pos, src})?;
                        }
                        "BadProcedureReturnType"=>{
                            let pos = parsePos(source, &content["pos"]);
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
                            let pos = parsePos(source, &content["unmatchedPos"]);
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
                    let pos = parsePos(source, content);
                    return Err(SyntaxError{reason: "tokenizing failed.".into(), src, pos: pos})?;
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