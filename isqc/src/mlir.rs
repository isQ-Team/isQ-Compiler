use miette::*;
use serde_json::*;

use crate::error::*;
use serde_json::Value as V;

use std::{fs::File, io::{Read}};

use crate::frontend::parse_pos;

pub fn resolve_mlir_output(input: &str, err_msg: String)->miette::Result<String>{
    //let src = NamedSource::new(name.to_owned(), source.to_owned());
    let json = serde_json::from_str(input).map_err(|_| InvalidMLIRJson)?;
    if let V::Object(kv) = json{
        if let Some(V::String(s)) = kv.get("Right"){
            return Ok(String::from(s));
        }else{
            if let Some(V::Array(err)) = kv.get("Left"){
                
                let mut optimization_err_list:Vec<OptimizationError> = Vec::new();
                for mlir_err in err.iter(){
                    let tag = mlir_err["tag"].as_str().unwrap();
                    if tag == "FileNotFound"{
                        let filename = mlir_err["pos"]["filename"].as_str().unwrap();
                        return Err(MLIRFileNotFound(filename.into()))?;
                    }else if tag == "BackendError"{
                        return Err(InvalidMLIRBackend)?;
                    }else if tag == "OptimizationError"{
                        let (src, pos) = parse_pos(&mlir_err["pos"])?;
                        let msg = mlir_err["msg"].as_str().unwrap();
                        optimization_err_list.push(OptimizationError{src: src, pos: pos, msg: msg.into()});
                    }
                }
                return Err(OptimizationFailed{related: optimization_err_list, msg: err_msg})?;

            }else{
                return Err(InvalidMLIRJson)?;
            }    
        }
    }else{
        return Err(InvalidMLIRJson)?;
    }
}