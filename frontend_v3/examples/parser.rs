#![feature(try_blocks)]
use std::io::Read;

use frontend_v3::lang::{tokenizer::{tokenizer, tokenizer_all, tokenizer_entry}, parser::{parse_program, parse_entry, ParseErrorDiagnostic}};
fn main()->miette::Result<()>{
    let input = std::io::stdin();
    let mut lock = input.lock();
    let mut buf = String::new();
    lock.read_to_string(&mut buf).unwrap();
    println!("{:?}", buf);
    let exec : miette::Result<()> = try{
        let tokens = tokenizer_entry(&buf)?;

        println!("{:?}", tokens);

        let ast = parse_entry(&tokens).map_err(|x| <_ as Into<ParseErrorDiagnostic>>::into(x))?;
        println!("{:?}", ast);
        ()
    };
    exec.map_err(|x| x.with_source_code(buf))?;
    return Ok(());

}