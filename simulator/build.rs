use std::io::BufRead;

fn main(){
    let f = std::fs::File::open("src/facades/qir/shim/exports.txt").unwrap();
    for line in std::io::BufReader::new(f).lines(){
        if let Ok(line) = line{
            println!("cargo:rustc-link-arg=-Wl,--undefined={}", line);
        }
    }
    println!("cargo:rustc-link-arg=-rdynamic");
}