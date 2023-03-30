#[cfg(windows)]
pub const LINE_ENDING: &'static str = "\r\n";
#[cfg(not(windows))]
pub const LINE_ENDING: &'static str = "\n";

pub fn merge(arr: &[&str]) -> String
{
    let mut res = "".to_string();
    for a in arr.iter() {
        res += a;
        res += LINE_ENDING;
    }
    res
}
