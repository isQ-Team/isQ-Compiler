use alloc::vec::Vec;

#[derive(Eq, PartialEq)]
pub struct QString{
    data: Vec<i8>
}

impl QString{
    pub unsafe fn from_i8_array(arr: *const i8)->Self{
        let mut s = QString{
            data: Vec::new()
        };
        let mut i = 0;
        while *arr.offset(i) != 0{
            s.data.push(*arr.offset(i));
            i+=1;
        }
        s.data.push(0);
        s
    }
    pub fn from_str(arr: &str)->Self{
        let mut s = QString{
            data: Vec::new()
        };
        for c in arr.as_bytes(){
            s.data.push(*c as i8);
        }
        s.data.push(0);
        s
    }
    pub fn get_raw(&self)->&[i8]{
        &self.data
    }
    pub fn concat(&self, other: &QString)->Self{
        let mut s = QString{
            data: Vec::new()
        };
        s.data.extend_from_slice(&self.data.split_last().unwrap().1);
        s.data.extend_from_slice(&other.data);
        s
    }
}

pub trait ToQIRString{
    fn to_qir_string(&self)->QString;
}

impl<T: core::fmt::Debug> ToQIRString for T {
    fn to_qir_string(&self)->QString{
        QString::from_str(&format!("{:?}", self))
    }
}


pub type QIRString = QString;