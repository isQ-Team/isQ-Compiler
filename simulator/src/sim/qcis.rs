use crate::{
    devices::naive::NaiveSimulator,
    devices::sq2u3,
    qdevice::{QuantumOp, QDevice}
};

extern crate std;
use core::hash::Hash;
use std::{io::Write, collections::HashMap, os::unix::raw::dev_t};
use alloc::{string::{String, ToString}, vec::Vec};
use regex::Regex;
use serde_json::de;

fn reg(qcis: &Vec<&str>) -> bool{
    
    let mut reg_map = HashMap::new();
    reg_map.insert("X", Regex::new(r"^X(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("Y", Regex::new(r"^Y(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("Z", Regex::new(r"^Z(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("X2P", Regex::new(r"^X2P(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("X2M", Regex::new(r"^X2M(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("Y2P", Regex::new(r"^Y2P(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("Y2M", Regex::new(r"^Y2M(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("H", Regex::new(r"^H(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("S", Regex::new(r"^S(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("SD", Regex::new(r"^SD(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("T", Regex::new(r"^T(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("TD", Regex::new(r"^TD(?:\s(?:Q(?:[1][0-2]|[1-9]))){1}$").unwrap());
    reg_map.insert("CZ", Regex::new(r"^CZ(?:\s+(?:Q(?:[1][0-2]|[1-9]))){2}$").unwrap());
    reg_map.insert("RX", Regex::new(r"^RX\s(?:Q(?:[1][0-2]|[1-9]))\s([+-]?([0-9]*[.])?([0-9]+|[0-9]+[E][+-]?[0-9]+))$").unwrap());
    reg_map.insert("RY", Regex::new(r"^RY\s(?:Q(?:[1][0-2]|[1-9]))\s([+-]?([0-9]*[.])?([0-9]+|[0-9]+[E][+-]?[0-9]+))$").unwrap());
    reg_map.insert("RZ", Regex::new(r"^RZ\s(?:Q(?:[1][0-2]|[1-9]))\s([+-]?([0-9]*[.])?([0-9]+|[0-9]+[E][+-]?[0-9]+))$").unwrap());
    reg_map.insert("RXY", Regex::new(r"^RXY\s(?:Q(?:[1-5][0-9]|60|[1-9]))\s([+-]?([0-9]*[.])?([0-9]+|[0-9]+[E][+-]?[0-9]+))\s([+-]?([0-9]*[.])?([0-9]+|[0-9]+[E][+-]?[0-9]+))$").unwrap());
    reg_map.insert("M", Regex::new(r"^M(?:\s+(?:Q(?:[1][0-2]|[1-9])))+$").unwrap());
    
    for &s in qcis{
        if s.is_empty(){continue;}
        let gate = s.split(' ').next().unwrap();
        if !reg_map.contains_key(gate){ return false; }
        let reg = reg_map.get(gate).unwrap();
        if !reg.is_match(s){ return false; }
    }
    return true;
}


fn get_single_mat(s: &str, parameters: &[f64]) -> [f64; 8]{
    let op = match s{
        "X" => QuantumOp::X,
        "Y" => QuantumOp::Y,
        "Z" => QuantumOp::Z,
        "X2P" => QuantumOp::X2P,
        "X2M" => QuantumOp::X2M,
        "Y2P" => QuantumOp::Y2P,
        "Y2M" => QuantumOp::Y2M,
        "H" => QuantumOp::H,
        "S" => QuantumOp::S,
        "SD" => QuantumOp::SInv,
        "T" => QuantumOp::T,
        "TD" => QuantumOp::TInv,
        "RX" => QuantumOp::Rx,
        "RY" => QuantumOp::Ry,
        "RZ" => QuantumOp::Rz,
        _ => QuantumOp::AnySQ
    };
    let mat = sq2u3::translate_sq_gate_to_matrix(op, parameters);
    let par = unsafe {core::mem::transmute::<_, [f64; 8]>(mat)};
    par
}

pub fn sim(code: String, shots: i64){
    
    let mut dev = NaiveSimulator::new();
    let mut qcis: Vec<&str> = code.split("\n").collect();

    if !reg(&qcis){
        panic!("format error!");
    }

    let mut qmap: HashMap<&str, usize> = HashMap::new();
    let mut qidx:usize = 0;
    let mut mi:Vec<usize> = vec![];

    for s in qcis{
        if s.is_empty(){continue;}
        let mut tmp = s.split(' ');
        let gate = tmp.next().unwrap();

        let qbit = tmp.next().unwrap();
        if !qmap.contains_key(qbit){
            qmap.insert(qbit, qidx);
            qidx += 1;
            dev.alloc_qubit();
        }

        if gate == "CZ"{
            let qbit1 = tmp.next().unwrap();
            if !qmap.contains_key(qbit1){
                qmap.insert(qbit1, qidx);
                qidx += 1;
                dev.alloc_qubit();
            }
            let q1 = qmap.get(qbit1).unwrap();
            dev.controlled_qop(QuantumOp::CZ, &[], &[qmap.get(qbit).unwrap(), q1], &[]);

        }else if gate.starts_with('R'){
            let theta = tmp.next().unwrap().parse::<f64>().unwrap();
            dev.controlled_qop(QuantumOp::AnySQ, &[], &[qmap.get(qbit).unwrap()], &get_single_mat(gate, &[theta]));
        }else if gate == "M"{
            mi.push(*qmap.get(qbit).unwrap());
        }else{
            dev.controlled_qop(QuantumOp::AnySQ, &[], &[qmap.get(qbit).unwrap()], &get_single_mat(gate, &[]));
        }
    }
    
    let mut res_map: HashMap<String, i32> = HashMap::new();
    for i in 0..shots{
        let mut tmp = dev.clone();
        let mut s = "".to_string();
        
        for qbit in mi.clone(){
            let x = tmp.measure(&qbit);
            match x{
                true => s += &"1".to_string(),
                false => s += &"0".to_string()
            };
        }
        
        let count = res_map.entry(s).or_insert(0);
        *count += 1;
    }

    std::println!("{:?}", res_map);

}