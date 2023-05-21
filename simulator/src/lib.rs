#![no_std]
//#![feature(vec_into_raw_parts)]
//#![feature(exit_status_error)]
#[cfg(test)]
#[macro_use]
extern crate std;

#[macro_use]
extern crate alloc;
extern crate core;
pub mod devices;
pub mod facades;
pub mod sim;
pub use devices::qdevice;
#[macro_use]
extern crate log;
