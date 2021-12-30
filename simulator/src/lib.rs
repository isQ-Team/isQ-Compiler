#![no_std]
#![feature(thread_local)]
#![feature(vec_into_raw_parts)]
#![feature(generic_associated_types)]
#[cfg(test)]
#[macro_use]
extern crate std;

#[macro_use]
extern crate alloc;
extern crate core;
pub mod devices;
pub mod facades;
pub use devices::qdevice;
#[macro_use]
extern crate log;
