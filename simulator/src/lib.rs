#![no_std]
#![feature(thread_local)]

#[cfg(test)]
#[macro_use]
extern crate std;

#[macro_use]
extern crate alloc;
extern crate core;
pub mod devices;
pub mod facades;
pub mod qdevice;

#[macro_use]
extern crate log;
