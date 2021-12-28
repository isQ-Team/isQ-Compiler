#![no_std]
#![feature(thread_local)]

#[cfg(test)]
#[macro_use]
extern crate std;

#[macro_use]
extern crate alloc;
extern crate core;
pub mod qdevice;
pub mod devices;
pub mod facades;

#[macro_use]
extern crate log;