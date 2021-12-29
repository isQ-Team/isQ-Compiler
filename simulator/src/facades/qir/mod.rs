// Facade for QIR.
// We introduce the concept of "QIR Context" to localize QIR calls.
// A LLVM shim is provided so that a QIR program can be linked against the shim to form a shared library.
//                                                     +----------------------------+
//                                                     |                            |
// +---------------+       +----------------+          |     +-----------------+    |
// |               |       |                |          |     |                 |    |
// | isQ Simulator +-------+ isQ QIR Facade |<---------+     | isQ QIR Shim    |    |
// |               |       |                |  Dynamic |     |                 |    |
// +---------------+       +----------------+  Linking |     +-----------------+    |
//                                                     |         ^                  |
//                                                     |         |Link against      |
//                                                     |         |                  |
//                                                     |     +---+-------------+    |
//                                                     |     |                 |    |
//                                                     |     | Quantum program |    |
//                                                     |     | in QIR          |    |
//                                                     |     |                 |    |
//                                                     |     +-----------------+    |
//                                                     |                            |
//                                                     |                            |
//                                                     |      Shared library        |
//                                                     +----------------------------+
pub mod array;
pub mod bigint;
pub mod callable;
pub mod context;
pub mod resource;
pub mod string;
pub mod tuple;
pub mod types;
mod shim;