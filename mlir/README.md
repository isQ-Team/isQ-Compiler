ISQ MLIR Dialect
=====================

Building
---------------------

We use out-of-tree building for the dialect.
Make sure you have `ninja install`-ed MLIR
so that CMake can find it.

IR Definition
---------------------

see [IR Definition](@ref ISQDialectDef)

Notes on compilation speed
---------------------
A simple executable linked with LLVM and MLIR costs ~1.5GiB, dragging down compilation speed and wasting disk space.

Building MLIR with `-DBUILD_SHARED_LIBS=ON` can reduce target size dramatically (from ~1.5GiB to ~300KiB).