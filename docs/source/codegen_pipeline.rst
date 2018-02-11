Code Generation Pipeline
==============================
 explain how codegen works here and maybe give example


Pipeline
--------
talk about TC mapper





CUDA Code Generation
--------------------

if needed, make it a separate doc





CPU Code Generation
-------------------

The early compilation stages, that is, TC parsing and lowering to Halide plus
polyhedral representation are the same as before. However, instead of
generating CUDA code LLVM-IR is emmited. The IR is then optimized and JIT
compiled.
