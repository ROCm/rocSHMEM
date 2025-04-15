Hacking and Debugging RocSHMEM
==============================

This documentation is mostly for core RocSHMEM developpers. Power users may find it useful.

How to debug parallel programs
------------------------------

When using Open MPI as the launch  mechanism, you can use `OMPI_MCA_mpi_abort_delay=-1` to keep parallel processes active after a crash. You can then use `ssh -t $nodename rocgdb -p $pid` to connect a `rocgdb` to the failed process. Sometimes errors are caught by UCX, `UCX_HANDLE_ERRORS=freeze` will have the same effect in such cases.

Look into `scripts/functional_tests/GDB_README` about an alternative technique that deploys multiple `xterm` to gdb into parallel processes.

How to use the address sanitizer (ASAN)
---------------------------------------

Refer to [General documentation for ASAN on AMD GPUs][1].

### Compiling with ASAN

If this is a fresh build directory, simply add `-DASAN=ON` to the `cmake` invocation.
  `cmake . <...> -DASAN=ON`

If you are enabling ASAN in a previously used build directory, use `ccmake` to alter the CMake Cache
  `ccmake .`

In the `ccmake` interface:
1. find and toggle `ASAN` ON
2. find and delete `COMPILING_TARGETS` (keybind `d`)

Do not forget to delete `COMPILING_TARGETS` again when disabling ASAN (otherwise xnack will remain active, impacting performance).

### Running with ASAN

You may need to add path to `libclang_rt.asan-x86_64.so` to `LD_LIBRARY_PATH` by hand. Depending on the ROCm version, it may be in an unusual place, e.g., `$ROCM_ROOT/lib/llvm/lib/clang/19/lib/linux/libclang_rt.asan-x86_64.so`; `find /opt/rocm -name libclang_rt.asan-x86_64.so` may be required to find it.

ASAN may [crash when using Open MPI][2]. If that happensi, set environment variable `OMPI_MCA_memory=^patcher`. Do not forget to unset this variable when not using ASAN (it will impact performance).

When running the program, the behavior of ASAN can be controlled with the `ASAN_OPTIONS` environment variable.



### References

[1]: https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.4.0/conceptual/using-gpu-sanitizer.html
[2]: https://github.com/open-mpi/ompi/issues/13069
