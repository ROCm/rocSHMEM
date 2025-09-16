# Changelog for rocSHMEM

## rocSHMEM 3.0.0 for ROCm 7.0.0

### Added

* Added the Reverse Offload conduit
* Added new APIs:
   * `rocshmem_ctx_barrier`
   * `rocshmem_ctx_barrier_wave`
   * `rocshmem_ctx_barrier_wg`
   * `rocshmem_barrier_all`
   * `rocshmem_barrier_all_wave`
   * `rocshmem_barrier_all_wg`
   * `rocshmem_ctx_sync`
   * `rocshmem_ctx_sync_wave`
   * `rocshmem_ctx_sync_wg`
   * `rocshmem_sync_all`
   * `rocshmem_sync_all_wave`
   * `rocshmem_sync_all_wg`
   * `rocshmem_init_attr`
   * `rocshmem_get_uniqueid`
   * `rocshmem_set_attr_uniqueid_args`
* Added dlmalloc based allocator
* Added XNACK support
* Added support for initialization with MPI communicators other than `MPI_COMM_WORLD`

### Changed

* Changed collective APIs to use `_wg` suffix rather than `_wg_` infix

### Resolved Issues
* Resolved segfault in `rocshmem_wg_ctx_create`, now provides nullptr if ctx cannot be created

## rocSHMEM 2.0.1 for ROCm 6.4.2

### Resolved Issues

* Resolved incorrect output for `rocshmem_ctx_my_pe` and `rocshmem_ctx_n_pes`
* Resolved multi-team errors by providing team specific buffers in `rocshmem_ctx_wg_team_sync`
* Resolved missing implementation of `rocshmem_g` for IPC conduit
