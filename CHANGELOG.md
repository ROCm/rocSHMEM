# Changelog for rocSHMEM

## Unreleased - rocSHMEM 2.x.x for ROCm x.x.x

### Added

* Added the Reverse Offload conduit
* Added new APIs:
   * `rocshmem_barrier`
   * `rocshmem_barrier_wave`
   * `rocshmem_barrier_wg`
   * `rocshmem_barrier_all`
   * `rocshmem_barrier_all_wave`
   * `rocshmem_barrier_all_wg`
   * `rocshmem_sync`
   * `rocshmem_sync_wave`
   * `rocshmem_sync_wg`
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

* Resolved incorrect output for `rocshmem_ctx_my_pe` and `rocshmem_ctx_n_pes`
* Resolved multi-team errors by providing team specific buffers in `rocshmem_ctx_wg_team_sync`
* Resolved segfault in `rocshmem_wg_ctx_create`, now provides nullptr if ctx cannot be created
* Resolved missing implementation of `rocshmem_g` for IPC conduit
