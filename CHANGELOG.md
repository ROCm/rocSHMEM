# Changelog for rocSHMEM

## rocSHMEM 2.0.1 for ROCm 6.4.2

### Resolved Issues

* Resolved incorrect output for `rocshmem_ctx_my_pe` and `rocshmem_ctx_n_pes`
* Resolved multi-team errors by providing team specific buffers in `rocshmem_ctx_wg_team_sync`
* Resolved missing implementation of `rocshmem_g` for IPC conduit
