#ifndef BENCH_PACKED_LAPLACIAN_CUDA_HPP
#define BENCH_PACKED_LAPLACIAN_CUDA_HPP

#include "sfem_base.hpp"

#include <stddef.h>
#include <stdint.h>

using bench_pack_idx_t = uint16_t;

extern "C" {

int bench_cuda_copy_device_to_host(void *dst, const void *src, size_t nbytes);
int bench_cuda_copy_host_to_device(void *dst, const void *src, size_t nbytes);
int bench_cuda_memset(void *ptr, int value, size_t nbytes);
int bench_cuda_device_synchronize();
int bench_cuda_peek_at_last_error();
int bench_cuda_shared_memory_limits(int *max_shmem, int *max_optin_shmem);

double bench_cuda_time(int repeat, void (*callback)(void *), void *ctx);

size_t bench_packed_laplacian_shared_workspace_bytes(ptrdiff_t max_pack_nodes);
ptrdiff_t bench_packed_laplacian_atomic_physical_size(ptrdiff_t max_pack_nodes);
int bench_packed_laplacian_set_dynamic_shared_memory_limit(int element_type, size_t shmem_size);

int bench_packed_laplacian_launch_atomic(int element_type,
                                         ptrdiff_t n_packs,
                                         ptrdiff_t n_elements_per_pack,
                                         ptrdiff_t n_elements,
                                         ptrdiff_t max_pack_nodes,
                                         size_t shmem_size,
                                         int block_size,
                                         bench_pack_idx_t **packed_elements,
                                         const ptrdiff_t *owned_nodes_ptr,
                                         const ptrdiff_t *n_shared,
                                         const ptrdiff_t *ghost_ptr,
                                         const idx_t *ghost_idx,
                                         const jacobian_t *fff,
                                         const real_t *x,
                                         real_t *y,
                                         void *stream);

int bench_packed_laplacian_launch_two_pass(int element_type,
                                           ptrdiff_t n_packs,
                                           ptrdiff_t n_elements_per_pack,
                                           ptrdiff_t n_elements,
                                           ptrdiff_t n_ghost_reduce_rows,
                                           ptrdiff_t max_pack_nodes,
                                           size_t shmem_size,
                                           int block_size,
                                           bench_pack_idx_t **packed_elements,
                                           const ptrdiff_t *owned_nodes_ptr,
                                           const ptrdiff_t *ghost_ptr,
                                           const idx_t *ghost_idx,
                                           const ptrdiff_t *ghost_reduce_ptr,
                                           const ptrdiff_t *ghost_reduce_idx,
                                           const idx_t *ghost_reduce_dest,
                                           const jacobian_t *fff,
                                           const real_t *x,
                                           real_t *y,
                                           real_t *ghost_buf,
                                           void *stream);
}

#endif  // BENCH_PACKED_LAPLACIAN_CUDA_HPP
