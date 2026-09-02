#ifndef CVFEM_HEX8_NS_CUDA_HPP
#define CVFEM_HEX8_NS_CUDA_HPP

// C ABI for the packed CVFEM HEX8 Navier-Stokes CUDA kernels.
//
// Deliberately free of CUDA types so the benchmark driver is compiled by the ordinary
// C++ compiler and never needs nvcc, mirroring bench/cuda/bench_packed_laplacian_cuda.hpp.
// Conventions follow SFEM's: extern "C", int return codes (0 == success), `void *stream`
// last where a stream is taken.
//
// The context owns every device allocation and is created once from host arrays. Field
// upload, compute and download are separate calls so a benchmark can time the kernel
// without host transfers in the measured region.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cvfem_cuda_ctx cvfem_cuda_ctx;

// Flush strategy. See PACKED_FORMAT.md sections 3 and 5.
enum {
    // One pass. Owned nodes below (n_contiguous - n_shared) have a single writing pack
    // and are accumulated without atomics; the shared owned tail and the ghosts use
    // atomicAdd. Not bit-reproducible, because atomics fix no summation order.
    CVFEM_CUDA_FLUSH_ATOMIC = 0,
    // Two passes. Owned nodes are written directly, ghosts are staged per entry and
    // then gathered through ghost_reduce_*, which visits each destination exactly once.
    // Atomics-free and therefore bit-deterministic run to run.
    CVFEM_CUDA_FLUSH_TWO_PASS = 1,
};

int cvfem_cuda_device_info(int *sm_count, int *max_shmem_per_block,
                           int *max_optin_shmem_per_block, int *warp_size);

// Bytes of dynamic shared memory a residual block needs for `max_pack_nodes` nodes:
// N_FIELDS staged in plus N_FIELDS accumulated out, i.e. 64 B/node in fp64.
size_t cvfem_cuda_residual_shmem_bytes(ptrdiff_t max_pack_nodes);

// All pointers are HOST pointers; the context copies what it needs to the device.
// `elems_flat` is [8 * nelements] pack-local node ids, indexed v * nelements + e.
// `adj_flat` is [9 * nelements] indexed c * nelements + e (SoA, so the element loop
// reads it coalesced).
int cvfem_cuda_create(cvfem_cuda_ctx **out_ctx,
                      ptrdiff_t nnodes, ptrdiff_t nelements,
                      ptrdiff_t n_packs, ptrdiff_t n_elements_per_pack,
                      ptrdiff_t max_pack_nodes,
                      ptrdiff_t n_ghost_entries, ptrdiff_t n_ghost_reduce_rows,
                      const uint16_t  *elems_flat,
                      const ptrdiff_t *owned_nodes_ptr,
                      const ptrdiff_t *n_shared,
                      const ptrdiff_t *ghost_ptr,
                      const int32_t   *ghost_idx,
                      const ptrdiff_t *ghost_reduce_ptr,
                      const ptrdiff_t *ghost_reduce_idx,
                      const int32_t   *ghost_reduce_dest,
                      const double    *adj_flat,
                      const double    *det);

int cvfem_cuda_destroy(cvfem_cuda_ctx *ctx);

// u and r are interleaved [4 * nnodes], node-major: u[node * 4 + field].
int cvfem_cuda_upload_u(cvfem_cuda_ctx *ctx, const double *u);
int cvfem_cuda_download_r(cvfem_cuda_ctx *ctx, double *r);

// Device-only; no host transfers. `block_size` <= 0 picks a default.
int cvfem_cuda_residual(cvfem_cuda_ctx *ctx, double rho, double mu,
                        int flush_mode, int block_size, void *stream);

// Residual with the Rhie-Chow mass-flux interpolation active. Stages six more arrays per
// node (coordinates and the nodal pressure gradient), so 112 B/node instead of 64 --
// this is the case that most constrains pack size. Call cvfem_cuda_nodal_p_grad first.
size_t cvfem_cuda_residual_rc_shmem_bytes(ptrdiff_t max_pack_nodes);
int    cvfem_cuda_residual_rc(cvfem_cuda_ctx *ctx, double rho, double mu, double rc_scale,
                              int flush_mode, int block_size, void *stream);

// ---- matrix-free Jacobian action, y = J(u) v -------------------------------
//
// Same block-per-pack shape as the residual, with a third staged array for v. Shared
// memory therefore goes from 8 arrays per node to 12 (96 B/node in fp64).
size_t cvfem_cuda_jacobian_action_shmem_bytes(ptrdiff_t max_pack_nodes);

int cvfem_cuda_upload_v(cvfem_cuda_ctx *ctx, const double *v);

int cvfem_cuda_jacobian_action(cvfem_cuda_ctx *ctx, double rho, double mu,
                               int flush_mode, int block_size, void *stream);

double cvfem_cuda_time_jacobian_action(cvfem_cuda_ctx *ctx, double rho, double mu,
                                       int flush_mode, int block_size, int repeat);

// ---- assembled BSR Jacobian ------------------------------------------------
//
// Assembly is element-parallel with global atomicAdd, NOT block-per-pack. A single
// element produces 64 blocks x 16 doubles = 8 KiB of matrix, so a pack-local BSR is
// orders of magnitude too large for shared memory; the pack machinery earns its keep
// for the matrix-free operators and not here.
enum {
    CVFEM_CUDA_JAC_HANDWRITTEN = 0,  // cvfem_hex8_ns_upwind_jacobian_add_slots
    CVFEM_CUDA_JAC_SYMPY       = 1,  // generated, flat CSE
    CVFEM_CUDA_JAC_SYMPY_BLOCK = 2,  // generated, blockwise write scheduling
    CVFEM_CUDA_JAC_SYMPY_ROW   = 3,  // generated, rowwise
    CVFEM_CUDA_JAC_SYMPY_FACE  = 4,  // generated, per sub-control-surface
    CVFEM_CUDA_JAC_N_VARIANTS  = 5,
};

const char *cvfem_cuda_jac_variant_name(int variant);

// elements_global_flat is [8 * nelements] GLOBAL node ids (v * nelements + e);
// element_slots is [64 * nelements] precomputed global BSR block ids.
int cvfem_cuda_bsr_attach(cvfem_cuda_ctx *ctx, ptrdiff_t nnz,
                          const int32_t *elements_global_flat,
                          const int32_t *element_slots,
                          const int32_t *rowptr, const int32_t *colidx);

int cvfem_cuda_assemble(cvfem_cuda_ctx *ctx, double rho, double mu,
                        int variant, int block_size, void *stream);

int cvfem_cuda_download_values(cvfem_cuda_ctx *ctx, double *values);

// ---- cuSPARSE BSR SpMV -------------------------------------------------------
//
// y = J v using the assembled matrix, as the reference point for the matrix-free
// Jacobian action. The layouts already line up: the BSR blocks are row-major 4x4 (so
// CUSPARSE_DIRECTION_ROW) and the vectors are interleaved node-major, which is exactly
// what bsrmv wants for a block dimension of 4.
//
// Call cvfem_cuda_assemble first; this multiplies whatever is in the values array.
// Result goes to the same device residual buffer the other operators write.
int cvfem_cuda_spmv(cvfem_cuda_ctx *ctx, void *stream);

double cvfem_cuda_time_spmv(cvfem_cuda_ctx *ctx, int repeat);

// Coloured assembly: one kernel launch per colour, no atomics anywhere.
//
// Two packs of the same colour share no nodes (that is what the colouring guarantees),
// so they can never write the same BSR block (i,j) -- a block is written only by an
// element containing both i and j. This exists to answer one question: is assembly
// limited by atomic throughput? Compare it against the atomic variants.
int cvfem_cuda_coloring_attach(cvfem_cuda_ctx *ctx, int n_colors,
                               const ptrdiff_t *pack_order, const ptrdiff_t *color_ptr);

int cvfem_cuda_assemble_colored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                int use_sympy, int block_size, void *stream);

double cvfem_cuda_time_assemble_colored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                        int use_sympy, int block_size, int repeat);

double cvfem_cuda_time_assemble(cvfem_cuda_ctx *ctx, double rho, double mu,
                                int variant, int block_size, int repeat);

// ---- boundary sub-control-surface terms -------------------------------------
//
// Element-parallel over a HOST-BUILT list of boundary elements. The host predicate
// hex8_face_on_domain is false for essentially every interior element, so launching
// over all elements would waste almost the whole grid; the list is built once.
int cvfem_cuda_boundary_attach(cvfem_cuda_ctx *ctx, ptrdiff_t n_boundary,
                               const int32_t *boundary_elems,
                               const double *px, const double *py, const double *pz,
                               double Lx, double Ly, double Lz);

// Adds the boundary contribution into the residual already in the context, so call it
// after cvfem_cuda_residual.
int cvfem_cuda_boundary_residual(cvfem_cuda_ctx *ctx, double rho, double mu,
                                 int block_size, void *stream);

// Adds the boundary contribution to y = J v already in the context.
int cvfem_cuda_boundary_jacobian_action(cvfem_cuda_ctx *ctx, double rho, double mu,
                                        int block_size, void *stream);

// Adds the boundary contribution into the assembled BSR values.
int cvfem_cuda_boundary_assemble(cvfem_cuda_ctx *ctx, double rho, double mu,
                                 int block_size, void *stream);

// ---- Rhie-Chow: nodal pressure gradient -------------------------------------
//
// Volume-weighted average of the element pressure gradient at each node, in two passes:
// accumulate, then divide by the accumulated volume. Needs cvfem_cuda_boundary_attach
// for the coordinates. Writes into device buffers the context owns.
int cvfem_cuda_nodal_p_grad(cvfem_cuda_ctx *ctx, int block_size, void *stream);
int cvfem_cuda_download_p_grad(cvfem_cuda_ctx *ctx, double *pgx, double *pgy, double *pgz);

int cvfem_cuda_synchronize(void);

// Wall time in seconds for `repeat` back-to-back residual launches, measured with CUDA
// events so host-side overhead is excluded.
double cvfem_cuda_time_residual(cvfem_cuda_ctx *ctx, double rho, double mu,
                                int flush_mode, int block_size, int repeat);

#ifdef __cplusplus
}
#endif

#endif  // CVFEM_HEX8_NS_CUDA_HPP
