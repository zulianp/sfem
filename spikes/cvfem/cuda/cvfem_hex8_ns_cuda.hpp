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
    //
    // This removes the atomics from the GLOBAL reduction only. It is not bit-reproducible:
    // both modes still accumulate a pack's element contributions with atomicAdd into
    // shared memory, and that fixes no order. Measured -- both modes differ run to run.
    CVFEM_CUDA_FLUSH_TWO_PASS = 1,
    // Bit-reproducible. Each element writes its 32 residual values to a scratch array
    // with no accumulation at all, and a second pass gives one thread per node, summing
    // that node's element contributions in increasing element order. Every sum happens in
    // a fixed order, so the result is identical run to run and across block sizes.
    //
    // It is not free: it materialises 32 doubles per element and reads them back, which
    // is why it is a third mode rather than a fix to the other two. Requires
    // cvfem_cuda_attach_node_to_element.
    CVFEM_CUDA_FLUSH_DETERMINISTIC = 2,
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
    // Moved to subpar/: neither is the fastest arrangement in any measured
    // configuration. sympy_row is within noise of sympy on the atomic layout and 6.5%
    // behind sympy_block when element-coloured; sympy_face is roughly half the speed of
    // everything else on both platforms. The enumerators stay so the numbering is stable
    // across a -DCVFEM_ENABLE_SUBPAR build, but N_VARIANTS -- which is what the driver
    // sweeps -- stops before them.
    CVFEM_CUDA_JAC_SYMPY_ROW   = 3,  // generated, rowwise    (subpar)
    CVFEM_CUDA_JAC_SYMPY_FACE  = 4,  // generated, facewise   (subpar)
#ifdef CVFEM_ENABLE_SUBPAR
    CVFEM_CUDA_JAC_N_VARIANTS  = 5,
#else
    CVFEM_CUDA_JAC_N_VARIANTS  = 3,
#endif
    // Isoparametric geometry. Kept outside the N_VARIANTS range because it is not an
    // alternative CSE arrangement of the same element matrix -- it computes a different
    // one -- so it must not be swept alongside the five above.
    CVFEM_CUDA_JAC_ISOPARAM    = 16,
    // Generated (CSE) isoparametric kernel. Separate from the five affine CSE shapes for
    // the same reason as above: it computes a different element matrix, not a different
    // arrangement of the same one.
    CVFEM_CUDA_JAC_ISOPARAM_SYMPY = 17,
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

// ---- isoparametric geometry ------------------------------------------------
//
// The affine path reads one precomputed adjugate and determinant per element. The
// isoparametric path evaluates the trilinear Jacobian at each of the 12
// sub-control-surface points from the element's node coordinates -- 12 3x3 inversions
// per element rather than a lookup -- which is what a mesh with non-parallel faces
// needs. The element kernels were already __host__ __device__ and templated, so these
// entry points supply the geometry the device path did not previously have.
//
// Upload the coordinates once before calling any of them. This is a no-op if
// cvfem_cuda_boundary_attach has already uploaded them.
// ---- packed-mesh assembly ---------------------------------------------------
//
// A pack-local BSR cannot be staged in shared memory -- one element alone produces 64
// blocks x 16 doubles -- so this stages the pack's *fields* and gathers them through the
// packed mesh's uint16 local ids, writing into the global BSR exactly as the
// element-parallel form does. Both write identically, so comparing them measures what
// the packed mesh addressing is worth on the read side, which is the only side it can
// affect for assembly.
int    cvfem_cuda_assemble_packed(cvfem_cuda_ctx *ctx, double rho, double mu,
                                  int variant, int geom, int block_size, void *stream);
double cvfem_cuda_time_assemble_packed(cvfem_cuda_ctx *ctx, double rho, double mu,
                                       int variant, int geom, int block_size, int repeat);

// ---- standard-mesh matrix-free, the baseline the packed form has to beat --------
//
// One thread per element, grid-stride, global node ids, atomicAdd straight into the
// global vector: no packs, no shared memory, no ghost reduction. This is what the
// operators look like on an ordinary element->node connectivity, and it is what the
// CPU's `atomic` layout has always been -- so it makes the packed-vs-standard question
// answerable on the device the same way it already was on the host.
//
// `geom` takes 0 for affine, 1 for isoparametric.
int    cvfem_cuda_residual_global(cvfem_cuda_ctx *ctx, double rho, double mu,
                                  int geom, int block_size, void *stream);
double cvfem_cuda_time_residual_global(cvfem_cuda_ctx *ctx, double rho, double mu,
                                       int geom, int block_size, int repeat);
int    cvfem_cuda_jacobian_action_global(cvfem_cuda_ctx *ctx, double rho, double mu,
                                         int geom, int block_size, void *stream);
double cvfem_cuda_time_jacobian_action_global(cvfem_cuda_ctx *ctx, double rho, double mu,
                                              int geom, int block_size, int repeat);

// Upload the global element->node connectivity on its own. bsr_attach also does this,
// but it allocates the matrix at the same time, which is not possible at the largest
// sizes; the standard-mesh matrix-free kernels need the connectivity and not the matrix.
int cvfem_cuda_attach_elements_global(cvfem_cuda_ctx *ctx, const int32_t *elements);

// Node-to-element adjacency in CSR, for CVFEM_CUDA_FLUSH_DETERMINISTIC. `enc` holds
// element * 8 + local_index, so the gather knows which of the element's eight slots the
// node occupies. Built on the host; see the driver.
int cvfem_cuda_attach_node_to_element(cvfem_cuda_ctx *ctx, const ptrdiff_t *n2e_ptr,
                                      const int32_t *n2e_enc, ptrdiff_t n_entries);

int    cvfem_cuda_residual_deterministic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                         int geom, int block_size, void *stream);
double cvfem_cuda_time_residual_deterministic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                              int geom, int block_size, int repeat);

int cvfem_cuda_attach_coords(cvfem_cuda_ctx *ctx,
                             const double *px, const double *py, const double *pz);

// Residual and J*v stage the coordinates per pack, so both need more shared memory than
// their affine counterparts: 64 -> 88 B/node and 96 -> 120 B/node. On a pack of 1,377
// nodes that is 121 KiB for J*v, still inside the 227 KiB opt-in but enough to matter
// when choosing the pack size.
size_t cvfem_cuda_residual_isoparam_shmem_bytes(ptrdiff_t max_pack_nodes);
size_t cvfem_cuda_jacobian_action_isoparam_shmem_bytes(ptrdiff_t max_pack_nodes);

int    cvfem_cuda_residual_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                    int flush_mode, int block_size, void *stream);
double cvfem_cuda_time_residual_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                         int flush_mode, int block_size, int repeat);

int    cvfem_cuda_jacobian_action_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                           int flush_mode, int block_size, void *stream);
double cvfem_cuda_time_jacobian_action_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                int flush_mode, int block_size, int repeat);

// Assembly gathers the coordinates from global memory instead: it is element-parallel
// with no pack structure to stage against, and the 24 doubles it reads per element are
// small against the 64 blocks x 16 doubles it writes.
int    cvfem_cuda_assemble_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                    int block_size, void *stream);
double cvfem_cuda_time_assemble_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                         int block_size, int repeat);

int    cvfem_cuda_assemble_isoparam_sympy(cvfem_cuda_ctx *ctx, double rho, double mu,
                                          int block_size, void *stream);
double cvfem_cuda_time_assemble_isoparam_sympy(cvfem_cuda_ctx *ctx, double rho, double mu,
                                               int block_size, int repeat);

// The same BSR-assembly strategies the affine path has, on isoparametric geometry.
//
// Element-coloured: eight colours, no atomics.
int    cvfem_cuda_assemble_ecolored_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                             int block_size, void *stream);
double cvfem_cuda_time_assemble_ecolored_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                  int block_size, int repeat);

// Split across Newton steps. The viscous half depends only on geometry and mu, so it stays
// constant even though the geometry is rebuilt per element; assemble_linear_isoparam builds
// it once and assemble_nonlinear_isoparam restores it and adds the convective half. The two
// halves are selected out of one kernel body, so together they reproduce the full assembly
// by construction.
int    cvfem_cuda_assemble_linear_isoparam(cvfem_cuda_ctx *ctx, double mu,
                                           int block_size, void *stream);
int    cvfem_cuda_assemble_nonlinear_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                              int block_size, void *stream);
double cvfem_cuda_time_assemble_nonlinear_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                   int block_size, int repeat);

// Block diagonal only, for the block-Jacobi preconditioner.
int    cvfem_cuda_assemble_diag_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                         int block_size, void *stream);
double cvfem_cuda_time_assemble_diag_isoparam(cvfem_cuda_ctx *ctx, double rho, double mu,
                                              int block_size, int repeat);

// ---- block diagonal, for the block-Jacobi preconditioner --------------------
//
// The steady solver preconditions with a 4x4 block Jacobi, which needs only the diagonal
// blocks -- one per node, not one per matrix nonzero. That is nnodes*16 doubles instead
// of nnz*16: at n=64, 33.5 MiB against 877 MiB, and 8 of an element's 64 blocks instead
// of all of them.
//
// The element kernel is the same one the full assembly uses. It is steered onto the
// diagonal by passing a slot array that is -1 off the diagonal, which the accumulate
// primitives drop -- so the values are identical to the diagonal of a full assembly by
// construction, not by a second derivation.
int cvfem_cuda_diag_alloc(cvfem_cuda_ctx *ctx);
int cvfem_cuda_assemble_diag(cvfem_cuda_ctx *ctx, double rho, double mu,
                             int block_size, void *stream);
int cvfem_cuda_download_diag(cvfem_cuda_ctx *ctx, double *diag);
double cvfem_cuda_time_assemble_diag(cvfem_cuda_ctx *ctx, double rho, double mu,
                                     int block_size, int repeat);

// Same split as the full matrix: the viscous diagonal is constant, so build it once and
// rebuild only the velocity-dependent part each Newton step.
int cvfem_cuda_assemble_diag_static(cvfem_cuda_ctx *ctx, double mu, int block_size, void *stream);
int cvfem_cuda_assemble_diag_dynamic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                     int block_size, void *stream);
double cvfem_cuda_time_assemble_diag_dynamic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                             int block_size, int repeat);

// In-place 4x4 inverse of every diagonal block, which is what the preconditioner applies.
// Blocks that are singular to working precision are left as the identity so a bad node
// degrades to no preconditioning rather than to NaNs.
int cvfem_cuda_invert_diag(cvfem_cuda_ctx *ctx, int block_size, void *stream);

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

// ---- split assembly: reuse the terms that do not change ---------------------
//
// The Jacobian's viscous part depends only on the mesh and mu, so in a Newton loop it is
// the same matrix every iteration. Assemble it once into a separate buffer; each
// iteration then restores it with a fully coalesced device-to-device copy and adds only
// the velocity-dependent convection and Rhie-Chow terms.
//
// Measured on one element: the constant half touches 528 of the element matrix's entries
// and the velocity-dependent half 250, so the constant half is also the larger share of
// the write traffic that the profiling showed to be the bottleneck.
int cvfem_cuda_assemble_linear(cvfem_cuda_ctx *ctx, double mu, int block_size, void *stream);

// Restores the stored linear part and adds the nonlinear terms. Call after
// cvfem_cuda_assemble_linear; this is what a Newton iteration runs.
int cvfem_cuda_assemble_nonlinear(cvfem_cuda_ctx *ctx, double rho, double mu,
                                  int block_size, void *stream);

double cvfem_cuda_time_assemble_nonlinear(cvfem_cuda_ctx *ctx, double rho, double mu,
                                          int block_size, int repeat);

// The restore alone, and the nonlinear kernel alone, so the split's cost can be
// attributed. Useful for answering whether the restore is worth engineering away.
// Restore only the blocks the nonlinear part actually writes.
//
// Measured on the assembled matrix: the velocity-dependent half touches 26.5% of blocks;
// the other 73.5% are written once at setup and never change. Copying the whole matrix
// back every iteration therefore moves ~4x more than necessary. A block is 16 contiguous
// doubles, so a block-wise restore is still coalesced.
// Attaching the block list also compacts the saved linear data: only the blocks that
// will be overwritten need saving, so the side buffer holds 26.5% of the matrix rather
// than a full duplicate. Call after cvfem_cuda_assemble_linear.
// `block_masks[b]` marks which of the 16 entries of block_ids[b] the velocity-dependent
// half writes. Zeroing only those, rather than whole blocks, is what lets the viscous
// entries elsewhere in the block survive from setup and never be rebuilt.
// `block_masks` is indexed by BLOCK ID over all nnz blocks (not by position in
// block_ids), because both halves of the split read it: the setup pass writes the
// entries it does not cover, the per-iteration pass rebuilds exactly the entries it
// does. One mask, so the two can never disagree.
int cvfem_cuda_nonlinear_blocks_attach(cvfem_cuda_ctx *ctx, ptrdiff_t n_blocks,
                                       const int32_t *block_ids,
                                       const uint16_t *block_masks_by_id);

// Bytes the split currently holds aside, so the memory cost is visible.
size_t cvfem_cuda_linear_side_bytes(cvfem_cuda_ctx *ctx);

// ---- single-matrix split: zero and recompute, no side buffer ----------------
//
// Same idea as above without the second matrix. The viscous contribution splits by
// element-local node pair: pairs with i == k or (i,k) a hex edge land in blocks the
// convection also writes, the other 32 do not. So:
//
//   setup      : assemble the viscous half for the 32 write-once pairs. Never touched again.
//   iteration  : zero the blocks convection writes, then recompute viscous(recomputed
//                pairs) + convection into them.
//
// Costs recomputing half the viscous work each iteration; saves holding any copy of the
// matrix aside.
int cvfem_cuda_assemble_static(cvfem_cuda_ctx *ctx, double mu, int block_size, void *stream);
int cvfem_cuda_assemble_dynamic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                int block_size, void *stream);
double cvfem_cuda_time_assemble_dynamic(cvfem_cuda_ctx *ctx, double rho, double mu,
                                        int block_size, int repeat);

int cvfem_cuda_assemble_nonlinear_sparse(cvfem_cuda_ctx *ctx, double rho, double mu,
                                         int block_size, void *stream);

double cvfem_cuda_time_assemble_nonlinear_sparse(cvfem_cuda_ctx *ctx, double rho, double mu,
                                                 int block_size, int repeat);

double cvfem_cuda_time_restore_only(cvfem_cuda_ctx *ctx, int repeat);
double cvfem_cuda_time_nonlinear_only(cvfem_cuda_ctx *ctx, double rho, double mu,
                                      int block_size, int repeat);

// ---- element-coloured assembly ----------------------------------------------
//
// One kernel launch per element colour, writing the matrix with a plain += instead of
// atomicAdd. Profiling showed the atomic variants issue ~17.9 M thread-level atomics per
// launch and sustain ~80 G atomics/s, at the L2 atomic-unit limit -- so removing the
// atomics, not tuning the arithmetic, is the lever.
//
// Unlike the pack colouring below, this removes the intra-block race too: within one
// element colour no two elements share a node, so no two threads target the same block.
int cvfem_cuda_element_coloring_attach(cvfem_cuda_ctx *ctx, int n_colors,
                                       const int32_t *element_order,
                                       const ptrdiff_t *color_ptr);

int cvfem_cuda_assemble_ecolored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                 int variant, int block_size, void *stream);

double cvfem_cuda_time_assemble_ecolored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                         int variant, int block_size, int repeat);

// Pack-coloured assembly -- MOVED TO subpar/, build with -DCVFEM_ENABLE_SUBPAR.
//
// The reasoning that produced it does not survive the move to a device. Two packs of the
// same colour share no nodes, so on the CPU -- where a pack is one thread -- colouring
// removes the atomics. On the device a pack is a whole block, so the race within the pack
// remains, and the kernel is correct only with blockDim.x == 1: 1.2 MDOF/s, about 200x
// slower than the atomic path. Element colouring (cvfem_cuda_assemble_ecolored, above) is
// the form that works here and is the fastest GPU assembly there is.
#ifdef CVFEM_ENABLE_SUBPAR
int cvfem_cuda_coloring_attach(cvfem_cuda_ctx *ctx, int n_colors,
                               const ptrdiff_t *pack_order, const ptrdiff_t *color_ptr);

int cvfem_cuda_assemble_colored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                int use_sympy, int block_size, void *stream);

double cvfem_cuda_time_assemble_colored(cvfem_cuda_ctx *ctx, double rho, double mu,
                                        int use_sympy, int block_size, int repeat);
#endif

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
