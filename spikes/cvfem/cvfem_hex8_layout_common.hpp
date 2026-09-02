#ifndef CVFEM_HEX8_LAYOUT_COMMON_HPP
#define CVFEM_HEX8_LAYOUT_COMMON_HPP

// Shared foundation for the HEX8 CVFEM Navier-Stokes assembly/apply layouts.
//
// Holds everything the layouts have in common: the mesh and matrix containers,
// the pack decomposition, the phase timers behind --breakdown, the per-thread
// scratch allocator, and the element gather/scatter primitives. Each layout then
// lives in its own header:
//
//   cvfem_hex8_layout_atomic.hpp   element sweep, #pragma omp atomic per entry
//   cvfem_hex8_layout_packed.hpp   pack-local buffer, reduced into the global one
//   cvfem_hex8_layout_colored.hpp  colored pack sweep straight into the global one
//   cvfem_hex8_layout_store.hpp    write-once packed assembly (packed variant)
//
// This header is self-contained: it pulls in the smesh/SFEM headers, the HEX8
// element kernels, and the scalar/index types the layouts are written against.

#include "smesh_mesh.hpp"
#include "smesh_mesh_reorder.hpp"
#include "smesh_packed_mesh.hpp"
#include "smesh_buffer.hpp"
#include "sfem_BSR.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#ifndef SFEM_INLINE
#define SFEM_INLINE inline __attribute__((always_inline))
#endif

#ifndef SFEM_NOINLINE
#define SFEM_NOINLINE __attribute__((noinline))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

using scalar_t = double;

static constexpr int N_FIELDS = 4;

#include "cvfem_portability.hpp"

// Needs scalar_t and N_FIELDS above.
#include "cvfem_hex8_pack_common.hpp"

#include "cvfem_hex8_ns_upwind_kernels.hpp"
#include "cvfem_hex8_ns_upwind_sympy_kernels.hpp"

enum class KernelKind {
    Current,
    Fd,
    Sumfact,
    Sympy,
    SympyBlock,
    SympyRow,
    SympyFace
};

static KernelKind parse_kernel(const std::string &name) {
    if (name == "current") return KernelKind::Current;
    if (name == "fd") return KernelKind::Fd;
    if (name == "sumfact") return KernelKind::Sumfact;
    if (name == "sympy") return KernelKind::Sympy;
    if (name == "sympy_block") return KernelKind::SympyBlock;
    if (name == "sympy_row") return KernelKind::SympyRow;
    if (name == "sympy_face") return KernelKind::SympyFace;
    return KernelKind::Sumfact;
}

static bool kernel_uses_sympy_residual(const KernelKind k) {
    return k == KernelKind::Sympy || k == KernelKind::SympyBlock || k == KernelKind::SympyRow || k == KernelKind::SympyFace;
}

static bool kernel_is_valid(const std::string &name) {
    return name == "current" || name == "fd" || name == "sumfact" || name == "sympy" || name == "sympy_block" ||
           name == "sympy_row" || name == "sympy_face";
}

enum class GeomKind { Affine, Isoparam };

static GeomKind parse_geom(const std::string &name) {
    if (name == "isoparam") return GeomKind::Isoparam;
    return GeomKind::Affine;
}

static constexpr scalar_t CVFEM_HEX8_UNIT_CUBE[CVFEM_HEX8_N_NODES][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {1, 1, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 1},
        {1, 1, 1},
        {0, 1, 1}};

struct MeshData {
    std::shared_ptr<smesh::Mesh> mesh;
    ptrdiff_t                    nnodes{0};
    ptrdiff_t                    nelements{0};
    smesh::idx_t               **elems{nullptr};
    smesh::geom_t              **points{nullptr};

    std::vector<scalar_t> ux, uy, uz, p;
    std::vector<scalar_t> rx, ry, rz, rc;
    std::vector<scalar_t> jacobian_adjugate[9];
    std::vector<scalar_t> jacobian_determinant;
};

struct BSR4 {
    std::shared_ptr<smesh::Mesh::NodeToNodeGraph> graph;
    const smesh::count_t                         *rowptr{nullptr};
    const smesh::idx_t                           *colidx{nullptr};
    smesh::SharedBuffer<scalar_t>                 values;
    std::vector<smesh::count_t>                   element_slots;
    ptrdiff_t                                     nnz{0};
};


static int threads_active() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static double wall_time() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ---- lightweight phase breakdown (enabled with --breakdown) -------------------
static int g_breakdown = 0;
static int g_dense_flush = 0;  // --dense-flush: stage ke densely, then flush 64 contiguous blocks
static int g_kernel_only = 0;  // --kernel-only: element kernel writes to a dense stack buffer (no scatter)
static int g_identity_slots[64];
enum PhaseId { PH_ZERO = 0, PH_LOCAL_MEMSET, PH_GATHER, PH_KERNEL, PH_LOCAL_TO_GLOBAL, PH_GHOST, PH_N };
static const char *const g_phase_name[PH_N] = {
        "zero_global", "zero_local", "gather_u", "element_kernel", "local_to_global", "ghost_reduce"};
static double g_phase[PH_N] = {0};
struct PhaseAcc {
    double t[PH_N] = {0};
    void   flush() {
        if (!g_breakdown) return;
#pragma omp critical
        for (int i = 0; i < PH_N; ++i) g_phase[i] += t[i];
    }
};
static SFEM_INLINE double phase_now() { return g_breakdown ? wall_time() : 0.0; }
static void phase_reset() {
    for (int i = 0; i < PH_N; ++i) g_phase[i] = 0;
}
static void phase_report(const char *tag, const int repeat, const int nthreads) {
    if (!g_breakdown) return;
    double total = 0;
    for (int i = 0; i < PH_N; ++i) total += g_phase[i];
    std::printf("  breakdown_%s (ms/call, summed over %d threads):\n", tag, nthreads);
    for (int i = 0; i < PH_N; ++i) {
        if (g_phase[i] == 0) continue;
        std::printf("    %-16s %8.3f  (%5.1f%%)\n",
                    g_phase_name[i],
                    1000.0 * g_phase[i] / double(repeat),
                    100.0 * g_phase[i] / total);
    }
    std::printf("    %-16s %8.3f\n", "TOTAL", 1000.0 * total / double(repeat));
}


static void fill_fields(MeshData &d) {
    d.ux.resize(d.nnodes);
    d.uy.resize(d.nnodes);
    d.uz.resize(d.nnodes);
    d.p.resize(d.nnodes);
    d.rx.assign(d.nnodes, 0.0);
    d.ry.assign(d.nnodes, 0.0);
    d.rz.assign(d.nnodes, 0.0);
    d.rc.assign(d.nnodes, 0.0);

    const auto *const x = d.points[0];
    const auto *const y = d.points[1];
    const auto *const z = d.points[2];

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = 1.0 + 0.3 * x[i] - 0.2 * y[i] + 0.1 * z[i];
        d.uy[i] = -0.4 + 0.2 * x[i] + 0.5 * y[i] - 0.15 * z[i];
        d.uz[i] = 0.2 - 0.1 * x[i] + 0.25 * y[i] + 0.4 * z[i];
        d.p[i]  = 1.0 + 0.1 * x[i] + 0.2 * y[i] - 0.05 * z[i];
    }
}

static void reset_residual(MeshData &d) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.rx[i] = scalar_t(0);
        d.ry[i] = scalar_t(0);
        d.rz[i] = scalar_t(0);
        d.rc[i] = scalar_t(0);
    }
}

static void precompute_affine_geometry(MeshData &d) {
    for (int c = 0; c < 9; ++c) d.jacobian_adjugate[c].resize((size_t)d.nelements);
    d.jacobian_determinant.resize((size_t)d.nelements);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], adj[9], det;
        const auto *const px = d.points[0];
        const auto *const py = d.points[1];
        const auto *const pz = d.points[2];
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            x[a]                 = scalar_t(px[g]);
            y[a]                 = scalar_t(py[g]);
            z[a]                 = scalar_t(pz[g]);
        }
        cvfem_hex8_affine_adj(x, y, z, adj, &det);
        for (int c = 0; c < 9; ++c) d.jacobian_adjugate[c][(size_t)e] = adj[c];
        d.jacobian_determinant[(size_t)e] = det;
    }
}

static SFEM_INLINE void load_hex8_adj(const MeshData &d, const ptrdiff_t e, scalar_t adj[9], scalar_t *det) {
    for (int c = 0; c < 9; ++c) adj[c] = d.jacobian_adjugate[c][(size_t)e];
    *det = d.jacobian_determinant[(size_t)e];
}

static BSR4 make_bsr4(const std::shared_ptr<smesh::Mesh> &mesh) {
    BSR4 b;
    b.graph  = mesh->node_to_node_graph();
    b.rowptr = b.graph->rowptr()->data();
    b.colidx = b.graph->colidx()->data();
    b.nnz    = b.graph->nnz();
    b.values = smesh::create_host_buffer<scalar_t>((size_t)b.nnz * 16);
    return b;
}

static void zero_bsr4(BSR4 &b) {
    const double t0 = phase_now();
    cvfem_zero_scalars(b.values->data(), b.nnz * 16);
    if (g_breakdown) g_phase[PH_ZERO] += wall_time() - t0;
}

static SFEM_INLINE void atomic_add(scalar_t *const SFEM_RESTRICT f, const smesh::idx_t id, const scalar_t value) {
    CVFEM_ATOMIC_ADD(f[id], value);
}

static SFEM_INLINE smesh::count_t find_bsr_slot(const smesh::count_t *const SFEM_RESTRICT rowptr,
                                                const smesh::idx_t *const SFEM_RESTRICT   colidx,
                                                const smesh::idx_t                        row,
                                                const smesh::idx_t                        col) {
    const smesh::count_t begin = rowptr[row];
    const smesh::count_t end   = rowptr[row + 1];
    for (smesh::count_t k = begin; k < end; ++k) {
        if (colidx[k] == col) return k;
    }
    return begin;
}






static SFEM_INLINE void bsr4_add16(scalar_t *const SFEM_RESTRICT dst, const scalar_t *const SFEM_RESTRICT src) {
#pragma omp simd
    for (int i = 0; i < 16; ++i) dst[i] += src[i];
}

static void precompute_element_bsr_slots(const MeshData &d, BSR4 &b) {
    b.element_slots.resize((size_t)d.nelements * CVFEM_HEX8_N_NODES * CVFEM_HEX8_N_NODES);

    smesh::idx_t **const SFEM_RESTRICT elems = d.elems;
    smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t row = elems[a][e];
            for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                const smesh::idx_t col = elems[bnode][e];
                slots[(size_t)e * 64 + a * 8 + bnode] = find_bsr_slot(b.rowptr, b.colidx, row, col);
            }
        }
    }
}


static SFEM_INLINE void gather_element_fields(const MeshData                  &d,
                                              const ptrdiff_t                  e,
                                              scalar_t *const SFEM_RESTRICT    ux,
                                              scalar_t *const SFEM_RESTRICT    uy,
                                              scalar_t *const SFEM_RESTRICT    uz,
                                              scalar_t *const SFEM_RESTRICT    p) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const smesh::idx_t g = d.elems[a][e];
        ux[a]                = d.ux[g];
        uy[a]                = d.uy[g];
        uz[a]                = d.uz[g];
        p[a]                 = d.p[g];
    }
}

static SFEM_INLINE void gather_element_coords(const MeshData               &d,
                                              const ptrdiff_t               e,
                                              scalar_t *const SFEM_RESTRICT x,
                                              scalar_t *const SFEM_RESTRICT y,
                                              scalar_t *const SFEM_RESTRICT z) {
    const auto *const px = d.points[0];
    const auto *const py = d.points[1];
    const auto *const pz = d.points[2];
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const smesh::idx_t g = d.elems[a][e];
        x[a]                 = scalar_t(px[g]);
        y[a]                 = scalar_t(py[g]);
        z[a]                 = scalar_t(pz[g]);
    }
}

static SFEM_INLINE void gather_hex8_adj_soa(const MeshData               &d,
                                            const ptrdiff_t               begin,
                                            const int                     nlanes,
                                            scalar_t *const SFEM_RESTRICT cof0,
                                            scalar_t *const SFEM_RESTRICT cof1,
                                            scalar_t *const SFEM_RESTRICT cof2,
                                            scalar_t *const SFEM_RESTRICT cof3,
                                            scalar_t *const SFEM_RESTRICT cof4,
                                            scalar_t *const SFEM_RESTRICT cof5,
                                            scalar_t *const SFEM_RESTRICT cof6,
                                            scalar_t *const SFEM_RESTRICT cof7,
                                            scalar_t *const SFEM_RESTRICT cof8,
                                            scalar_t *const SFEM_RESTRICT det) {
    const size_t n = (size_t)nlanes * sizeof(scalar_t);
    std::memcpy(cof0, d.jacobian_adjugate[0].data() + begin, n);
    std::memcpy(cof1, d.jacobian_adjugate[1].data() + begin, n);
    std::memcpy(cof2, d.jacobian_adjugate[2].data() + begin, n);
    std::memcpy(cof3, d.jacobian_adjugate[3].data() + begin, n);
    std::memcpy(cof4, d.jacobian_adjugate[4].data() + begin, n);
    std::memcpy(cof5, d.jacobian_adjugate[5].data() + begin, n);
    std::memcpy(cof6, d.jacobian_adjugate[6].data() + begin, n);
    std::memcpy(cof7, d.jacobian_adjugate[7].data() + begin, n);
    std::memcpy(cof8, d.jacobian_adjugate[8].data() + begin, n);
    std::memcpy(det, d.jacobian_determinant.data() + begin, n);
    if (nlanes < CVFEM_HEX8_VEC_SIZE) {
        const size_t pad = (size_t)(CVFEM_HEX8_VEC_SIZE - nlanes) * sizeof(scalar_t);
        std::memset(cof0 + nlanes, 0, pad);
        std::memset(cof1 + nlanes, 0, pad);
        std::memset(cof2 + nlanes, 0, pad);
        std::memset(cof3 + nlanes, 0, pad);
        std::memset(cof4 + nlanes, 0, pad);
        std::memset(cof5 + nlanes, 0, pad);
        std::memset(cof6 + nlanes, 0, pad);
        std::memset(cof7 + nlanes, 0, pad);
        std::memset(cof8 + nlanes, 0, pad);
        for (int lane = nlanes; lane < CVFEM_HEX8_VEC_SIZE; ++lane) det[lane] = scalar_t(1);
    }
}

static SFEM_INLINE void gather_hex8_simd_from_pack(pack_idx_t **const SFEM_RESTRICT   elems,
                                                   const scalar_t *const SFEM_RESTRICT pack_u,
                                                   const MeshData                     &d,
                                                   const ptrdiff_t                     begin,
                                                   const int                           nlanes,
                                                   Hex8InputPack                      &in,
                                                   scalar_t *const SFEM_RESTRICT       cof0,
                                                   scalar_t *const SFEM_RESTRICT       cof1,
                                                   scalar_t *const SFEM_RESTRICT       cof2,
                                                   scalar_t *const SFEM_RESTRICT       cof3,
                                                   scalar_t *const SFEM_RESTRICT       cof4,
                                                   scalar_t *const SFEM_RESTRICT       cof5,
                                                   scalar_t *const SFEM_RESTRICT       cof6,
                                                   scalar_t *const SFEM_RESTRICT       cof7,
                                                   scalar_t *const SFEM_RESTRICT       cof8,
                                                   scalar_t *const SFEM_RESTRICT       det) {
    gather_hex8_adj_soa(d, begin, nlanes, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)elems[a][e] * N_FIELDS;
                in.ux[a][lane]                        = u[0];
                in.uy[a][lane]                        = u[1];
                in.uz[a][lane]                        = u[2];
                in.p[a][lane]                         = u[3];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                in.ux[a][lane] = in.uy[a][lane] = in.uz[a][lane] = in.p[a][lane] = scalar_t(0);
            }
        }
    }
}

static SFEM_INLINE void scatter_hex8_simd_to_pack(pack_idx_t **const SFEM_RESTRICT elems,
                                                  scalar_t *const SFEM_RESTRICT    pack_out,
                                                  const ptrdiff_t                  begin,
                                                  const int                        nlanes,
                                                  const Hex8ResidualPack          &out) {
    for (int lane = 0; lane < nlanes; ++lane) {
        const ptrdiff_t e = begin + lane;
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            scalar_t *const SFEM_RESTRICT dst = pack_out + (ptrdiff_t)elems[a][e] * N_FIELDS;
            dst[0] += out.rx[a][lane];
            dst[1] += out.ry[a][lane];
            dst[2] += out.rz[a][lane];
            dst[3] += out.rc[a][lane];
        }
    }
}

static SFEM_INLINE void gather_hex8_action_simd_from_pack(pack_idx_t **const SFEM_RESTRICT   elems,
                                                          const scalar_t *const SFEM_RESTRICT pack_u,
                                                          const scalar_t *const SFEM_RESTRICT pack_dir,
                                                          const MeshData                     &d,
                                                          const ptrdiff_t                     begin,
                                                          const int                           nlanes,
                                                          Hex8InputPack                      &u,
                                                          Hex8InputPack                      &du,
                                                          scalar_t *const SFEM_RESTRICT       cof0,
                                                          scalar_t *const SFEM_RESTRICT       cof1,
                                                          scalar_t *const SFEM_RESTRICT       cof2,
                                                          scalar_t *const SFEM_RESTRICT       cof3,
                                                          scalar_t *const SFEM_RESTRICT       cof4,
                                                          scalar_t *const SFEM_RESTRICT       cof5,
                                                          scalar_t *const SFEM_RESTRICT       cof6,
                                                          scalar_t *const SFEM_RESTRICT       cof7,
                                                          scalar_t *const SFEM_RESTRICT       cof8,
                                                          scalar_t *const SFEM_RESTRICT       det) {
    gather_hex8_simd_from_pack(elems, pack_u, d, begin, nlanes, u, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const scalar_t *const SFEM_RESTRICT dvec = pack_dir + (ptrdiff_t)elems[a][e] * N_FIELDS;
                du.ux[a][lane]                           = dvec[0];
                du.uy[a][lane]                           = dvec[1];
                du.uz[a][lane]                           = dvec[2];
                du.p[a][lane]                            = dvec[3];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                du.ux[a][lane] = du.uy[a][lane] = du.uz[a][lane] = du.p[a][lane] = scalar_t(0);
            }
        }
    }
}

static SFEM_INLINE void fill_pack_xyz(const PackedData                  &p,
                                      const MeshData                    &d,
                                      const ptrdiff_t                    pack,
                                      const ptrdiff_t                    n_contiguous,
                                      const ptrdiff_t                    n_ghost,
                                      const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                      scalar_t *const SFEM_RESTRICT      pack_x,
                                      scalar_t *const SFEM_RESTRICT      pack_y,
                                      scalar_t *const SFEM_RESTRICT      pack_z) {
    const auto *const px    = d.points[0];
    const auto *const py    = d.points[1];
    const auto *const pz    = d.points[2];
    const ptrdiff_t   owned = p.owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        const ptrdiff_t g = owned + k;
        pack_x[k]         = scalar_t(px[g]);
        pack_y[k]         = scalar_t(py[g]);
        pack_z[k]         = scalar_t(pz[g]);
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const smesh::idx_t g = ghosts[k];
        pack_x[n_contiguous + k] = scalar_t(px[g]);
        pack_y[n_contiguous + k] = scalar_t(py[g]);
        pack_z[n_contiguous + k] = scalar_t(pz[g]);
    }
}

static SFEM_INLINE void gather_hex8_isoparam_simd_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                            const scalar_t *const SFEM_RESTRICT pack_u,
                                                            const scalar_t *const SFEM_RESTRICT pack_x,
                                                            const scalar_t *const SFEM_RESTRICT pack_y,
                                                            const scalar_t *const SFEM_RESTRICT pack_z,
                                                            const ptrdiff_t                     begin,
                                                            const int                           nlanes,
                                                            Hex8InputPack                      &in,
                                                            Hex8CoordPack                      &xyz) {
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t                    loc = elems[a][e];
                const scalar_t *const SFEM_RESTRICT u   = pack_u + (ptrdiff_t)loc * N_FIELDS;
                in.ux[a][lane]                          = u[0];
                in.uy[a][lane]                          = u[1];
                in.uz[a][lane]                          = u[2];
                in.p[a][lane]                           = u[3];
                xyz.x[a][lane]                          = pack_x[loc];
                xyz.y[a][lane]                          = pack_y[loc];
                xyz.z[a][lane]                          = pack_z[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                in.ux[a][lane] = in.uy[a][lane] = in.uz[a][lane] = in.p[a][lane] = scalar_t(0);
                xyz.x[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][0];
                xyz.y[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][1];
                xyz.z[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][2];
            }
        }
    }
}

static SFEM_INLINE void gather_hex8_isoparam_action_simd_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                                   const scalar_t *const SFEM_RESTRICT pack_u,
                                                                   const scalar_t *const SFEM_RESTRICT pack_dir,
                                                                   const scalar_t *const SFEM_RESTRICT pack_x,
                                                                   const scalar_t *const SFEM_RESTRICT pack_y,
                                                                   const scalar_t *const SFEM_RESTRICT pack_z,
                                                                   const ptrdiff_t                     begin,
                                                                   const int                           nlanes,
                                                                   Hex8InputPack                      &u,
                                                                   Hex8InputPack                      &du,
                                                                   Hex8CoordPack                      &xyz) {
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t                    loc  = elems[a][e];
                const scalar_t *const SFEM_RESTRICT usrc = pack_u + (ptrdiff_t)loc * N_FIELDS;
                const scalar_t *const SFEM_RESTRICT dsrc = pack_dir + (ptrdiff_t)loc * N_FIELDS;
                u.ux[a][lane]                            = usrc[0];
                u.uy[a][lane]                            = usrc[1];
                u.uz[a][lane]                            = usrc[2];
                u.p[a][lane]                             = usrc[3];
                du.ux[a][lane]                           = dsrc[0];
                du.uy[a][lane]                           = dsrc[1];
                du.uz[a][lane]                           = dsrc[2];
                du.p[a][lane]                            = dsrc[3];
                xyz.x[a][lane]                           = pack_x[loc];
                xyz.y[a][lane]                           = pack_y[loc];
                xyz.z[a][lane]                           = pack_z[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                u.ux[a][lane] = u.uy[a][lane] = u.uz[a][lane] = u.p[a][lane] = scalar_t(0);
                du.ux[a][lane] = du.uy[a][lane] = du.uz[a][lane] = du.p[a][lane] = scalar_t(0);
                xyz.x[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][0];
                xyz.y[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][1];
                xyz.z[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][2];
            }
        }
    }
}

static SFEM_INLINE void gather_hex8_coords_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                     const scalar_t *const SFEM_RESTRICT pack_x,
                                                     const scalar_t *const SFEM_RESTRICT pack_y,
                                                     const scalar_t *const SFEM_RESTRICT pack_z,
                                                     const ptrdiff_t                     e,
                                                     scalar_t *const SFEM_RESTRICT       x,
                                                     scalar_t *const SFEM_RESTRICT       y,
                                                     scalar_t *const SFEM_RESTRICT       z) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const pack_idx_t loc = elems[a][e];
        x[a]                 = pack_x[loc];
        y[a]                 = pack_y[loc];
        z[a]                 = pack_z[loc];
    }
}

// ---------------------------------------------------------------------------
// Pack staging
// ---------------------------------------------------------------------------

// Copy a pack's nodal fields into an interleaved pack-local buffer. Indexing the
// element kernels through pack-local ids turns four scattered global reads per
// node into one contiguous read, which is why the packed and colored layouts both
// stage through this buffer rather than gathering from d.ux/uy/uz/p directly.
static SFEM_INLINE void fill_pack_fields(const PackedData                       &p,
                                         const MeshData                         &d,
                                         const ptrdiff_t                         pack,
                                         const ptrdiff_t                         n_contiguous,
                                         const ptrdiff_t                         n_ghost,
                                         const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                         scalar_t *const SFEM_RESTRICT           pack_u) {
    const scalar_t *const SFEM_RESTRICT ux    = d.ux.data();
    const scalar_t *const SFEM_RESTRICT uy    = d.uy.data();
    const scalar_t *const SFEM_RESTRICT uz    = d.uz.data();
    const scalar_t *const SFEM_RESTRICT pr    = d.p.data();
    const ptrdiff_t                     owned = p.owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
        const ptrdiff_t               g   = owned + k;
        dst[0]                            = ux[g];
        dst[1]                            = uy[g];
        dst[2]                            = uz[g];
        dst[3]                            = pr[g];
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
        const smesh::idx_t            g   = ghosts[k];
        dst[0]                            = ux[g];
        dst[1]                            = uy[g];
        dst[2]                            = uz[g];
        dst[3]                            = pr[g];
    }
}

// Same, for an already-interleaved global vector (a Krylov direction).
static SFEM_INLINE void fill_pack_interleaved(const PackedData                       &p,
                                              const ptrdiff_t                         pack,
                                              const ptrdiff_t                         n_contiguous,
                                              const ptrdiff_t                         n_ghost,
                                              const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                              const scalar_t *const SFEM_RESTRICT     src,
                                              scalar_t *const SFEM_RESTRICT           pack_v) {
    const ptrdiff_t owned = p.owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        std::memcpy(pack_v + k * N_FIELDS, src + (owned + k) * N_FIELDS, N_FIELDS * sizeof(scalar_t));
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        std::memcpy(pack_v + (n_contiguous + k) * N_FIELDS,
                    src + (ptrdiff_t)ghosts[k] * N_FIELDS,
                    N_FIELDS * sizeof(scalar_t));
    }
}

// Pack-local node id -> global node id. The colored layout scatters into the
// global arrays directly, so it needs this to translate the pack-local element
// table it gathers through.
static SFEM_INLINE void fill_pack_l2g(const PackedData                       &p,
                                      const ptrdiff_t                         pack,
                                      const ptrdiff_t                         n_contiguous,
                                      const ptrdiff_t                         n_ghost,
                                      const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                      smesh::idx_t *const SFEM_RESTRICT       l2g) {
    const ptrdiff_t owned = p.owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) l2g[k] = (smesh::idx_t)(owned + k);
    for (ptrdiff_t k = 0; k < n_ghost; ++k) l2g[n_contiguous + k] = ghosts[k];
}

// ---------------------------------------------------------------------------
// Element matrix -> target matrix
// ---------------------------------------------------------------------------

// Flush a dense block-major element matrix ke[(i*8+k)*16 + c] into the target
// matrix: 64 contiguous 16-double adds instead of ~768 scattered scalar updates.
static SFEM_INLINE void hex8_blocks_to_slots(const int *const SFEM_RESTRICT      slots,
                                             const scalar_t *const SFEM_RESTRICT ke,
                                             scalar_t *const SFEM_RESTRICT       values) {
    for (int blk = 0; blk < 64; ++blk) {
        scalar_t *const SFEM_RESTRICT       dst = values + (ptrdiff_t)slots[blk] * 16;
        const scalar_t *const SFEM_RESTRICT src = ke + blk * 16;
#pragma omp simd
        for (int c = 0; c < 16; ++c) dst[c] += src[c];
    }
}

static SFEM_INLINE void hex8_local_slots_to_bsr4(const int *const SFEM_RESTRICT      slots,
                                                 const scalar_t *const SFEM_RESTRICT ke,
                                                 scalar_t *const SFEM_RESTRICT       values) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
            scalar_t *const SFEM_RESTRICT blk = values + (ptrdiff_t)slots[a * 8 + bnode] * 16;
            for (int rf = 0; rf < 4; ++rf) {
                for (int cf = 0; cf < 4; ++cf) {
                    blk[rf * 4 + cf] += ke[(a * 4 + rf) * CVFEM_HEX8_N_DOF + (bnode * 4 + cf)];
                }
            }
        }
    }
}

#endif  // CVFEM_HEX8_LAYOUT_COMMON_HPP
