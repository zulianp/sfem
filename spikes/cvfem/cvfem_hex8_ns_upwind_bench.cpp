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
using pack_idx_t = uint16_t;

static constexpr int N_FIELDS = 4;

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
    std::vector<Hex8Geom> geom;
};

struct BSR4 {
    std::shared_ptr<smesh::Mesh::NodeToNodeGraph> graph;
    const smesh::count_t                         *rowptr{nullptr};
    const smesh::idx_t                           *colidx{nullptr};
    smesh::SharedBuffer<scalar_t>                 values;
    std::vector<smesh::count_t>                   element_slots;
    ptrdiff_t                                     nnz{0};
};

struct PackedData {
    std::shared_ptr<smesh::PackedMesh<pack_idx_t>> packed;
    ptrdiff_t                                      n_packs{0};
    ptrdiff_t                                      n_elements_per_pack{0};
    ptrdiff_t                                      max_nodes_per_pack{0};
    pack_idx_t                                   **elems{nullptr};
    const ptrdiff_t                               *owned_nodes_ptr{nullptr};
    const ptrdiff_t                               *n_shared{nullptr};
    const ptrdiff_t                               *ghost_ptr{nullptr};
    const smesh::idx_t                            *ghost_idx{nullptr};
    ptrdiff_t                                      n_ghost_entries{0};
    ptrdiff_t                                      n_ghost_reduce_rows{0};
    const ptrdiff_t                               *ghost_reduce_ptr{nullptr};
    const ptrdiff_t                               *ghost_reduce_idx{nullptr};
    const smesh::idx_t                            *ghost_reduce_dest{nullptr};
    std::vector<scalar_t>                          ghost_buf;
    ptrdiff_t                                      mean_nodes_per_pack{0};
    ptrdiff_t                                      max_actual_nodes_per_pack{0};
    std::vector<std::vector<int>>                  local_rowptr;
    std::vector<std::vector<pack_idx_t>>           local_colidx;
    std::vector<std::vector<smesh::count_t>>       local_global_slot;
    std::vector<int>                               local_element_slot;
    ptrdiff_t                                      max_local_nnz{0};
    std::vector<ptrdiff_t>                         ghost_mat_ptr;
    std::vector<smesh::count_t>                    ghost_mat_slot;
    std::vector<scalar_t>                          ghost_mat_val;
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

template <typename T>
static T *thread_scratch(const int slot, const size_t n) {
    static thread_local T     *ptr[4] = {nullptr, nullptr, nullptr, nullptr};
    static thread_local size_t cap[4] = {0, 0, 0, 0};
    if (cap[slot] < n) {
        std::free(ptr[slot]);
        ptr[slot] = static_cast<T *>(std::calloc(n, sizeof(T)));
        cap[slot] = ptr[slot] ? n : 0;
    }
    return ptr[slot];
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
    d.geom.resize((size_t)d.nelements);

    const auto *const    x  = d.points[0];
    const auto *const    y  = d.points[1];
    const auto *const    z  = d.points[2];
    smesh::idx_t **const ev = d.elems;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        const smesh::idx_t i0 = ev[0][e];
        const smesh::idx_t i1 = ev[1][e];
        const smesh::idx_t i3 = ev[3][e];
        const smesh::idx_t i4 = ev[4][e];

        const scalar_t jx0 = scalar_t(x[i1] - x[i0]);
        const scalar_t jx1 = scalar_t(y[i1] - y[i0]);
        const scalar_t jx2 = scalar_t(z[i1] - z[i0]);
        const scalar_t jy0 = scalar_t(x[i3] - x[i0]);
        const scalar_t jy1 = scalar_t(y[i3] - y[i0]);
        const scalar_t jy2 = scalar_t(z[i3] - z[i0]);
        const scalar_t jz0 = scalar_t(x[i4] - x[i0]);
        const scalar_t jz1 = scalar_t(y[i4] - y[i0]);
        const scalar_t jz2 = scalar_t(z[i4] - z[i0]);

        Hex8Geom g;
        g.cof[0] = jy1 * jz2 - jy2 * jz1;
        g.cof[1] = jy2 * jz0 - jy0 * jz2;
        g.cof[2] = jy0 * jz1 - jy1 * jz0;
        g.cof[3] = jz1 * jx2 - jz2 * jx1;
        g.cof[4] = jz2 * jx0 - jz0 * jx2;
        g.cof[5] = jz0 * jx1 - jz1 * jx0;
        g.cof[6] = jx1 * jy2 - jx2 * jy1;
        g.cof[7] = jx2 * jy0 - jx0 * jy2;
        g.cof[8] = jx0 * jy1 - jx1 * jy0;
        g.det    = jx0 * g.cof[0] + jx1 * g.cof[1] + jx2 * g.cof[2];
        d.geom[(size_t)e] = g;
    }
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

static void zero_bsr4(BSR4 &b) { cvfem_zero_scalars(b.values->data(), b.nnz * 16); }

static SFEM_INLINE void atomic_add(scalar_t *const SFEM_RESTRICT f, const smesh::idx_t id, const scalar_t value) {
#pragma omp atomic update
    f[id] += value;
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

static PackedData make_packed(const std::shared_ptr<smesh::Mesh> &mesh, const int pack_size) {
    PackedData p;
    p.packed              = smesh::PackedMesh<pack_idx_t>::create(mesh, {}, true, pack_size);
    p.n_packs             = p.packed->n_packs(0);
    p.n_elements_per_pack = p.packed->n_elements_per_pack(0);
    p.max_nodes_per_pack  = p.packed->max_nodes_per_pack();
    p.elems               = p.packed->elements(0)->data();
    p.owned_nodes_ptr     = p.packed->owned_nodes_ptr(0)->data();
    p.n_shared            = p.packed->n_shared(0)->data();
    p.ghost_ptr           = p.packed->ghost_ptr(0)->data();
    p.ghost_idx           = p.packed->ghost_idx(0)->data();
    p.n_ghost_entries     = p.packed->n_ghost_entries(0);
    p.n_ghost_reduce_rows = p.packed->n_ghost_reduce_rows(0);
    p.ghost_reduce_ptr    = p.packed->ghost_reduce_ptr(0)->data();
    p.ghost_reduce_idx    = p.packed->ghost_reduce_idx(0)->data();
    p.ghost_reduce_dest   = p.packed->ghost_reduce_dest(0)->data();
    p.ghost_buf.assign((size_t)N_FIELDS * (size_t)p.n_ghost_entries, 0.0);

    ptrdiff_t sum_nodes = 0;
    ptrdiff_t max_nodes = 0;
    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_pack_nodes =
                (p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack]) + (p.ghost_ptr[pack + 1] - p.ghost_ptr[pack]);
        sum_nodes += n_pack_nodes;
        max_nodes = std::max(max_nodes, n_pack_nodes);
    }
    p.mean_nodes_per_pack       = p.n_packs ? sum_nodes / p.n_packs : 0;
    p.max_actual_nodes_per_pack = max_nodes;
    return p;
}

static SFEM_INLINE size_t packed_scratch_n(const PackedData &p) {
    const ptrdiff_t n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
    return (size_t)N_FIELDS * (size_t)n;
}

static SFEM_INLINE size_t packed_xyz_n(const PackedData &p) {
    const ptrdiff_t n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
    return 3 * (size_t)n;
}

static SFEM_INLINE smesh::idx_t pack_local_to_global(const PackedData &p,
                                                     const ptrdiff_t   pack,
                                                     const ptrdiff_t   n_contiguous,
                                                     const pack_idx_t  local) {
    if ((ptrdiff_t)local < n_contiguous) return smesh::idx_t(p.owned_nodes_ptr[pack] + (ptrdiff_t)local);
    return p.ghost_idx[p.ghost_ptr[pack] + ((ptrdiff_t)local - n_contiguous)];
}

static SFEM_INLINE int find_pack_col(const pack_idx_t target, const pack_idx_t *const SFEM_RESTRICT row, const int n) {
    for (int i = 0; i < n; ++i) {
        if (row[i] == target) return i;
    }
    return 0;
}

static SFEM_INLINE void bsr4_add16(scalar_t *const SFEM_RESTRICT dst, const scalar_t *const SFEM_RESTRICT src) {
#pragma omp simd
    for (int i = 0; i < 16; ++i) dst[i] += src[i];
}

static void build_pack_local_crs(PackedData               &p,
                                 const ptrdiff_t           nelements,
                                 const smesh::count_t     *rowptr_g,
                                 const smesh::idx_t       *colidx_g) {
    p.local_rowptr.resize((size_t)p.n_packs);
    p.local_colidx.resize((size_t)p.n_packs);
    p.local_global_slot.resize((size_t)p.n_packs);
    p.local_element_slot.assign((size_t)nelements * CVFEM_HEX8_N_NODES * CVFEM_HEX8_N_NODES, 0);
    p.max_local_nnz = 0;

    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
        const ptrdiff_t e_start      = pack * p.n_elements_per_pack;
        const ptrdiff_t e_end        = std::min(nelements, (pack + 1) * p.n_elements_per_pack);

        std::vector<std::vector<pack_idx_t>> adj((size_t)n_pack_nodes);
        for (ptrdiff_t e = e_start; e < e_end; ++e) {
            pack_idx_t ev[CVFEM_HEX8_N_NODES];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) ev[a] = p.elems[a][e];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                for (int b = 0; b < CVFEM_HEX8_N_NODES; ++b) adj[(size_t)ev[a]].push_back(ev[b]);
            }
        }

        auto &rowptr = p.local_rowptr[(size_t)pack];
        auto &colidx = p.local_colidx[(size_t)pack];
        rowptr.assign((size_t)n_pack_nodes + 1, 0);
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) {
            auto &row = adj[(size_t)i];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            rowptr[(size_t)i + 1] = (int)row.size();
        }
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) rowptr[(size_t)i + 1] += rowptr[(size_t)i];
        colidx.resize((size_t)rowptr[(size_t)n_pack_nodes]);
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) {
            const auto &row = adj[(size_t)i];
            std::memcpy(colidx.data() + rowptr[(size_t)i], row.data(), row.size() * sizeof(pack_idx_t));
        }

        auto &global_slots = p.local_global_slot[(size_t)pack];
        global_slots.resize(colidx.size());
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) {
            const smesh::idx_t grow  = pack_local_to_global(p, pack, n_contiguous, (pack_idx_t)i);
            const int          begin = rowptr[(size_t)i];
            const int          end   = rowptr[(size_t)i + 1];
            for (int t = begin; t < end; ++t) {
                const smesh::idx_t gcol = pack_local_to_global(p, pack, n_contiguous, colidx[(size_t)t]);
                global_slots[(size_t)t] = find_bsr_slot(rowptr_g, colidx_g, grow, gcol);
            }
        }
        p.max_local_nnz = std::max(p.max_local_nnz, (ptrdiff_t)colidx.size());

        for (ptrdiff_t e = e_start; e < e_end; ++e) {
            int *const slots = p.local_element_slot.data() + (size_t)e * 64;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t local_row = p.elems[a][e];
                const int        row_begin = rowptr[(size_t)local_row];
                const int        row_len   = rowptr[(size_t)local_row + 1] - row_begin;
                const pack_idx_t *row      = colidx.data() + row_begin;
                for (int b = 0; b < CVFEM_HEX8_N_NODES; ++b) {
                    slots[a * 8 + b] = row_begin + find_pack_col(p.elems[b][e], row, row_len);
                }
            }
        }
    }

    p.ghost_mat_ptr.assign((size_t)p.n_ghost_entries + 1, 0);
    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t ghost_off    = p.ghost_ptr[pack];
        const auto     &rowptr       = p.local_rowptr[(size_t)pack];
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
            const ptrdiff_t local_i = n_contiguous + k;
            p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k + 1] = rowptr[(size_t)local_i + 1] - rowptr[(size_t)local_i];
        }
    }
    for (ptrdiff_t i = 0; i < p.n_ghost_entries; ++i) p.ghost_mat_ptr[(size_t)i + 1] += p.ghost_mat_ptr[(size_t)i];

    const ptrdiff_t gnnz = p.ghost_mat_ptr[(size_t)p.n_ghost_entries];
    p.ghost_mat_slot.resize((size_t)gnnz);
    p.ghost_mat_val.assign((size_t)gnnz * 16, 0.0);

    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t ghost_off    = p.ghost_ptr[pack];
        const auto     &rowptr       = p.local_rowptr[(size_t)pack];
        const auto     &colidx       = p.local_colidx[(size_t)pack];
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
            const ptrdiff_t local_i = n_contiguous + k;
            const int       begin   = rowptr[(size_t)local_i];
            const int       end     = rowptr[(size_t)local_i + 1];
            const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
            const smesh::idx_t grow = p.ghost_idx[(size_t)ghost_off + (size_t)k];
            for (int t = 0; t < end - begin; ++t) {
                const smesh::idx_t gcol = pack_local_to_global(p, pack, n_contiguous, colidx[(size_t)begin + t]);
                p.ghost_mat_slot[(size_t)dest + (size_t)t] = find_bsr_slot(rowptr_g, colidx_g, grow, gcol);
            }
        }
    }
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

static SFEM_INLINE void gather_hex8_simd_from_pack(pack_idx_t **const SFEM_RESTRICT   elems,
                                                   const scalar_t *const SFEM_RESTRICT pack_u,
                                                   const Hex8Geom *const SFEM_RESTRICT geom,
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
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            const Hex8Geom &g = geom[e];
            cof0[lane]        = g.cof[0];
            cof1[lane]        = g.cof[1];
            cof2[lane]        = g.cof[2];
            cof3[lane]        = g.cof[3];
            cof4[lane]        = g.cof[4];
            cof5[lane]        = g.cof[5];
            cof6[lane]        = g.cof[6];
            cof7[lane]        = g.cof[7];
            cof8[lane]        = g.cof[8];
            det[lane]         = g.det;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)elems[a][e] * N_FIELDS;
                in.ux[a][lane]                        = u[0];
                in.uy[a][lane]                        = u[1];
                in.uz[a][lane]                        = u[2];
                in.p[a][lane]                         = u[3];
            }
        } else {
            cof0[lane] = cof1[lane] = cof2[lane] = scalar_t(0);
            cof3[lane] = cof4[lane] = cof5[lane] = scalar_t(0);
            cof6[lane] = cof7[lane] = cof8[lane] = scalar_t(0);
            det[lane]                            = scalar_t(1);
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
                                                          const Hex8Geom *const SFEM_RESTRICT geom,
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
    gather_hex8_simd_from_pack(elems, pack_u, geom, begin, nlanes, u, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
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

static SFEM_NOINLINE void apply_jacobian_action_atomic(MeshData             &d,
                                                       const scalar_t        rho,
                                                       const scalar_t        mu,
                                                       const scalar_t *const dir,
                                                       scalar_t *const       jv) {
    cvfem_zero_scalars(jv, d.nnodes * N_FIELDS);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(d, e, ux, uy, uz, p);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g         = d.elems[a][e];
            const scalar_t *const SFEM_RESTRICT dv = dir + (ptrdiff_t)g * N_FIELDS;
            vx[a]                            = dv[0];
            vy[a]                            = dv[1];
            vz[a]                            = dv[2];
            q[a]                             = dv[3];
        }
        cvfem_hex8_ns_upwind_jacobian_action(rho, mu, d.geom[(size_t)e], ux, uy, uz, vx, vy, vz, q, r);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 0, 0, r[a * 4 + 0]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 1, 0, r[a * 4 + 1]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 2, 0, r[a * 4 + 2]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 3, 0, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_jacobian_action_atomic_isoparam(MeshData             &d,
                                                                const scalar_t        rho,
                                                                const scalar_t        mu,
                                                                const scalar_t *const dir,
                                                                scalar_t *const       jv) {
    cvfem_zero_scalars(jv, d.nnodes * N_FIELDS);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], vx[8], vy[8], vz[8], q[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t                  g  = d.elems[a][e];
            const scalar_t *const SFEM_RESTRICT dv = dir + (ptrdiff_t)g * N_FIELDS;
            vx[a]                                  = dv[0];
            vy[a]                                  = dv[1];
            vz[a]                                  = dv[2];
            q[a]                                   = dv[3];
        }
        cvfem_hex8_ns_upwind_jacobian_action_isoparam(rho, mu, x, y, z, ux, uy, uz, vx, vy, vz, q, r);
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 0, 0, r[a * 4 + 0]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 1, 0, r[a * 4 + 1]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 2, 0, r[a * 4 + 2]);
            atomic_add(jv + (ptrdiff_t)g * N_FIELDS + 3, 0, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic(MeshData &d, const scalar_t rho, const scalar_t mu) {
    reset_residual(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_residual(rho, mu, d.geom[(size_t)e], ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(d.rx.data(), g, r[a * 4 + 0]);
            atomic_add(d.ry.data(), g, r[a * 4 + 1]);
            atomic_add(d.rz.data(), g, r[a * 4 + 2]);
            atomic_add(d.rc.data(), g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic_sumfact(MeshData &d, const scalar_t rho, const scalar_t mu) {
    reset_residual(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, d.geom[(size_t)e], ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(d.rx.data(), g, r[a * 4 + 0]);
            atomic_add(d.ry.data(), g, r[a * 4 + 1]);
            atomic_add(d.rz.data(), g, r[a * 4 + 2]);
            atomic_add(d.rc.data(), g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic_isoparam(MeshData &d, const scalar_t rho, const scalar_t mu) {
    reset_residual(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, x, y, z, ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(d.rx.data(), g, r[a * 4 + 0]);
            atomic_add(d.ry.data(), g, r[a * 4 + 1]);
            atomic_add(d.rz.data(), g, r[a * 4 + 2]);
            atomic_add(d.rc.data(), g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void apply_residual_atomic_sympy(MeshData &d, const scalar_t rho, const scalar_t mu) {
    reset_residual(d);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], r[CVFEM_HEX8_N_DOF];
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_sympy_residual(rho, mu, d.geom[(size_t)e], ux, uy, uz, p, r);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            atomic_add(d.rx.data(), g, r[a * 4 + 0]);
            atomic_add(d.ry.data(), g, r[a * 4 + 1]);
            atomic_add(d.rz.data(), g, r[a * 4 + 2]);
            atomic_add(d.rc.data(), g, r[a * 4 + 3]);
        }
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_fd(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.empty() ? nullptr : b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8], ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_fd(rho, mu, d.geom[(size_t)e], ux, uy, uz, p, ke);

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t row = d.elems[a][e];
            for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                const smesh::count_t slot =
                        slots ? slots[(size_t)e * 64 + a * 8 + bnode] : find_bsr_slot(b.rowptr, b.colidx, row, d.elems[bnode][e]);
                scalar_t *const      blk  = values + (ptrdiff_t)slot * 16;
                for (int rf = 0; rf < 4; ++rf) {
                    for (int cf = 0; cf < 4; ++cf) {
                        const scalar_t v = ke[(a * 4 + rf) * CVFEM_HEX8_N_DOF + (bnode * 4 + cf)];
#pragma omp atomic update
                        blk[rf * 4 + cf] += v;
                    }
                }
            }
        }
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots(rho, mu, d.geom[(size_t)e], ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_block(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_blockwise(
                rho, mu, d.geom[(size_t)e], ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_row(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_rowwise(
                rho, mu, d.geom[(size_t)e], ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sympy_face(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_facewise(
                rho, mu, d.geom[(size_t)e], ux, uy, uz, slots + (size_t)e * 64, values);
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_sumfact(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT                 values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT     slots  = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t ux[8], uy[8], uz[8], p[8];
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_add_slots<true>(
                rho, mu, d.geom[(size_t)e], ux, uy, uz, slots + (size_t)e * 64, values);
        (void)p;
    }
}

static SFEM_NOINLINE void assemble_jacobian_atomic_isoparam(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    scalar_t *const SFEM_RESTRICT             values = b.values->data();
    const smesh::count_t *const SFEM_RESTRICT slots  = b.element_slots.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        gather_element_coords(d, e, x, y, z);
        gather_element_fields(d, e, ux, uy, uz, p);
        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true>(
                rho, mu, x, y, z, ux, uy, uz, slots + (size_t)e * 64, values);
        (void)p;
    }
}

static SFEM_NOINLINE void apply_residual_packed(MeshData        &d,
                                                PackedData      &p,
                                                const scalar_t   rho,
                                                const scalar_t   mu,
                                                const KernelKind kernel_kind,
                                                const GeomKind   geom_kind) {
    const scalar_t *const SFEM_RESTRICT ux = d.ux.data();
    const scalar_t *const SFEM_RESTRICT uy = d.uy.data();
    const scalar_t *const SFEM_RESTRICT uz = d.uz.data();
    const scalar_t *const SFEM_RESTRICT pr = d.p.data();
    scalar_t *const SFEM_RESTRICT       rx = d.rx.data();
    scalar_t *const SFEM_RESTRICT       ry = d.ry.data();
    scalar_t *const SFEM_RESTRICT       rz = d.rz.data();
    scalar_t *const SFEM_RESTRICT       rc = d.rc.data();
    const size_t                        scratch_n = packed_scratch_n(p);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                geom_kind == GeomKind::Isoparam ? thread_scratch<scalar_t>(3, packed_xyz_n(p)) : nullptr;
        const ptrdiff_t xyz_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off    = p.ghost_ptr[pack];

            std::memset(pack_out, 0, (size_t)n_pack_nodes * (size_t)N_FIELDS * sizeof(scalar_t));

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

            if (geom_kind == GeomKind::Isoparam) {
                fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
                Hex8InputPack    in;
                Hex8CoordPack    xyz;
                Hex8ResidualPack outp;
                for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                    const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                    gather_hex8_isoparam_simd_from_pack(
                            p.elems, pack_u, pack_x, pack_y, pack_z, begin, nlanes, in, xyz);
                    cvfem_hex8_ns_upwind_residual_isoparam_simd(rho, mu, xyz, in, outp);
                    scatter_hex8_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
                }
            } else if (kernel_kind == KernelKind::Sumfact) {
                alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE], cof2[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE], cof5[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE], cof8[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t det[CVFEM_HEX8_VEC_SIZE];
                Hex8InputPack    in;
                Hex8ResidualPack outp;
                for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                    const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                    gather_hex8_simd_from_pack(p.elems,
                                               pack_u,
                                               d.geom.data(),
                                               begin,
                                               nlanes,
                                               in,
                                               cof0,
                                               cof1,
                                               cof2,
                                               cof3,
                                               cof4,
                                               cof5,
                                               cof6,
                                               cof7,
                                               cof8,
                                               det);
                    cvfem_hex8_ns_upwind_residual_sumfact_simd(
                            rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det, in, outp);
                    scatter_hex8_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
                }
            } else {
                const bool sympy = kernel_uses_sympy_residual(kernel_kind);
                for (ptrdiff_t e = e_start; e < e_end; ++e) {
                    scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8], r[CVFEM_HEX8_N_DOF];
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)p.elems[a][e] * N_FIELDS;
                        ux_e[a]                              = u[0];
                        uy_e[a]                              = u[1];
                        uz_e[a]                              = u[2];
                        p_e[a]                               = u[3];
                    }
                    if (sympy)
                        cvfem_hex8_ns_upwind_sympy_residual(rho, mu, d.geom[(size_t)e], ux_e, uy_e, uz_e, p_e, r);
                    else
                        cvfem_hex8_ns_upwind_residual(rho, mu, d.geom[(size_t)e], ux_e, uy_e, uz_e, p_e, r);

                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        scalar_t *const SFEM_RESTRICT out = pack_out + (ptrdiff_t)p.elems[a][e] * N_FIELDS;
                        out[0] += r[a * 4 + 0];
                        out[1] += r[a * 4 + 1];
                        out[2] += r[a * 4 + 2];
                        out[3] += r[a * 4 + 3];
                    }
                }
            }

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + k * N_FIELDS;
                const ptrdiff_t                     g   = owned + k;
                rx[g]                                   = out[0];
                ry[g]                                   = out[1];
                rz[g]                                   = out[2];
                rc[g]                                   = out[3];
            }

            scalar_t *const SFEM_RESTRICT gx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = p.ghost_buf.data() + 3 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (n_contiguous + k) * N_FIELDS;
                gx[ghost_off + k]                       = out[0];
                gy[ghost_off + k]                       = out[1];
                gz[ghost_off + k]                       = out[2];
                gc[ghost_off + k]                       = out[3];
            }
        }
    }

    scalar_t *const fields[N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            fields[f][dest] += sum;
        }
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

static SFEM_NOINLINE void assemble_jacobian_packed(MeshData        &d,
                                                   PackedData      &p,
                                                   BSR4            &b,
                                                   const scalar_t   rho,
                                                   const scalar_t   mu,
                                                   const KernelKind kernel_kind,
                                                   const GeomKind   geom_kind) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                geom_kind == GeomKind::Isoparam ? thread_scratch<scalar_t>(3, packed_xyz_n(p)) : nullptr;
        const ptrdiff_t xyz_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto                             &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto                             &lslots       = p.local_global_slot[(size_t)pack];
            const int                               local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
                const ptrdiff_t               g   = owned + k;
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
                const smesh::idx_t            g   = ghosts[k];
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }
            if (geom_kind == GeomKind::Isoparam)
                fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);

            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8];
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)p.elems[a][e] * N_FIELDS;
                    ux_e[a]                              = u[0];
                    uy_e[a]                              = u[1];
                    uz_e[a]                              = u[2];
                    p_e[a]                               = u[3];
                }

                const int *const SFEM_RESTRICT slots = p.local_element_slot.data() + (size_t)e * 64;
                if (geom_kind == GeomKind::Isoparam) {
                    scalar_t x[8], y[8], z[8];
                    gather_hex8_coords_from_pack(p.elems, pack_x, pack_y, pack_z, e, x, y, z);
                    if (kernel_kind == KernelKind::Fd) {
                        scalar_t ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
                        cvfem_hex8_ns_upwind_jacobian_fd_isoparam(rho, mu, x, y, z, ux_e, uy_e, uz_e, p_e, ke);
                        hex8_local_slots_to_bsr4(slots, ke, local_vals);
                    } else {
                        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                                rho, mu, x, y, z, ux_e, uy_e, uz_e, slots, local_vals);
                    }
                } else if (kernel_kind == KernelKind::Sumfact) {
                    cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                            rho, mu, d.geom[(size_t)e], ux_e, uy_e, uz_e, slots, local_vals);
                } else if (kernel_kind == KernelKind::Sympy) {
                    cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots(rho, mu, d.geom[(size_t)e], ux_e, uy_e, uz_e, slots, local_vals);
                } else if (kernel_kind == KernelKind::SympyBlock) {
                    cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_blockwise(
                            rho, mu, d.geom[(size_t)e], ux_e, uy_e, uz_e, slots, local_vals);
                } else if (kernel_kind == KernelKind::SympyRow) {
                    cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_rowwise(
                            rho, mu, d.geom[(size_t)e], ux_e, uy_e, uz_e, slots, local_vals);
                } else if (kernel_kind == KernelKind::SympyFace) {
                    cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_facewise(
                            rho, mu, d.geom[(size_t)e], ux_e, uy_e, uz_e, slots, local_vals);
                } else {
                    scalar_t ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
                    cvfem_hex8_ns_upwind_jacobian_fd(rho, mu, d.geom[(size_t)e], ux_e, uy_e, uz_e, p_e, ke);
                    hex8_local_slots_to_bsr4(slots, ke, local_vals);
                }
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);

            const ptrdiff_t ghost_off = p.ghost_ptr[pack];
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const ptrdiff_t local_i = n_contiguous + k;
                const int       begin   = lrowptr[(size_t)local_i];
                const int       end     = lrowptr[(size_t)local_i + 1];
                const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
                std::memcpy(p.ghost_mat_val.data() + dest * 16,
                            local_vals + (ptrdiff_t)begin * 16,
                            (size_t)(end - begin) * 16 * sizeof(scalar_t));
            }
        }
    }

    scalar_t *const SFEM_RESTRICT gvalues = b.values->data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const ptrdiff_t begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t end   = p.ghost_reduce_ptr[row + 1];
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t ghost_entry = p.ghost_reduce_idx[j];
            const ptrdiff_t k0          = p.ghost_mat_ptr[(size_t)ghost_entry];
            const ptrdiff_t k1          = p.ghost_mat_ptr[(size_t)ghost_entry + 1];
            for (ptrdiff_t t = k0; t < k1; ++t) {
                bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16], p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}

static SFEM_NOINLINE void apply_jacobian_action_packed(MeshData              &d,
                                                       PackedData            &p,
                                                       const scalar_t         rho,
                                                       const scalar_t         mu,
                                                       const scalar_t *const  dir,
                                                       scalar_t *const        jv,
                                                       const GeomKind         geom_kind) {
    const size_t scratch_n = packed_scratch_n(p);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                geom_kind == GeomKind::Isoparam ? thread_scratch<scalar_t>(3, packed_xyz_n(p)) : nullptr;
        const ptrdiff_t xyz_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off    = p.ghost_ptr[pack];

            std::memset(pack_out, 0, (size_t)n_pack_nodes * (size_t)N_FIELDS * sizeof(scalar_t));

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                scalar_t *const SFEM_RESTRICT dst  = pack_u + k * N_FIELDS;
                scalar_t *const SFEM_RESTRICT dstd = pack_dir + k * N_FIELDS;
                const ptrdiff_t               g    = owned + k;
                dst[0]                             = d.ux[g];
                dst[1]                             = d.uy[g];
                dst[2]                             = d.uz[g];
                dst[3]                             = d.p[g];
                dstd[0]                            = dir[(ptrdiff_t)g * N_FIELDS + 0];
                dstd[1]                            = dir[(ptrdiff_t)g * N_FIELDS + 1];
                dstd[2]                            = dir[(ptrdiff_t)g * N_FIELDS + 2];
                dstd[3]                            = dir[(ptrdiff_t)g * N_FIELDS + 3];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT dst  = pack_u + (n_contiguous + k) * N_FIELDS;
                scalar_t *const SFEM_RESTRICT dstd = pack_dir + (n_contiguous + k) * N_FIELDS;
                const smesh::idx_t            g    = ghosts[k];
                dst[0]                             = d.ux[g];
                dst[1]                             = d.uy[g];
                dst[2]                             = d.uz[g];
                dst[3]                             = d.p[g];
                dstd[0]                            = dir[(ptrdiff_t)g * N_FIELDS + 0];
                dstd[1]                            = dir[(ptrdiff_t)g * N_FIELDS + 1];
                dstd[2]                            = dir[(ptrdiff_t)g * N_FIELDS + 2];
                dstd[3]                            = dir[(ptrdiff_t)g * N_FIELDS + 3];
            }

            Hex8InputPack    u_pack;
            Hex8InputPack    du_pack;
            Hex8ResidualPack outp;
            Hex8CoordPack    xyz;
            if (geom_kind == GeomKind::Isoparam)
                fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
            for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                if (geom_kind == GeomKind::Isoparam) {
                    gather_hex8_isoparam_action_simd_from_pack(p.elems,
                                                               pack_u,
                                                               pack_dir,
                                                               pack_x,
                                                               pack_y,
                                                               pack_z,
                                                               begin,
                                                               nlanes,
                                                               u_pack,
                                                               du_pack,
                                                               xyz);
                    cvfem_hex8_ns_upwind_jacobian_action_isoparam_simd(rho, mu, xyz, u_pack, du_pack, outp);
                } else {
                    alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE],
                            cof2[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE],
                            cof5[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE],
                            cof8[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t det[CVFEM_HEX8_VEC_SIZE];
                    gather_hex8_action_simd_from_pack(p.elems,
                                                      pack_u,
                                                      pack_dir,
                                                      d.geom.data(),
                                                      begin,
                                                      nlanes,
                                                      u_pack,
                                                      du_pack,
                                                      cof0,
                                                      cof1,
                                                      cof2,
                                                      cof3,
                                                      cof4,
                                                      cof5,
                                                      cof6,
                                                      cof7,
                                                      cof8,
                                                      det);
                    cvfem_hex8_ns_upwind_jacobian_action_simd(
                            rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det, u_pack, du_pack, outp);
                }
                scatter_hex8_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
            }

            std::memcpy(jv + owned * N_FIELDS, pack_out, (size_t)n_contiguous * (size_t)N_FIELDS * sizeof(scalar_t));

            scalar_t *const SFEM_RESTRICT gx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = p.ghost_buf.data() + 3 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (n_contiguous + k) * N_FIELDS;
                gx[ghost_off + k]                       = out[0];
                gy[ghost_off + k]                       = out[1];
                gz[ghost_off + k]                       = out[2];
                gc[ghost_off + k]                       = out[3];
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        scalar_t *const    out   = jv + (ptrdiff_t)dest * N_FIELDS;
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            out[f] += sum;
        }
    }
}

static void pack_residual(const MeshData &d, std::vector<scalar_t> &r) {
    r.resize((size_t)d.nnodes * 4);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        r[(size_t)i * 4 + 0] = d.rx[i];
        r[(size_t)i * 4 + 1] = d.ry[i];
        r[(size_t)i * 4 + 2] = d.rz[i];
        r[(size_t)i * 4 + 3] = d.rc[i];
    }
}

static void bsr4_spmv(const BSR4 &b, const ptrdiff_t nnodes, const scalar_t *const x, scalar_t *const y) {
    std::fill(y, y + nnodes * 4, scalar_t(0));

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < nnodes; ++row) {
        scalar_t acc[4] = {0, 0, 0, 0};
        for (smesh::count_t k = b.rowptr[row]; k < b.rowptr[row + 1]; ++k) {
            const scalar_t *const blk = b.values->data() + (ptrdiff_t)k * 16;
            const scalar_t *const xx  = x + (ptrdiff_t)b.colidx[k] * 4;
            acc[0] += blk[0] * xx[0] + blk[1] * xx[1] + blk[2] * xx[2] + blk[3] * xx[3];
            acc[1] += blk[4] * xx[0] + blk[5] * xx[1] + blk[6] * xx[2] + blk[7] * xx[3];
            acc[2] += blk[8] * xx[0] + blk[9] * xx[1] + blk[10] * xx[2] + blk[11] * xx[3];
            acc[3] += blk[12] * xx[0] + blk[13] * xx[1] + blk[14] * xx[2] + blk[15] * xx[3];
        }
        y[(ptrdiff_t)row * 4 + 0] = acc[0];
        y[(ptrdiff_t)row * 4 + 1] = acc[1];
        y[(ptrdiff_t)row * 4 + 2] = acc[2];
        y[(ptrdiff_t)row * 4 + 3] = acc[3];
    }
}

static scalar_t max_abs_diff(const scalar_t *const a, const scalar_t *const b, const ptrdiff_t n) {
    scalar_t m = 0;
    for (ptrdiff_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static scalar_t verify_jacobian_fd(MeshData        &d,
                                   BSR4            &b,
                                   const scalar_t   rho,
                                   const scalar_t   mu,
                                   const GeomKind   geom_kind) {
    const ptrdiff_t ndof = d.nnodes * 4;
    std::vector<scalar_t> x0((size_t)ndof), dir((size_t)ndof), rm, rp, jv((size_t)ndof);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        x0[(size_t)i * 4 + 0] = d.ux[i];
        x0[(size_t)i * 4 + 1] = d.uy[i];
        x0[(size_t)i * 4 + 2] = d.uz[i];
        x0[(size_t)i * 4 + 3] = d.p[i];
    }
    std::fill(dir.begin(), dir.end(), scalar_t(0));
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) dir[(size_t)i * 4 + 3] = scalar_t(1);

    const scalar_t eps = scalar_t(1.0e-6);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x0[(size_t)i * 4 + 0] - eps * dir[(size_t)i * 4 + 0];
        d.uy[i] = x0[(size_t)i * 4 + 1] - eps * dir[(size_t)i * 4 + 1];
        d.uz[i] = x0[(size_t)i * 4 + 2] - eps * dir[(size_t)i * 4 + 2];
        d.p[i]  = x0[(size_t)i * 4 + 3] - eps * dir[(size_t)i * 4 + 3];
    }
    if (geom_kind == GeomKind::Isoparam)
        apply_residual_atomic_isoparam(d, rho, mu);
    else
        apply_residual_atomic(d, rho, mu);
    pack_residual(d, rm);

    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x0[(size_t)i * 4 + 0] + eps * dir[(size_t)i * 4 + 0];
        d.uy[i] = x0[(size_t)i * 4 + 1] + eps * dir[(size_t)i * 4 + 1];
        d.uz[i] = x0[(size_t)i * 4 + 2] + eps * dir[(size_t)i * 4 + 2];
        d.p[i]  = x0[(size_t)i * 4 + 3] + eps * dir[(size_t)i * 4 + 3];
    }
    if (geom_kind == GeomKind::Isoparam)
        apply_residual_atomic_isoparam(d, rho, mu);
    else
        apply_residual_atomic(d, rho, mu);
    pack_residual(d, rp);

    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x0[(size_t)i * 4 + 0];
        d.uy[i] = x0[(size_t)i * 4 + 1];
        d.uz[i] = x0[(size_t)i * 4 + 2];
        d.p[i]  = x0[(size_t)i * 4 + 3];
    }

    bsr4_spmv(b, d.nnodes, dir.data(), jv.data());

    scalar_t max_fd = 0;
    scalar_t max_er = 0;
    for (ptrdiff_t i = 0; i < ndof; ++i) {
        const scalar_t fd = (rp[(size_t)i] - rm[(size_t)i]) / (2 * eps);
        max_fd            = std::max(max_fd, std::fabs(fd));
        max_er            = std::max(max_er, std::fabs(fd - jv[(size_t)i]));
    }
    return max_er / std::max(max_fd, scalar_t(1.0e-30));
}

int main(int argc, char **argv) {
    int own_mpi = 0;
    MPI_Initialized(&own_mpi);
    own_mpi = !own_mpi;
    if (own_mpi) MPI_Init(&argc, &argv);

    int         n          = 8;
    int         repeat     = 10;
    int         warmup     = 2;
    int         assemble   = 0;
    int         jac_action = 0;
    int         bsr_apply  = 0;
    int         verify     = 0;
    int         verify_jac = 0;
    int         use_sfc    = 1;
    scalar_t    rho        = 1.0;
    scalar_t    mu         = 0.01;
    std::string layout     = "atomic";
    std::string kernel     = "sumfact";
    std::string geom       = "affine";
    scalar_t    warp       = 0;
    int         pack_size  = 2048;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc)
            n = std::atoi(argv[++i]);
        else if (arg == "--repeat" && i + 1 < argc)
            repeat = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc)
            warmup = std::atoi(argv[++i]);
        else if (arg == "--rho" && i + 1 < argc)
            rho = std::atof(argv[++i]);
        else if (arg == "--mu" && i + 1 < argc)
            mu = std::atof(argv[++i]);
        else if (arg == "--kernel" && i + 1 < argc)
            kernel = argv[++i];
        else if (arg == "--geom" && i + 1 < argc)
            geom = argv[++i];
        else if (arg == "--warp" && i + 1 < argc)
            warp = std::atof(argv[++i]);
        else if (arg == "--layout" && i + 1 < argc)
            layout = argv[++i];
        else if (arg == "--pack-size" && i + 1 < argc)
            pack_size = std::atoi(argv[++i]);
        else if (arg == "--assemble")
            assemble = 1;
        else if (arg == "--jac-action")
            jac_action = 1;
        else if (arg == "--bsr-apply")
            bsr_apply = 1;
        else if (arg == "--verify")
            verify = 1;
        else if (arg == "--verify-jac")
            verify_jac = 1;
        else if (arg == "--no-sfc")
            use_sfc = 0;
        else if (arg == "--help") {
            std::printf(
                    "usage: %s [--n N] [--repeat N] [--warmup N] [--assemble] [--jac-action] [--bsr-apply]\n"
                    "          [--verify] [--verify-jac] [--layout packed|atomic]\n"
                    "          [--kernel sumfact|current|fd|sympy|sympy_block|sympy_row|sympy_face]\n"
                    "          [--geom affine|isoparam] [--warp EPS] [--pack-size N] [--no-sfc]\n"
                    "  --layout NAME  assembly/apply layout (default atomic)\n"
                    "  --kernel NAME  residual/Jacobian micro-kernel variant (default sumfact)\n"
                    "  --geom NAME    affine (constant J) or isoparam (12 SCS trilinear J)\n"
                    "  --warp EPS     x += EPS * sin(pi y) nodal perturbation\n"
                    "  --bsr-apply    assemble once, then time BSR SpMV y = J(u) v\n",
                    argv[0]);
            if (own_mpi) MPI_Finalize();
            return 0;
        }
    }

    if (!kernel_is_valid(kernel)) {
        std::fprintf(stderr,
                     "invalid --kernel '%s' (expected sumfact, current, fd, sympy, sympy_block, sympy_row, or sympy_face)\n",
                     kernel.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    if ((assemble ? 1 : 0) + (jac_action ? 1 : 0) + (bsr_apply ? 1 : 0) > 1) {
        std::fprintf(stderr, "specify at most one of --assemble, --jac-action, --bsr-apply\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    const KernelKind kernel_kind = parse_kernel(kernel);
    if (geom != "affine" && geom != "isoparam") {
        std::fprintf(stderr, "invalid --geom '%s' (expected affine or isoparam)\n", geom.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    const GeomKind geom_kind = parse_geom(geom);
    if (geom_kind == GeomKind::Isoparam && kernel_uses_sympy_residual(kernel_kind)) {
        std::fprintf(stderr, "--geom isoparam is incompatible with sympy kernels\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    if (layout != "packed" && layout != "atomic") {
        std::fprintf(stderr, "invalid --layout '%s' (expected packed or atomic)\n", layout.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    MeshData d;
    d.mesh = smesh::Mesh::create_hex8_cube(smesh::Communicator::self(), n, n, n, 0, 0, 0, 1, 1, 1);
    if (!d.mesh || d.mesh->element_type(0) != smesh::HEX8) {
        std::fprintf(stderr, "failed to create HEX8 mesh\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    if (use_sfc) {
        auto sfc = smesh::SFC::create_from_env();
        sfc->reorder(*d.mesh);
    }

    PackedData packed;
    if (layout == "packed" || verify || verify_jac || jac_action || bsr_apply) packed = make_packed(d.mesh, pack_size);

    d.nnodes    = d.mesh->n_nodes();
    d.nelements = d.mesh->n_elements(0);
    d.elems     = d.mesh->elements(0)->data();
    d.points    = d.mesh->points()->data();

    if (warp != scalar_t(0)) {
        const scalar_t pi = std::acos(scalar_t(-1));
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
            d.points[0][i] += smesh::geom_t(warp * std::sin(pi * scalar_t(d.points[1][i])));
        }
    }

    fill_fields(d);
    precompute_affine_geometry(d);

    BSR4 bsr;
    if (assemble || verify_jac || bsr_apply) bsr = make_bsr4(d.mesh);
    if (assemble || verify_jac || bsr_apply) {
        if (layout == "packed" || verify_jac || bsr_apply)
            build_pack_local_crs(packed, d.nelements, bsr.rowptr, bsr.colidx);
        if (layout == "atomic") precompute_element_bsr_slots(d, bsr);
    }

    if (layout == "packed" || verify || verify_jac || jac_action || bsr_apply) {
        const size_t scratch_n = packed_scratch_n(packed);
        const size_t bsr_n     = 16 * (size_t)std::max<ptrdiff_t>(packed.max_local_nnz, 1);
        const size_t slot2_n   = std::max(scratch_n, bsr_n);
#pragma omp parallel
        {
            (void)thread_scratch<scalar_t>(0, scratch_n);
            (void)thread_scratch<scalar_t>(1, scratch_n);
            if (assemble || verify_jac || jac_action || bsr_apply) (void)thread_scratch<scalar_t>(2, slot2_n);
            if (geom_kind == GeomKind::Isoparam || verify) (void)thread_scratch<scalar_t>(3, packed_xyz_n(packed));
        }
    }

    if (verify) {
        apply_residual_atomic(d, rho, mu);
        std::vector<scalar_t> current_r;
        pack_residual(d, current_r);

        apply_residual_atomic_sumfact(d, rho, mu);
        std::vector<scalar_t> sumfact_r;
        pack_residual(d, sumfact_r);
        const scalar_t sumfact_err = max_abs_diff(current_r.data(), sumfact_r.data(), (ptrdiff_t)current_r.size());
        std::printf("verify_sumfact_residual_vs_current_abs: %.6e\n", sumfact_err);
        if (sumfact_err > 1.0e-10) {
            std::fprintf(stderr, "HEX8 sumfact residual mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }

        apply_residual_atomic_isoparam(d, rho, mu);
        std::vector<scalar_t> isoparam_r;
        pack_residual(d, isoparam_r);
        const scalar_t iso_err = max_abs_diff(current_r.data(), isoparam_r.data(), (ptrdiff_t)current_r.size());
        std::printf("verify_isoparam_residual_vs_affine_abs: %.6e\n", iso_err);
        if (warp == scalar_t(0)) {
            if (iso_err > 1.0e-12) {
                std::fprintf(stderr, "HEX8 cube isoparam residual mismatch vs affine\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }
        } else if (iso_err <= 1.0e-12) {
            std::fprintf(stderr, "HEX8 warped isoparam residual unexpectedly matches affine\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }

        if (layout == "packed" || verify_jac) {
            apply_residual_packed(d, packed, rho, mu, KernelKind::Current, GeomKind::Affine);
            std::vector<scalar_t> packed_current_r;
            pack_residual(d, packed_current_r);
            const scalar_t packed_err =
                    max_abs_diff(current_r.data(), packed_current_r.data(), (ptrdiff_t)current_r.size());
            std::printf("verify_packed_residual_vs_atomic_abs: %.6e\n", packed_err);
            if (packed_err > 1.0e-10) {
                std::fprintf(stderr, "HEX8 packed residual mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }

            apply_residual_packed(d, packed, rho, mu, KernelKind::Sumfact, GeomKind::Affine);
            std::vector<scalar_t> packed_sumfact_r;
            pack_residual(d, packed_sumfact_r);
            const scalar_t packed_sf_err =
                    max_abs_diff(current_r.data(), packed_sumfact_r.data(), (ptrdiff_t)current_r.size());
            std::printf("verify_packed_sumfact_residual_vs_current_abs: %.6e\n", packed_sf_err);
            if (packed_sf_err > 1.0e-10) {
                std::fprintf(stderr, "HEX8 packed sumfact residual mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }

            apply_residual_packed(d, packed, rho, mu, KernelKind::Sumfact, GeomKind::Isoparam);
            std::vector<scalar_t> packed_iso_r;
            pack_residual(d, packed_iso_r);
            const scalar_t packed_iso_err =
                    max_abs_diff(isoparam_r.data(), packed_iso_r.data(), (ptrdiff_t)packed_iso_r.size());
            std::printf("verify_packed_isoparam_residual_vs_atomic_abs: %.6e\n", packed_iso_err);
            if (packed_iso_err > 1.0e-10) {
                std::fprintf(stderr, "HEX8 packed isoparam residual mismatch\n");
                if (own_mpi) MPI_Finalize();
                return 1;
            }
        }

        if (layout == "packed")
            apply_residual_packed(d, packed, rho, mu, KernelKind::Sympy, GeomKind::Affine);
        else
            apply_residual_atomic_sympy(d, rho, mu);
        std::vector<scalar_t> sympy_r;
        pack_residual(d, sympy_r);
        const scalar_t max_err = max_abs_diff(current_r.data(), sympy_r.data(), (ptrdiff_t)current_r.size());
        std::printf("verify_sympy_residual_vs_current_abs: %.6e\n", max_err);
        if (max_err > 1.0e-10) {
            std::fprintf(stderr, "HEX8 SymPy residual mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    auto apply_fn = [&]() {
        if (geom_kind == GeomKind::Isoparam) {
            if (layout == "packed")
                apply_residual_packed(d, packed, rho, mu, kernel_kind, GeomKind::Isoparam);
            else
                apply_residual_atomic_isoparam(d, rho, mu);
        } else if (layout == "packed")
            apply_residual_packed(d, packed, rho, mu, kernel_kind, GeomKind::Affine);
        else if (kernel_uses_sympy_residual(kernel_kind))
            apply_residual_atomic_sympy(d, rho, mu);
        else if (kernel_kind == KernelKind::Sumfact)
            apply_residual_atomic_sumfact(d, rho, mu);
        else
            apply_residual_atomic(d, rho, mu);
    };
    auto jac_fn = [&]() {
        if (geom_kind == GeomKind::Isoparam) {
            if (layout == "packed")
                assemble_jacobian_packed(d, packed, bsr, rho, mu, kernel_kind, GeomKind::Isoparam);
            else
                assemble_jacobian_atomic_isoparam(d, bsr, rho, mu);
        } else if (layout == "packed") {
            assemble_jacobian_packed(d, packed, bsr, rho, mu, kernel_kind, GeomKind::Affine);
        } else if (kernel_kind == KernelKind::Sumfact)
            assemble_jacobian_atomic_sumfact(d, bsr, rho, mu);
        else if (kernel_kind == KernelKind::Sympy)
            assemble_jacobian_atomic_sympy(d, bsr, rho, mu);
        else if (kernel_kind == KernelKind::SympyBlock)
            assemble_jacobian_atomic_sympy_block(d, bsr, rho, mu);
        else if (kernel_kind == KernelKind::SympyRow)
            assemble_jacobian_atomic_sympy_row(d, bsr, rho, mu);
        else if (kernel_kind == KernelKind::SympyFace)
            assemble_jacobian_atomic_sympy_face(d, bsr, rho, mu);
        else
            assemble_jacobian_atomic_fd(d, bsr, rho, mu);
    };

    std::vector<scalar_t> jac_dir, jac_out;
    if (jac_action || verify_jac || bsr_apply) {
        jac_dir.resize((size_t)d.nnodes * N_FIELDS);
        jac_out.assign((size_t)d.nnodes * N_FIELDS, 0.0);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < d.nnodes * N_FIELDS; ++i) jac_dir[(size_t)i] = 1.0 + 0.01 * scalar_t(i % 7);
    }
    auto jac_action_fn = [&]() {
        if (layout == "packed")
            apply_jacobian_action_packed(d, packed, rho, mu, jac_dir.data(), jac_out.data(), geom_kind);
        else if (geom_kind == GeomKind::Isoparam)
            apply_jacobian_action_atomic_isoparam(d, rho, mu, jac_dir.data(), jac_out.data());
        else
            apply_jacobian_action_atomic(d, rho, mu, jac_dir.data(), jac_out.data());
    };

    if (bsr_apply) jac_fn();
    decltype(sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
            d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(), bsr.values, scalar_t(0))) bsr_apply_op;
    if (bsr_apply) {
        bsr_apply_op = sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
                d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(), bsr.values, scalar_t(0));
    }
    auto bsr_apply_fn = [&]() { bsr_apply_op->apply(jac_dir.data(), jac_out.data()); };

    if (verify_jac) {
        jac_fn();
        const scalar_t rel = verify_jacobian_fd(d, bsr, rho, mu, geom_kind);
        std::printf("verify_jac_spmv_vs_fd_rel: %.6e\n", rel);
        if (rel > 1.0e-6) {
            std::fprintf(stderr, "HEX8 BSR Jacobian mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }

        std::vector<scalar_t> jv_spmv((size_t)d.nnodes * N_FIELDS), jv_mf((size_t)d.nnodes * N_FIELDS),
                jv_mf_atomic((size_t)d.nnodes * N_FIELDS);
        bsr4_spmv(bsr, d.nnodes, jac_dir.data(), jv_spmv.data());
        apply_jacobian_action_packed(d, packed, rho, mu, jac_dir.data(), jv_mf.data(), geom_kind);
        if (geom_kind == GeomKind::Isoparam)
            apply_jacobian_action_atomic_isoparam(d, rho, mu, jac_dir.data(), jv_mf_atomic.data());
        else
            apply_jacobian_action_atomic(d, rho, mu, jac_dir.data(), jv_mf_atomic.data());
        const scalar_t mf_err     = max_abs_diff(jv_spmv.data(), jv_mf.data(), d.nnodes * N_FIELDS);
        const scalar_t atomic_err = max_abs_diff(jv_mf.data(), jv_mf_atomic.data(), d.nnodes * N_FIELDS);
        std::printf("verify_jac_mf_action_vs_spmv_abs: %.6e\n", mf_err);
        std::printf("verify_jac_mf_atomic_action_vs_packed_abs: %.6e\n", atomic_err);
        if (mf_err > 1.0e-8 || atomic_err > 1.0e-12) {
            std::fprintf(stderr, "HEX8 Jacobian-action mismatch\n");
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    for (int i = 0; i < warmup; ++i) {
        if (assemble)
            jac_fn();
        else if (jac_action)
            jac_action_fn();
        else if (bsr_apply)
            bsr_apply_fn();
        else
            apply_fn();
    }

    const double t0 = wall_time();
    for (int i = 0; i < repeat; ++i) {
        if (assemble)
            jac_fn();
        else if (jac_action)
            jac_action_fn();
        else if (bsr_apply)
            bsr_apply_fn();
        else
            apply_fn();
    }
    const double t1 = wall_time();

    const double seconds          = t1 - t0;
    const double seconds_per_call = seconds / double(repeat);
    const double melems           = double(d.nelements) / seconds_per_call / 1.0e6;
    const double unique_mdofs     = double(d.nnodes) * 4.0 / seconds_per_call / 1.0e6;
    const double visit_mdofs      = double(d.nelements) * double(CVFEM_HEX8_N_DOF) / seconds_per_call / 1.0e6;

    const double residual_flops =
            geom_kind == GeomKind::Isoparam ? CVFEM_HEX8_ISOPARAM_RESIDUAL_FLOPS_PER_ELEMENT
                                            : CVFEM_HEX8_RESIDUAL_FLOPS_PER_ELEMENT;
    const double jac_action_flops =
            geom_kind == GeomKind::Isoparam ? CVFEM_HEX8_ISOPARAM_JAC_ACTION_FLOPS_PER_ELEMENT
                                            : CVFEM_HEX8_JAC_ACTION_FLOPS_PER_ELEMENT;
    const double assemble_flops =
            geom_kind == GeomKind::Isoparam ? CVFEM_HEX8_ISOPARAM_ASSEMBLE_FLOPS_PER_ELEMENT
                                            : CVFEM_HEX8_ASSEMBLE_FLOPS_PER_ELEMENT;
    const double elem_apps = double(repeat) * double(d.nelements);

    scalar_t checksum = 0;
    if (assemble) {
        for (ptrdiff_t i = 0; i < bsr.nnz * 16; ++i) checksum += bsr.values->data()[i];
    } else if (jac_action || bsr_apply) {
        for (ptrdiff_t i = 0; i < d.nnodes * N_FIELDS; ++i) checksum += jac_out[(size_t)i];
    } else {
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) checksum += d.rx[i] + d.ry[i] + d.rz[i] + d.rc[i];
    }

    std::printf("cvfem_hex8_ns_upwind_smesh\n");
    std::printf("  mesh_manager: smesh::Mesh::create_hex8_cube\n");
    std::printf("  operation: %s\n",
                bsr_apply ? "bsr_apply" : (jac_action ? "jacobian_action" : (assemble ? "jacobian_assemble" : "residual")));
    std::printf("  layout: %s\n", layout.c_str());
    std::printf("  kernel: %s\n", kernel.c_str());
    std::printf("  geom: %s\n", geom.c_str());
    std::printf("  warp: %.6e\n", warp);
    std::printf("  OpenMP_threads: %d\n", threads_active());
    if (layout == "packed") {
        std::printf("  pack_size: %d\n", pack_size);
        std::printf("  n_packs: %td\n", packed.n_packs);
        std::printf("  n_elements_per_pack: %td\n", packed.n_elements_per_pack);
        std::printf("  mean_nodes_per_pack: %td\n", packed.mean_nodes_per_pack);
        std::printf("  max_actual_nodes_per_pack: %td\n", packed.max_actual_nodes_per_pack);
        if (assemble || bsr_apply) std::printf("  max_local_nnz: %td\n", packed.max_local_nnz);
    }
    std::printf("  cube_n: %d\n", n);
    std::printf("  nodes: %td\n", d.nnodes);
    std::printf("  elements: %td\n", d.nelements);
    std::printf("  repeat: %d\n", repeat);
    if (!bsr_apply) {
        std::printf("  MELEM/s: %.3f\n", melems);
        std::printf("  MDOF/s_element_visits: %.3f\n", visit_mdofs);
    }
    std::printf("  MDOF/s_unique_mesh_dofs: %.3f\n", unique_mdofs);
    std::printf("  checksum: %.16e\n", checksum);
    if (!assemble && !jac_action && !bsr_apply) {
        std::printf("  seconds_per_apply: %.6e\n", seconds_per_call);
        std::printf("  GFLOP/s_model: %.3f\n", elem_apps * residual_flops / seconds / 1.0e9);
        std::printf("  flops_per_element_model: %.1f\n", residual_flops);
    }
    if (assemble || bsr_apply) {
        std::printf("  bsr_nnz: %td\n", bsr.nnz);
        std::printf("  bsr_nnz_per_node: %.3f\n", double(bsr.nnz) / double(d.nnodes));
        {
            smesh::count_t dmin = bsr.rowptr[1] - bsr.rowptr[0];
            smesh::count_t dmax = dmin;
            for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                const smesh::count_t deg = bsr.rowptr[i + 1] - bsr.rowptr[i];
                dmin                     = std::min(dmin, deg);
                dmax                     = std::max(dmax, deg);
            }
            std::printf("  bsr_row_nnz_min: %d\n", (int)dmin);
            std::printf("  bsr_row_nnz_max: %d\n", (int)dmax);
            std::printf("  bsr_values_MiB: %.3f\n", double(bsr.nnz) * 16.0 * 8.0 / (1024.0 * 1024.0));
            std::printf("  bsr_x_KiB: %.3f\n", double(d.nnodes) * 4.0 * 8.0 / 1024.0);
        }
    }
    if (assemble) {
        std::printf("  seconds_per_assemble: %.6e\n", seconds_per_call);
        std::printf("  MELEM/s_assemble: %.3f\n", melems);
        std::printf("  GFLOP/s_assemble_model: %.3f\n", elem_apps * assemble_flops / seconds / 1.0e9);
        std::printf("  flops_per_element_assemble_model: %.1f\n", assemble_flops);
    }
    if (jac_action) {
        std::printf("  seconds_per_jac_action: %.6e\n", seconds_per_call);
        std::printf("  MELEM/s_jac_action: %.3f\n", melems);
        std::printf("  GFLOP/s_jac_action_model: %.3f\n", elem_apps * jac_action_flops / seconds / 1.0e9);
        std::printf("  flops_per_element_jac_action_model: %.1f\n", jac_action_flops);
    }
    if (bsr_apply) {
        const double bsr_apply_flops = double(bsr.nnz) * 2.0 * 16.0;
        const double bsr_apply_bytes = double(bsr.nnz) * 16.0 * double(sizeof(scalar_t)) +
                                       double(d.nnodes) * 8.0 * double(sizeof(scalar_t)) +
                                       double(bsr.nnz) * double(sizeof(smesh::idx_t));
        std::printf("  seconds_per_bsr_apply: %.6e\n", seconds_per_call);
        std::printf("  GFLOP/s_bsr_apply_model: %.3f\n", double(repeat) * bsr_apply_flops / seconds / 1.0e9);
        std::printf("  GB/s_bsr_apply_model: %.3f\n", double(repeat) * bsr_apply_bytes / seconds / 1.0e9);
        std::printf("  flops_per_bsr_apply_model: %.1f\n", bsr_apply_flops);
        std::printf("  bytes_per_bsr_apply_model: %.1f\n", bsr_apply_bytes);
    }

    d.mesh.reset();
    if (own_mpi) MPI_Finalize();
    return 0;
}
