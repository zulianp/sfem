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

using scalar_t   = double;
using jacobian_t = smesh::jacobian_t;
using pack_idx_t = uint16_t;

#ifndef VEC_BYTES
#define VEC_BYTES 128
#endif

static constexpr int VEC_SIZE = VEC_BYTES / int(sizeof(scalar_t));
static_assert(VEC_SIZE >= 1, "invalid vector size");
static constexpr int N_FIELDS    = 4;
static constexpr int N_ACTION_BASE_FIELDS = 4;
static constexpr int ALIGN_BYTES = 64;

#include "cvfem_tet4_ns_upwind_kernels.hpp"

enum class KernelKind {
    Current,
    Sympy,
    CurrentSlots,
    SympySlots,
    SympyDirect,
    SympyBlock,
    SympyFace,
    SympySimd,
    SympySimdClean,
    SympyBlockSimd,
    SympyRowSimd,
    SympyRowSimdFused,
    SympyFaceSimd
};

static KernelKind parse_kernel(const std::string &name) {
    if (name == "sympy") return KernelKind::Sympy;
    if (name == "current_slots") return KernelKind::CurrentSlots;
    if (name == "sympy_slots") return KernelKind::SympySlots;
    if (name == "sympy_direct") return KernelKind::SympyDirect;
    if (name == "sympy_block") return KernelKind::SympyBlock;
    if (name == "sympy_face") return KernelKind::SympyFace;
    if (name == "sympy_simd") return KernelKind::SympySimd;
    if (name == "sympy_simd_clean") return KernelKind::SympySimdClean;
    if (name == "sympy_block_simd") return KernelKind::SympyBlockSimd;
    if (name == "sympy_row_simd") return KernelKind::SympyRowSimd;
    if (name == "sympy_row_simd_fused") return KernelKind::SympyRowSimdFused;
    if (name == "sympy_face_simd") return KernelKind::SympyFaceSimd;
    return KernelKind::Current;
}

static bool kernel_is_sympy_residual(const KernelKind k) {
    return k == KernelKind::Sympy || k == KernelKind::SympySlots || k == KernelKind::SympyDirect || k == KernelKind::SympyBlock ||
           k == KernelKind::SympyFace || k == KernelKind::SympySimd || k == KernelKind::SympySimdClean || k == KernelKind::SympyBlockSimd ||
           k == KernelKind::SympyRowSimd || k == KernelKind::SympyRowSimdFused || k == KernelKind::SympyFaceSimd;
}

template <typename T>
struct AlignedBuffer {
    T     *ptr{nullptr};
    size_t n{0};

    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer &)            = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;
    AlignedBuffer(AlignedBuffer &&o) noexcept : ptr(o.ptr), n(o.n) {
        o.ptr = nullptr;
        o.n   = 0;
    }
    AlignedBuffer &operator=(AlignedBuffer &&o) noexcept {
        if (this != &o) {
            std::free(ptr);
            ptr   = o.ptr;
            n     = o.n;
            o.ptr = nullptr;
            o.n   = 0;
        }
        return *this;
    }
    ~AlignedBuffer() { std::free(ptr); }

    void resize(const size_t count) {
        std::free(ptr);
        ptr = nullptr;
        n   = 0;
        if (!count) return;
        const size_t bytes = ((count * sizeof(T) + (size_t)ALIGN_BYTES - 1) / (size_t)ALIGN_BYTES) * (size_t)ALIGN_BYTES;
        void        *p     = nullptr;
        if (posix_memalign(&p, (size_t)ALIGN_BYTES, bytes) != 0) return;
        std::memset(p, 0, bytes);
        ptr = static_cast<T *>(p);
        n   = count;
    }

    T       *data() { return ptr; }
    const T *data() const { return ptr; }
    T       &operator[](ptrdiff_t i) { return ptr[i]; }
    const T &operator[](ptrdiff_t i) const { return ptr[i]; }
};

struct MeshData {
    std::shared_ptr<smesh::Mesh> mesh;
    ptrdiff_t                    nnodes{0};
    ptrdiff_t                    nelements{0};
    smesh::idx_t               **elems{nullptr};
    smesh::geom_t              **points{nullptr};

    std::vector<scalar_t> ux, uy, uz, p;
    std::vector<scalar_t> rx, ry, rz, rc;

    AlignedBuffer<jacobian_t> adj[9];
    AlignedBuffer<jacobian_t> det;
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
    std::vector<smesh::idx_t>                      ghost_mat_col;
    std::vector<smesh::count_t>                    ghost_mat_slot;
    std::vector<scalar_t>                          ghost_mat_val;
    ptrdiff_t                                      action_base_stride{0};
    std::vector<scalar_t>                          action_base_velocity;
};

static ptrdiff_t parse_size(const char *s) {
    char        *end   = nullptr;
    const double v     = std::strtod(s, &end);
    ptrdiff_t    scale = 1;
    if (end && *end) {
        if (*end == 'k' || *end == 'K') scale = 1024LL;
        if (*end == 'm' || *end == 'M') scale = 1024LL * 1024LL;
    }
    return ptrdiff_t(v * double(scale));
}

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
        d.rx[i] = 0.0;
        d.ry[i] = 0.0;
        d.rz[i] = 0.0;
        d.rc[i] = 0.0;
    }
}

static void precompute_affine_geometry(MeshData &d) {
    const ptrdiff_t padded_nelements = ((d.nelements + VEC_SIZE - 1) / VEC_SIZE) * VEC_SIZE;

    for (int k = 0; k < 9; ++k) d.adj[k].resize(padded_nelements);
    d.det.resize(padded_nelements);

    const auto *const    x  = d.points[0];
    const auto *const    y  = d.points[1];
    const auto *const    z  = d.points[2];
    smesh::idx_t **const ev = d.elems;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        const smesh::idx_t i0 = ev[0][e];
        const smesh::idx_t i1 = ev[1][e];
        const smesh::idx_t i2 = ev[2][e];
        const smesh::idx_t i3 = ev[3][e];

        const scalar_t j00 = scalar_t(x[i1] - x[i0]);
        const scalar_t j10 = scalar_t(y[i1] - y[i0]);
        const scalar_t j20 = scalar_t(z[i1] - z[i0]);
        const scalar_t j01 = scalar_t(x[i2] - x[i0]);
        const scalar_t j11 = scalar_t(y[i2] - y[i0]);
        const scalar_t j21 = scalar_t(z[i2] - z[i0]);
        const scalar_t j02 = scalar_t(x[i3] - x[i0]);
        const scalar_t j12 = scalar_t(y[i3] - y[i0]);
        const scalar_t j22 = scalar_t(z[i3] - z[i0]);

        const scalar_t a00 = j11 * j22 - j12 * j21;
        const scalar_t a01 = -(j10 * j22 - j12 * j20);
        const scalar_t a02 = j10 * j21 - j11 * j20;
        const scalar_t a10 = -(j01 * j22 - j02 * j21);
        const scalar_t a11 = j00 * j22 - j02 * j20;
        const scalar_t a12 = -(j00 * j21 - j01 * j20);
        const scalar_t a20 = j01 * j12 - j02 * j11;
        const scalar_t a21 = -(j00 * j12 - j02 * j10);
        const scalar_t a22 = j00 * j11 - j01 * j10;

        const scalar_t determinant = j00 * a00 + j01 * a01 + j02 * a02;

        d.adj[0][e] = jacobian_t(a00);
        d.adj[1][e] = jacobian_t(a10);
        d.adj[2][e] = jacobian_t(a20);
        d.adj[3][e] = jacobian_t(a01);
        d.adj[4][e] = jacobian_t(a11);
        d.adj[5][e] = jacobian_t(a21);
        d.adj[6][e] = jacobian_t(a02);
        d.adj[7][e] = jacobian_t(a12);
        d.adj[8][e] = jacobian_t(a22);
        d.det[e]    = jacobian_t(determinant);
    }

    if (padded_nelements > d.nelements) {
        const ptrdiff_t last = d.nelements - 1;
        for (ptrdiff_t e = d.nelements; e < padded_nelements; ++e) {
            for (int k = 0; k < 9; ++k) d.adj[k][e] = d.adj[k][last];
            d.det[e] = d.det[last];
        }
    }
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

static SFEM_INLINE smesh::idx_t pack_local_to_global(const PackedData &p,
                                                     const ptrdiff_t   pack,
                                                     const ptrdiff_t   n_contiguous,
                                                     const pack_idx_t  local) {
    if ((ptrdiff_t)local < n_contiguous) return smesh::idx_t(p.owned_nodes_ptr[pack] + (ptrdiff_t)local);
    return p.ghost_idx[p.ghost_ptr[pack] + ((ptrdiff_t)local - n_contiguous)];
}

static SFEM_INLINE smesh::count_t bsr4_find_slot(const smesh::count_t *const SFEM_RESTRICT rowptr,
                                                 const smesh::idx_t *const SFEM_RESTRICT   colidx,
                                                 const smesh::idx_t                        row,
                                                 const smesh::idx_t                        col) {
    const int          len = int(rowptr[row + 1] - rowptr[row]);
    const smesh::idx_t ks  = cvfem_linear_search(col, &colidx[rowptr[row]], len);
    return rowptr[row] + (smesh::count_t)ks;
}

static void build_pack_local_crs(PackedData               &p,
                                 const ptrdiff_t           nelements,
                                 const smesh::count_t     *rowptr_g,
                                 const smesh::idx_t       *colidx_g) {
    p.local_rowptr.resize((size_t)p.n_packs);
    p.local_colidx.resize((size_t)p.n_packs);
    p.local_global_slot.resize((size_t)p.n_packs);
    p.local_element_slot.assign((size_t)nelements * 16, 0);
    p.max_local_nnz = 0;

    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
        const ptrdiff_t e_start      = pack * p.n_elements_per_pack;
        const ptrdiff_t e_end        = std::min(nelements, (pack + 1) * p.n_elements_per_pack);

        std::vector<std::vector<pack_idx_t>> adj((size_t)n_pack_nodes);
        for (ptrdiff_t e = e_start; e < e_end; ++e) {
            const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) adj[(size_t)ev[a]].push_back(ev[b]);
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
        auto &slots = p.local_global_slot[(size_t)pack];
        slots.resize(colidx.size());
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) {
            const smesh::idx_t grow  = pack_local_to_global(p, pack, n_contiguous, (pack_idx_t)i);
            const int          begin = rowptr[(size_t)i];
            const int          end   = rowptr[(size_t)i + 1];
            for (int t = begin; t < end; ++t) {
                const smesh::idx_t gcol = pack_local_to_global(p, pack, n_contiguous, colidx[(size_t)t]);
                slots[(size_t)t]        = bsr4_find_slot(rowptr_g, colidx_g, grow, gcol);
            }
        }
        p.max_local_nnz = std::max(p.max_local_nnz, (ptrdiff_t)colidx.size());

        for (ptrdiff_t e = e_start; e < e_end; ++e) {
            const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
            int             *slots = p.local_element_slot.data() + (size_t)e * 16;
            for (int a = 0; a < 4; ++a) {
                const int       row_begin = rowptr[(size_t)ev[a]];
                const int       lenrow    = rowptr[(size_t)ev[a] + 1] - row_begin;
                const pack_idx_t *row     = colidx.data() + row_begin;
                pack_idx_t       ks[4];
                cvfem_find_cols4(ev, row, lenrow, ks);
                for (int b = 0; b < 4; ++b) {
                    slots[a * 4 + b] = row_begin + (int)ks[b];
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
    p.ghost_mat_col.resize((size_t)gnnz);
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
            const ptrdiff_t    dest = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
            const smesh::idx_t grow = p.ghost_idx[(size_t)ghost_off + (size_t)k];
            for (int t = 0; t < end - begin; ++t) {
                const smesh::idx_t gcol                    = pack_local_to_global(p, pack, n_contiguous, colidx[(size_t)begin + t]);
                p.ghost_mat_col[(size_t)dest + (size_t)t]  = gcol;
                p.ghost_mat_slot[(size_t)dest + (size_t)t] = bsr4_find_slot(rowptr_g, colidx_g, grow, gcol);
            }
        }
    }
}

struct BSR4 {
    std::shared_ptr<smesh::Mesh::NodeToNodeGraph> graph;
    const smesh::count_t                         *rowptr{nullptr};
    const smesh::idx_t                           *colidx{nullptr};
    smesh::SharedBuffer<scalar_t>                 values;
    ptrdiff_t                                     nnz{0};
};

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

static SFEM_INLINE size_t packed_scratch_n(const PackedData &p) {
    const ptrdiff_t n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
    return (size_t)N_FIELDS * (size_t)n;
}

static SFEM_INLINE void atomic_add(scalar_t *const SFEM_RESTRICT f, const ptrdiff_t id, const scalar_t value) {
#pragma omp atomic update
    f[id] += value;
}

static SFEM_INLINE void gather_tet4_pack_global(const MeshData &d, const ptrdiff_t begin, const int nlanes, Tet4InputPack &pack) {
    smesh::idx_t **const SFEM_RESTRICT  elems = d.elems;
    const scalar_t *const SFEM_RESTRICT ux    = d.ux.data();
    const scalar_t *const SFEM_RESTRICT uy    = d.uy.data();
    const scalar_t *const SFEM_RESTRICT uz    = d.uz.data();
    const scalar_t *const SFEM_RESTRICT p     = d.p.data();

    const int last_active_lane = nlanes - 1;

    for (int lane = 0; lane < VEC_SIZE; ++lane) {
        const int          active_lane = lane < nlanes ? lane : last_active_lane;
        const ptrdiff_t    e           = begin + active_lane;
        const smesh::idx_t n0          = elems[0][e];
        const smesh::idx_t n1          = elems[1][e];
        const smesh::idx_t n2          = elems[2][e];
        const smesh::idx_t n3          = elems[3][e];

        pack.ux[0][lane] = ux[n0];
        pack.ux[1][lane] = ux[n1];
        pack.ux[2][lane] = ux[n2];
        pack.ux[3][lane] = ux[n3];
        pack.uy[0][lane] = uy[n0];
        pack.uy[1][lane] = uy[n1];
        pack.uy[2][lane] = uy[n2];
        pack.uy[3][lane] = uy[n3];
        pack.uz[0][lane] = uz[n0];
        pack.uz[1][lane] = uz[n1];
        pack.uz[2][lane] = uz[n2];
        pack.uz[3][lane] = uz[n3];
        pack.p[0][lane]  = p[n0];
        pack.p[1][lane]  = p[n1];
        pack.p[2][lane]  = p[n2];
        pack.p[3][lane]  = p[n3];
    }
}

static SFEM_INLINE void gather_tet4_pack_local(pack_idx_t **const SFEM_RESTRICT    elems,
                                               const scalar_t *const SFEM_RESTRICT pack_u,
                                               const ptrdiff_t                     begin,
                                               const int                           nlanes,
                                               Tet4InputPack                      &pack) {
    const pack_idx_t *const SFEM_RESTRICT e0 = elems[0] + begin;
    const pack_idx_t *const SFEM_RESTRICT e1 = elems[1] + begin;
    const pack_idx_t *const SFEM_RESTRICT e2 = elems[2] + begin;
    const pack_idx_t *const SFEM_RESTRICT e3 = elems[3] + begin;

#pragma omp simd
    for (int lane = 0; lane < nlanes; ++lane) {
        const scalar_t *const SFEM_RESTRICT u0 = pack_u + (ptrdiff_t)e0[lane] * N_FIELDS;
        const scalar_t *const SFEM_RESTRICT u1 = pack_u + (ptrdiff_t)e1[lane] * N_FIELDS;
        const scalar_t *const SFEM_RESTRICT u2 = pack_u + (ptrdiff_t)e2[lane] * N_FIELDS;
        const scalar_t *const SFEM_RESTRICT u3 = pack_u + (ptrdiff_t)e3[lane] * N_FIELDS;
        pack.ux[0][lane]                       = u0[0];
        pack.uy[0][lane]                       = u0[1];
        pack.uz[0][lane]                       = u0[2];
        pack.p[0][lane]                        = u0[3];
        pack.ux[1][lane]                       = u1[0];
        pack.uy[1][lane]                       = u1[1];
        pack.uz[1][lane]                       = u1[2];
        pack.p[1][lane]                        = u1[3];
        pack.ux[2][lane]                       = u2[0];
        pack.uy[2][lane]                       = u2[1];
        pack.uz[2][lane]                       = u2[2];
        pack.p[2][lane]                        = u2[3];
        pack.ux[3][lane]                       = u3[0];
        pack.uy[3][lane]                       = u3[1];
        pack.uz[3][lane]                       = u3[2];
        pack.p[3][lane]                        = u3[3];
    }

    if (nlanes < VEC_SIZE) {
        const int last = nlanes - 1;
        for (int lane = nlanes; lane < VEC_SIZE; ++lane) {
            for (int a = 0; a < 4; ++a) {
                pack.ux[a][lane] = pack.ux[a][last];
                pack.uy[a][lane] = pack.uy[a][last];
                pack.uz[a][lane] = pack.uz[a][last];
                pack.p[a][lane]  = pack.p[a][last];
            }
        }
    }
}

static SFEM_INLINE void gather_tet4_action_pack_local(pack_idx_t **const SFEM_RESTRICT    elems,
                                                      const scalar_t *const SFEM_RESTRICT pack_velocity,
                                                      const scalar_t *const SFEM_RESTRICT pack_dir,
                                                      const ptrdiff_t                     begin,
                                                      const int                           nlanes,
                                                      Tet4InputPack                      &u_pack,
                                                      Tet4InputPack                      &du_pack) {
    const pack_idx_t *const SFEM_RESTRICT e0 = elems[0] + begin;
    const pack_idx_t *const SFEM_RESTRICT e1 = elems[1] + begin;
    const pack_idx_t *const SFEM_RESTRICT e2 = elems[2] + begin;
    const pack_idx_t *const SFEM_RESTRICT e3 = elems[3] + begin;

#pragma omp simd
    for (int lane = 0; lane < nlanes; ++lane) {
        const scalar_t *const SFEM_RESTRICT u0 = pack_velocity + (ptrdiff_t)e0[lane] * N_ACTION_BASE_FIELDS;
        const scalar_t *const SFEM_RESTRICT u1 = pack_velocity + (ptrdiff_t)e1[lane] * N_ACTION_BASE_FIELDS;
        const scalar_t *const SFEM_RESTRICT u2 = pack_velocity + (ptrdiff_t)e2[lane] * N_ACTION_BASE_FIELDS;
        const scalar_t *const SFEM_RESTRICT u3 = pack_velocity + (ptrdiff_t)e3[lane] * N_ACTION_BASE_FIELDS;
        const scalar_t *const SFEM_RESTRICT d0 = pack_dir + (ptrdiff_t)e0[lane] * N_FIELDS;
        const scalar_t *const SFEM_RESTRICT d1 = pack_dir + (ptrdiff_t)e1[lane] * N_FIELDS;
        const scalar_t *const SFEM_RESTRICT d2 = pack_dir + (ptrdiff_t)e2[lane] * N_FIELDS;
        const scalar_t *const SFEM_RESTRICT d3 = pack_dir + (ptrdiff_t)e3[lane] * N_FIELDS;
        u_pack.ux[0][lane]                     = u0[0];
        u_pack.uy[0][lane]                     = u0[1];
        u_pack.uz[0][lane]                     = u0[2];
        u_pack.ux[1][lane]                     = u1[0];
        u_pack.uy[1][lane]                     = u1[1];
        u_pack.uz[1][lane]                     = u1[2];
        u_pack.ux[2][lane]                     = u2[0];
        u_pack.uy[2][lane]                     = u2[1];
        u_pack.uz[2][lane]                     = u2[2];
        u_pack.ux[3][lane]                     = u3[0];
        u_pack.uy[3][lane]                     = u3[1];
        u_pack.uz[3][lane]                     = u3[2];
        du_pack.ux[0][lane]                    = d0[0];
        du_pack.uy[0][lane]                    = d0[1];
        du_pack.uz[0][lane]                    = d0[2];
        du_pack.p[0][lane]                     = d0[3];
        du_pack.ux[1][lane]                    = d1[0];
        du_pack.uy[1][lane]                    = d1[1];
        du_pack.uz[1][lane]                    = d1[2];
        du_pack.p[1][lane]                     = d1[3];
        du_pack.ux[2][lane]                    = d2[0];
        du_pack.uy[2][lane]                    = d2[1];
        du_pack.uz[2][lane]                    = d2[2];
        du_pack.p[2][lane]                     = d2[3];
        du_pack.ux[3][lane]                    = d3[0];
        du_pack.uy[3][lane]                    = d3[1];
        du_pack.uz[3][lane]                    = d3[2];
        du_pack.p[3][lane]                     = d3[3];
    }

    if (nlanes < VEC_SIZE) {
        const int last = nlanes - 1;
        for (int lane = nlanes; lane < VEC_SIZE; ++lane) {
            for (int a = 0; a < 4; ++a) {
                u_pack.ux[a][lane]  = u_pack.ux[a][last];
                u_pack.uy[a][lane]  = u_pack.uy[a][last];
                u_pack.uz[a][lane]  = u_pack.uz[a][last];
                du_pack.ux[a][lane] = du_pack.ux[a][last];
                du_pack.uy[a][lane] = du_pack.uy[a][last];
                du_pack.uz[a][lane] = du_pack.uz[a][last];
                du_pack.p[a][lane]  = du_pack.p[a][last];
            }
        }
    }
}

static SFEM_INLINE void gather_tet4_action_pack_global(const MeshData &d,
                                                       const scalar_t *const SFEM_RESTRICT dir,
                                                       const ptrdiff_t begin,
                                                       const int       nlanes,
                                                       Tet4InputPack  &u_pack,
                                                       Tet4InputPack  &du_pack) {
    smesh::idx_t **const SFEM_RESTRICT  elems = d.elems;
    const scalar_t *const SFEM_RESTRICT ux    = d.ux.data();
    const scalar_t *const SFEM_RESTRICT uy    = d.uy.data();
    const scalar_t *const SFEM_RESTRICT uz    = d.uz.data();

    const int last_active_lane = nlanes - 1;

    for (int lane = 0; lane < VEC_SIZE; ++lane) {
        const int          active_lane = lane < nlanes ? lane : last_active_lane;
        const ptrdiff_t    e           = begin + active_lane;
        const smesh::idx_t n0          = elems[0][e];
        const smesh::idx_t n1          = elems[1][e];
        const smesh::idx_t n2          = elems[2][e];
        const smesh::idx_t n3          = elems[3][e];
        const scalar_t *const SFEM_RESTRICT d0 = dir + (ptrdiff_t)n0 * N_FIELDS;
        const scalar_t *const SFEM_RESTRICT d1 = dir + (ptrdiff_t)n1 * N_FIELDS;
        const scalar_t *const SFEM_RESTRICT d2 = dir + (ptrdiff_t)n2 * N_FIELDS;
        const scalar_t *const SFEM_RESTRICT d3 = dir + (ptrdiff_t)n3 * N_FIELDS;

        u_pack.ux[0][lane]  = ux[n0];
        u_pack.uy[0][lane]  = uy[n0];
        u_pack.uz[0][lane]  = uz[n0];
        u_pack.ux[1][lane]  = ux[n1];
        u_pack.uy[1][lane]  = uy[n1];
        u_pack.uz[1][lane]  = uz[n1];
        u_pack.ux[2][lane]  = ux[n2];
        u_pack.uy[2][lane]  = uy[n2];
        u_pack.uz[2][lane]  = uz[n2];
        u_pack.ux[3][lane]  = ux[n3];
        u_pack.uy[3][lane]  = uy[n3];
        u_pack.uz[3][lane]  = uz[n3];
        du_pack.ux[0][lane] = d0[0];
        du_pack.uy[0][lane] = d0[1];
        du_pack.uz[0][lane] = d0[2];
        du_pack.p[0][lane]  = d0[3];
        du_pack.ux[1][lane] = d1[0];
        du_pack.uy[1][lane] = d1[1];
        du_pack.uz[1][lane] = d1[2];
        du_pack.p[1][lane]  = d1[3];
        du_pack.ux[2][lane] = d2[0];
        du_pack.uy[2][lane] = d2[1];
        du_pack.uz[2][lane] = d2[2];
        du_pack.p[2][lane]  = d2[3];
        du_pack.ux[3][lane] = d3[0];
        du_pack.uy[3][lane] = d3[1];
        du_pack.uz[3][lane] = d3[2];
        du_pack.p[3][lane]  = d3[3];
    }
}

static SFEM_INLINE void run_microkernel(const MeshData      &d,
                                        const scalar_t       rho,
                                        const scalar_t       mu,
                                        const ptrdiff_t      begin,
                                        const int            nlanes,
                                        const Tet4InputPack &in,
                                        Tet4ResidualPack    &out) {
    cvfem_run_residual_kernel(rho,
                              mu,
                              d.adj[0].data() + begin,
                              d.adj[1].data() + begin,
                              d.adj[2].data() + begin,
                              d.adj[3].data() + begin,
                              d.adj[4].data() + begin,
                              d.adj[5].data() + begin,
                              d.adj[6].data() + begin,
                              d.adj[7].data() + begin,
                              d.adj[8].data() + begin,
                              d.det.data() + begin,
                              nlanes,
                              in,
                              out);
}

static SFEM_INLINE void run_microkernel_sympy(const MeshData      &d,
                                              const scalar_t       rho,
                                              const scalar_t       mu,
                                              const ptrdiff_t      begin,
                                              const int            nlanes,
                                              const Tet4InputPack &in,
                                              Tet4ResidualPack    &out) {
    cvfem_run_residual_sympy_kernel(rho,
                                    mu,
                                    d.adj[0].data() + begin,
                                    d.adj[1].data() + begin,
                                    d.adj[2].data() + begin,
                                    d.adj[3].data() + begin,
                                    d.adj[4].data() + begin,
                                    d.adj[5].data() + begin,
                                    d.adj[6].data() + begin,
                                    d.adj[7].data() + begin,
                                    d.adj[8].data() + begin,
                                    d.det.data() + begin,
                                    nlanes,
                                    in,
                                    out);
}

static SFEM_INLINE void run_jacobian_action_microkernel(const MeshData      &d,
                                                        const scalar_t       rho,
                                                        const scalar_t       mu,
                                                        const ptrdiff_t      begin,
                                                        const int            nlanes,
                                                        const Tet4InputPack &u,
                                                        const Tet4InputPack &du,
                                                        Tet4ResidualPack    &out) {
    cvfem_run_jacobian_action_kernel(rho,
                                     mu,
                                     d.adj[0].data() + begin,
                                     d.adj[1].data() + begin,
                                     d.adj[2].data() + begin,
                                     d.adj[3].data() + begin,
                                     d.adj[4].data() + begin,
                                     d.adj[5].data() + begin,
                                     d.adj[6].data() + begin,
                                     d.adj[7].data() + begin,
                                     d.adj[8].data() + begin,
                                     d.det.data() + begin,
                                     nlanes,
                                     u,
                                     du,
                                     out);
}

static SFEM_INLINE void run_jacobian_action_microkernel_sympy(const MeshData      &d,
                                                              const scalar_t       rho,
                                                              const scalar_t       mu,
                                                              const ptrdiff_t      begin,
                                                              const int            nlanes,
                                                              const Tet4InputPack &u,
                                                              const Tet4InputPack &du,
                                                              Tet4ResidualPack    &out) {
    cvfem_run_jacobian_action_sympy_kernel(rho,
                                           mu,
                                           d.adj[0].data() + begin,
                                           d.adj[1].data() + begin,
                                           d.adj[2].data() + begin,
                                           d.adj[3].data() + begin,
                                           d.adj[4].data() + begin,
                                           d.adj[5].data() + begin,
                                           d.adj[6].data() + begin,
                                           d.adj[7].data() + begin,
                                           d.adj[8].data() + begin,
                                           d.det.data() + begin,
                                           nlanes,
                                           u,
                                           du,
                                           out);
}

static SFEM_INLINE void jacobian_element_global(const MeshData                 &d,
                                                const ptrdiff_t                e,
                                                const smesh::idx_t *const      ev,
                                                const scalar_t                 rho,
                                                const scalar_t                 mu,
                                                scalar_t *const SFEM_RESTRICT  Ke) {
    const scalar_t ux[4] = {d.ux[ev[0]], d.ux[ev[1]], d.ux[ev[2]], d.ux[ev[3]]};
    const scalar_t uy[4] = {d.uy[ev[0]], d.uy[ev[1]], d.uy[ev[2]], d.uy[ev[3]]};
    const scalar_t uz[4] = {d.uz[ev[0]], d.uz[ev[1]], d.uz[ev[2]], d.uz[ev[3]]};
    cvfem_tet4_ns_upwind_jacobian_dense(rho,
                                        mu,
                                        scalar_t(d.adj[0][e]),
                                        scalar_t(d.adj[1][e]),
                                        scalar_t(d.adj[2][e]),
                                        scalar_t(d.adj[3][e]),
                                        scalar_t(d.adj[4][e]),
                                        scalar_t(d.adj[5][e]),
                                        scalar_t(d.adj[6][e]),
                                        scalar_t(d.adj[7][e]),
                                        scalar_t(d.adj[8][e]),
                                        scalar_t(d.det[e]),
                                        ux,
                                        uy,
                                        uz,
                                        Ke);
}

static SFEM_INLINE void jacobian_element_global_sympy(const MeshData                &d,
                                                      const ptrdiff_t               e,
                                                      const smesh::idx_t *const     ev,
                                                      const scalar_t                rho,
                                                      const scalar_t                mu,
                                                      scalar_t *const SFEM_RESTRICT Ke) {
    const scalar_t ux[4] = {d.ux[ev[0]], d.ux[ev[1]], d.ux[ev[2]], d.ux[ev[3]]};
    const scalar_t uy[4] = {d.uy[ev[0]], d.uy[ev[1]], d.uy[ev[2]], d.uy[ev[3]]};
    const scalar_t uz[4] = {d.uz[ev[0]], d.uz[ev[1]], d.uz[ev[2]], d.uz[ev[3]]};
    cvfem_tet4_ns_upwind_sympy_jacobian_dense(rho,
                                              mu,
                                              scalar_t(d.adj[0][e]),
                                              scalar_t(d.adj[1][e]),
                                              scalar_t(d.adj[2][e]),
                                              scalar_t(d.adj[3][e]),
                                              scalar_t(d.adj[4][e]),
                                              scalar_t(d.adj[5][e]),
                                              scalar_t(d.adj[6][e]),
                                              scalar_t(d.adj[7][e]),
                                              scalar_t(d.adj[8][e]),
                                              scalar_t(d.det[e]),
                                              ux,
                                              uy,
                                              uz,
                                              Ke);
}

static SFEM_INLINE void jacobian_element_packed(const MeshData                 &d,
                                                const ptrdiff_t                e,
                                                const pack_idx_t *const        ev,
                                                const scalar_t *const          pack_u,
                                                const scalar_t                 rho,
                                                const scalar_t                 mu,
                                                scalar_t *const SFEM_RESTRICT  Ke) {
    const scalar_t *const u0 = pack_u + (ptrdiff_t)ev[0] * N_FIELDS;
    const scalar_t *const u1 = pack_u + (ptrdiff_t)ev[1] * N_FIELDS;
    const scalar_t *const u2 = pack_u + (ptrdiff_t)ev[2] * N_FIELDS;
    const scalar_t *const u3 = pack_u + (ptrdiff_t)ev[3] * N_FIELDS;
    const scalar_t        ux[4] = {u0[0], u1[0], u2[0], u3[0]};
    const scalar_t        uy[4] = {u0[1], u1[1], u2[1], u3[1]};
    const scalar_t        uz[4] = {u0[2], u1[2], u2[2], u3[2]};
    cvfem_tet4_ns_upwind_jacobian_dense(rho,
                                        mu,
                                        scalar_t(d.adj[0][e]),
                                        scalar_t(d.adj[1][e]),
                                        scalar_t(d.adj[2][e]),
                                        scalar_t(d.adj[3][e]),
                                        scalar_t(d.adj[4][e]),
                                        scalar_t(d.adj[5][e]),
                                        scalar_t(d.adj[6][e]),
                                        scalar_t(d.adj[7][e]),
                                        scalar_t(d.adj[8][e]),
                                        scalar_t(d.det[e]),
                                        ux,
                                        uy,
                                        uz,
                                        Ke);
}

static SFEM_INLINE void jacobian_element_packed_sympy(const MeshData                &d,
                                                      const ptrdiff_t               e,
                                                      const pack_idx_t *const       ev,
                                                      const scalar_t *const         pack_u,
                                                      const scalar_t                rho,
                                                      const scalar_t                mu,
                                                      scalar_t *const SFEM_RESTRICT Ke) {
    const scalar_t *const u0 = pack_u + (ptrdiff_t)ev[0] * N_FIELDS;
    const scalar_t *const u1 = pack_u + (ptrdiff_t)ev[1] * N_FIELDS;
    const scalar_t *const u2 = pack_u + (ptrdiff_t)ev[2] * N_FIELDS;
    const scalar_t *const u3 = pack_u + (ptrdiff_t)ev[3] * N_FIELDS;
    const scalar_t        ux[4] = {u0[0], u1[0], u2[0], u3[0]};
    const scalar_t        uy[4] = {u0[1], u1[1], u2[1], u3[1]};
    const scalar_t        uz[4] = {u0[2], u1[2], u2[2], u3[2]};
    cvfem_tet4_ns_upwind_sympy_jacobian_dense(rho,
                                              mu,
                                              scalar_t(d.adj[0][e]),
                                              scalar_t(d.adj[1][e]),
                                              scalar_t(d.adj[2][e]),
                                              scalar_t(d.adj[3][e]),
                                              scalar_t(d.adj[4][e]),
                                              scalar_t(d.adj[5][e]),
                                              scalar_t(d.adj[6][e]),
                                              scalar_t(d.adj[7][e]),
                                              scalar_t(d.adj[8][e]),
                                              scalar_t(d.det[e]),
                                              ux,
                                              uy,
                                              uz,
                                              Ke);
}

static SFEM_INLINE void jacobian_element_packed_sympy_add_slots(const MeshData                &d,
                                                                const ptrdiff_t               e,
                                                                const pack_idx_t *const       ev,
                                                                const scalar_t *const         pack_u,
                                                                const scalar_t                rho,
                                                                const scalar_t                mu,
                                                                const int *const SFEM_RESTRICT slots,
                                                                scalar_t *const SFEM_RESTRICT values) {
    const scalar_t *const u0 = pack_u + (ptrdiff_t)ev[0] * N_FIELDS;
    const scalar_t *const u1 = pack_u + (ptrdiff_t)ev[1] * N_FIELDS;
    const scalar_t *const u2 = pack_u + (ptrdiff_t)ev[2] * N_FIELDS;
    const scalar_t *const u3 = pack_u + (ptrdiff_t)ev[3] * N_FIELDS;
    const scalar_t        ux[4] = {u0[0], u1[0], u2[0], u3[0]};
    const scalar_t        uy[4] = {u0[1], u1[1], u2[1], u3[1]};
    const scalar_t        uz[4] = {u0[2], u1[2], u2[2], u3[2]};
    cvfem_tet4_ns_upwind_sympy_jacobian_add_bsr_slots(rho,
                                                      mu,
                                                      scalar_t(d.adj[0][e]),
                                                      scalar_t(d.adj[1][e]),
                                                      scalar_t(d.adj[2][e]),
                                                      scalar_t(d.adj[3][e]),
                                                      scalar_t(d.adj[4][e]),
                                                      scalar_t(d.adj[5][e]),
                                                      scalar_t(d.adj[6][e]),
                                                      scalar_t(d.adj[7][e]),
                                                      scalar_t(d.adj[8][e]),
                                                      scalar_t(d.det[e]),
                                                      ux,
                                                      uy,
                                                      uz,
                                                      slots,
                                                      values);
}

static SFEM_INLINE void jacobian_element_packed_sympy_add_slots_blockwise(const MeshData                &d,
                                                                          const ptrdiff_t               e,
                                                                          const pack_idx_t *const       ev,
                                                                          const scalar_t *const         pack_u,
                                                                          const scalar_t                rho,
                                                                          const scalar_t                mu,
                                                                          const int *const SFEM_RESTRICT slots,
                                                                          scalar_t *const SFEM_RESTRICT values) {
    const scalar_t *const u0 = pack_u + (ptrdiff_t)ev[0] * N_FIELDS;
    const scalar_t *const u1 = pack_u + (ptrdiff_t)ev[1] * N_FIELDS;
    const scalar_t *const u2 = pack_u + (ptrdiff_t)ev[2] * N_FIELDS;
    const scalar_t *const u3 = pack_u + (ptrdiff_t)ev[3] * N_FIELDS;
    const scalar_t        ux[4] = {u0[0], u1[0], u2[0], u3[0]};
    const scalar_t        uy[4] = {u0[1], u1[1], u2[1], u3[1]};
    const scalar_t        uz[4] = {u0[2], u1[2], u2[2], u3[2]};
    cvfem_tet4_ns_upwind_sympy_jacobian_add_bsr_slots_blockwise(rho,
                                                                mu,
                                                                scalar_t(d.adj[0][e]),
                                                                scalar_t(d.adj[1][e]),
                                                                scalar_t(d.adj[2][e]),
                                                                scalar_t(d.adj[3][e]),
                                                                scalar_t(d.adj[4][e]),
                                                                scalar_t(d.adj[5][e]),
                                                                scalar_t(d.adj[6][e]),
                                                                scalar_t(d.adj[7][e]),
                                                                scalar_t(d.adj[8][e]),
                                                                scalar_t(d.det[e]),
                                                                ux,
                                                                uy,
                                                                uz,
                                                                slots,
                                                                values);
}

static SFEM_INLINE void jacobian_element_packed_sympy_add_slots_facewise(const MeshData                &d,
                                                                         const ptrdiff_t               e,
                                                                         const pack_idx_t *const       ev,
                                                                         const scalar_t *const         pack_u,
                                                                         const scalar_t                rho,
                                                                         const scalar_t                mu,
                                                                         const int *const SFEM_RESTRICT slots,
                                                                         scalar_t *const SFEM_RESTRICT values) {
    const scalar_t *const u0 = pack_u + (ptrdiff_t)ev[0] * N_FIELDS;
    const scalar_t *const u1 = pack_u + (ptrdiff_t)ev[1] * N_FIELDS;
    const scalar_t *const u2 = pack_u + (ptrdiff_t)ev[2] * N_FIELDS;
    const scalar_t *const u3 = pack_u + (ptrdiff_t)ev[3] * N_FIELDS;
    const scalar_t        ux[4] = {u0[0], u1[0], u2[0], u3[0]};
    const scalar_t        uy[4] = {u0[1], u1[1], u2[1], u3[1]};
    const scalar_t        uz[4] = {u0[2], u1[2], u2[2], u3[2]};
    cvfem_tet4_ns_upwind_sympy_jacobian_add_bsr_slots_facewise(rho,
                                                               mu,
                                                               scalar_t(d.adj[0][e]),
                                                               scalar_t(d.adj[1][e]),
                                                               scalar_t(d.adj[2][e]),
                                                               scalar_t(d.adj[3][e]),
                                                               scalar_t(d.adj[4][e]),
                                                               scalar_t(d.adj[5][e]),
                                                               scalar_t(d.adj[6][e]),
                                                               scalar_t(d.adj[7][e]),
                                                               scalar_t(d.adj[8][e]),
                                                               scalar_t(d.det[e]),
                                                               ux,
                                                               uy,
                                                               uz,
                                                               slots,
                                                               values);
}

static SFEM_INLINE void scatter_tet4_pack_global(MeshData               &d,
                                                 const ptrdiff_t         begin,
                                                 const int               nlanes,
                                                 const Tet4ResidualPack &pack) {
    smesh::idx_t **const SFEM_RESTRICT elems  = d.elems;
    scalar_t *const SFEM_RESTRICT      rx_ptr = d.rx.data();
    scalar_t *const SFEM_RESTRICT      ry_ptr = d.ry.data();
    scalar_t *const SFEM_RESTRICT      rz_ptr = d.rz.data();
    scalar_t *const SFEM_RESTRICT      rc_ptr = d.rc.data();

    for (int lane = 0; lane < nlanes; ++lane) {
        const ptrdiff_t    e  = begin + lane;
        const smesh::idx_t n0 = elems[0][e];
        const smesh::idx_t n1 = elems[1][e];
        const smesh::idx_t n2 = elems[2][e];
        const smesh::idx_t n3 = elems[3][e];

        atomic_add(rx_ptr, n0, pack.rx[0][lane]);
        atomic_add(rx_ptr, n1, pack.rx[1][lane]);
        atomic_add(rx_ptr, n2, pack.rx[2][lane]);
        atomic_add(rx_ptr, n3, pack.rx[3][lane]);
        atomic_add(ry_ptr, n0, pack.ry[0][lane]);
        atomic_add(ry_ptr, n1, pack.ry[1][lane]);
        atomic_add(ry_ptr, n2, pack.ry[2][lane]);
        atomic_add(ry_ptr, n3, pack.ry[3][lane]);
        atomic_add(rz_ptr, n0, pack.rz[0][lane]);
        atomic_add(rz_ptr, n1, pack.rz[1][lane]);
        atomic_add(rz_ptr, n2, pack.rz[2][lane]);
        atomic_add(rz_ptr, n3, pack.rz[3][lane]);
        atomic_add(rc_ptr, n0, pack.rc[0][lane]);
        atomic_add(rc_ptr, n1, pack.rc[1][lane]);
        atomic_add(rc_ptr, n2, pack.rc[2][lane]);
        atomic_add(rc_ptr, n3, pack.rc[3][lane]);
    }
}

static SFEM_INLINE void scatter_tet4_action_pack_global(smesh::idx_t **const SFEM_RESTRICT elems,
                                                        scalar_t *const SFEM_RESTRICT      jv,
                                                        const ptrdiff_t                    begin,
                                                        const int                          nlanes,
                                                        const Tet4ResidualPack            &pack) {
    for (int lane = 0; lane < nlanes; ++lane) {
        const ptrdiff_t    e  = begin + lane;
        const smesh::idx_t n0 = elems[0][e];
        const smesh::idx_t n1 = elems[1][e];
        const smesh::idx_t n2 = elems[2][e];
        const smesh::idx_t n3 = elems[3][e];

        atomic_add(jv, (ptrdiff_t)n0 * N_FIELDS + 0, pack.rx[0][lane]);
        atomic_add(jv, (ptrdiff_t)n0 * N_FIELDS + 1, pack.ry[0][lane]);
        atomic_add(jv, (ptrdiff_t)n0 * N_FIELDS + 2, pack.rz[0][lane]);
        atomic_add(jv, (ptrdiff_t)n0 * N_FIELDS + 3, pack.rc[0][lane]);
        atomic_add(jv, (ptrdiff_t)n1 * N_FIELDS + 0, pack.rx[1][lane]);
        atomic_add(jv, (ptrdiff_t)n1 * N_FIELDS + 1, pack.ry[1][lane]);
        atomic_add(jv, (ptrdiff_t)n1 * N_FIELDS + 2, pack.rz[1][lane]);
        atomic_add(jv, (ptrdiff_t)n1 * N_FIELDS + 3, pack.rc[1][lane]);
        atomic_add(jv, (ptrdiff_t)n2 * N_FIELDS + 0, pack.rx[2][lane]);
        atomic_add(jv, (ptrdiff_t)n2 * N_FIELDS + 1, pack.ry[2][lane]);
        atomic_add(jv, (ptrdiff_t)n2 * N_FIELDS + 2, pack.rz[2][lane]);
        atomic_add(jv, (ptrdiff_t)n2 * N_FIELDS + 3, pack.rc[2][lane]);
        atomic_add(jv, (ptrdiff_t)n3 * N_FIELDS + 0, pack.rx[3][lane]);
        atomic_add(jv, (ptrdiff_t)n3 * N_FIELDS + 1, pack.ry[3][lane]);
        atomic_add(jv, (ptrdiff_t)n3 * N_FIELDS + 2, pack.rz[3][lane]);
        atomic_add(jv, (ptrdiff_t)n3 * N_FIELDS + 3, pack.rc[3][lane]);
    }
}

static SFEM_INLINE void scatter_tet4_pack_local(pack_idx_t **const SFEM_RESTRICT elems,
                                                scalar_t *const SFEM_RESTRICT    pack_out,
                                                const ptrdiff_t                  begin,
                                                const int                        nlanes,
                                                const Tet4ResidualPack          &pack) {
    const pack_idx_t *const SFEM_RESTRICT e0 = elems[0] + begin;
    const pack_idx_t *const SFEM_RESTRICT e1 = elems[1] + begin;
    const pack_idx_t *const SFEM_RESTRICT e2 = elems[2] + begin;
    const pack_idx_t *const SFEM_RESTRICT e3 = elems[3] + begin;

    for (int lane = 0; lane < nlanes; ++lane) {
        scalar_t *const SFEM_RESTRICT r0 = pack_out + (ptrdiff_t)e0[lane] * N_FIELDS;
        scalar_t *const SFEM_RESTRICT r1 = pack_out + (ptrdiff_t)e1[lane] * N_FIELDS;
        scalar_t *const SFEM_RESTRICT r2 = pack_out + (ptrdiff_t)e2[lane] * N_FIELDS;
        scalar_t *const SFEM_RESTRICT r3 = pack_out + (ptrdiff_t)e3[lane] * N_FIELDS;
        r0[0] += pack.rx[0][lane];
        r0[1] += pack.ry[0][lane];
        r0[2] += pack.rz[0][lane];
        r0[3] += pack.rc[0][lane];
        r1[0] += pack.rx[1][lane];
        r1[1] += pack.ry[1][lane];
        r1[2] += pack.rz[1][lane];
        r1[3] += pack.rc[1][lane];
        r2[0] += pack.rx[2][lane];
        r2[1] += pack.ry[2][lane];
        r2[2] += pack.rz[2][lane];
        r2[3] += pack.rc[2][lane];
        r3[0] += pack.rx[3][lane];
        r3[1] += pack.ry[3][lane];
        r3[2] += pack.rz[3][lane];
        r3[3] += pack.rc[3][lane];
    }
}

static SFEM_NOINLINE void cvfem_tet4_ns_upwind_apply_atomic(MeshData &d, const scalar_t rho, const scalar_t mu) {
    reset_residual(d);
    const ptrdiff_t ne = d.nelements;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t begin = 0; begin < ne; begin += VEC_SIZE) {
        const int        nlanes = int(std::min<ptrdiff_t>(ne - begin, VEC_SIZE));
        Tet4InputPack    in;
        Tet4ResidualPack out;
        gather_tet4_pack_global(d, begin, nlanes, in);
        run_microkernel(d, rho, mu, begin, nlanes, in, out);
        scatter_tet4_pack_global(d, begin, nlanes, out);
    }
}

static SFEM_NOINLINE void cvfem_tet4_ns_upwind_jacobian_action_atomic(const MeshData          &d,
                                                                      const scalar_t          rho,
                                                                      const scalar_t          mu,
                                                                      const scalar_t *const   dir,
                                                                      scalar_t *const         jv,
                                                                      const bool              use_sympy_action) {
    const ptrdiff_t ndof = d.nnodes * N_FIELDS;
    cvfem_zero_scalars(jv, ndof);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t begin = 0; begin < d.nelements; begin += VEC_SIZE) {
        const int        nlanes = int(std::min<ptrdiff_t>(d.nelements - begin, VEC_SIZE));
        Tet4InputPack    u_pack;
        Tet4InputPack    du_pack;
        Tet4ResidualPack out;
        gather_tet4_action_pack_global(d, dir, begin, nlanes, u_pack, du_pack);
        if (use_sympy_action)
            run_jacobian_action_microkernel_sympy(d, rho, mu, begin, nlanes, u_pack, du_pack, out);
        else
            run_jacobian_action_microkernel(d, rho, mu, begin, nlanes, u_pack, du_pack, out);
        scatter_tet4_action_pack_global(d.elems, jv, begin, nlanes, out);
    }
}

static SFEM_NOINLINE void cvfem_tet4_ns_upwind_apply_packed(MeshData &d, PackedData &p, const scalar_t rho, const scalar_t mu) {
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

            for (ptrdiff_t begin = e_start; begin < e_end; begin += VEC_SIZE) {
                const int        nlanes = int(MIN((ptrdiff_t)VEC_SIZE, e_end - begin));
                Tet4InputPack    in;
                Tet4ResidualPack out;
                gather_tet4_pack_local(p.elems, pack_u, begin, nlanes, in);
                run_microkernel(d, rho, mu, begin, nlanes, in, out);
                scatter_tet4_pack_local(p.elems, pack_out, begin, nlanes, out);
            }

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t *const SFEM_RESTRICT po = pack_out + k * N_FIELDS;
                const ptrdiff_t                     g  = owned + k;
                rx[g]                                  = po[0];
                ry[g]                                  = po[1];
                rz[g]                                  = po[2];
                rc[g]                                  = po[3];
            }

            scalar_t *const SFEM_RESTRICT gx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = p.ghost_buf.data() + 3 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT po = pack_out + (n_contiguous + k) * N_FIELDS;
                gx[ghost_off + k]                      = po[0];
                gy[ghost_off + k]                      = po[1];
                gz[ghost_off + k]                      = po[2];
                gc[ghost_off + k]                      = po[3];
            }
        }
    }

    scalar_t *const out_fields[N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + f * p.n_ghost_entries;
            scalar_t                            sum   = 0.0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            out_fields[f][dest] += sum;
        }
    }
}

static SFEM_NOINLINE void cvfem_tet4_ns_upwind_apply_sympy_atomic(MeshData &d, const scalar_t rho, const scalar_t mu) {
    reset_residual(d);
    const ptrdiff_t ne = d.nelements;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t begin = 0; begin < ne; begin += VEC_SIZE) {
        const int        nlanes = int(std::min<ptrdiff_t>(ne - begin, VEC_SIZE));
        Tet4InputPack    in;
        Tet4ResidualPack out;
        gather_tet4_pack_global(d, begin, nlanes, in);
        run_microkernel_sympy(d, rho, mu, begin, nlanes, in, out);
        scatter_tet4_pack_global(d, begin, nlanes, out);
    }
}

static SFEM_NOINLINE void cvfem_tet4_ns_upwind_apply_sympy_packed(MeshData &d,
                                                                  PackedData &p,
                                                                  const scalar_t rho,
                                                                  const scalar_t mu) {
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

            for (ptrdiff_t begin = e_start; begin < e_end; begin += VEC_SIZE) {
                const int        nlanes = int(MIN((ptrdiff_t)VEC_SIZE, e_end - begin));
                Tet4InputPack    in;
                Tet4ResidualPack out;
                gather_tet4_pack_local(p.elems, pack_u, begin, nlanes, in);
                run_microkernel_sympy(d, rho, mu, begin, nlanes, in, out);
                scatter_tet4_pack_local(p.elems, pack_out, begin, nlanes, out);
            }

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t *const SFEM_RESTRICT po = pack_out + k * N_FIELDS;
                const ptrdiff_t                     g  = owned + k;
                rx[g]                                  = po[0];
                ry[g]                                  = po[1];
                rz[g]                                  = po[2];
                rc[g]                                  = po[3];
            }

            scalar_t *const SFEM_RESTRICT gx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = p.ghost_buf.data() + 3 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT po = pack_out + (n_contiguous + k) * N_FIELDS;
                gx[ghost_off + k]                      = po[0];
                gy[ghost_off + k]                      = po[1];
                gz[ghost_off + k]                      = po[2];
                gc[ghost_off + k]                      = po[3];
            }
        }
    }

    scalar_t *const out_fields[N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + f * p.n_ghost_entries;
            scalar_t                            sum   = 0.0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            out_fields[f][dest] += sum;
        }
    }
}

static void cvfem_tet4_ns_upwind_prepack_action_base_velocity(MeshData &d, PackedData &p) {
    p.action_base_stride = std::max<ptrdiff_t>(p.max_actual_nodes_per_pack, 1);
    p.action_base_velocity.assign(
            (size_t)p.n_packs * (size_t)p.action_base_stride * (size_t)N_ACTION_BASE_FIELDS, scalar_t(0));

    const scalar_t *const SFEM_RESTRICT ux = d.ux.data();
    const scalar_t *const SFEM_RESTRICT uy = d.uy.data();
    const scalar_t *const SFEM_RESTRICT uz = d.uz.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        scalar_t *const SFEM_RESTRICT base =
                p.action_base_velocity.data() + (size_t)pack * (size_t)p.action_base_stride * (size_t)N_ACTION_BASE_FIELDS;
        const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
        const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
        const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];

        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
            scalar_t *const SFEM_RESTRICT dst = base + k * N_ACTION_BASE_FIELDS;
            const ptrdiff_t               g   = owned + k;
            dst[0]                            = ux[g];
            dst[1]                            = uy[g];
            dst[2]                            = uz[g];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
            scalar_t *const SFEM_RESTRICT dst = base + (n_contiguous + k) * N_ACTION_BASE_FIELDS;
            const smesh::idx_t            g   = ghosts[k];
            dst[0]                            = ux[g];
            dst[1]                            = uy[g];
            dst[2]                            = uz[g];
        }
    }
}

static SFEM_NOINLINE void cvfem_tet4_ns_upwind_jacobian_action_packed(MeshData              &d,
                                                                      PackedData            &p,
                                                                      const scalar_t         rho,
                                                                      const scalar_t         mu,
                                                                      const scalar_t *const  dir,
                                                                      scalar_t *const        jv,
                                                                      const bool             use_sympy_action) {
    const size_t                        scratch_n = packed_scratch_n(p);
    if (p.action_base_velocity.empty()) cvfem_tet4_ns_upwind_prepack_action_base_velocity(d, p);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);

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
            const scalar_t *const SFEM_RESTRICT      pack_u       =
                    p.action_base_velocity.data() + (size_t)pack * (size_t)p.action_base_stride * (size_t)N_ACTION_BASE_FIELDS;

            std::memset(pack_out, 0, (size_t)n_pack_nodes * (size_t)N_FIELDS * sizeof(scalar_t));

            std::memcpy(pack_dir, dir + owned * N_FIELDS, (size_t)n_contiguous * (size_t)N_FIELDS * sizeof(scalar_t));
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT       dst_d = pack_dir + (n_contiguous + k) * N_FIELDS;
                const smesh::idx_t                  g     = ghosts[k];
                const scalar_t *const SFEM_RESTRICT src_d = dir + (ptrdiff_t)g * N_FIELDS;
                dst_d[0]                                  = src_d[0];
                dst_d[1]                                  = src_d[1];
                dst_d[2]                                  = src_d[2];
                dst_d[3]                                  = src_d[3];
            }

            for (ptrdiff_t begin = e_start; begin < e_end; begin += VEC_SIZE) {
                const int        nlanes = int(MIN((ptrdiff_t)VEC_SIZE, e_end - begin));
                Tet4InputPack    u_pack;
                Tet4InputPack    du_pack;
                Tet4ResidualPack out;
                gather_tet4_action_pack_local(p.elems, pack_u, pack_dir, begin, nlanes, u_pack, du_pack);
                if (use_sympy_action)
                    run_jacobian_action_microkernel_sympy(d, rho, mu, begin, nlanes, u_pack, du_pack, out);
                else
                    run_jacobian_action_microkernel(d, rho, mu, begin, nlanes, u_pack, du_pack, out);
                scatter_tet4_pack_local(p.elems, pack_out, begin, nlanes, out);
            }

            std::memcpy(jv + owned * N_FIELDS, pack_out, (size_t)n_contiguous * (size_t)N_FIELDS * sizeof(scalar_t));

            scalar_t *const SFEM_RESTRICT gx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = p.ghost_buf.data() + 3 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT po = pack_out + (n_contiguous + k) * N_FIELDS;
                gx[ghost_off + k]                      = po[0];
                gy[ghost_off + k]                      = po[1];
                gz[ghost_off + k]                      = po[2];
                gc[ghost_off + k]                      = po[3];
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
            scalar_t                            sum   = 0.0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            out[f] += sum;
        }
    }
}


static SFEM_NOINLINE void assemble_bsr4_atomic(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    const ptrdiff_t                    ne     = d.nelements;
    smesh::idx_t **const SFEM_RESTRICT elems  = d.elems;
    scalar_t *const SFEM_RESTRICT      values = b.values->data();

#pragma omp parallel
    {
        alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < ne; ++e) {
            const smesh::idx_t ev[4] = {elems[0][e], elems[1][e], elems[2][e], elems[3][e]};
            jacobian_element_global(d, e, ev, rho, mu, Ke);
            tet4_local_to_global_bsr4<true>(ev, Ke, b.rowptr, b.colidx, values);
        }
    }
}

static SFEM_NOINLINE void assemble_bsr4_packed(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto                             &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto                             &lcolidx      = p.local_colidx[(size_t)pack];
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

            alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                jacobian_element_packed(d, e, ev, pack_u, rho, mu, Ke);
                tet4_local_to_global_bsr4<false>(ev, Ke, lrowptr.data(), lcolidx.data(), local_vals);
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

            const ptrdiff_t ghost_off = p.ghost_ptr[pack];
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const ptrdiff_t local_i = n_contiguous + k;
                const int       begin   = lrowptr[(size_t)local_i];
                const int       end     = lrowptr[(size_t)local_i + 1];
                const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
                std::memcpy(p.ghost_mat_val.data() + dest * 16, local_vals + (ptrdiff_t)begin * 16, (size_t)(end - begin) * 16 * sizeof(scalar_t));
            }
            (void)n_pack_nodes;
        }
    }

    scalar_t *const SFEM_RESTRICT gvalues = b.values->data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        (void)p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t ghost_entry = p.ghost_reduce_idx[j];
            const ptrdiff_t k0          = p.ghost_mat_ptr[(size_t)ghost_entry];
            const ptrdiff_t k1          = p.ghost_mat_ptr[(size_t)ghost_entry + 1];
            for (ptrdiff_t t = k0; t < k1; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}

static SFEM_NOINLINE void assemble_bsr4_atomic_sympy(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    const ptrdiff_t                    ne     = d.nelements;
    smesh::idx_t **const SFEM_RESTRICT elems  = d.elems;
    scalar_t *const SFEM_RESTRICT      values = b.values->data();

#pragma omp parallel
    {
        alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < ne; ++e) {
            const smesh::idx_t ev[4] = {elems[0][e], elems[1][e], elems[2][e], elems[3][e]};
            jacobian_element_global_sympy(d, e, ev, rho, mu, Ke);
            tet4_local_to_global_bsr4<true>(ev, Ke, b.rowptr, b.colidx, values);
        }
    }
}

static SFEM_NOINLINE void assemble_bsr4_packed_sympy(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto                             &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto                             &lcolidx      = p.local_colidx[(size_t)pack];
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

            alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                jacobian_element_packed_sympy(d, e, ev, pack_u, rho, mu, Ke);
                tet4_local_to_global_bsr4<false>(ev, Ke, lrowptr.data(), lcolidx.data(), local_vals);
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

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
            (void)n_pack_nodes;
        }
    }

    scalar_t *const SFEM_RESTRICT gvalues = b.values->data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        (void)p.ghost_reduce_dest[row];
        const ptrdiff_t begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t end   = p.ghost_reduce_ptr[row + 1];
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t ghost_entry = p.ghost_reduce_idx[j];
            const ptrdiff_t k0          = p.ghost_mat_ptr[(size_t)ghost_entry];
            const ptrdiff_t k1          = p.ghost_mat_ptr[(size_t)ghost_entry + 1];
            for (ptrdiff_t t = k0; t < k1; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}

template <bool Sympy, bool DirectToSlots, bool Blockwise, bool Facewise>
static SFEM_NOINLINE void assemble_bsr4_packed_slots_variant(MeshData       &d,
                                                             PackedData     &p,
                                                             BSR4           &b,
                                                             const scalar_t  rho,
                                                             const scalar_t  mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t                   e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                   e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                   owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                   n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                   n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const         ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto                       &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto                       &lslots       = p.local_global_slot[(size_t)pack];
            const int                         local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

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

            if constexpr (DirectToSlots) {
                for (ptrdiff_t e = e_start; e < e_end; ++e) {
                    const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                    if constexpr (Facewise) {
                        jacobian_element_packed_sympy_add_slots_facewise(
                                d, e, ev, pack_u, rho, mu, p.local_element_slot.data() + (size_t)e * 16, local_vals);
                    } else if constexpr (Blockwise) {
                        jacobian_element_packed_sympy_add_slots_blockwise(
                                d, e, ev, pack_u, rho, mu, p.local_element_slot.data() + (size_t)e * 16, local_vals);
                    } else {
                        jacobian_element_packed_sympy_add_slots(
                                d, e, ev, pack_u, rho, mu, p.local_element_slot.data() + (size_t)e * 16, local_vals);
                    }
                }
            } else {
                alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
                for (ptrdiff_t e = e_start; e < e_end; ++e) {
                    const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                    if constexpr (Sympy) {
                        jacobian_element_packed_sympy(d, e, ev, pack_u, rho, mu, Ke);
                    } else {
                        jacobian_element_packed(d, e, ev, pack_u, rho, mu, Ke);
                    }
                    tet4_local_slots_to_bsr4(p.local_element_slot.data() + (size_t)e * 16, Ke, local_vals);
                }
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

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
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}

static SFEM_NOINLINE void assemble_bsr4_packed_current_slots(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    assemble_bsr4_packed_slots_variant<false, false, false, false>(d, p, b, rho, mu);
}

static SFEM_NOINLINE void assemble_bsr4_packed_sympy_slots(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    assemble_bsr4_packed_slots_variant<true, false, false, false>(d, p, b, rho, mu);
}

static SFEM_NOINLINE void assemble_bsr4_packed_sympy_direct(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    assemble_bsr4_packed_slots_variant<true, true, false, false>(d, p, b, rho, mu);
}

static SFEM_NOINLINE void assemble_bsr4_packed_sympy_block(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    assemble_bsr4_packed_slots_variant<true, true, true, false>(d, p, b, rho, mu);
}

static SFEM_NOINLINE void assemble_bsr4_packed_sympy_face(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    assemble_bsr4_packed_slots_variant<true, true, false, true>(d, p, b, rho, mu);
}

static SFEM_NOINLINE void assemble_bsr4_packed_sympy_simd(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        alignas(ALIGN_BYTES) scalar_t Ke_vec[CVFEM_N_DOF * CVFEM_N_DOF * SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t a0[SIMD_SIZE], a1[SIMD_SIZE], a2[SIMD_SIZE], a3[SIMD_SIZE], a4[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t a5[SIMD_SIZE], a6[SIMD_SIZE], a7[SIMD_SIZE], a8[SIMD_SIZE], detv[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t ux0[SIMD_SIZE], ux1[SIMD_SIZE], ux2[SIMD_SIZE], ux3[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t uy0[SIMD_SIZE], uy1[SIMD_SIZE], uy2[SIMD_SIZE], uy3[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t uz0[SIMD_SIZE], uz1[SIMD_SIZE], uz2[SIMD_SIZE], uz3[SIMD_SIZE];

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t           e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t           e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t           owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t           n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t           n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto               &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto               &lslots       = p.local_global_slot[(size_t)pack];
            const int                 local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

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

            ptrdiff_t e = e_start;
            for (; e + SIMD_SIZE <= e_end; e += SIMD_SIZE) {
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    const ptrdiff_t  ee    = e + lane;
                    const pack_idx_t ev[4] = {p.elems[0][ee], p.elems[1][ee], p.elems[2][ee], p.elems[3][ee]};
                    const scalar_t *const u0 = pack_u + (ptrdiff_t)ev[0] * N_FIELDS;
                    const scalar_t *const u1 = pack_u + (ptrdiff_t)ev[1] * N_FIELDS;
                    const scalar_t *const u2 = pack_u + (ptrdiff_t)ev[2] * N_FIELDS;
                    const scalar_t *const u3 = pack_u + (ptrdiff_t)ev[3] * N_FIELDS;
                    a0[lane]                 = scalar_t(d.adj[0][ee]);
                    a1[lane]                 = scalar_t(d.adj[1][ee]);
                    a2[lane]                 = scalar_t(d.adj[2][ee]);
                    a3[lane]                 = scalar_t(d.adj[3][ee]);
                    a4[lane]                 = scalar_t(d.adj[4][ee]);
                    a5[lane]                 = scalar_t(d.adj[5][ee]);
                    a6[lane]                 = scalar_t(d.adj[6][ee]);
                    a7[lane]                 = scalar_t(d.adj[7][ee]);
                    a8[lane]                 = scalar_t(d.adj[8][ee]);
                    detv[lane]               = scalar_t(d.det[ee]);
                    ux0[lane]                = u0[0];
                    ux1[lane]                = u1[0];
                    ux2[lane]                = u2[0];
                    ux3[lane]                = u3[0];
                    uy0[lane]                = u0[1];
                    uy1[lane]                = u1[1];
                    uy2[lane]                = u2[1];
                    uy3[lane]                = u3[1];
                    uz0[lane]                = u0[2];
                    uz1[lane]                = u1[2];
                    uz2[lane]                = u2[2];
                    uz3[lane]                = u3[2];
                }
                cvfem_tet4_ns_upwind_sympy_jacobian_dense_vector(rho,
                                                                  mu,
                                                                  a0,
                                                                  a1,
                                                                  a2,
                                                                  a3,
                                                                  a4,
                                                                  a5,
                                                                  a6,
                                                                  a7,
                                                                  a8,
                                                                  detv,
                                                                  ux0,
                                                                  ux1,
                                                                  ux2,
                                                                  ux3,
                                                                  uy0,
                                                                  uy1,
                                                                  uy2,
                                                                  uy3,
                                                                  uz0,
                                                                  uz1,
                                                                  uz2,
                                                                  uz3,
                                                                  Ke_vec);
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    tet4_local_slots_to_bsr4_vec_lane(p.local_element_slot.data() + (size_t)(e + lane) * 16, Ke_vec, lane, local_vals);
                }
            }
            for (; e < e_end; ++e) {
                const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
                jacobian_element_packed_sympy(d, e, ev, pack_u, rho, mu, Ke);
                tet4_local_slots_to_bsr4(p.local_element_slot.data() + (size_t)e * 16, Ke, local_vals);
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

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
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}

static SFEM_NOINLINE void assemble_bsr4_packed_sympy_simd_clean(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        alignas(ALIGN_BYTES) scalar_t Ke_vec[CVFEM_N_DOF * CVFEM_N_DOF * SIMD_SIZE];

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t           e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t           e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t           owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t           n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t           n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto               &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto               &lslots       = p.local_global_slot[(size_t)pack];
            const int                 local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
                const ptrdiff_t               g   = owned + k;
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
                const smesh::idx_t            g   = ghosts[k];
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
            }

            ptrdiff_t e = e_start;
            for (; e + SIMD_SIZE <= e_end; e += SIMD_SIZE) {
#if CVFEM_SIMD_BYTES == 64
#define CVFEM_VEC8(X) scalar_v{X(0), X(1), X(2), X(3), X(4), X(5), X(6), X(7)}
#define CVFEM_VEC(X)  CVFEM_VEC8(X)
#elif CVFEM_SIMD_BYTES == 32
#define CVFEM_VEC4(X) scalar_v{X(0), X(1), X(2), X(3)}
#define CVFEM_VEC(X)  CVFEM_VEC4(X)
#elif CVFEM_SIMD_BYTES == 16
#define CVFEM_VEC2(X) scalar_v{X(0), X(1)}
#define CVFEM_VEC(X)  CVFEM_VEC2(X)
#else
#error "assemble_bsr4_packed_sympy_simd_clean supports 16, 32, or 64 byte SIMD vectors"
#endif
#define CVFEM_ADJ_LANE(K, L) scalar_t(d.adj[K][e + (L)])
#define CVFEM_DET_LANE(L)    scalar_t(d.det[e + (L)])
#define CVFEM_U_LANE(N, F, L) pack_u[(ptrdiff_t)p.elems[N][e + (L)] * N_FIELDS + (F)]
#define CVFEM_ADJ0(L) CVFEM_ADJ_LANE(0, L)
#define CVFEM_ADJ1(L) CVFEM_ADJ_LANE(1, L)
#define CVFEM_ADJ2(L) CVFEM_ADJ_LANE(2, L)
#define CVFEM_ADJ3(L) CVFEM_ADJ_LANE(3, L)
#define CVFEM_ADJ4(L) CVFEM_ADJ_LANE(4, L)
#define CVFEM_ADJ5(L) CVFEM_ADJ_LANE(5, L)
#define CVFEM_ADJ6(L) CVFEM_ADJ_LANE(6, L)
#define CVFEM_ADJ7(L) CVFEM_ADJ_LANE(7, L)
#define CVFEM_ADJ8(L) CVFEM_ADJ_LANE(8, L)
#define CVFEM_DET(L)  CVFEM_DET_LANE(L)
#define CVFEM_UX0(L)  CVFEM_U_LANE(0, 0, L)
#define CVFEM_UX1(L)  CVFEM_U_LANE(1, 0, L)
#define CVFEM_UX2(L)  CVFEM_U_LANE(2, 0, L)
#define CVFEM_UX3(L)  CVFEM_U_LANE(3, 0, L)
#define CVFEM_UY0(L)  CVFEM_U_LANE(0, 1, L)
#define CVFEM_UY1(L)  CVFEM_U_LANE(1, 1, L)
#define CVFEM_UY2(L)  CVFEM_U_LANE(2, 1, L)
#define CVFEM_UY3(L)  CVFEM_U_LANE(3, 1, L)
#define CVFEM_UZ0(L)  CVFEM_U_LANE(0, 2, L)
#define CVFEM_UZ1(L)  CVFEM_U_LANE(1, 2, L)
#define CVFEM_UZ2(L)  CVFEM_U_LANE(2, 2, L)
#define CVFEM_UZ3(L)  CVFEM_U_LANE(3, 2, L)
                const scalar_v a0   = CVFEM_VEC(CVFEM_ADJ0);
                const scalar_v a1   = CVFEM_VEC(CVFEM_ADJ1);
                const scalar_v a2   = CVFEM_VEC(CVFEM_ADJ2);
                const scalar_v a3   = CVFEM_VEC(CVFEM_ADJ3);
                const scalar_v a4   = CVFEM_VEC(CVFEM_ADJ4);
                const scalar_v a5   = CVFEM_VEC(CVFEM_ADJ5);
                const scalar_v a6   = CVFEM_VEC(CVFEM_ADJ6);
                const scalar_v a7   = CVFEM_VEC(CVFEM_ADJ7);
                const scalar_v a8   = CVFEM_VEC(CVFEM_ADJ8);
                const scalar_v detv = CVFEM_VEC(CVFEM_DET);
                const scalar_v ux0  = CVFEM_VEC(CVFEM_UX0);
                const scalar_v ux1  = CVFEM_VEC(CVFEM_UX1);
                const scalar_v ux2  = CVFEM_VEC(CVFEM_UX2);
                const scalar_v ux3  = CVFEM_VEC(CVFEM_UX3);
                const scalar_v uy0  = CVFEM_VEC(CVFEM_UY0);
                const scalar_v uy1  = CVFEM_VEC(CVFEM_UY1);
                const scalar_v uy2  = CVFEM_VEC(CVFEM_UY2);
                const scalar_v uy3  = CVFEM_VEC(CVFEM_UY3);
                const scalar_v uz0  = CVFEM_VEC(CVFEM_UZ0);
                const scalar_v uz1  = CVFEM_VEC(CVFEM_UZ1);
                const scalar_v uz2  = CVFEM_VEC(CVFEM_UZ2);
                const scalar_v uz3  = CVFEM_VEC(CVFEM_UZ3);
#undef CVFEM_UZ3
#undef CVFEM_UZ2
#undef CVFEM_UZ1
#undef CVFEM_UZ0
#undef CVFEM_UY3
#undef CVFEM_UY2
#undef CVFEM_UY1
#undef CVFEM_UY0
#undef CVFEM_UX3
#undef CVFEM_UX2
#undef CVFEM_UX1
#undef CVFEM_UX0
#undef CVFEM_DET
#undef CVFEM_ADJ8
#undef CVFEM_ADJ7
#undef CVFEM_ADJ6
#undef CVFEM_ADJ5
#undef CVFEM_ADJ4
#undef CVFEM_ADJ3
#undef CVFEM_ADJ2
#undef CVFEM_ADJ1
#undef CVFEM_ADJ0
#undef CVFEM_U_LANE
#undef CVFEM_DET_LANE
#undef CVFEM_ADJ_LANE
#undef CVFEM_VEC
#if CVFEM_SIMD_BYTES == 64
#undef CVFEM_VEC8
#elif CVFEM_SIMD_BYTES == 32
#undef CVFEM_VEC4
#elif CVFEM_SIMD_BYTES == 16
#undef CVFEM_VEC2
#endif
                cvfem_tet4_ns_upwind_sympy_jacobian_dense_vector_values(rho,
                                                                         mu,
                                                                         a0,
                                                                         a1,
                                                                         a2,
                                                                         a3,
                                                                         a4,
                                                                         a5,
                                                                         a6,
                                                                         a7,
                                                                         a8,
                                                                         detv,
                                                                         ux0,
                                                                         ux1,
                                                                         ux2,
                                                                         ux3,
                                                                         uy0,
                                                                         uy1,
                                                                         uy2,
                                                                         uy3,
                                                                         uz0,
                                                                         uz1,
                                                                         uz2,
                                                                         uz3,
                                                                         Ke_vec);
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    tet4_local_slots_to_bsr4_vec_lane(p.local_element_slot.data() + (size_t)(e + lane) * 16, Ke_vec, lane, local_vals);
                }
            }
            for (; e < e_end; ++e) {
                const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
                jacobian_element_packed_sympy(d, e, ev, pack_u, rho, mu, Ke);
                tet4_local_slots_to_bsr4(p.local_element_slot.data() + (size_t)e * 16, Ke, local_vals);
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

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
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}

static SFEM_NOINLINE void assemble_bsr4_packed_sympy_block_simd(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        alignas(ALIGN_BYTES) scalar_t a0[SIMD_SIZE], a1[SIMD_SIZE], a2[SIMD_SIZE], a3[SIMD_SIZE], a4[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t a5[SIMD_SIZE], a6[SIMD_SIZE], a7[SIMD_SIZE], a8[SIMD_SIZE], detv[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t ux0[SIMD_SIZE], ux1[SIMD_SIZE], ux2[SIMD_SIZE], ux3[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t uy0[SIMD_SIZE], uy1[SIMD_SIZE], uy2[SIMD_SIZE], uy3[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t uz0[SIMD_SIZE], uz1[SIMD_SIZE], uz2[SIMD_SIZE], uz3[SIMD_SIZE];

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t           e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t           e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t           owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t           n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t           n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto               &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto               &lslots       = p.local_global_slot[(size_t)pack];
            const int                 local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

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

            ptrdiff_t e = e_start;
            for (; e + SIMD_SIZE <= e_end; e += SIMD_SIZE) {
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    const ptrdiff_t  ee    = e + lane;
                    const pack_idx_t ev[4] = {p.elems[0][ee], p.elems[1][ee], p.elems[2][ee], p.elems[3][ee]};
                    const scalar_t *const u0 = pack_u + (ptrdiff_t)ev[0] * N_FIELDS;
                    const scalar_t *const u1 = pack_u + (ptrdiff_t)ev[1] * N_FIELDS;
                    const scalar_t *const u2 = pack_u + (ptrdiff_t)ev[2] * N_FIELDS;
                    const scalar_t *const u3 = pack_u + (ptrdiff_t)ev[3] * N_FIELDS;
                    a0[lane]                 = scalar_t(d.adj[0][ee]);
                    a1[lane]                 = scalar_t(d.adj[1][ee]);
                    a2[lane]                 = scalar_t(d.adj[2][ee]);
                    a3[lane]                 = scalar_t(d.adj[3][ee]);
                    a4[lane]                 = scalar_t(d.adj[4][ee]);
                    a5[lane]                 = scalar_t(d.adj[5][ee]);
                    a6[lane]                 = scalar_t(d.adj[6][ee]);
                    a7[lane]                 = scalar_t(d.adj[7][ee]);
                    a8[lane]                 = scalar_t(d.adj[8][ee]);
                    detv[lane]               = scalar_t(d.det[ee]);
                    ux0[lane]                = u0[0];
                    ux1[lane]                = u1[0];
                    ux2[lane]                = u2[0];
                    ux3[lane]                = u3[0];
                    uy0[lane]                = u0[1];
                    uy1[lane]                = u1[1];
                    uy2[lane]                = u2[1];
                    uy3[lane]                = u3[1];
                    uz0[lane]                = u0[2];
                    uz1[lane]                = u1[2];
                    uz2[lane]                = u2[2];
                    uz3[lane]                = u3[2];
                }
                cvfem_tet4_ns_upwind_sympy_jacobian_add_bsr_slots_blockwise_vector(rho,
                                                                                   mu,
                                                                                   a0,
                                                                                   a1,
                                                                                   a2,
                                                                                   a3,
                                                                                   a4,
                                                                                   a5,
                                                                                   a6,
                                                                                   a7,
                                                                                   a8,
                                                                                   detv,
                                                                                   ux0,
                                                                                   ux1,
                                                                                   ux2,
                                                                                   ux3,
                                                                                   uy0,
                                                                                   uy1,
                                                                                   uy2,
                                                                                   uy3,
                                                                                   uz0,
                                                                                   uz1,
                                                                                   uz2,
                                                                                   uz3,
                                                                                   p.local_element_slot.data() + (size_t)e * 16,
                                                                                   local_vals);
            }
            for (; e < e_end; ++e) {
                const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                jacobian_element_packed_sympy_add_slots_blockwise(
                        d, e, ev, pack_u, rho, mu, p.local_element_slot.data() + (size_t)e * 16, local_vals);
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

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
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}

static SFEM_INLINE void assemble_bsr4_packed_sympy_row_simd_pack(MeshData &d,
                                                                 PackedData &p,
                                                                 BSR4 &b,
                                                                 const scalar_t rho,
                                                                 const scalar_t mu,
                                                                 const ptrdiff_t pack,
                                                                 scalar_t *const SFEM_RESTRICT pack_u,
                                                                 scalar_t *const SFEM_RESTRICT local_vals) {
    const ptrdiff_t           e_start      = pack * p.n_elements_per_pack;
    const ptrdiff_t           e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
    const ptrdiff_t           owned        = p.owned_nodes_ptr[pack];
    const ptrdiff_t           n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
    const ptrdiff_t           n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
    const smesh::idx_t *const ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
    const auto               &lrowptr      = p.local_rowptr[(size_t)pack];
    const auto               &lslots       = p.local_global_slot[(size_t)pack];

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

    ptrdiff_t e = e_start;
    for (; e + SIMD_SIZE <= e_end; e += SIMD_SIZE) {
        scalar_v adj0 = scalar_v{}, adj1 = scalar_v{}, adj2 = scalar_v{};
        scalar_v adj3 = scalar_v{}, adj4 = scalar_v{}, adj5 = scalar_v{};
        scalar_v adj6 = scalar_v{}, adj7 = scalar_v{}, adj8 = scalar_v{}, detv = scalar_v{};
        scalar_v ux0 = scalar_v{}, ux1 = scalar_v{}, ux2 = scalar_v{}, ux3 = scalar_v{};
        scalar_v uy0 = scalar_v{}, uy1 = scalar_v{}, uy2 = scalar_v{}, uy3 = scalar_v{};
        scalar_v uz0 = scalar_v{}, uz1 = scalar_v{}, uz2 = scalar_v{}, uz3 = scalar_v{};

#pragma unroll
        for (int lane = 0; lane < SIMD_SIZE; ++lane) {
            const ptrdiff_t  ee  = e + lane;
            const pack_idx_t ev0 = p.elems[0][ee];
            const pack_idx_t ev1 = p.elems[1][ee];
            const pack_idx_t ev2 = p.elems[2][ee];
            const pack_idx_t ev3 = p.elems[3][ee];

            const scalar_t *const u0 = pack_u + (ptrdiff_t)ev0 * N_FIELDS;
            const scalar_t *const u1 = pack_u + (ptrdiff_t)ev1 * N_FIELDS;
            const scalar_t *const u2 = pack_u + (ptrdiff_t)ev2 * N_FIELDS;
            const scalar_t *const u3 = pack_u + (ptrdiff_t)ev3 * N_FIELDS;

            adj0[lane] = scalar_t(d.adj[0][ee]);
            adj1[lane] = scalar_t(d.adj[1][ee]);
            adj2[lane] = scalar_t(d.adj[2][ee]);
            adj3[lane] = scalar_t(d.adj[3][ee]);
            adj4[lane] = scalar_t(d.adj[4][ee]);
            adj5[lane] = scalar_t(d.adj[5][ee]);
            adj6[lane] = scalar_t(d.adj[6][ee]);
            adj7[lane] = scalar_t(d.adj[7][ee]);
            adj8[lane] = scalar_t(d.adj[8][ee]);
            detv[lane] = scalar_t(d.det[ee]);

            ux0[lane] = u0[0];
            ux1[lane] = u1[0];
            ux2[lane] = u2[0];
            ux3[lane] = u3[0];
            uy0[lane] = u0[1];
            uy1[lane] = u1[1];
            uy2[lane] = u2[1];
            uy3[lane] = u3[1];
            uz0[lane] = u0[2];
            uz1[lane] = u1[2];
            uz2[lane] = u2[2];
            uz3[lane] = u3[2];
        }

        cvfem_tet4_ns_upwind_sympy_jacobian_add_bsr_slots_rowwise_vector_values(rho,
                                                                                mu,
                                                                                adj0,
                                                                                adj1,
                                                                                adj2,
                                                                                adj3,
                                                                                adj4,
                                                                                adj5,
                                                                                adj6,
                                                                                adj7,
                                                                                adj8,
                                                                                detv,
                                                                                ux0,
                                                                                ux1,
                                                                                ux2,
                                                                                ux3,
                                                                                uy0,
                                                                                uy1,
                                                                                uy2,
                                                                                uy3,
                                                                                uz0,
                                                                                uz1,
                                                                                uz2,
                                                                                uz3,
                                                                                p.local_element_slot.data() + (size_t)e * 16,
                                                                                local_vals);
    }
    for (; e < e_end; ++e) {
        const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
        jacobian_element_packed_sympy_add_slots_blockwise(
                d, e, ev, pack_u, rho, mu, p.local_element_slot.data() + (size_t)e * 16, local_vals);
    }

    scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
    const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
    for (int t = 0; t < owned_nnz; ++t) {
        cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
    }

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

static SFEM_NOINLINE void assemble_bsr4_packed_sympy_row_simd(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const auto &lrowptr   = p.local_rowptr[(size_t)pack];
            const int   local_nnz = lrowptr.empty() ? 0 : lrowptr.back();
            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));
            assemble_bsr4_packed_sympy_row_simd_pack(d, p, b, rho, mu, pack, pack_u, local_vals);
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
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}

static SFEM_NOINLINE void assemble_bsr4_packed_sympy_row_simd_fused(MeshData &d,
                                                                    PackedData &p,
                                                                    BSR4 &b,
                                                                    const scalar_t rho,
                                                                    const scalar_t mu) {
    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        scalar_t *const SFEM_RESTRICT gvalues    = b.values->data();

#pragma omp for schedule(static)
        for (ptrdiff_t i = 0; i < b.nnz * 16; ++i) {
            gvalues[i] = scalar_t(0);
        }

        scalar_t *const SFEM_RESTRICT ghost_values = p.ghost_mat_val.data();
        const ptrdiff_t               n_ghost_vals = (ptrdiff_t)p.ghost_mat_val.size();
#pragma omp for schedule(static)
        for (ptrdiff_t i = 0; i < n_ghost_vals; ++i) {
            ghost_values[i] = scalar_t(0);
        }

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const auto &lrowptr   = p.local_rowptr[(size_t)pack];
            const int   local_nnz = lrowptr.empty() ? 0 : lrowptr.back();
            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));
            assemble_bsr4_packed_sympy_row_simd_pack(d, p, b, rho, mu, pack, pack_u, local_vals);
        }

#pragma omp for schedule(static)
        for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
            const ptrdiff_t begin = p.ghost_reduce_ptr[row];
            const ptrdiff_t end   = p.ghost_reduce_ptr[row + 1];
            for (ptrdiff_t j = begin; j < end; ++j) {
                const ptrdiff_t ghost_entry = p.ghost_reduce_idx[j];
                const ptrdiff_t k0          = p.ghost_mat_ptr[(size_t)ghost_entry];
                const ptrdiff_t k1          = p.ghost_mat_ptr[(size_t)ghost_entry + 1];
                for (ptrdiff_t t = k0; t < k1; ++t) {
                    cvfem_bsr4_add16_vec(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                         ghost_values + t * 16);
                }
            }
        }
    }
}

static SFEM_NOINLINE void assemble_bsr4_packed_sympy_face_simd(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        alignas(ALIGN_BYTES) scalar_t face_ke[CVFEM_TET4_NS_UPWIND_FACE_SIMD_MAX_NNZ * SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t a0[SIMD_SIZE], a1[SIMD_SIZE], a2[SIMD_SIZE], a3[SIMD_SIZE], a4[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t a5[SIMD_SIZE], a6[SIMD_SIZE], a7[SIMD_SIZE], a8[SIMD_SIZE], detv[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t ux0[SIMD_SIZE], ux1[SIMD_SIZE], ux2[SIMD_SIZE], ux3[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t uy0[SIMD_SIZE], uy1[SIMD_SIZE], uy2[SIMD_SIZE], uy3[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t uz0[SIMD_SIZE], uz1[SIMD_SIZE], uz2[SIMD_SIZE], uz3[SIMD_SIZE];

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t           e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t           e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t           owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t           n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t           n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto               &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto               &lslots       = p.local_global_slot[(size_t)pack];
            const int                 local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

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

            ptrdiff_t e = e_start;
            for (; e + SIMD_SIZE <= e_end; e += SIMD_SIZE) {
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    const ptrdiff_t  ee    = e + lane;
                    const pack_idx_t ev[4] = {p.elems[0][ee], p.elems[1][ee], p.elems[2][ee], p.elems[3][ee]};
                    const scalar_t *const u0 = pack_u + (ptrdiff_t)ev[0] * N_FIELDS;
                    const scalar_t *const u1 = pack_u + (ptrdiff_t)ev[1] * N_FIELDS;
                    const scalar_t *const u2 = pack_u + (ptrdiff_t)ev[2] * N_FIELDS;
                    const scalar_t *const u3 = pack_u + (ptrdiff_t)ev[3] * N_FIELDS;
                    a0[lane]                 = scalar_t(d.adj[0][ee]);
                    a1[lane]                 = scalar_t(d.adj[1][ee]);
                    a2[lane]                 = scalar_t(d.adj[2][ee]);
                    a3[lane]                 = scalar_t(d.adj[3][ee]);
                    a4[lane]                 = scalar_t(d.adj[4][ee]);
                    a5[lane]                 = scalar_t(d.adj[5][ee]);
                    a6[lane]                 = scalar_t(d.adj[6][ee]);
                    a7[lane]                 = scalar_t(d.adj[7][ee]);
                    a8[lane]                 = scalar_t(d.adj[8][ee]);
                    detv[lane]               = scalar_t(d.det[ee]);
                    ux0[lane]                = u0[0];
                    ux1[lane]                = u1[0];
                    ux2[lane]                = u2[0];
                    ux3[lane]                = u3[0];
                    uy0[lane]                = u0[1];
                    uy1[lane]                = u1[1];
                    uy2[lane]                = u2[1];
                    uy3[lane]                = u3[1];
                    uz0[lane]                = u0[2];
                    uz1[lane]                = u1[2];
                    uz2[lane]                = u2[2];
                    uz3[lane]                = u3[2];
                }

#define CVFEM_FACE_SIMD(FACE)                                                                                                      \
    do {                                                                                                                           \
        cvfem_tet4_ns_upwind_sympy_jacobian_face##FACE##_vector(                                                                   \
                rho, mu, a0, a1, a2, a3, a4, a5, a6, a7, a8, detv, ux0, ux1, ux2, ux3, uy0, uy1, uy2, uy3, uz0, uz1, uz2, uz3,    \
                face_ke);                                                                                                          \
        for (int lane = 0; lane < SIMD_SIZE; ++lane) {                                                                             \
            cvfem_tet4_ns_upwind_sympy_jacobian_face##FACE##_vector_lane_to_bsr_slots(                                             \
                    p.local_element_slot.data() + (size_t)(e + lane) * 16, face_ke, lane, local_vals);                             \
        }                                                                                                                          \
    } while (0)

                CVFEM_FACE_SIMD(0);
                CVFEM_FACE_SIMD(1);
                CVFEM_FACE_SIMD(2);
                CVFEM_FACE_SIMD(3);
                CVFEM_FACE_SIMD(4);
                CVFEM_FACE_SIMD(5);

#undef CVFEM_FACE_SIMD
            }
            for (; e < e_end; ++e) {
                const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                jacobian_element_packed_sympy_add_slots_facewise(
                        d, e, ev, pack_u, rho, mu, p.local_element_slot.data() + (size_t)e * 16, local_vals);
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

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
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}

static void pack_state(const MeshData &d, std::vector<scalar_t> &x) {
    x.resize((size_t)d.nnodes * 4);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        x[(size_t)i * 4 + 0] = d.ux[i];
        x[(size_t)i * 4 + 1] = d.uy[i];
        x[(size_t)i * 4 + 2] = d.uz[i];
        x[(size_t)i * 4 + 3] = d.p[i];
    }
}

static void unpack_residual(const MeshData &d, std::vector<scalar_t> &r) {
    r.resize((size_t)d.nnodes * 4);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        r[(size_t)i * 4 + 0] = d.rx[i];
        r[(size_t)i * 4 + 1] = d.ry[i];
        r[(size_t)i * 4 + 2] = d.rz[i];
        r[(size_t)i * 4 + 3] = d.rc[i];
    }
}

static scalar_t max_abs_diff_ptr(const scalar_t *a, const scalar_t *b, ptrdiff_t n) {
    scalar_t m = 0.0;
    for (ptrdiff_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static void residual_lane_to_dof(const Tet4ResidualPack &out, const int lane, scalar_t *const r) {
    for (int a = 0; a < 4; ++a) {
        r[a * 4 + 0] = out.rx[a][lane];
        r[a * 4 + 1] = out.ry[a][lane];
        r[a * 4 + 2] = out.rz[a][lane];
        r[a * 4 + 3] = out.rc[a][lane];
    }
}

static void apply_ke_to_dir(MeshData &d, const scalar_t rho, const scalar_t mu, const scalar_t *const dir,
                            scalar_t *const jv) {
    const ptrdiff_t ndof = d.nnodes * 4;
    for (ptrdiff_t i = 0; i < ndof; ++i) jv[i] = 0.0;
    smesh::idx_t **const elems = d.elems;
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
        const smesh::idx_t ev[4] = {elems[0][e], elems[1][e], elems[2][e], elems[3][e]};
        jacobian_element_global(d, e, ev, rho, mu, Ke);
        scalar_t loc[16];
        for (int a = 0; a < 4; ++a)
            for (int f = 0; f < 4; ++f) loc[a * 4 + f] = dir[(ptrdiff_t)ev[a] * 4 + f];
        for (int r = 0; r < 16; ++r) {
            scalar_t acc = 0.0;
            for (int c = 0; c < 16; ++c) acc += Ke[r * 16 + c] * loc[c];
            jv[(ptrdiff_t)ev[r / 4] * 4 + (r % 4)] += acc;
        }
    }
}

static scalar_t verify_element_kernel_jac(MeshData &d, const scalar_t rho, const scalar_t mu) {
    Tet4InputPack in;
    gather_tet4_pack_global(d, 0, 1, in);
    alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
    smesh::idx_t **const elems = d.elems;
    const smesh::idx_t ev[4] = {elems[0][0], elems[1][0], elems[2][0], elems[3][0]};
    jacobian_element_global(d, 0, ev, rho, mu, Ke);
    const scalar_t *const ke = Ke;

    const scalar_t adj0 = scalar_t(d.adj[0][0]);
    const scalar_t adj1 = scalar_t(d.adj[1][0]);
    const scalar_t adj2 = scalar_t(d.adj[2][0]);
    const scalar_t adj3 = scalar_t(d.adj[3][0]);
    const scalar_t adj4 = scalar_t(d.adj[4][0]);
    const scalar_t adj5 = scalar_t(d.adj[5][0]);
    const scalar_t adj6 = scalar_t(d.adj[6][0]);
    const scalar_t adj7 = scalar_t(d.adj[7][0]);
    const scalar_t adj8 = scalar_t(d.adj[8][0]);
    const scalar_t det  = scalar_t(d.det[0]);
    const scalar_t ux[4] = {in.ux[0][0], in.ux[1][0], in.ux[2][0], in.ux[3][0]};
    const scalar_t uy[4] = {in.uy[0][0], in.uy[1][0], in.uy[2][0], in.uy[3][0]};
    const scalar_t uz[4] = {in.uz[0][0], in.uz[1][0], in.uz[2][0], in.uz[3][0]};
    const scalar_t p[4]  = {in.p[0][0], in.p[1][0], in.p[2][0], in.p[3][0]};

    Tet4ResidualPack current_residual;
    run_microkernel(d, rho, mu, 0, 1, in, current_residual);
    scalar_t r_current[CVFEM_N_DOF];
    scalar_t r_sympy[CVFEM_N_DOF];
    residual_lane_to_dof(current_residual, 0, r_current);
    cvfem_tet4_ns_upwind_sympy_residual_dense(
            rho, mu, adj0, adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, det, ux, uy, uz, p, r_sympy);
    const scalar_t sympy_residual_err = max_abs_diff_ptr(r_current, r_sympy, CVFEM_N_DOF);

    alignas(ALIGN_BYTES) scalar_t Ke_sympy[CVFEM_N_DOF * CVFEM_N_DOF];
    cvfem_tet4_ns_upwind_sympy_jacobian_dense(
            rho, mu, adj0, adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, det, ux, uy, uz, Ke_sympy);
    const scalar_t sympy_jacobian_err = max_abs_diff_ptr(ke, Ke_sympy, CVFEM_N_DOF * CVFEM_N_DOF);
    std::printf("verify_sympy_residual_vs_current_abs: %.6e\n", sympy_residual_err);
    std::printf("verify_sympy_jacobian_vs_current_abs: %.6e\n", sympy_jacobian_err);
    if (sympy_residual_err > 1.0e-10 || sympy_jacobian_err > 1.0e-10) return 1.0;

    const scalar_t eps     = 1.0e-6;
    scalar_t       max_abs = 0.0;
    scalar_t       scale   = 0.0;
    for (int col = 0; col < 16; ++col) {
        const int         node  = col / 4;
        const int         field = col % 4;
        Tet4InputPack     in_p  = in;
        scalar_t         *comp  = field == 0   ? in_p.ux[node]
                                  : field == 1 ? in_p.uy[node]
                                  : field == 2 ? in_p.uz[node]
                                               : in_p.p[node];
        comp[0] += eps;
        Tet4ResidualPack outp;
        run_microkernel(d, rho, mu, 0, 1, in_p, outp);
        scalar_t rp[16];
        residual_lane_to_dof(outp, 0, rp);
        Tet4InputPack in_m   = in;
        scalar_t     *comp_m = field == 0   ? in_m.ux[node]
                               : field == 1 ? in_m.uy[node]
                               : field == 2 ? in_m.uz[node]
                                            : in_m.p[node];
        comp_m[0] -= eps;
        Tet4ResidualPack outm;
        run_microkernel(d, rho, mu, 0, 1, in_m, outm);
        scalar_t rm[16];
        residual_lane_to_dof(outm, 0, rm);
        for (int row = 0; row < 16; ++row) {
            const scalar_t fd  = (rp[row] - rm[row]) / (2.0 * eps);
            const scalar_t ana = ke[row * 16 + col];
            scale              = std::max(scale, std::fabs(ana));
            max_abs            = std::max(max_abs, std::fabs(ana - fd));
        }
    }
    const scalar_t max_rel = max_abs / std::max(scale, 1.0e-30);
    std::printf("verify_jac_element0_vs_fd_rel: %.6e\n", max_rel);
    std::printf("verify_jac_element0_vs_fd_abs: %.6e\n", max_abs);
    return max_rel;
}

template <typename ApplyFn>
static scalar_t spmv_vs_central_fd(MeshData                   &d,
                                   ApplyFn                   &&apply,
                                   const BSR4                 &b,
                                   const std::vector<scalar_t> &dir,
                                   const scalar_t              rho,
                                   const scalar_t              mu,
                                   scalar_t                   &abs_err) {
    std::vector<scalar_t> x0, rm, rp, jv((size_t)d.nnodes * 4, 0.0);
    pack_state(d, x0);
    const scalar_t eps = 1.0e-6;
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x0[(size_t)i * 4 + 0] - eps * dir[(size_t)i * 4 + 0];
        d.uy[i] = x0[(size_t)i * 4 + 1] - eps * dir[(size_t)i * 4 + 1];
        d.uz[i] = x0[(size_t)i * 4 + 2] - eps * dir[(size_t)i * 4 + 2];
        d.p[i]  = x0[(size_t)i * 4 + 3] - eps * dir[(size_t)i * 4 + 3];
    }
    apply(d, rho, mu);
    unpack_residual(d, rm);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x0[(size_t)i * 4 + 0] + eps * dir[(size_t)i * 4 + 0];
        d.uy[i] = x0[(size_t)i * 4 + 1] + eps * dir[(size_t)i * 4 + 1];
        d.uz[i] = x0[(size_t)i * 4 + 2] + eps * dir[(size_t)i * 4 + 2];
        d.p[i]  = x0[(size_t)i * 4 + 3] + eps * dir[(size_t)i * 4 + 3];
    }
    apply(d, rho, mu);
    unpack_residual(d, rp);
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        d.ux[i] = x0[(size_t)i * 4 + 0];
        d.uy[i] = x0[(size_t)i * 4 + 1];
        d.uz[i] = x0[(size_t)i * 4 + 2];
        d.p[i]  = x0[(size_t)i * 4 + 3];
    }
    auto spmv = sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
            d.nnodes, d.nnodes, 4, b.graph->rowptr(), b.graph->colidx(), b.values, scalar_t(0));
    spmv->apply(dir.data(), jv.data());
    scalar_t max_fd = 0.0;
    abs_err         = 0.0;
    for (ptrdiff_t i = 0; i < d.nnodes * 4; ++i) {
        const scalar_t fd = (rp[(size_t)i] - rm[(size_t)i]) / (2.0 * eps);
        max_fd            = std::max(max_fd, std::fabs(fd));
        abs_err           = std::max(abs_err, std::fabs(jv[(size_t)i] - fd));
    }
    return abs_err / std::max(max_fd, 1.0e-30);
}

static scalar_t checksum(const MeshData &d) {
    scalar_t        sum    = 0.0;
    const ptrdiff_t stride = std::max<ptrdiff_t>(1, d.nnodes / 4096);
    for (ptrdiff_t i = 0; i < d.nnodes; i += stride) {
        const scalar_t w = 1.0 + scalar_t(i % 17) * 0.01;
        sum += w * (d.rx[i] + 1.3 * d.ry[i] + 1.7 * d.rz[i] + 2.1 * d.rc[i]);
    }
    return sum;
}

static scalar_t checksum_vec(const scalar_t *const x, const ptrdiff_t nnodes) {
    scalar_t        sum    = 0.0;
    const ptrdiff_t stride = std::max<ptrdiff_t>(1, nnodes / 4096);
    for (ptrdiff_t i = 0; i < nnodes; i += stride) {
        const scalar_t *const xi = x + i * N_FIELDS;
        const scalar_t        w  = 1.0 + scalar_t(i % 17) * 0.01;
        sum += w * (xi[0] + 1.3 * xi[1] + 1.7 * xi[2] + 2.1 * xi[3]);
    }
    return sum;
}

static scalar_t max_abs_diff(const std::vector<scalar_t> &a, const std::vector<scalar_t> &b) {
    scalar_t        m = 0.0;
    const ptrdiff_t n = ptrdiff_t(a.size());
    for (ptrdiff_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char **argv) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    const int own_mpi = !mpi_initialized;
    if (own_mpi) MPI_Init(&argc, &argv);

    ptrdiff_t   n         = 48;
    int         repeat    = 20;
    int         warmup    = 3;
    scalar_t    rho       = 1.0;
    scalar_t    mu        = 0.01;
    std::string layout    = "packed";
    std::string kernel    = "sympy_row_simd";
    int         verify    = 0;
    int         verify_jac = 0;
    int         assemble  = 0;
    int         jac_action = 0;
    int         bsr_apply  = 0;
    int         pack_size = 2048;
    int         use_sfc   = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--n" || arg == "--nx") && i + 1 < argc)
            n = parse_size(argv[++i]);
        else if (arg == "--repeat" && i + 1 < argc)
            repeat = std::atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc)
            warmup = std::atoi(argv[++i]);
        else if (arg == "--rho" && i + 1 < argc)
            rho = std::atof(argv[++i]);
        else if (arg == "--mu" && i + 1 < argc)
            mu = std::atof(argv[++i]);
        else if (arg == "--layout" && i + 1 < argc)
            layout = argv[++i];
        else if (arg == "--kernel" && i + 1 < argc)
            kernel = argv[++i];
        else if (arg == "--pack-size" && i + 1 < argc)
            pack_size = std::atoi(argv[++i]);
        else if (arg == "--no-sfc")
            use_sfc = 0;
        else if (arg == "--verify")
            verify = 1;
        else if (arg == "--verify-jac")
            verify_jac = 1;
        else if (arg == "--assemble")
            assemble = 1;
        else if (arg == "--jac-action")
            jac_action = 1;
        else if (arg == "--bsr-apply")
            bsr_apply = 1;
        else if (arg == "--help") {
            std::printf(
                    "usage: %s [--n cube_cells_per_dim] [--repeat N] [--warmup N]\n"
                    "          [--layout packed|atomic] [--kernel current|sympy|current_slots|sympy_slots|sympy_direct|sympy_block|sympy_face|sympy_simd|sympy_simd_clean|sympy_block_simd|sympy_row_simd|sympy_row_simd_fused|sympy_face_simd]\n"
                    "          [--pack-size N] [--no-sfc]\n"
                    "          [--verify] [--verify-jac] [--assemble] [--jac-action] [--bsr-apply]\n"
                    "  --kernel NAME  micro-kernel variant (default sympy_row_simd)\n"
                    "  --jac-action   apply the matrix-free TET4 Jacobian action J(u) v\n"
                    "  --bsr-apply    assemble once, then time BSR SpMV y = J(u) v\n"
                    "  --pack-size N   elements per pack (0 = fill uint16; default 2048)\n",
                    argv[0]);
            if (own_mpi) MPI_Finalize();
            return 0;
        }
    }

    if (kernel != "current" && kernel != "sympy" && kernel != "current_slots" && kernel != "sympy_slots" &&
        kernel != "sympy_direct" && kernel != "sympy_block" && kernel != "sympy_face" && kernel != "sympy_simd" &&
        kernel != "sympy_simd_clean" &&
        kernel != "sympy_block_simd" && kernel != "sympy_row_simd" && kernel != "sympy_row_simd_fused" && kernel != "sympy_face_simd") {
        std::fprintf(stderr,
                     "invalid --kernel '%s' (expected current, sympy, current_slots, sympy_slots, sympy_direct, sympy_block, sympy_face, sympy_simd, sympy_simd_clean, sympy_block_simd, sympy_row_simd, sympy_row_simd_fused, or sympy_face_simd)\n",
                     kernel.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    if ((assemble ? 1 : 0) + (jac_action ? 1 : 0) + (bsr_apply ? 1 : 0) > 1) {
        std::fprintf(stderr, "--assemble, --jac-action, and --bsr-apply are separate benchmark modes\n");
        if (own_mpi) MPI_Finalize();
        return 1;
    }
    if (layout != "packed" && layout != "atomic") {
        std::fprintf(stderr, "invalid --layout '%s' (expected packed or atomic)\n", layout.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    const int use_packed = (layout == "packed") || verify || verify_jac;
    const KernelKind kernel_kind = parse_kernel(kernel);
    if ((assemble || bsr_apply) && layout == "atomic" &&
        (kernel_kind == KernelKind::CurrentSlots || kernel_kind == KernelKind::SympySlots || kernel_kind == KernelKind::SympyDirect ||
         kernel_kind == KernelKind::SympyBlock || kernel_kind == KernelKind::SympyFace || kernel_kind == KernelKind::SympySimd ||
         kernel_kind == KernelKind::SympyBlockSimd || kernel_kind == KernelKind::SympyRowSimd || kernel_kind == KernelKind::SympyRowSimdFused ||
         kernel_kind == KernelKind::SympyFaceSimd)) {
        std::fprintf(stderr, "--kernel %s is a packed Jacobian assembly variant; use --layout packed for --assemble/--bsr-apply\n", kernel.c_str());
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    MeshData d;
    d.mesh = smesh::Mesh::create_tet4_cube(smesh::Communicator::self(), n, n, n, 0, 0, 0, 1, 1, 1);
    if (!d.mesh || d.mesh->element_type(0) != smesh::TET4) {
        std::fprintf(stderr, "failed to create TET4 mesh\n");
        d.mesh.reset();
        if (own_mpi) MPI_Finalize();
        return 1;
    }

    if (use_sfc) {
        auto sfc = smesh::SFC::create_from_env();
        sfc->reorder(*d.mesh);
    }

    PackedData packed;
    if (use_packed) packed = make_packed(d.mesh, pack_size);

    d.nnodes    = d.mesh->n_nodes();
    d.nelements = d.mesh->n_elements(0);
    d.elems     = d.mesh->elements(0)->data();
    d.points    = d.mesh->points()->data();

    fill_fields(d);
    precompute_affine_geometry(d);

    BSR4 bsr;
    if (assemble || verify_jac || bsr_apply) bsr = make_bsr4(d.mesh);
    if (use_packed && (assemble || verify_jac || bsr_apply)) build_pack_local_crs(packed, d.nelements, bsr.rowptr, bsr.colidx);

    if (use_packed) {
        const size_t scratch_n = packed_scratch_n(packed);
        const size_t bsr_n     = 16 * (size_t)std::max<ptrdiff_t>(packed.max_local_nnz, 1);
        const size_t slot2_n   = (jac_action || bsr_apply) ? std::max(scratch_n, bsr_n) : bsr_n;
#pragma omp parallel
        {
            (void)thread_scratch<scalar_t>(0, scratch_n);
            (void)thread_scratch<scalar_t>(1, scratch_n);
            if (assemble || verify_jac || jac_action || bsr_apply) (void)thread_scratch<scalar_t>(2, slot2_n);
        }
    }

    if (verify) {
        cvfem_tet4_ns_upwind_apply_packed(d, packed, rho, mu);
        std::vector<scalar_t> rx = d.rx, ry = d.ry, rz = d.rz, rc = d.rc;
        cvfem_tet4_ns_upwind_apply_atomic(d, rho, mu);
        const scalar_t err = std::max(std::max(max_abs_diff(rx, d.rx), max_abs_diff(ry, d.ry)),
                                      std::max(max_abs_diff(rz, d.rz), max_abs_diff(rc, d.rc)));
        std::printf("verify_packed_vs_atomic_max_abs: %.6e\n", err);
        if (err > 1.0e-10) {
            std::fprintf(stderr, "packed vs atomic residual mismatch\n");
            d.mesh.reset();
            if (own_mpi) MPI_Finalize();
            return 1;
        }
        cvfem_tet4_ns_upwind_apply_sympy_packed(d, packed, rho, mu);
        const scalar_t sympy_err = std::max(std::max(max_abs_diff(rx, d.rx), max_abs_diff(ry, d.ry)),
                                            std::max(max_abs_diff(rz, d.rz), max_abs_diff(rc, d.rc)));
        std::printf("verify_sympy_residual_mesh_vs_current_abs: %.6e\n", sympy_err);
        if (sympy_err > 1.0e-10) {
            std::fprintf(stderr, "SymPy residual mismatch\n");
            d.mesh.reset();
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    if (verify_jac) {
        assemble_bsr4_packed(d, packed, bsr, rho, mu);
        std::vector<scalar_t> packed_vals(bsr.values->data(), bsr.values->data() + bsr.nnz * 16);
        assemble_bsr4_atomic(d, bsr, rho, mu);
        std::vector<scalar_t> current_vals(bsr.values->data(), bsr.values->data() + bsr.nnz * 16);
        const scalar_t jac_err = max_abs_diff_ptr(packed_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_jac_packed_vs_atomic_max_abs: %.6e\n", jac_err);
        if (jac_err > 1.0e-12) {
            std::fprintf(stderr, "packed vs atomic BSR Jacobian mismatch\n");
            d.mesh.reset();
            if (own_mpi) MPI_Finalize();
            return 1;
        }
        assemble_bsr4_atomic_sympy(d, bsr, rho, mu);
        const scalar_t sympy_jac_atomic_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_jac_atomic_mesh_vs_current_abs: %.6e\n", sympy_jac_atomic_err);
        assemble_bsr4_packed_sympy(d, packed, bsr, rho, mu);
        const scalar_t sympy_jac_packed_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_jac_packed_mesh_vs_current_abs: %.6e\n", sympy_jac_packed_err);
        assemble_bsr4_packed_current_slots(d, packed, bsr, rho, mu);
        const scalar_t current_slots_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_current_slots_jac_packed_mesh_vs_current_abs: %.6e\n", current_slots_err);
        assemble_bsr4_packed_sympy_slots(d, packed, bsr, rho, mu);
        const scalar_t sympy_slots_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_slots_jac_packed_mesh_vs_current_abs: %.6e\n", sympy_slots_err);
        assemble_bsr4_packed_sympy_direct(d, packed, bsr, rho, mu);
        const scalar_t sympy_direct_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_direct_jac_packed_mesh_vs_current_abs: %.6e\n", sympy_direct_err);
        assemble_bsr4_packed_sympy_block(d, packed, bsr, rho, mu);
        const scalar_t sympy_block_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_block_jac_packed_mesh_vs_current_abs: %.6e\n", sympy_block_err);
        assemble_bsr4_packed_sympy_face(d, packed, bsr, rho, mu);
        const scalar_t sympy_face_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_face_jac_packed_mesh_vs_current_abs: %.6e\n", sympy_face_err);
        assemble_bsr4_packed_sympy_simd(d, packed, bsr, rho, mu);
        const scalar_t sympy_simd_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_simd_jac_packed_mesh_vs_current_abs: %.6e\n", sympy_simd_err);
        assemble_bsr4_packed_sympy_simd_clean(d, packed, bsr, rho, mu);
        const scalar_t sympy_simd_clean_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_simd_clean_jac_packed_mesh_vs_current_abs: %.6e\n", sympy_simd_clean_err);
        assemble_bsr4_packed_sympy_block_simd(d, packed, bsr, rho, mu);
        const scalar_t sympy_block_simd_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_block_simd_jac_packed_mesh_vs_current_abs: %.6e\n", sympy_block_simd_err);
        assemble_bsr4_packed_sympy_row_simd(d, packed, bsr, rho, mu);
        const scalar_t sympy_row_simd_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_row_simd_jac_packed_mesh_vs_current_abs: %.6e\n", sympy_row_simd_err);
        assemble_bsr4_packed_sympy_row_simd_fused(d, packed, bsr, rho, mu);
        const scalar_t sympy_row_simd_fused_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_row_simd_fused_jac_packed_mesh_vs_current_abs: %.6e\n", sympy_row_simd_fused_err);
        assemble_bsr4_packed_sympy_face_simd(d, packed, bsr, rho, mu);
        const scalar_t sympy_face_simd_err = max_abs_diff_ptr(current_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_sympy_face_simd_jac_packed_mesh_vs_current_abs: %.6e\n", sympy_face_simd_err);
        if (sympy_jac_atomic_err > 1.0e-12 || sympy_jac_packed_err > 1.0e-12 || current_slots_err > 1.0e-12 ||
            sympy_slots_err > 1.0e-12 || sympy_direct_err > 1.0e-12 || sympy_block_err > 1.0e-12 ||
            sympy_face_err > 1.0e-12 || sympy_simd_err > 1.0e-12 || sympy_simd_clean_err > 1.0e-12 || sympy_block_simd_err > 1.0e-12 ||
            sympy_row_simd_err > 1.0e-12 || sympy_row_simd_fused_err > 1.0e-12 || sympy_face_simd_err > 1.0e-12) {
            std::fprintf(stderr, "SymPy BSR Jacobian mismatch\n");
            d.mesh.reset();
            if (own_mpi) MPI_Finalize();
            return 1;
        }
        assemble_bsr4_atomic(d, bsr, rho, mu);

        const scalar_t elem_rel = verify_element_kernel_jac(d, rho, mu);

        std::vector<scalar_t> dir((size_t)d.nnodes * 4), jv((size_t)d.nnodes * 4, 0.0), jv_e((size_t)d.nnodes * 4, 0.0),
                jv_mf((size_t)d.nnodes * 4, 0.0), jv_mf_atomic((size_t)d.nnodes * 4, 0.0), jv_mf_sympy((size_t)d.nnodes * 4, 0.0);
        for (ptrdiff_t i = 0; i < d.nnodes * 4; ++i) dir[(size_t)i] = 1.0 + 0.01 * scalar_t(i % 7);
        apply_ke_to_dir(d, rho, mu, dir.data(), jv_e.data());
        cvfem_tet4_ns_upwind_jacobian_action_packed(d, packed, rho, mu, dir.data(), jv_mf.data(), false);
        cvfem_tet4_ns_upwind_jacobian_action_atomic(d, rho, mu, dir.data(), jv_mf_atomic.data(), false);
        cvfem_tet4_ns_upwind_jacobian_action_packed(d, packed, rho, mu, dir.data(), jv_mf_sympy.data(), true);
        auto spmv = sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
                d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(), bsr.values, scalar_t(0));
        spmv->apply(dir.data(), jv.data());
        const scalar_t spmv_elem_err = max_abs_diff_ptr(jv.data(), jv_e.data(), d.nnodes * 4);
        const scalar_t mf_spmv_err   = max_abs_diff_ptr(jv.data(), jv_mf.data(), d.nnodes * 4);
        const scalar_t mf_atomic_err = max_abs_diff_ptr(jv_mf.data(), jv_mf_atomic.data(), d.nnodes * 4);
        const scalar_t mf_sympy_err  = max_abs_diff_ptr(jv_mf.data(), jv_mf_sympy.data(), d.nnodes * 4);
        std::printf("verify_jac_spmv_vs_elem_abs: %.6e\n", spmv_elem_err);
        std::printf("verify_jac_mf_action_vs_spmv_abs: %.6e\n", mf_spmv_err);
        std::printf("verify_jac_mf_atomic_action_vs_packed_abs: %.6e\n", mf_atomic_err);
        std::printf("verify_jac_mf_sympy_action_vs_hand_abs: %.6e\n", mf_sympy_err);

        // Global residual FD uses a pressure direction: momentum convection is
        // non-differentiable at mdot == 0, so a mixed velocity probe hits kinks.
        std::fill(dir.begin(), dir.end(), 0.0);
        for (ptrdiff_t i = 0; i < d.nnodes; ++i) dir[(size_t)i * 4 + 3] = 1.0;
        scalar_t       fd_abs = 0.0;
        const scalar_t rel    = spmv_vs_central_fd(
                d,
                [](MeshData &md, const scalar_t r, const scalar_t m) { cvfem_tet4_ns_upwind_apply_atomic(md, r, m); },
                bsr,
                dir,
                rho,
                mu,
                fd_abs);
        std::printf("verify_jac_spmv_vs_fd_rel: %.6e\n", rel);
        std::printf("verify_jac_spmv_vs_fd_abs: %.6e\n", fd_abs);
        if (elem_rel > 1.0e-6 || spmv_elem_err > 1.0e-12 || mf_spmv_err > 1.0e-12 || mf_atomic_err > 1.0e-12 ||
            mf_sympy_err > 1.0e-12 || rel > 1.0e-6) {
            std::fprintf(stderr, "SpMV J d vs FD residual mismatch\n");
            d.mesh.reset();
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    std::vector<scalar_t> jac_dir, jac_out;
    if (jac_action || bsr_apply) {
        jac_dir.resize((size_t)d.nnodes * N_FIELDS);
        jac_out.assign((size_t)d.nnodes * N_FIELDS, 0.0);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < d.nnodes * N_FIELDS; ++i) {
            jac_dir[(size_t)i] = 1.0 + 0.01 * scalar_t(i % 7);
        }
    }
    if (jac_action && layout == "packed") cvfem_tet4_ns_upwind_prepack_action_base_velocity(d, packed);

    auto apply = [&]() {
        if (kernel_is_sympy_residual(kernel_kind)) {
            if (layout == "atomic")
                cvfem_tet4_ns_upwind_apply_sympy_atomic(d, rho, mu);
            else
                cvfem_tet4_ns_upwind_apply_sympy_packed(d, packed, rho, mu);
        } else {
            if (layout == "atomic")
                cvfem_tet4_ns_upwind_apply_atomic(d, rho, mu);
            else
                cvfem_tet4_ns_upwind_apply_packed(d, packed, rho, mu);
        }
    };

    const bool use_sympy_action = kernel_is_sympy_residual(kernel_kind);
    auto       jac_action_fn    = [&]() {
        if (layout == "atomic")
            cvfem_tet4_ns_upwind_jacobian_action_atomic(d, rho, mu, jac_dir.data(), jac_out.data(), use_sympy_action);
        else
            cvfem_tet4_ns_upwind_jacobian_action_packed(d, packed, rho, mu, jac_dir.data(), jac_out.data(), use_sympy_action);
    };

    auto assemble_fn = [&]() {
        if (kernel_kind == KernelKind::Sympy) {
            if (layout == "atomic")
                assemble_bsr4_atomic_sympy(d, bsr, rho, mu);
            else
                assemble_bsr4_packed_sympy(d, packed, bsr, rho, mu);
        } else if (kernel_kind == KernelKind::CurrentSlots) {
            assemble_bsr4_packed_current_slots(d, packed, bsr, rho, mu);
        } else if (kernel_kind == KernelKind::SympySlots) {
            assemble_bsr4_packed_sympy_slots(d, packed, bsr, rho, mu);
        } else if (kernel_kind == KernelKind::SympyDirect) {
            assemble_bsr4_packed_sympy_direct(d, packed, bsr, rho, mu);
        } else if (kernel_kind == KernelKind::SympyBlock) {
            assemble_bsr4_packed_sympy_block(d, packed, bsr, rho, mu);
        } else if (kernel_kind == KernelKind::SympyFace) {
            assemble_bsr4_packed_sympy_face(d, packed, bsr, rho, mu);
        } else if (kernel_kind == KernelKind::SympySimd) {
            assemble_bsr4_packed_sympy_simd(d, packed, bsr, rho, mu);
        } else if (kernel_kind == KernelKind::SympySimdClean) {
            assemble_bsr4_packed_sympy_simd_clean(d, packed, bsr, rho, mu);
        } else if (kernel_kind == KernelKind::SympyBlockSimd) {
            assemble_bsr4_packed_sympy_block_simd(d, packed, bsr, rho, mu);
        } else if (kernel_kind == KernelKind::SympyRowSimd) {
            assemble_bsr4_packed_sympy_row_simd(d, packed, bsr, rho, mu);
        } else if (kernel_kind == KernelKind::SympyRowSimdFused) {
            assemble_bsr4_packed_sympy_row_simd_fused(d, packed, bsr, rho, mu);
        } else if (kernel_kind == KernelKind::SympyFaceSimd) {
            assemble_bsr4_packed_sympy_face_simd(d, packed, bsr, rho, mu);
        } else {
            if (layout == "atomic")
                assemble_bsr4_atomic(d, bsr, rho, mu);
            else
                assemble_bsr4_packed(d, packed, bsr, rho, mu);
        }
    };

    if (bsr_apply) assemble_fn();
    decltype(sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
            d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(), bsr.values, scalar_t(0))) bsr_apply_op;
    if (bsr_apply) {
        bsr_apply_op = sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
                d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(), bsr.values, scalar_t(0));
    }
    auto bsr_apply_fn = [&]() {
        bsr_apply_op->apply(jac_dir.data(), jac_out.data());
    };

    if (jac_action) {
        for (int i = 0; i < warmup; ++i) jac_action_fn();
    } else if (bsr_apply) {
        for (int i = 0; i < warmup; ++i) bsr_apply_fn();
    } else if (!assemble) {
        for (int i = 0; i < warmup; ++i) apply();
    } else {
        for (int i = 0; i < warmup; ++i) assemble_fn();
    }

    const double t0 = wall_time();
    if (jac_action) {
        for (int i = 0; i < repeat; ++i) jac_action_fn();
    } else if (bsr_apply) {
        for (int i = 0; i < repeat; ++i) bsr_apply_fn();
    } else if (!assemble) {
        for (int i = 0; i < repeat; ++i) apply();
    } else {
        for (int i = 0; i < repeat; ++i) assemble_fn();
    }
    const double t1 = wall_time();

    const double seconds          = t1 - t0;
    const double seconds_per_call = seconds / double(repeat);
    const double elem_apps        = double(d.nelements) * double(repeat);

    // Source add/mul/div in cvfem_tet4_ns_upwind_simd_microkernel.
    // Not counted: 6 abs/ternary negs, float→double casts, residual zero-stores.
    constexpr double flops_inv_det  = 1.0;
    constexpr double flops_ref_diff = 9.0;
    constexpr double flops_grad     = 9.0 * 6.0;
    constexpr double flops_area     = 3.0 * 15.0 + 3.0 * 9.0;
    constexpr double flops_scs_body = 6.0 + 6.0 + 4.0 + 2.0 + 27.0 + 18.0 + 8.0;
    constexpr double flops_per_element =
            flops_inv_det + flops_ref_diff + flops_grad + flops_area + 6.0 * flops_scs_body;
    static_assert(flops_per_element == 562.0, "element flop model drifted from kernel");
    constexpr double bytes_per_element =
            8.0 * double(sizeof(smesh::idx_t)) + 10.0 * double(sizeof(jacobian_t)) + (16.0 + 32.0) * double(sizeof(scalar_t));
    constexpr double dofs_per_element_visit = 4.0 * 4.0;

    const double melems       = double(d.nelements) / seconds_per_call / 1.0e6;
    const double mdofs        = double(d.nelements) * dofs_per_element_visit / seconds_per_call / 1.0e6;
    const double unique_mdofs = double(d.nnodes) * 4.0 / seconds_per_call / 1.0e6;
    const double gflops       = elem_apps * flops_per_element / seconds / 1.0e9;
    const double gbps         = elem_apps * bytes_per_element / seconds / 1.0e9;

    if (layout == "atomic")
        std::printf("cvfem_tet4_ns_upwind_smesh_gather_simd_atomic_scatter\n");
    else
        std::printf("cvfem_tet4_ns_upwind_smesh_packed_two_pass\n");
    std::printf("  mesh_manager: smesh::Mesh::create_tet4_cube\n");
    std::printf("  operation: %s\n", bsr_apply ? "bsr_apply" : (jac_action ? "jacobian_action" : (assemble ? "jacobian_assemble" : "residual")));
    std::printf("  kernel: %s\n", kernel.c_str());
    if (jac_action) std::printf("  jac_action_kernel: %s\n", use_sympy_action ? "sympy_face_cse" : "hand");
    std::printf("  OpenMP_threads: %d\n", threads_active());
    std::printf("  LANE_PACK_BYTES: %d\n", VEC_BYTES);
    std::printf("  LANES_PER_PACK: %d\n", VEC_SIZE);
    std::printf("  SIMD_BYTES: %d\n", CVFEM_SIMD_BYTES);
    std::printf("  SIMD_LANES: %d\n", SIMD_SIZE);
    std::printf("  ALIGN_BYTES: %d\n", ALIGN_BYTES);
    std::printf("  geom_t_bytes: %zu\n", sizeof(smesh::geom_t));
    std::printf("  jacobian_t_bytes: %zu\n", sizeof(jacobian_t));
    if (use_packed) {
        std::printf("  sfc_reorder: %d\n", use_sfc);
        std::printf("  pack_size: %d\n", pack_size);
        std::printf("  n_packs: %td\n", packed.n_packs);
        std::printf("  n_elements_per_pack: %td\n", packed.n_elements_per_pack);
        std::printf("  mean_nodes_per_pack: %td\n", packed.mean_nodes_per_pack);
        std::printf("  max_actual_nodes_per_pack: %td\n", packed.max_actual_nodes_per_pack);
        std::printf("  scratch_nodes: %td\n", packed.max_actual_nodes_per_pack);
        std::printf("  max_nodes_per_pack_capacity: %td\n", packed.max_nodes_per_pack);
    }
    std::printf("  cube_n: %td\n", n);
    std::printf("  nodes: %td\n", d.nnodes);
    std::printf("  elements: %td\n", d.nelements);
    std::printf("  repeat: %d\n", repeat);
    std::printf("  seconds_per_call: %.6e\n", seconds_per_call);
    if (!assemble && !jac_action && !bsr_apply) std::printf("  seconds_per_apply: %.6e\n", seconds_per_call);
    if (!bsr_apply) {
        std::printf("  MELEM/s: %.3f\n", melems);
        std::printf("  MDOF/s_element_visits: %.3f\n", mdofs);
    }
    std::printf("  MDOF/s_unique_mesh_dofs: %.3f\n", unique_mdofs);
    if (!bsr_apply) {
        std::printf("  GFLOP/s_model: %.3f\n", gflops);
        std::printf("  GB/s_gather_scatter_model: %.3f\n", gbps);
        std::printf("  flops_per_element_model: %.1f\n", flops_per_element);
        std::printf("  bytes_per_element_model: %.1f\n", bytes_per_element);
    }
    std::printf("  checksum: %.16e\n", (jac_action || bsr_apply) ? checksum_vec(jac_out.data(), d.nnodes) : checksum(d));
    if (assemble || verify_jac) {
        std::printf("  bsr_nnz: %td\n", bsr.nnz);
        std::printf("  flops_per_element_jacobian_model: %.1f\n", CVFEM_JACOBIAN_FLOPS_PER_ELEMENT);
        if (use_packed) std::printf("  max_local_nnz: %td\n", packed.max_local_nnz);
    }
    if (bsr_apply) {
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
        if (use_packed) std::printf("  max_local_nnz: %td\n", packed.max_local_nnz);
    }
    if (assemble) {
        const double bytes_assemble = double(bsr.nnz) * 16.0 * 8.0 + double(d.nelements) * (8.0 * double(sizeof(smesh::idx_t)) +
                                                                                           10.0 * double(sizeof(jacobian_t)) +
                                                                                           16.0 * double(sizeof(scalar_t)));
        std::printf("  seconds_per_assemble: %.6e\n", seconds_per_call);
        std::printf("  MELEM/s_assemble: %.3f\n", melems);
        std::printf("  GB/s_assemble_model: %.3f\n", double(repeat) * bytes_assemble / seconds / 1.0e9);
        std::printf("  GFLOP/s_assemble_model: %.3f\n",
                    elem_apps * CVFEM_JACOBIAN_FLOPS_PER_ELEMENT / seconds / 1.0e9);
    }
    if (jac_action) {
        constexpr double jac_action_flops_per_element = CVFEM_RESIDUAL_FLOPS_PER_ELEMENT + 4.0 * 6.0 * 8.0;
        const double     bytes_jac_action =
                8.0 * double(sizeof(smesh::idx_t)) + 10.0 * double(sizeof(jacobian_t)) + (32.0 + 32.0) * double(sizeof(scalar_t));
        std::printf("  seconds_per_jac_action: %.6e\n", seconds_per_call);
        std::printf("  MELEM/s_jac_action: %.3f\n", melems);
        std::printf("  GB/s_jac_action_model: %.3f\n", elem_apps * bytes_jac_action / seconds / 1.0e9);
        std::printf("  GFLOP/s_jac_action_model: %.3f\n", elem_apps * jac_action_flops_per_element / seconds / 1.0e9);
        std::printf("  flops_per_element_jac_action_model: %.1f\n", jac_action_flops_per_element);
        std::printf("  bytes_per_element_jac_action_model: %.1f\n", bytes_jac_action);
    }
    if (bsr_apply) {
        const double bsr_apply_flops = double(bsr.nnz) * 2.0 * 16.0;
        const double bsr_apply_bytes = double(bsr.nnz) * 16.0 * double(sizeof(scalar_t)) +
                                       double(bsr.nnz) * double(sizeof(smesh::idx_t)) +
                                       2.0 * double(d.nnodes) * N_FIELDS * double(sizeof(scalar_t));
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

