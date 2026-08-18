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
static constexpr int ALIGN_BYTES = 64;

#include "cvfem_tet4_ns_upwind_kernels.hpp"

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
    ptrdiff_t                                      max_local_nnz{0};
    std::vector<ptrdiff_t>                         ghost_mat_ptr;
    std::vector<smesh::idx_t>                      ghost_mat_col;
    std::vector<scalar_t>                          ghost_mat_val;
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

static void build_pack_local_crs(PackedData &p, const ptrdiff_t nelements) {
    p.local_rowptr.resize((size_t)p.n_packs);
    p.local_colidx.resize((size_t)p.n_packs);
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
        p.max_local_nnz = std::max(p.max_local_nnz, (ptrdiff_t)colidx.size());
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
            for (int t = 0; t < end - begin; ++t) {
                p.ghost_mat_col[(size_t)dest + (size_t)t] = pack_local_to_global(p, pack, n_contiguous, colidx[(size_t)begin + t]);
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

static void zero_bsr4(BSR4 &b) {
    scalar_t *const v = b.values->data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < b.nnz * 16; ++i) v[i] = 0.0;
}

static SFEM_INLINE void bsr4_add_block(const smesh::count_t *const SFEM_RESTRICT rowptr,
                                       const smesh::idx_t *const SFEM_RESTRICT   colidx,
                                       scalar_t *const SFEM_RESTRICT             values,
                                       const smesh::idx_t                        row,
                                       const smesh::idx_t                        col,
                                       const scalar_t *const SFEM_RESTRICT       block,
                                       const bool                                use_atomic) {
    const int              len  = int(rowptr[row + 1] - rowptr[row]);
    const smesh::idx_t     ks   = cvfem_linear_search(col, &colidx[rowptr[row]], len);
    scalar_t *const SFEM_RESTRICT dst = &values[(rowptr[row] + ks) * 16];
    if (use_atomic) {
        for (int t = 0; t < 16; ++t) {
#pragma omp atomic update
            dst[t] += block[t];
        }
    } else {
        for (int t = 0; t < 16; ++t) dst[t] += block[t];
    }
}



static SFEM_INLINE size_t packed_scratch_n(const PackedData &p) {
    const ptrdiff_t n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
    return (size_t)N_FIELDS * (size_t)n;
}

static SFEM_INLINE void atomic_add(scalar_t *const SFEM_RESTRICT f, const smesh::idx_t id, const scalar_t value) {
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

static SFEM_INLINE void run_jacobian_kernel(const MeshData      &d,
                                            const scalar_t       rho,
                                            const scalar_t       mu,
                                            const ptrdiff_t      begin,
                                            const int            nlanes,
                                            const Tet4InputPack &in,
                                            scalar_t             Ke[16][16][VEC_SIZE]) {
    cvfem_run_jacobian_kernel(rho,
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
                              Ke);
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


static SFEM_NOINLINE void assemble_bsr4_atomic(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    const ptrdiff_t                    ne     = d.nelements;
    smesh::idx_t **const SFEM_RESTRICT elems  = d.elems;
    scalar_t *const SFEM_RESTRICT      values = b.values->data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t begin = 0; begin < ne; begin += VEC_SIZE) {
        const int        nlanes = int(std::min<ptrdiff_t>(ne - begin, VEC_SIZE));
        Tet4InputPack    in;
        alignas(ALIGN_BYTES) scalar_t Ke[16][16][VEC_SIZE];
        gather_tet4_pack_global(d, begin, nlanes, in);
        run_jacobian_kernel(d, rho, mu, begin, nlanes, in, Ke);
        for (int lane = 0; lane < nlanes; ++lane) {
            const ptrdiff_t e = begin + lane;
            const smesh::idx_t ev[4] = {elems[0][e], elems[1][e], elems[2][e], elems[3][e]};
            scalar_t ke[16 * 16];
            cvfem_extract_ke_lane(Ke, lane, ke);
            tet4_local_to_global_bsr4<true>(ev, ke, b.rowptr, b.colidx, values);
        }
    }
}

static SFEM_NOINLINE void assemble_bsr4_packed(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    std::fill(p.ghost_mat_val.begin(), p.ghost_mat_val.end(), 0.0);

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

            for (ptrdiff_t begin = e_start; begin < e_end; begin += VEC_SIZE) {
                const int        nlanes = int(MIN((ptrdiff_t)VEC_SIZE, e_end - begin));
                Tet4InputPack    in;
                alignas(ALIGN_BYTES) scalar_t Ke[16][16][VEC_SIZE];
                gather_tet4_pack_local(p.elems, pack_u, begin, nlanes, in);
                run_jacobian_kernel(d, rho, mu, begin, nlanes, in, Ke);
                for (int lane = 0; lane < nlanes; ++lane) {
                    const ptrdiff_t e = begin + lane;
                    const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                    scalar_t ke[16 * 16];
                    cvfem_extract_ke_lane(Ke, lane, ke);
                    tet4_local_to_global_bsr4<false>(ev, ke, lrowptr.data(), lcolidx.data(), local_vals);
                }
            }

            scalar_t *const SFEM_RESTRICT gvalues = b.values->data();
            for (ptrdiff_t i = 0; i < n_contiguous; ++i) {
                const smesh::idx_t grow  = smesh::idx_t(owned + i);
                const int          begin = lrowptr[(size_t)i];
                const int          end   = lrowptr[(size_t)i + 1];
                for (int t = begin; t < end; ++t) {
                    const smesh::idx_t gcol = pack_local_to_global(p, pack, n_contiguous, lcolidx[(size_t)t]);
                    bsr4_add_block(b.rowptr, b.colidx, gvalues, grow, gcol, local_vals + (ptrdiff_t)t * 16, false);
                }
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
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t ghost_entry = p.ghost_reduce_idx[j];
            const ptrdiff_t k0          = p.ghost_mat_ptr[(size_t)ghost_entry];
            const ptrdiff_t k1          = p.ghost_mat_ptr[(size_t)ghost_entry + 1];
            for (ptrdiff_t t = k0; t < k1; ++t) {
                bsr4_add_block(b.rowptr, b.colidx, gvalues, dest, p.ghost_mat_col[(size_t)t], p.ghost_mat_val.data() + t * 16, false);
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
    for (ptrdiff_t begin = 0; begin < d.nelements; begin += VEC_SIZE) {
        const int     nlanes = int(std::min<ptrdiff_t>(d.nelements - begin, VEC_SIZE));
        Tet4InputPack in;
        alignas(ALIGN_BYTES) scalar_t Ke[16][16][VEC_SIZE];
        gather_tet4_pack_global(d, begin, nlanes, in);
        run_jacobian_kernel(d, rho, mu, begin, nlanes, in, Ke);
        for (int lane = 0; lane < nlanes; ++lane) {
            const ptrdiff_t    e     = begin + lane;
            const smesh::idx_t ev[4] = {elems[0][e], elems[1][e], elems[2][e], elems[3][e]};
            scalar_t           ke[16 * 16];
            cvfem_extract_ke_lane(Ke, lane, ke);
            scalar_t loc[16];
            for (int a = 0; a < 4; ++a)
                for (int f = 0; f < 4; ++f) loc[a * 4 + f] = dir[(ptrdiff_t)ev[a] * 4 + f];
            for (int r = 0; r < 16; ++r) {
                scalar_t acc = 0.0;
                for (int c = 0; c < 16; ++c) acc += ke[r * 16 + c] * loc[c];
                jv[(ptrdiff_t)ev[r / 4] * 4 + (r % 4)] += acc;
            }
        }
    }
}

static scalar_t verify_element_kernel_jac(MeshData &d, const scalar_t rho, const scalar_t mu) {
    Tet4InputPack in;
    gather_tet4_pack_global(d, 0, 1, in);
    alignas(ALIGN_BYTES) scalar_t Ke[16][16][VEC_SIZE];
    run_jacobian_kernel(d, rho, mu, 0, 1, in, Ke);
    scalar_t ke[16 * 16];
    cvfem_extract_ke_lane(Ke, 0, ke);

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
    int         verify    = 0;
    int         verify_jac = 0;
    int         assemble  = 0;
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
        else if (arg == "--help") {
            std::printf(
                    "usage: %s [--n cube_cells_per_dim] [--repeat N] [--warmup N]\n"
                    "          [--layout packed|atomic] [--pack-size N] [--no-sfc]\n"
                    "          [--verify] [--verify-jac] [--assemble]\n"
                    "  --pack-size N   elements per pack (0 = fill uint16; default 2048)\n",
                    argv[0]);
            if (own_mpi) MPI_Finalize();
            return 0;
        }
    }

    const int use_packed = (layout == "packed") || verify || verify_jac;

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
    if (assemble || verify_jac) bsr = make_bsr4(d.mesh);
    if (use_packed && (assemble || verify_jac)) build_pack_local_crs(packed, d.nelements);

    if (use_packed) {
        const size_t scratch_n = packed_scratch_n(packed);
        const size_t bsr_n     = 16 * (size_t)std::max<ptrdiff_t>(packed.max_local_nnz, 1);
#pragma omp parallel
        {
            (void)thread_scratch<scalar_t>(0, scratch_n);
            (void)thread_scratch<scalar_t>(1, scratch_n);
            if (assemble || verify_jac) (void)thread_scratch<scalar_t>(2, bsr_n);
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
    }

    if (verify_jac) {
        assemble_bsr4_packed(d, packed, bsr, rho, mu);
        std::vector<scalar_t> packed_vals(bsr.values->data(), bsr.values->data() + bsr.nnz * 16);
        assemble_bsr4_atomic(d, bsr, rho, mu);
        const scalar_t jac_err = max_abs_diff_ptr(packed_vals.data(), bsr.values->data(), bsr.nnz * 16);
        std::printf("verify_jac_packed_vs_atomic_max_abs: %.6e\n", jac_err);
        if (jac_err > 1.0e-12) {
            std::fprintf(stderr, "packed vs atomic BSR Jacobian mismatch\n");
            d.mesh.reset();
            if (own_mpi) MPI_Finalize();
            return 1;
        }

        const scalar_t elem_rel = verify_element_kernel_jac(d, rho, mu);

        std::vector<scalar_t> dir((size_t)d.nnodes * 4), jv((size_t)d.nnodes * 4, 0.0), jv_e((size_t)d.nnodes * 4, 0.0);
        for (ptrdiff_t i = 0; i < d.nnodes * 4; ++i) dir[(size_t)i] = 1.0 + 0.01 * scalar_t(i % 7);
        apply_ke_to_dir(d, rho, mu, dir.data(), jv_e.data());
        auto spmv = sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
                d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(), bsr.values, scalar_t(0));
        spmv->apply(dir.data(), jv.data());
        const scalar_t spmv_elem_err = max_abs_diff_ptr(jv.data(), jv_e.data(), d.nnodes * 4);
        std::printf("verify_jac_spmv_vs_elem_abs: %.6e\n", spmv_elem_err);

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
        if (elem_rel > 1.0e-6 || spmv_elem_err > 1.0e-12 || rel > 1.0e-6) {
            std::fprintf(stderr, "SpMV J d vs FD residual mismatch\n");
            d.mesh.reset();
            if (own_mpi) MPI_Finalize();
            return 1;
        }
    }

    auto apply = [&]() {
        if (layout == "atomic")
            cvfem_tet4_ns_upwind_apply_atomic(d, rho, mu);
        else
            cvfem_tet4_ns_upwind_apply_packed(d, packed, rho, mu);
    };

    auto assemble_fn = [&]() {
        if (layout == "atomic")
            assemble_bsr4_atomic(d, bsr, rho, mu);
        else
            assemble_bsr4_packed(d, packed, bsr, rho, mu);
    };

    if (!assemble) {
        for (int i = 0; i < warmup; ++i) apply();
    } else {
        for (int i = 0; i < warmup; ++i) assemble_fn();
    }

    const double t0 = wall_time();
    if (!assemble) {
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
    std::printf("  OpenMP_threads: %d\n", threads_active());
    std::printf("  LANE_PACK_BYTES: %d\n", VEC_BYTES);
    std::printf("  LANES_PER_PACK: %d\n", VEC_SIZE);
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
    std::printf("  seconds_per_apply: %.6e\n", seconds_per_call);
    std::printf("  MELEM/s: %.3f\n", melems);
    std::printf("  MDOF/s_element_visits: %.3f\n", mdofs);
    std::printf("  MDOF/s_unique_mesh_dofs: %.3f\n", unique_mdofs);
    std::printf("  GFLOP/s_model: %.3f\n", gflops);
    std::printf("  GB/s_gather_scatter_model: %.3f\n", gbps);
    std::printf("  flops_per_element_model: %.1f\n", flops_per_element);
    std::printf("  bytes_per_element_model: %.1f\n", bytes_per_element);
    std::printf("  checksum: %.16e\n", checksum(d));
    if (assemble || verify_jac) {
        std::printf("  bsr_nnz: %td\n", bsr.nnz);
        std::printf("  flops_per_element_jacobian_model: %.1f\n", CVFEM_JACOBIAN_FLOPS_PER_ELEMENT);
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

    d.mesh.reset();
    if (own_mpi) MPI_Finalize();
    return 0;
}

