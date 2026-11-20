// The device assembly strategies of the Mooney-Rivlin Prony TET10 tangent (bench_patch_assembly_cuda.hpp).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bench_patch_assembly_cuda.hpp"
#include "mooney_rivlin_prony_d3_simplex_hessian.cuh"
#include "quad_tet_q11.cuh"
#include "sfem_cuda_CooperativeAssembly.cuh"
#include "tet10_q11.cuh"

extern "C" int cu_mooney_rivlin_prony_tet10_hessian_bsr_i_msoa(const int             scalar_bytes,
                                                               const int             values_bytes,
                                                               const ptrdiff_t       nelements,
                                                               const ptrdiff_t       nnodes,
                                                               idx_t **const         elements,
                                                               const geom_t *const *const points,
                                                               const real_t          C01,
                                                               const real_t          C10,
                                                               const real_t          K,
                                                               const real_t          iso_relaxation_gamma,
                                                               const void *const     iso_history,
                                                               const ptrdiff_t       u_stride,
                                                               const void *const     ux,
                                                               const void *const     uy,
                                                               const void *const     uz,
                                                               const count_t *const  rowptr,
                                                               const idx_t *const    colidx,
                                                               void *const           values,
                                                               void *const           stream);

namespace {
    constexpr int NS = 10, NQ = 11, NC = 3, ND = 3, NA = NC * ND;
    // The material: C01/C10 = 4 as the valves', a bulk modulus well above both, a relaxation factor
    // and a fixed history of the size of the stresses (the tangent reads both).
    constexpr double C10 = 1, C01 = 4, KB = 50, GAMMA = 0.3;

#define CHECK(x)                                                                                      \
    do {                                                                                              \
        const cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                      \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));        \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

    template <typename T>
    T *to_device(const T *host, const size_t n) {
        T *d = nullptr;
        CHECK(cudaMalloc(&d, std::max<size_t>(n, 1) * sizeof(T)));
        if (n) CHECK(cudaMemcpy(d, host, n * sizeof(T), cudaMemcpyHostToDevice));
        return d;
    }

    __global__ void init_state(const ptrdiff_t n, const geom_t *const x, const geom_t *const y, const geom_t *const z,
                               const double L, double *const u) {
        const ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x;
        if (i >= n) return;
        const double a = x[i] / L, b = y[i] / L, c = z[i] / L;
        // A stretch and a bend, small against the mesh's size: no element inverts.
        u[i * 3 + 0] = L * (0.05 * a + 0.004 * sin(6.0 * b + 2.0 * c));
        u[i * 3 + 1] = L * (0.02 * a * a + 0.004 * sin(4.0 * c + 2.0 * a));
        u[i * 3 + 2] = L * (0.004 * sin(6.0 * a + 2.0 * b));
    }

    __global__ void init_history(const ptrdiff_t n, double *const h) {
        const ptrdiff_t k = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x;
        if (k >= n) return;
        h[k] = 0.05 * C10 * sin(0.37 * (double)(k % 1000003));
    }

    // The window [r0, r1) of the graph: rows outside it empty, its values from 0.
    __global__ void window_rowptr(const ptrdiff_t n, const count_t *const rp, const ptrdiff_t r0, const ptrdiff_t r1,
                                  count_t *const out) {
        const ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x;
        if (i > n) return;
        const ptrdiff_t j = i < r0 ? r0 : (i > r1 ? r1 : i);
        out[i]            = rp[j] - rp[r0];
    }

    // Patch assembly: a thread per row i of [r0, r1). It visits the elements of its patch in the
    // map's (ascending) order; for each it recomputes the points' tangents (generated, as the
    // cooperative kernel's) and forms the row's blocks B(a, t) = sum_q g_a^T T_q g_t in double, then
    // adds them to its row, which no other thread writes.
    template <typename v_t>
    __global__ void __launch_bounds__(128) patch_rows(const ptrdiff_t r0,
                                                      const ptrdiff_t r1,
                                                      const ptrdiff_t nelements,
                                                      idx_t **const   elements,
                                                      const geom_t *const *const points,
                                                      const double *const iso_history,
                                                      const double *const u,
                                                      const count_t *const rp,
                                                      const idx_t *const   ci,
                                                      const count_t *const n2e_ptr,
                                                      const element_idx_t *const n2e_idx,
                                                      v_t *const                 values) {
        const ptrdiff_t i = r0 + blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x;
        if (i >= r1) return;
        const count_t rb = rp[i], re = rp[i + 1];
        if (rb == re) return;
        const double *const gx = sfem::codegen::ref_tet10_q11<double>::grad_ref_x();
        const double *const gy = sfem::codegen::ref_tet10_q11<double>::grad_ref_y();
        const double *const gz = sfem::codegen::ref_tet10_q11<double>::grad_ref_z();
        const double *const w  = sfem::codegen::quad_tet_q11<double>::q_weight();
        const double *const grad_ref[ND] = {gx, gy, gz};
        for (count_t p = n2e_ptr[i]; p < n2e_ptr[i + 1]; ++p) {
            const ptrdiff_t e = n2e_idx[p];
            idx_t           ev[NS];
            double          xe[ND][NS], bu[NS * NC];
            int             a = 0;
            for (int s = 0; s < NS; ++s) {
                ev[s] = elements[s][e];
                if (ev[s] == i) a = s;
                for (int d = 0; d < ND; ++d) xe[d][s] = points[d][ev[s]];
                for (int c = 0; c < NC; ++c) bu[s * NC + c] = u[(ptrdiff_t)ev[s] * 3 + c];
            }
            count_t pos[NS];
            for (int t = 0; t < NS; ++t) pos[t] = rb;
            for (count_t k = rb; k < re; ++k) {
                const idx_t c = ci[k];
                for (int t = 0; t < NS; ++t)
                    if (c == ev[t]) pos[t] = k;
            }
            double B[NS][NC * NC];
            for (int t = 0; t < NS; ++t)
                for (int k = 0; k < NC * NC; ++k) B[t][k] = 0;
            for (int q = 0; q < NQ; ++q) {
                double adj[ND * ND], det;
                sfem::cooperative::point_geometry<double, NS, ND>(xe, grad_ref, q, adj, det);
                double T[sfem::cooperative::n_packed<NA>()];
                sfem::codegen::mooney_rivlin_prony_d3_simplex_direct_hessian_reference_point_tangent<double, NQ, NS, false>(
                        q, adj, det, gx, gy, gz, w, C01, C10, KB, GAMMA, iso_history + e, nelements, bu, T);
                const double ga[ND] = {gx[q * NS + a], gy[q * NS + a], gz[q * NS + a]};
                // G[i][j][n] = sum_m g_a[m] T[(i, m), (j, n)]
                double G[NC][NC][ND];
                for (int ii = 0; ii < NC; ++ii)
                    for (int jj = 0; jj < NC; ++jj)
                        for (int n = 0; n < ND; ++n) {
                            double acc = 0;
                            for (int m = 0; m < ND; ++m) acc += ga[m] * T[sfem::cooperative::packed<NA>(ii * ND + m, jj * ND + n)];
                            G[ii][jj][n] = acc;
                        }
                for (int t = 0; t < NS; ++t) {
                    const double gt[ND] = {gx[q * NS + t], gy[q * NS + t], gz[q * NS + t]};
                    for (int ii = 0; ii < NC; ++ii)
                        for (int jj = 0; jj < NC; ++jj) B[t][ii * NC + jj] += G[ii][jj][0] * gt[0] + G[ii][jj][1] * gt[1] + G[ii][jj][2] * gt[2];
                }
            }
            for (int t = 0; t < NS; ++t) {
                v_t *const block = values + (ptrdiff_t)pos[t] * (NC * NC);
                for (int k = 0; k < NC * NC; ++k) block[k] += v_t(B[t][k]);
            }
        }
    }

    // Owner computes on the device: a block of OWNER_ROWS threads owns as many rows [b0, b1) of the
    // window, a thread each, and visits the elements of their patches, each once (at its first owned
    // node), in a fixed order: by row, then the map's order. OE of them at a time, a thread per element
    // and point forms the point's tangent in shared memory (generated, as the cooperative kernel's);
    // then the elements in turn, the block's threads form the owned rows' blocks
    // B(s, t) = sum_q g_s^T T_q g_t, an entry each, and add them to the window's values, which only
    // this block writes there: no atomics, one order of additions. An element with owned nodes in two
    // blocks is formed by both (the only repeated work; `formed` counts the elements formed).
    constexpr int OE = 16, OWNER_ROWS = 192, LIST = 4096;
    static_assert(OE * NQ <= OWNER_ROWS, "a thread per element and point");

    struct OwnerShared {
        double        T[OE][NQ][sfem::cooperative::n_packed<NA>()];
        double        xs[OE][ND][NS];
        double        bus[OE][NS * NC];
        idx_t         evs[OE][NS];
        count_t       pos[OE][NS][NS];
        unsigned      owned[OE];
        int           offset[OWNER_ROWS + 1];
        element_idx_t list[LIST];
    };

    template <typename v_t>
    __global__ void __launch_bounds__(OWNER_ROWS, 1) owner_rows(const ptrdiff_t r0,
                                                                const ptrdiff_t r1,
                                                                const ptrdiff_t nelements,
                                                                idx_t **const   elements,
                                                                const geom_t *const *const points,
                                                                const double *const iso_history,
                                                                const double *const u,
                                                                const count_t *const rp,
                                                                const idx_t *const   ci,
                                                                const count_t *const n2e_ptr,
                                                                const element_idx_t *const n2e_idx,
                                                                v_t *const                 values,
                                                                unsigned long long *const  formed) {
        extern __shared__ __align__(16) unsigned char raw[];
        OwnerShared &sh = *reinterpret_cast<OwnerShared *>(raw);
        const ptrdiff_t b0 = r0 + (ptrdiff_t)blockIdx.x * OWNER_ROWS;
        const ptrdiff_t b1 = b0 + OWNER_ROWS < r1 ? b0 + OWNER_ROWS : r1;
        const double *const gx = sfem::codegen::ref_tet10_q11<double>::grad_ref_x();
        const double *const gy = sfem::codegen::ref_tet10_q11<double>::grad_ref_y();
        const double *const gz = sfem::codegen::ref_tet10_q11<double>::grad_ref_z();
        const double *const w  = sfem::codegen::quad_tet_q11<double>::q_weight();
        const double *const grad_ref[ND] = {gx, gy, gz};
        auto is_owned = [&](const ptrdiff_t n) { return n >= b0 && n < b1 && rp[n] < rp[n + 1]; };
        auto first_at = [&](const ptrdiff_t e, const ptrdiff_t i) {
            for (int s = 0; s < NS; ++s) {
                const ptrdiff_t n = elements[s][e];
                if (n < i && is_owned(n)) return false;
            }
            return true;
        };

        // The block's elements: thread l counts those its row i = b0 + l is first in, a scan places
        // them, in the order of the rows and of each row's map.
        const int       l = threadIdx.x;
        const ptrdiff_t i = b0 + l;
        int             mine = 0;
        if (i < b1 && is_owned(i))
            for (count_t p = n2e_ptr[i]; p < n2e_ptr[i + 1]; ++p) mine += first_at(n2e_idx[p], i);
        sh.offset[l + 1] = mine;
        if (l == 0) sh.offset[0] = 0;
        __syncthreads();
        if (l == 0)
            for (int k = 1; k <= OWNER_ROWS; ++k) sh.offset[k] += sh.offset[k - 1];
        __syncthreads();
        const int total = sh.offset[OWNER_ROWS];
        if (total > LIST) {
            if (l == 0) printf("owner_rows: %d elements in a block, more than %d\n", total, LIST);
            __trap();
        }
        if (mine) {
            int at = sh.offset[l];
            for (count_t p = n2e_ptr[i]; p < n2e_ptr[i + 1]; ++p)
                if (first_at(n2e_idx[p], i)) sh.list[at++] = n2e_idx[p];
        }
        if (l == 0 && formed) atomicAdd(formed, (unsigned long long)total);
        __syncthreads();

        for (int g = 0; g < total; g += OE) {
            const int count = total - g < OE ? total - g : OE;
            for (int task = l; task < count * NS; task += blockDim.x) {
                const int       le = task / NS, s = task % NS;
                const ptrdiff_t e  = sh.list[g + le];
                const idx_t     n  = elements[s][e];
                sh.evs[le][s]      = n;
                for (int d = 0; d < ND; ++d) sh.xs[le][d][s] = points[d][n];
                for (int c = 0; c < NC; ++c) sh.bus[le][s * NC + c] = u[(ptrdiff_t)n * 3 + c];
            }
            __syncthreads();
            for (int le = l; le < count; le += blockDim.x) {
                unsigned bits = 0;
                for (int s = 0; s < NS; ++s) bits |= is_owned(sh.evs[le][s]) ? 1u << s : 0u;
                sh.owned[le] = bits;
            }
            for (int task = l; task < count * NS; task += blockDim.x) {
                const int le = task / NS, s = task % NS;
                if (is_owned(sh.evs[le][s])) sfem::cooperative::row_map<NS>(sh.evs[le], s, rp, ci, sh.pos[le]);
            }
            if (l < count * NQ) {
                const int le = l / NQ, q = l % NQ;
                double    adj[ND * ND], det;
                sfem::cooperative::point_geometry<double, NS, ND>(sh.xs[le], grad_ref, q, adj, det);
                sfem::codegen::mooney_rivlin_prony_d3_simplex_direct_hessian_reference_point_tangent<double, NQ, NS, false>(
                        q, adj, det, gx, gy, gz, w, C01, C10, KB, GAMMA, iso_history + sh.list[g + le], nelements, sh.bus[le], sh.T[le][q]);
            }
            __syncthreads();
            for (int le = 0; le < count; ++le) {
                const unsigned bits = sh.owned[le];
                for (int task = l; task < NS * NS * NC * NC; task += blockDim.x) {
                    const int s = task / (NS * NC * NC), rest = task % (NS * NC * NC), t = rest / (NC * NC), k = rest % (NC * NC);
                    if (!(bits >> s & 1u)) continue;
                    const int ii = k / NC, jj = k % NC;
                    double    acc = 0;
                    for (int q = 0; q < NQ; ++q) {
                        const double *const Tq = sh.T[le][q];
                        for (int m = 0; m < ND; ++m) {
                            const double gs = grad_ref[m][q * NS + s];
                            for (int n = 0; n < ND; ++n)
                                acc += gs * Tq[sfem::cooperative::packed<NA>(ii * ND + m, jj * ND + n)] * grad_ref[n][q * NS + t];
                        }
                    }
                    values[(ptrdiff_t)sh.pos[le][s][t] * (NC * NC) + k] += v_t(acc);
                }
                __syncthreads();
            }
        }
    }

    uint64_t digest(const void *data, const size_t bytes) {
        const unsigned char *p = (const unsigned char *)data;
        uint64_t             h = 1469598103934665603ull;
        for (size_t k = 0; k < bytes; ++k) h = (h ^ p[k]) * 1099511628211ull;
        return h;
    }

    struct Device {
        ptrdiff_t      ne{0}, n{0};
        idx_t        **elements{nullptr};  // device array of device arrays
        const geom_t *const *points{nullptr};
        double        *history{nullptr}, *u{nullptr};
        count_t       *rowptr{nullptr}, *n2e_ptr{nullptr}, *wrp{nullptr};
        idx_t         *colidx{nullptr};
        element_idx_t *n2e_idx{nullptr};
        std::vector<count_t> h_rowptr;
    };

    float elapsed(cudaEvent_t a, cudaEvent_t b) {
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }

    enum Strategy { COOP, PATCH, OWNER };
    unsigned long long *g_formed = nullptr;

    // One assembly of rows [r0, r1) into `values` (zeroed here) by `s`.
    template <typename v_t>
    void assemble(const Device &d, const Strategy s, const ptrdiff_t r0, const ptrdiff_t r1, v_t *const values) {
        const ptrdiff_t nnz = d.h_rowptr[r1] - d.h_rowptr[r0];
        window_rowptr<<<(unsigned)((d.n + 1 + 255) / 256), 256>>>(d.n, d.rowptr, r0, r1, d.wrp);
        CHECK(cudaMemsetAsync(values, 0, nnz * 9 * sizeof(v_t)));
        const idx_t *const ci = d.colidx + d.h_rowptr[r0];
        if (s == COOP) {
            const int status = cu_mooney_rivlin_prony_tet10_hessian_bsr_i_msoa(
                    8, (int)sizeof(v_t), d.ne, d.n, d.elements, d.points, C01, C10, KB, GAMMA, d.history, 3, d.u, d.u + 1, d.u + 2, d.wrp, ci,
                    values, nullptr);
            if (status != SFEM_SUCCESS) {
                fprintf(stderr, "cooperative assembly failed\n");
                exit(EXIT_FAILURE);
            }
        } else if (s == OWNER) {
            const unsigned blocks = (unsigned)((r1 - r0 + OWNER_ROWS - 1) / OWNER_ROWS);
            owner_rows<v_t><<<blocks, OWNER_ROWS, sizeof(OwnerShared)>>>(r0, r1, d.ne, d.elements, d.points, d.history, d.u, d.wrp, ci,
                                                                         d.n2e_ptr, d.n2e_idx, values, g_formed);
            CHECK(cudaGetLastError());
        } else {
            const int threads = 128;
            patch_rows<v_t><<<(unsigned)((r1 - r0 + threads - 1) / threads), threads>>>(
                    r0, r1, d.ne, d.elements, d.points, d.history, d.u, d.wrp, ci, d.n2e_ptr, d.n2e_idx, values);
            CHECK(cudaGetLastError());
        }
    }

    // All windows in turn into one buffer of the largest window's size: the best time over `repeats`,
    // and the last window's values compared against the reference's rows.
    template <typename v_t>
    void row(const Device &d, const char *name, const Strategy s, const std::vector<ptrdiff_t> &bounds, const int repeats,
             const std::vector<double> &reference) {
        const int W   = (int)bounds.size() - 1;
        ptrdiff_t big = 0;
        for (int k = 0; k < W; ++k) big = std::max<ptrdiff_t>(big, d.h_rowptr[bounds[k + 1]] - d.h_rowptr[bounds[k]]);
        v_t *values = nullptr;
        CHECK(cudaMalloc(&values, big * 9 * sizeof(v_t)));
        cudaEvent_t a, b;
        CHECK(cudaEventCreate(&a));
        CHECK(cudaEventCreate(&b));
        float  best = 1e30f;
        double diff = 0, top = 0;
        std::vector<uint64_t> digests;
        for (int r = 0; r < repeats; ++r) {
            CHECK(cudaEventRecord(a));
            uint64_t h = 1469598103934665603ull;
            for (int k = 0; k < W; ++k) assemble<v_t>(d, s, bounds[k], bounds[k + 1], values);
            CHECK(cudaEventRecord(b));
            CHECK(cudaEventSynchronize(b));
            best = std::min(best, elapsed(a, b));
            // The bits of every window (assembled again, off the clock) for the digest and the error.
            if (r == 0 || r == repeats - 1) {
                for (int k = 0; k < W; ++k) {
                    assemble<v_t>(d, s, bounds[k], bounds[k + 1], values);
                    const ptrdiff_t   o = d.h_rowptr[bounds[k]], nnz = d.h_rowptr[bounds[k + 1]] - o;
                    std::vector<v_t>  host(nnz * 9);
                    CHECK(cudaMemcpy(host.data(), values, host.size() * sizeof(v_t), cudaMemcpyDeviceToHost));
                    h ^= digest(host.data(), host.size() * sizeof(v_t)) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
                    if (r == 0 && !reference.empty()) {
                        for (ptrdiff_t j = 0; j < nnz * 9; ++j) {
                            diff = std::max(diff, std::fabs((double)host[j] - reference[o * 9 + j]));
                            top  = std::max(top, std::fabs(reference[o * 9 + j]));
                        }
                    }
                }
                digests.push_back(h);
            }
        }
        const bool same = digests.size() < 2 || digests.front() == digests.back();
        printf("%-10s %-4s %3d windows %10.3f ms  scratch %8.1f MB  rel max diff %9.2e  repeatable %s  %016llx\n", name,
               sizeof(v_t) == 8 ? "f64" : "f32", W, best, 1e-6 * big * 9 * sizeof(v_t), top > 0 ? diff / top : 0.0, same ? "yes" : "no ",
               (unsigned long long)digests.back());
        CHECK(cudaFree(values));
        CHECK(cudaEventDestroy(a));
        CHECK(cudaEventDestroy(b));
    }
}  // namespace

int run_patch_bench(const PatchBenchInput &in) {
    Device d;
    d.ne = in.nelements;
    d.n  = in.nnodes;
    d.h_rowptr.assign(in.rowptr, in.rowptr + in.nnodes + 1);
    std::vector<idx_t *> eptr(NS);
    for (int s = 0; s < NS; ++s) eptr[s] = to_device(in.elements[s], d.ne);
    d.elements = to_device(eptr.data(), NS);
    std::vector<const geom_t *> pptr(3);
    double                      L = 0;
    for (int k = 0; k < 3; ++k) {
        pptr[k]                     = to_device(in.points[k], d.n);
        const auto mm               = std::minmax_element(in.points[k], in.points[k] + d.n);
        L                           = std::max(L, (double)(*mm.second - *mm.first));
    }
    d.points  = to_device(pptr.data(), 3);
    d.rowptr  = to_device(in.rowptr, d.n + 1);
    d.colidx  = to_device(in.colidx, (size_t)in.rowptr[d.n]);
    d.n2e_ptr = to_device(in.n2e_ptr, d.n + 1);
    d.n2e_idx = to_device(in.n2e_idx, (size_t)in.n2e_ptr[d.n]);
    CHECK(cudaMalloc(&d.wrp, (d.n + 1) * sizeof(count_t)));
    CHECK(cudaMalloc(&d.u, d.n * 3 * sizeof(double)));
    CHECK(cudaMalloc(&d.history, (size_t)6 * NQ * d.ne * sizeof(double)));
    init_state<<<(unsigned)((d.n + 255) / 256), 256>>>(d.n, pptr[0], pptr[1], pptr[2], L, d.u);
    init_history<<<(unsigned)((6 * NQ * d.ne + 255) / 256), 256>>>(6 * NQ * d.ne, d.history);
    CHECK(cudaDeviceSynchronize());

    cudaFuncAttributes attr;
    CHECK(cudaFuncGetAttributes(&attr, patch_rows<double>));
    printf("patch kernel: %d registers, %zu B local memory per thread; n2e map %.1f MB\n", attr.numRegs, attr.localSizeBytes,
           1e-6 * ((d.n + 1) * sizeof(count_t) + (size_t)in.n2e_ptr[d.n] * sizeof(element_idx_t)));
    CHECK(cudaMalloc(&g_formed, sizeof(unsigned long long)));
    CHECK(cudaMemset(g_formed, 0, sizeof(unsigned long long)));
    CHECK(cudaFuncSetAttribute(owner_rows<double>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(OwnerShared)));
    CHECK(cudaFuncSetAttribute(owner_rows<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(OwnerShared)));
    CHECK(cudaFuncGetAttributes(&attr, owner_rows<double>));
    printf("owner kernel: %d registers, %zu B local memory per thread, %zu B dynamic shared\n", attr.numRegs, attr.localSizeBytes, sizeof(OwnerShared));

    // The reference: the cooperative kernel into f64 values, all rows.
    const ptrdiff_t nnz = in.rowptr[d.n];
    std::vector<double> reference(nnz * 9);
    {
        double *values = nullptr;
        CHECK(cudaMalloc(&values, nnz * 9 * sizeof(double)));
        assemble<double>(d, COOP, 0, d.n, values);
        CHECK(cudaMemcpy(reference.data(), values, reference.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(values));
    }

    for (const int W : {1, in.windows, 2 * in.windows}) {
        // Windows of about equal blocks, as the stored tangent's update cuts them.
        std::vector<ptrdiff_t> bounds{0};
        for (int k = 1; k < W; ++k)
            bounds.push_back(std::lower_bound(in.rowptr, in.rowptr + d.n, (count_t)((double)nnz * k / W)) - in.rowptr);
        bounds.push_back(d.n);
        row<double>(d, "coop", COOP, bounds, in.repeats, reference);
        row<float>(d, "coop", COOP, bounds, in.repeats, reference);
        if (in.patch) {
            row<double>(d, "patch", PATCH, bounds, in.repeats, reference);
            row<float>(d, "patch", PATCH, bounds, in.repeats, reference);
        }
        row<double>(d, "owner", OWNER, bounds, in.repeats, reference);
        row<float>(d, "owner", OWNER, bounds, in.repeats, reference);
        {
            // Elements formed per element of the mesh, all windows once (owner: the repeated work).
            CHECK(cudaMemset(g_formed, 0, sizeof(unsigned long long)));
            double *values = nullptr;
            ptrdiff_t big = 0;
            for (int k = 0; k < W; ++k) big = std::max<ptrdiff_t>(big, in.rowptr[bounds[k + 1]] - in.rowptr[bounds[k]]);
            CHECK(cudaMalloc(&values, big * 9 * sizeof(double)));
            for (int k = 0; k < W; ++k) assemble<double>(d, OWNER, bounds[k], bounds[k + 1], values);
            unsigned long long n = 0;
            CHECK(cudaMemcpy(&n, g_formed, sizeof(n), cudaMemcpyDeviceToHost));
            CHECK(cudaFree(values));
            printf("owner      %3d windows: %.3f elements formed per element\n", W, (double)n / d.ne);
            CHECK(cudaMemset(g_formed, 0, sizeof(unsigned long long)));
        }
    }
    return EXIT_SUCCESS;
}
