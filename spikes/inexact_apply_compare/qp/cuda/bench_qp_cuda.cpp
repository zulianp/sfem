// The compressed formats on one GPU against BSR: bench_qp's problem (bench_problem.inc), its
// state, increment and reference, with every operator on the device.
//
//   bench_qp_cuda [n] [repeats]
//
// Rows as bench_qp's (set-up ms, MDOF/s, B/dof, rel l2, rel energy).  BSR f64 and f32 are the
// library's device assembly (cu_neohookean_ogden_hessian_bsr_3d_*) and product
// (sfem::d_bsr_spmv_native); the quantized BSR encodes the f32 assembly; the stores are
// cuda/qp_device.cu's.  Each store row is followed by its set-up gate: the device store's arrays
// against the host set-up's (LANE_VS = 32, the device's block): the codes that differ and by how
// much, and the largest relative difference of the scales and of the means.  Timings are host
// clocks around calls that end in a stream synchronisation; set-up is the best of three updates,
// application the best of three runs of `repeats` calls.  QP_REPEAT=k runs the store rows k times
// (the spread between them is the measurement's).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "sfem_base.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#include "kernel_math.hpp"
#include "qp_setup.hpp"
#include "qp_apply.hpp"

#include "generated_abi.inc"
#include "element_mesh.inc"
#include "packed_mesh.inc"
#include "bsr_matrix.inc"
#include "bench_problem.inc"

#include "sfem_BSR_cuda.hpp"
#include "sfem_GeneratedNeoHookeanOgden_cuda_c_abi.hpp"

#include "qp_device.hpp"

#if defined(ELEMENT_PROTEUS_HEX27)
static const smesh::ElemType ELEMENT_TYPE = smesh::PROTEUS_HEX27;
#elif defined(ELEMENT_HEX8)
static const smesh::ElemType ELEMENT_TYPE = smesh::HEX8;
#elif defined(ELEMENT_TET10)
static const smesh::ElemType ELEMENT_TYPE = smesh::TET10;
#else
static const smesh::ElemType ELEMENT_TYPE = smesh::TET4;
#endif

#define CHECK_CUDA(x)                                                                   \
    do {                                                                                \
        const cudaError_t e_ = (x);                                                     \
        if (e_ != cudaSuccess) {                                                        \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

// Device copies, freed at exit.
struct Dev {
    std::vector<void *> ptrs;
    template <typename T>
    T *put(const T *src, const size_t n) {
        void *p = nullptr;
        CHECK_CUDA(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
        if (n) CHECK_CUDA(cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice));
        ptrs.push_back(p);
        return (T *)p;
    }
    template <typename T>
    T *put(const std::vector<T> &v) {
        return put(v.data(), v.size());
    }
    template <typename T>
    T *zeros(const size_t n) {
        void *p = nullptr;
        CHECK_CUDA(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
        CHECK_CUDA(cudaMemset(p, 0, std::max<size_t>(1, n) * sizeof(T)));
        ptrs.push_back(p);
        return (T *)p;
    }
    ~Dev() {
        for (void *p : ptrs) cudaFree(p);
    }
};

// What the device rows need of the problem.
struct DeviceProblem {
    Dev d;
    ptrdiff_t nnodes = 0;
    double *u = nullptr, *h = nullptr, *out = nullptr;  // interleaved
    qpd::Geometry g, gb;  // gb: with the block-local scatter
    std::vector<int32_t> h_kel;
    std::shared_ptr<void> blocks;
    // The library's assembly: the mesh-order connectivity and the points, as device pointer arrays.
    idx_t **elements = nullptr;
    const geom_t *const *points = nullptr;
    const geom_t *adj[9] = {}, *det = nullptr;
    count_t *rowptr = nullptr;
    idx_t *colidx = nullptr;
    BsrGraph graph;
};

static std::vector<double> interleave(const std::vector<double> &x, const std::vector<double> &y,
                                      const std::vector<double> &z) {
    std::vector<double> v(x.size() * 3);
    for (size_t i = 0; i < x.size(); ++i) {
        v[i * 3 + 0] = x[i];
        v[i * 3 + 1] = y[i];
        v[i * 3 + 2] = z[i];
    }
    return v;
}

static void upload(Problem &P, DeviceProblem &D) {
    Mesh &m = P.m;
    D.nnodes = m.nnodes;
    D.u = D.d.put(interleave(P.ux, P.uy, P.uz));
    D.h = D.d.put(interleave(P.hx, P.hy, P.hz));
    D.out = D.d.zeros<double>((size_t)m.nnodes * 3);
    // Kernel order, [node][element].
    std::vector<int32_t> kel((size_t)NXE * m.nelements);
    for (int p = 0; p < NXE; ++p) std::copy(m.kevp[p], m.kevp[p] + m.nelements, kel.begin() + (size_t)p * m.nelements);
    D.g.nelements = m.nelements;
    D.g.elements = D.d.put(kel);
    D.h_kel = kel;
    std::vector<float> adj((size_t)9 * m.nelements);
    for (int c = 0; c < 9; ++c) std::copy(m.adj[c].begin(), m.adj[c].end(), adj.begin() + (size_t)c * m.nelements);
    D.g.adj = D.d.put(adj);
    D.g.det = D.d.put(m.det);
    D.g.mu = P.mu;
    D.g.lmbda = P.lmbda;
    D.gb = D.g;
    D.blocks = qpd::attach_blocks(D.gb, D.h_kel.data());
    for (int c = 0; c < 9; ++c) D.adj[c] = D.g.adj + (size_t)c * m.nelements;
    D.det = D.g.det;
    // Mesh order for the library's assembly.
    std::vector<idx_t *> eptr(NXE);
    for (int p = 0; p < NXE; ++p) eptr[p] = D.d.put(m.evp[p], (size_t)m.nelements);
    D.elements = D.d.put(eptr);
    std::vector<const geom_t *> pptr = {D.d.put(P.gpx), D.d.put(P.gpy), D.d.put(P.gpz)};
    D.points = D.d.put(pptr);
    D.graph = build_bsr_graph(m);
    D.rowptr = D.d.put(D.graph.rowptr);
    D.colidx = D.d.put(D.graph.colidx);
}

// The library's device assembly of the material's tangent at x into values of type V (zeroed first).
template <typename V>
static int assemble(const Problem &P, const DeviceProblem &D, const double *x, V *values) {
    CHECK_CUDA(cudaMemset(values, 0, (size_t)D.graph.nnz() * 9 * sizeof(V)));
    const smesh::PrimitiveType vt = sizeof(V) == 8 ? smesh::SMESH_FLOAT64 : smesh::SMESH_FLOAT32;
    int status;
    if (ELEMENT_TYPE == smesh::TET4) {
        status = cu_neohookean_ogden_hessian_bsr_3d_a_msoa(
                ELEMENT_TYPE, smesh::SMESH_FLOAT64, vt, P.m.nelements, P.m.nnodes, D.elements, D.adj[0], D.adj[1],
                D.adj[2], D.adj[3], D.adj[4], D.adj[5], D.adj[6], D.adj[7], D.adj[8], D.det, P.lmbda, P.mu, 3, x, x + 1,
                x + 2, D.rowptr, D.colidx, values, nullptr);
    } else {
        status = cu_neohookean_ogden_hessian_bsr_3d_i_msoa(ELEMENT_TYPE, smesh::SMESH_FLOAT64, vt, P.m.nelements,
                                                            P.m.nnodes, D.elements, D.points, P.lmbda, P.mu, 3, x,
                                                            x + 1, x + 2, D.rowptr, D.colidx, values, nullptr);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return status;
}

// The device product out (interleaved) against the reference.
static void device_errors(const Problem &P, const double *dout, Row &row) {
    const size_t n = (size_t)P.m.nnodes;
    std::vector<double> y(n * 3), a(n), b(n), c(n);
    CHECK_CUDA(cudaMemcpy(y.data(), dout, y.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        a[i] = y[i * 3 + 0];
        b[i] = y[i * 3 + 1];
        c[i] = y[i * 3 + 2];
    }
    errors(P, a, b, c, row);
}

// A failing call inside a timed region would time as a fast one: stop instead.
static void must(const int status, const std::string &what) {
    if (status) {
        fprintf(stderr, "bench_qp_cuda: %s failed (%d)\n", what.c_str(), status);
        std::exit(1);
    }
}

template <typename F>
static double best_of_three(F &&fn) {
    double best = 1e300;
    for (int attempt = 0; attempt < 3; ++attempt) {
        const double t0 = now();
        fn();
        best = std::min(best, now() - t0);
    }
    return best;
}

// The set-up gate: the device store's arrays against the host store's.
template <class St>
static void setup_gate(const char *name, Problem &P, const qpd::Kernels &k) {
    using code_t = typename St::code_type;
    Mesh &m = P.m;
    St st(m.nelements);
    qp::qp_setup_impl<double, geom_t, St, St::VS>(m.nelements, m.kevp.data(), m.adj[0].data(), m.adj[1].data(),
                                                  m.adj[2].data(), m.adj[3].data(), m.adj[4].data(), m.adj[5].data(),
                                                  m.adj[6].data(), m.adj[7].data(), m.adj[8].data(), m.det.data(),
                                                  P.lmbda, P.mu, 1, P.ux.data(), P.uy.data(), P.uz.data(), st);
    std::vector<uint8_t> dc, dx;
    std::vector<float> ds;
    if (k.download(dc, ds, dx)) {
        printf("  gate %-29s download failed\n", name);
        return;
    }
    const code_t *const hc = st.codes.data();
    const code_t *const devc = reinterpret_cast<const code_t *>(dc.data());
    const size_t nc = std::min(st.codes.size(), dc.size() / sizeof(code_t));
    size_t differ = 0;
    double worst = 0;  // int8: code steps; f16: units in the last place; f64: relative
    for (size_t i = 0; i < nc; ++i) {
        if (std::memcmp(&hc[i], &devc[i], sizeof(code_t)) == 0) continue;
        ++differ;
        if constexpr (std::is_same<code_t, int8_t>::value) {
            worst = std::max(worst, std::fabs((double)hc[i] - (double)devc[i]));
        } else if constexpr (std::is_same<code_t, double>::value) {
            worst = std::max(worst, std::fabs(hc[i] - devc[i]));
        } else {
            uint16_t a, b;
            std::memcpy(&a, &hc[i], 2);
            std::memcpy(&b, &devc[i], 2);
            auto ordered = [](uint16_t u) { return (u & 0x8000) ? -(int)(u & 0x7fff) : (int)u; };
            worst = std::max(worst, (double)std::abs(ordered(a) - ordered(b)));
        }
    }
    double scale_diff = 0;
    for (size_t i = 0; i < std::min({st.scales.size(), ds.size(), (size_t)m.nelements}); ++i)
        if (st.scales[i] != 0) scale_diff = std::max(scale_diff, std::fabs((double)st.scales[i] - ds[i]) / std::fabs((double)st.scales[i]));
    double extra_diff = 0;
    size_t extra_differ = 0;
    if constexpr (St::is_element_matrix) {
        if (St::per_pair)
            for (size_t i = 0; i < std::min(st.pair_factors.size(), dx.size()); ++i) extra_differ += st.pair_factors[i] != dx[i];
    } else if constexpr (St::has_aux) {
        const float *const da = reinterpret_cast<const float *>(dx.data());
        for (size_t i = 0; i < std::min(st.aux.size(), dx.size() / 4); ++i)
            if (st.aux[i] != 0) extra_diff = std::max(extra_diff, std::fabs((double)st.aux[i] - da[i]) / std::fabs((double)st.aux[i]));
    }
    if constexpr (std::is_same<code_t, double>::value) {
        double top = 0;
        for (size_t i = 0; i < nc; ++i) top = std::max(top, std::fabs(hc[i]));
        worst /= top > 0 ? top : 1;  // f64: relative to the largest entry
    }
    printf("  gate %-29s codes differing %zu of %zu (%.2e), worst %.3g; scales %.2e; means %.2e; pair factors differing %zu\n",
           name, differ, nc, nc ? (double)differ / nc : 0.0, worst, scale_diff, extra_diff, extra_differ);
}

template <class St>
static void store_row(const std::string &name, const char *variant, const qpd::Geometry &g, Problem &P, DeviceProblem &D,
                      const int repeats) {
    if (!selected(name.c_str())) return;
    auto k = qpd::create(variant, g, nullptr);
    if (!k) {
        printf("  %-34s not on the device for %s\n", name.c_str(), qpd::element_name());
        return;
    }
    Row row;
    row.name = name;
    row.setup_s = best_of_three([&] { must(k->update(D.u), name + " set-up"); });
    CHECK_CUDA(cudaMemset(D.out, 0, (size_t)D.nnodes * 3 * sizeof(double)));
    must(k->apply(D.h, D.out), name);
    device_errors(P, D.out, row);
    row.apply_s = best_of_three([&] {
                      for (int r = 0; r < repeats; ++r) must(k->apply(D.h, D.out), name);
                  }) /
                  repeats;
    row.bytes = (double)k->bytes();
    print_row(P, row);
    setup_gate<St>(name.c_str(), P, *k);
    fflush(stdout);
}

int main(int argc, char **argv) {
    static const int sizes[] = SIZES;
    const int n = argc > 1 ? std::atoi(argv[1]) : sizes[2];
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 10;
    const double warp = std::getenv("QP_WARP") ? std::atof(std::getenv("QP_WARP")) : 0.4;
    Problem P;
    make_problem(P, n, warp);
    Mesh &m = P.m;
    DeviceProblem D;
    upload(P, D);
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("%s %s: n=%d, %ld elements, %ld nodes, %ld dof, %s, mu=%g lambda=%g, warp %g\n", MATERIAL_LABEL,
                ELEMENT_NAME, n, (long)m.nelements, (long)m.nnodes, (long)P.ndof, prop.name, P.mu, P.lmbda, warp);
    std::printf("  %-34s %10s %10s %10s %10s %10s\n", "operator", "setup ms", "MDOF/s", "B/dof", "rel l2", "rel energy");

    const BsrGraph &g = D.graph;
    const double graph_bytes = g.colidx.size() * sizeof(idx_t) + g.rowptr.size() * sizeof(count_t);
    // BSR, f64 and f32 values: the library's device assembly and product.
    auto bsr = [&](auto value) {
        using V = decltype(value);
        const char *name = sizeof(V) == 8 ? "BSR f64" : "BSR f32";
        if (!selected(name)) return;
        V *values = nullptr;
        CHECK_CUDA(cudaMalloc(&values, (size_t)g.nnz() * 9 * sizeof(V)));
        Row row;
        row.name = name;
        row.setup_s = best_of_three([&] { must(assemble<V>(P, D, D.u, values), std::string(name) + " assembly"); });
        auto product = [&] {
            must(sfem::d_bsr_spmv_native<count_t, idx_t, V, double>(g.n_rows, 3, D.rowptr, D.colidx, values, 0.0, D.h,
                                                                     D.out, nullptr),
                 name);
        };
        product();
        CHECK_CUDA(cudaDeviceSynchronize());
        device_errors(P, D.out, row);
        row.apply_s = best_of_three([&] {
                          for (int r = 0; r < repeats; ++r) product();
                          CHECK_CUDA(cudaDeviceSynchronize());
                      }) /
                      repeats;
        row.bytes = graph_bytes + (double)g.nnz() * 9 * sizeof(V);
        print_row(P, row);
        // cuSPARSE's BSR product (bsrmv) on the same values, as a reference: f64 with double vectors,
        // f32 with float vectors (it has no single-precision matrix with double vectors).
        const std::string cname = sizeof(V) == 8 ? "cusparse BSR f64" : "cusparse BSR f32";
        // Its legacy BSR interface indexes the values with 32 bits.
        const bool cfits = (size_t)g.nnz() * 9 <= (size_t)INT32_MAX;
        if (selected(cname.c_str()) && !cfits)
            printf("  %-34s skipped: %zu values exceed its 32-bit indices\n", cname.c_str(), (size_t)g.nnz() * 9);
        if (selected(cname.c_str()) && cfits) {
            cusparseHandle_t handle;
            cusparseMatDescr_t descr;
            cusparseCreate(&handle);
            cusparseCreateMatDescr(&descr);
            const size_t nv = (size_t)g.n_rows * 3;
            V *hv = nullptr, *ov = nullptr;
            CHECK_CUDA(cudaMalloc(&hv, nv * sizeof(V)));
            CHECK_CUDA(cudaMalloc(&ov, nv * sizeof(V)));
            {
                std::vector<double> hh(nv);
                CHECK_CUDA(cudaMemcpy(hh.data(), D.h, nv * sizeof(double), cudaMemcpyDeviceToHost));
                std::vector<V> hc(hh.begin(), hh.end());
                CHECK_CUDA(cudaMemcpy(hv, hc.data(), nv * sizeof(V), cudaMemcpyHostToDevice));
            }
            const V one = 1, zero = 0;
            auto cproduct = [&] {
                cusparseStatus_t st;
                if constexpr (sizeof(V) == 8)
                    st = cusparseDbsrmv(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, (int)g.n_rows,
                                        (int)g.n_rows, (int)g.nnz(), &one, descr, values, D.rowptr, D.colidx, 3, hv, &zero, ov);
                else
                    st = cusparseSbsrmv(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, (int)g.n_rows,
                                        (int)g.n_rows, (int)g.nnz(), &one, descr, values, D.rowptr, D.colidx, 3, hv, &zero, ov);
                must(st != CUSPARSE_STATUS_SUCCESS, cname);
            };
            cproduct();
            CHECK_CUDA(cudaDeviceSynchronize());
            {
                std::vector<V> oc(nv);
                CHECK_CUDA(cudaMemcpy(oc.data(), ov, nv * sizeof(V), cudaMemcpyDeviceToHost));
                std::vector<double> od(oc.begin(), oc.end());
                CHECK_CUDA(cudaMemcpy(D.out, od.data(), nv * sizeof(double), cudaMemcpyHostToDevice));
            }
            Row crow;
            crow.name = cname;
            crow.setup_s = row.setup_s;
            device_errors(P, D.out, crow);
            crow.apply_s = best_of_three([&] {
                               for (int r = 0; r < repeats; ++r) cproduct();
                               CHECK_CUDA(cudaDeviceSynchronize());
                           }) /
                           repeats;
            crow.bytes = row.bytes;
            print_row(P, crow);
            cudaFree(hv);
            cudaFree(ov);
            cusparseDestroyMatDescr(descr);
            cusparseDestroy(handle);
        }
        cudaFree(values);
        fflush(stdout);
    };
    bsr(double(0));
    bsr(float(0));

    // Quantized BSR: the f32 assembly encoded on the device; set-up is both.
    for (const char *variant : {"bsro16", "bsroi8"}) {
        const std::string name = std::string(variant[4] == '1' ? "BSR f16 full " : "BSR i8/blk full ") +
                                 (ELEMENT_TYPE == smesh::PROTEUS_HEX27 ? "i32 packed" : "off16 packed");
        if (!selected(name.c_str())) continue;
        qpd::Assembly a;
        a.n_rows = g.n_rows;
        a.rowptr = D.rowptr;
        a.colidx = D.colidx;
        a.h_rowptr = g.rowptr.data();
        a.h_colidx = g.colidx.data();
        a.assemble = [&](const double *x, float *values) { return assemble<float>(P, D, x, values); };
        auto k = qpd::create_bsr(variant, a, nullptr);
        Row row;
        row.name = name;
        row.setup_s = best_of_three([&] { must(k->update(D.u), name + " set-up"); });
        CHECK_CUDA(cudaMemset(D.out, 0, (size_t)D.nnodes * 3 * sizeof(double)));
        must(k->apply(D.h, D.out), name);
        device_errors(P, D.out, row);
        row.apply_s = best_of_three([&] {
                          for (int r = 0; r < repeats; ++r) must(k->apply(D.h, D.out), name);
                      }) /
                      repeats;
        row.bytes = (double)k->bytes();
        print_row(P, row);
        fflush(stdout);
    }

    // The stores (the host stores of the set-up gate have the device's block of 32 elements).
    static_assert(LANE_VS == 32, "bench_qp_cuda compares with host stores of 32 lanes");
    using qp::Layout;
    using qp::Scheme;
    using qp::EaForm;
    // Each with the atomic scatter and with the block-local one (" block").
    printf("  block-local scatter: blocks of 128 elements, at most %d nodes\n", D.gb.max_block_nodes);
    // QP_SHAPES: the element stores' kernel shapes to run, words of scatter and resident blocks per SM
    // (atomic1 atomic2 atomic4 block1 block2 block4); default the stores' own.  The row name gets
    // " <shape>" when shapes are given.
    std::vector<std::string> shapes = {""};
    if (const char *v = std::getenv("QP_SHAPES")) {
        shapes.clear();
        std::string all(v), w;
        for (size_t a = 0, b; a < all.size(); a = b + 1) {
            b = all.find(' ', a);
            if (b == std::string::npos) b = all.size();
            if (b > a) shapes.push_back(all.substr(a, b - a));
        }
    }
    const int passes = std::getenv("QP_REPEAT") ? std::atoi(std::getenv("QP_REPEAT")) : 1;
    for (int pass = 0; pass < passes; ++pass)
    for (const std::string &shape : shapes) {
        if (!shape.empty()) {
            setenv("QPD_MINB", shape.substr(shape.size() - 1).c_str(), 1);
            setenv("QPD_SCATTER", shape.rfind("block", 0) == 0 ? "block" : "atomic", 1);
        }
        const qpd::Geometry &gg = D.gb;
        const std::string sfx = shape.empty() ? "" : " " + shape;
        store_row<qp::Store<El, double, Scheme::Plain, Layout::Blocked, 32>>("qp f64" + sfx, "f64", gg, P, D, repeats);
        store_row<qp::Store<El, half_t, Scheme::Plain, Layout::Blocked, 32>>("qp f16" + sfx, "f16", gg, P, D, repeats);
        if constexpr (El::NV > 1)
            store_row<qp::Store<El, int8_t, Scheme::MeanDeviation, Layout::Blocked, 32>>("qp i8 mean+dev" + sfx, "i8md", gg,
                                                                                         P, D, repeats);
        store_row<qp::EaStore<El, half_t, EaForm::Difference, 32>>("ea diff f16" + sfx, "ead16", gg, P, D, repeats);
        store_row<qp::EaStore<El, int8_t, EaForm::PairScale, 32>>("ea diff i8 pair" + sfx, "eadi8p", gg, P, D, repeats);
    }
    return 0;
}
