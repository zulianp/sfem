// Newton on the GPU with a compressed format as the Krylov operator: newton_qp's flow with the
// case on the device (run it with --set execution_space=device).
//
//   newton_qp_cuda case.yaml [--set key.path=value]... [--qp f64|f16|i8md|ead16|eadi8p|bsro16|bsroi8]
//
// The device material operator is wrapped as newton_qp wraps the host one (qp_op.hpp); the inexact
// pair runs cuda/qp_device.cu's kernels on the device vectors the Newton hands it.  The references
// need no --qp: solver.linear.format=BSR with storage_bytes 8 or 4 (the library's device BSR).
// The element stores take the kernel shape the GPU kernel-shape test chose (QPD_SCATTER and
// QPD_MINB override it).
// Besides newton_qp's lines: the device memory in use after the set-up, after the store and at its
// peak (sampled every 20 ms during the run), and the name of the device.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "sfem_API.hpp"
#include "sfem_ElasticityCase.hpp"
#include "sfem_Playground.hpp"
#include "sfem_Function.hpp"
#include "sfem_FunctionSpace.hpp"

#include "kernel_math.hpp"
#include "qp_contract.hpp"
#include "qp_element.hpp"
#include "qp_op.hpp"

#include "qp_device.hpp"

namespace {

static double device_used_mb() {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    return (double)(total_b - free_b) / 1048576.0;
}

// The device memory's peak while it lives, sampled.
class DevicePeak {
public:
    DevicePeak() : peak_(device_used_mb()), run_(true), t_([this] {
                       while (run_) {
                           const double u = device_used_mb();
                           double p = peak_.load();
                           while (u > p && !peak_.compare_exchange_weak(p, u)) {
                           }
                           std::this_thread::sleep_for(std::chrono::milliseconds(20));
                       }
                   }) {}
    ~DevicePeak() { stop(); }
    double stop() {
        if (run_) {
            run_ = false;
            t_.join();
        }
        return peak_.load();
    }

private:
    std::atomic<double> peak_;
    std::atomic<bool> run_;
    std::thread t_;
};

template <typename T>
static T *upload(const T *src, const size_t n, std::vector<std::shared_ptr<void>> &keep) {
    void *p = nullptr;
    if (cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)) != cudaSuccess) SFEM_ERROR("newton_qp_cuda: cudaMalloc failed\n");
    if (n) cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice);
    keep.emplace_back(p, [](void *q) { cudaFree(q); });
    return (T *)p;
}

// A device store behind the wrapper's interface; it owns the device geometry it reads.
struct DeviceKernels final : QpKernels {
    std::vector<std::shared_ptr<void>> keep;
    std::unique_ptr<qpd::Kernels> k;
    int update(const real_t *const x) override { return k->update(x); }
    int apply(const real_t *const h, real_t *const out) override { return k->apply(h, out); }
    size_t bytes() const override { return k->bytes(); }
};

// The element stores: the connectivity in the kernels' order and the affine geometry, uploaded.
static std::unique_ptr<QpKernels> element_kernels(const std::string &variant, const Geometry &h) {
    auto d = std::make_unique<DeviceKernels>();
    const ptrdiff_t ne = h.nelements;
    std::vector<int32_t> el((size_t)El::N * ne);
    for (int p = 0; p < El::N; ++p) std::copy(h.kernel_elements[p], h.kernel_elements[p] + ne, el.begin() + (size_t)p * ne);
    std::vector<float> adj((size_t)9 * ne);
    for (int c = 0; c < 9; ++c) std::copy(h.adj[c].begin(), h.adj[c].end(), adj.begin() + (size_t)c * ne);
    qpd::Geometry g;
    g.nelements = ne;
    g.elements = upload(el.data(), el.size(), d->keep);
    g.adj = upload(adj.data(), adj.size(), d->keep);
    g.det = upload(h.det.data(), h.det.size(), d->keep);
    g.mu = h.mu;
    g.lmbda = h.lmbda;
    d->keep.push_back(qpd::attach_blocks(g, el.data()));
    d->k = qpd::create(variant, g, nullptr);
    if (!d->k) return nullptr;
    return d;
}

// Quantized BSR: the device operator's f32 assembly on the space's node graph, encoded.
static std::unique_ptr<QpKernels> bsr_kernels(const std::string &variant, const std::shared_ptr<sfem::Op> &inner,
                                              const std::shared_ptr<sfem::CRSGraph> &graph) {
    auto d = std::make_unique<DeviceKernels>();
    const ptrdiff_t n = graph->n_nodes();
    const count_t *const rp = graph->rowptr()->data();
    const idx_t *const ci = graph->colidx()->data();
    qpd::Assembly a;
    a.n_rows = n;
    a.h_rowptr = rp;
    a.h_colidx = ci;
    a.rowptr = upload(rp, (size_t)n + 1, d->keep);
    a.colidx = upload(ci, (size_t)rp[n], d->keep);
    const size_t nvalues = (size_t)rp[n] * 9;
    a.assemble = [inner, rp = a.rowptr, ci = a.colidx, nvalues](const double *x, float *values) {
        if (cudaMemset(values, 0, nvalues * sizeof(float)) != cudaSuccess) return 1;
        return inner->hessian_bsr_typed(x, rp, ci, smesh::SMESH_FLOAT32, values);
    };
    d->k = qpd::create_bsr(variant, a, nullptr);
    if (!d->k) return nullptr;
    return d;
}

}  // namespace

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    std::string path, variant;
    std::vector<std::string> overrides;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--set") && i + 1 < argc)
            overrides.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--qp") && i + 1 < argc)
            variant = argv[++i];
        else if (path.empty())
            path = argv[i];
    }
    if (path.empty()) {
        fprintf(stderr, "usage: %s case.yaml [--set key.path=value]... [--qp f64|f16|i8md|ead16|eadi8p|bsro16|bsroi8]\n", argv[0]);
        return SFEM_FAILURE;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("newton_qp: device %s\n", prop.name);
    auto config = sfem::ElasticityCase::read(path, overrides);
    if (!config) return SFEM_FAILURE;
    auto playground = sfem::Playground::create(ctx->communicator(), *config);
    if (!playground) return SFEM_FAILURE;
    printf("newton_qp: peak resident after set-up %.1f MB\n", peak_resident_mb());
    printf("newton_qp: device memory after set-up %.1f MB\n", device_used_mb());

    if (!variant.empty()) {
        auto f = playground->function();
        auto material = playground->material();
        const bool assembled = variant.rfind("bsr", 0) == 0;
        std::unique_ptr<Geometry> geometry;
        std::unique_ptr<QpKernels> kernels;
        if (assembled) {
            geometry = std::make_unique<Geometry>();
            geometry->nelements = f->space()->mesh()->block(0)->n_elements();
            kernels = bsr_kernels(variant, material, f->space()->element_node_graph());
        } else {
            geometry = affine_geometry(*f->space()->mesh());
            if (!material_parameters(*config, geometry->mu, geometry->lmbda)) {
                fprintf(stderr, "newton_qp: material needs mu and lmbda, or young_modulus and poisson_ratio\n");
                return SFEM_FAILURE;
            }
            kernels = element_kernels(variant, *geometry);
        }
        if (!kernels) {
            fprintf(stderr, "newton_qp: no device store '%s' for %s\n", variant.c_str(), El::name);
            return SFEM_FAILURE;
        }
        auto wrapped = std::make_shared<QpOp>(material, std::move(geometry), std::move(kernels), El::name);
        f->remove_operator(material);
        f->add_operator(wrapped);
        printf("newton_qp: %s wrapped with the %s qp store (%s, device)\n", material->name(), variant.c_str(), El::name);
        printf("newton_qp: peak resident after the store %.1f MB\n", peak_resident_mb());
        printf("newton_qp: device memory after the store %.1f MB\n", device_used_mb());
    }
    DevicePeak peak;
    const double t0 = now();
    const int status = playground->run();
    cudaDeviceSynchronize();
    printf("newton_qp: run %.3f s, status %d\n", now() - t0, status);
    printf("newton_qp: device peak %.1f MB\n", peak.stop());
    printf("newton_qp: peak resident %.1f MB\n", peak_resident_mb());
    return status;
}
