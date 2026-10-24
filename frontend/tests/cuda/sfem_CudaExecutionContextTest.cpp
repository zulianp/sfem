// The device BLAS in an execution context: it reports the context it was built with, every
// operation is ordered on the context's stream (not the legacy default stream), and the reduction
// into device memory allocates nothing and does not synchronise, so a CUDA graph can capture it.
//
// The stream test closes a gate on the context's stream: a host function enqueued there waits for
// a flag. Work ordered on that stream cannot have run while the gate is closed; work that went to
// the legacy stream instead has run once the legacy stream is synchronised, which does not wait for
// a non-blocking stream. Both outcomes are certain, not a race.

#include <atomic>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "sfem_API.hpp"
#include "sfem_cuda_blas.hpp"
#include "sfem_test.hpp"

namespace {
    struct Gate {
        std::atomic<bool> open{false};
    };

    void CUDART_CB wait_for_gate(void *data) {
        auto *gate = static_cast<Gate *>(data);
        while (!gate->open.load()) {
        }
    }

    std::vector<real_t> to_host(const real_t *const d, const ptrdiff_t n) {
        std::vector<real_t> h(n);
        cudaMemcpy(h.data(), d, n * sizeof(real_t), cudaMemcpyDeviceToHost);
        return h;
    }

    real_t *to_device(const std::vector<real_t> &h) {
        real_t *d = nullptr;
        cudaMalloc((void **)&d, h.size() * sizeof(real_t));
        cudaMemcpy(d, h.data(), h.size() * sizeof(real_t), cudaMemcpyHostToDevice);
        return d;
    }
}  // namespace

int test_blas_reports_its_context() {
    cudaStream_t s;
    SFEM_TEST_ASSERT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);

    auto host   = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
    auto device = sfem::blas<real_t>(sfem::ExecutionContext::device(s));
    auto legacy = sfem::blas<real_t>(sfem::EXECUTION_SPACE_DEVICE);

    SFEM_TEST_ASSERT(host->execution_space() == sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(host->context().stream() == nullptr);
    SFEM_TEST_ASSERT(device->execution_space() == sfem::EXECUTION_SPACE_DEVICE);
    SFEM_TEST_ASSERT(device->context().stream() == (void *)s);
    SFEM_TEST_ASSERT(legacy->execution_space() == sfem::EXECUTION_SPACE_DEVICE);
    SFEM_TEST_ASSERT(legacy->context().stream() == nullptr);

    cudaStreamDestroy(s);
    return SFEM_TEST_SUCCESS;
}

int test_blas_runs_on_its_stream() {
    const ptrdiff_t     n = 100000;
    std::vector<real_t> hx(n), hy(n);
    for (ptrdiff_t i = 0; i < n; ++i) {
        hx[i] = 1 + i % 7;
        hy[i] = 2;
    }
    real_t *x = to_device(hx), *y = to_device(hy), *z = to_device(hy), *d = nullptr;
    cudaMalloc((void **)&d, sizeof(real_t));
    cudaMemset(d, 0, sizeof(real_t));

    cudaStream_t s;
    SFEM_TEST_ASSERT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
    auto blas = sfem::blas<real_t>(sfem::ExecutionContext::device(s));

    // Each operation once before the gate closes: the first launch of a kernel loads its module
    // (CUDA's lazy loading), which can wait for the whole device, the held stream included.
    {
        real_t *w = to_device(hy);
        blas->axpy(n, 0, x, w);
        blas->copy(n, w, z);
        blas->scal(n, 1, w);
        blas->dot(n, w, w, d);
        blas->zeros(n, w);
        SFEM_TEST_ASSERT(cudaStreamSynchronize(s) == cudaSuccess);
        cudaFree(w);
        cudaMemset(d, 0, sizeof(real_t));
        SFEM_TEST_ASSERT(cudaDeviceSynchronize() == cudaSuccess);
    }

    Gate gate;
    SFEM_TEST_ASSERT(cudaLaunchHostFunc(s, wait_for_gate, &gate) == cudaSuccess);
    blas->axpy(n, 1, x, y);       // y = 2 + x
    blas->copy(n, y, z);          // z = y
    blas->scal(n, 2, z);          // z = 2 (2 + x)
    blas->dot(n, x, x, d);        // d = x . x, into device memory
    blas->zeros(n, x);            // x = 0
    // Whatever went to the legacy stream has run once it is synchronised; the context's stream is
    // still held by the gate.
    SFEM_TEST_ASSERT(cudaStreamSynchronize(0) == cudaSuccess);
    std::vector<real_t> early_y(n), early_x(n);
    real_t              early_d = -1;
    cudaStream_t        probe;
    cudaStreamCreateWithFlags(&probe, cudaStreamNonBlocking);
    cudaMemcpyAsync(early_y.data(), y, n * sizeof(real_t), cudaMemcpyDeviceToHost, probe);
    cudaMemcpyAsync(early_x.data(), x, n * sizeof(real_t), cudaMemcpyDeviceToHost, probe);
    cudaMemcpyAsync(&early_d, d, sizeof(real_t), cudaMemcpyDeviceToHost, probe);
    cudaStreamSynchronize(probe);
    int ran_early = early_d != 0;
    for (ptrdiff_t i = 0; i < n; ++i) ran_early += early_y[i] != hy[i] || early_x[i] != hx[i];

    gate.open = true;
    SFEM_TEST_ASSERT(cudaStreamSynchronize(s) == cudaSuccess);
    const auto ry = to_host(y, n), rz = to_host(z, n), rx = to_host(x, n);
    real_t     rd = 0;
    cudaMemcpy(&rd, d, sizeof(real_t), cudaMemcpyDeviceToHost);

    double dot = 0;
    int    wrong = 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        dot += hx[i] * hx[i];
        wrong += ry[i] != 2 + hx[i] || rz[i] != 2 * (2 + hx[i]) || rx[i] != 0;
    }
    printf("  on its stream: %d values changed before the gate opened, %d wrong after; dot %g (exact %g)\n",
           ran_early, wrong, (double)rd, dot);

    cudaStreamDestroy(probe);
    cudaStreamDestroy(s);
    cudaFree(x), cudaFree(y), cudaFree(z), cudaFree(d);
    SFEM_TEST_ASSERT(ran_early == 0);
    SFEM_TEST_ASSERT(wrong == 0);
    SFEM_TEST_ASSERT(std::abs(rd - dot) <= 1e-12 * dot);
    return SFEM_TEST_SUCCESS;
}

int test_dot_into_device_memory_is_captured() {
    const ptrdiff_t     n = 300000;
    std::vector<real_t> hl(n), hr(n);
    double              exact = 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        hl[i] = 1 + i % 5;
        hr[i] = 0.5 * (1 + i % 3);
        exact += hl[i] * hr[i];
    }
    real_t *l = to_device(hl), *r = to_device(hr), *d = nullptr;
    cudaMalloc((void **)&d, sizeof(real_t));

    cudaStream_t s;
    SFEM_TEST_ASSERT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
    auto blas = sfem::blas<real_t>(sfem::ExecutionContext::device(s));

    // Global capture mode: an allocation or a synchronising call anywhere invalidates the capture.
    cudaGraph_t graph = nullptr;
    SFEM_TEST_ASSERT(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal) == cudaSuccess);
    blas->dot(n, l, r, d);
    const cudaError_t captured = cudaStreamEndCapture(s, &graph);
    printf("  dot into device memory captured: %s\n", cudaGetErrorString(captured));
    SFEM_TEST_ASSERT(captured == cudaSuccess && graph);
    cudaGetLastError();

    cudaGraphExec_t exec = nullptr;
    SFEM_TEST_ASSERT(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess);
    real_t value = 0;
    for (int k = 0; k < 3; ++k) {
        SFEM_TEST_ASSERT(cudaGraphLaunch(exec, s) == cudaSuccess);
        SFEM_TEST_ASSERT(cudaStreamSynchronize(s) == cudaSuccess);
        cudaMemcpy(&value, d, sizeof(real_t), cudaMemcpyDeviceToHost);
        SFEM_TEST_ASSERT(std::abs(value - exact) <= 1e-12 * exact);
    }
    // The host dot is the same reduction, read back through the reusable slot.
    SFEM_TEST_ASSERT(std::abs(blas->dot(n, l, r) - exact) <= 1e-12 * exact);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    cudaFree(l), cudaFree(r), cudaFree(d);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_blas_reports_its_context);
    // Before the gate: an allocation in the reduction fails the capture here, while under the
    // gate it would wait for the held stream.
    SFEM_RUN_TEST(test_dot_into_device_memory_is_captured);
    SFEM_RUN_TEST(test_blas_runs_on_its_stream);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
