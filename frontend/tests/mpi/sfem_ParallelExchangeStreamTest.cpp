// The ghost exchange of a distributed space on a stream: with every rank on one node and NCCL
// (Communicator::enable_device), synchronize_begin/end on a non-blocking stream queue the pack,
// the messages and the unpack on that stream and return without waiting. A host-function gate
// holds the stream: the ghosts must be unchanged while it is closed (the default stream, synchronised,
// does not wait for it) and right once it opens. Without NCCL or with ranks on several nodes the
// exchange waits for MPI on the host, and the test says so and passes.

#include <atomic>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "sfem_FunctionSpace.hpp"
#include "sfem_test.hpp"

#include "smesh_device_communicator.hpp"
#include "smesh_mesh.hpp"

namespace {
    struct Gate {
        std::atomic<bool> open{false};
    };

    void CUDART_CB wait_for_gate(void *data) {
        auto *gate = static_cast<Gate *>(data);
        while (!gate->open.load()) {
        }
    }
}  // namespace

int test_exchange_on_its_stream() {
    auto comm = sfem::Communicator::world();
    auto mesh = sfem::Mesh::create_hex8_cube(comm, 12, 12, 12, 0, 0, 0, 1, 1, 1);
    SFEM_TEST_ASSERT(mesh && mesh->is_distributed());
    auto space = sfem::FunctionSpace::create(mesh, 1);

    comm->enable_device();
    auto device = comm->device();
    int  usable = device && device->whole();
    usable      = -comm->max(-usable);
    if (!usable) {
        if (comm->rank() == 0) printf("  no NCCL on one node: the exchange waits on the host, nothing to test\n");
        return SFEM_TEST_SUCCESS;
    }

    // Each node's value is its id in the mesh plus one; the ghosts start at -1.
    const ptrdiff_t      n     = space->n_dofs();
    const ptrdiff_t      owned = space->n_owned_dofs();
    auto                 ids   = mesh->distributed()->node_mapping();
    std::vector<real_t>  expected(n), start(n);
    for (ptrdiff_t i = 0; i < n; ++i) {
        expected[i] = (real_t)ids->data()[i] + 1;
        start[i]    = i < owned ? expected[i] : real_t(-1);
    }
    real_t *x = nullptr;
    cudaMalloc((void **)&x, n * sizeof(real_t));

    cudaStream_t s, probe;
    SFEM_TEST_ASSERT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
    SFEM_TEST_ASSERT(cudaStreamCreateWithFlags(&probe, cudaStreamNonBlocking) == cudaSuccess);
    const auto context = sfem::ExecutionContext::device(s);

    // Once before the gate: modules load and NCCL connects at first use, and either can wait
    // for the whole device.
    cudaMemcpy(x, start.data(), n * sizeof(real_t), cudaMemcpyHostToDevice);
    SFEM_TEST_ASSERT(space->synchronize(x, context) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(cudaDeviceSynchronize() == cudaSuccess);
    cudaMemcpy(x, start.data(), n * sizeof(real_t), cudaMemcpyHostToDevice);
    SFEM_TEST_ASSERT(cudaDeviceSynchronize() == cudaSuccess);
    comm->barrier();

    Gate gate;
    SFEM_TEST_ASSERT(cudaLaunchHostFunc(s, wait_for_gate, &gate) == cudaSuccess);
    SFEM_TEST_ASSERT(space->synchronize_begin(x, context) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(space->synchronize_end(context) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(cudaStreamSynchronize(0) == cudaSuccess);
    std::vector<real_t> early(n);
    cudaMemcpyAsync(early.data(), x, n * sizeof(real_t), cudaMemcpyDeviceToHost, probe);
    cudaStreamSynchronize(probe);
    long ran_early = 0;
    for (ptrdiff_t i = owned; i < n; ++i) ran_early += early[i] != real_t(-1);

    gate.open = true;
    SFEM_TEST_ASSERT(cudaStreamSynchronize(s) == cudaSuccess);
    std::vector<real_t> after(n);
    cudaMemcpy(after.data(), x, n * sizeof(real_t), cudaMemcpyDeviceToHost);
    long wrong = 0;
    for (ptrdiff_t i = 0; i < n; ++i) wrong += after[i] != expected[i];

    const long ghosts = comm->sum((long)(n - owned));
    ran_early         = comm->sum(ran_early);
    wrong             = comm->sum(wrong);
    if (comm->rank() == 0) {
        printf("  exchange on its stream, %d ranks: %ld of %ld ghosts set before the gate opened, %ld values wrong after\n",
               comm->size(),
               ran_early,
               ghosts,
               wrong);
    }

    cudaStreamDestroy(probe);
    cudaStreamDestroy(s);
    cudaFree(x);
    SFEM_TEST_ASSERT(ghosts > 0);
    SFEM_TEST_ASSERT(ran_early == 0);
    SFEM_TEST_ASSERT(wrong == 0);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_exchange_on_its_stream);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
