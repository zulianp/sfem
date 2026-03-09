#include "sfem_API.hpp"

#include "sfem_aliases.hpp"
#include "smesh_env.hpp"
#include "sfem_base.hpp"
#include "smesh_mesh_reorder.hpp"

int main(int argc, char *argv[]) {
    using namespace sfem;
    Context context(argc, argv);
    auto    comm = context.communicator();

    if (comm->size() > 1) {
        SFEM_ERROR("Parallel execution not supported!\n");
    }

    const int base_resolution = smesh::Env::read("SFEM_BASE_RESOLUTION", 64);
    const int warmup          = smesh::Env::read("SFEM_WARMUP", 3);
    const int repeat          = smesh::Env::read("SFEM_REPEAT", 20);

    auto mesh = Mesh::create_cube(
            comm, smesh::TET4, base_resolution, base_resolution, base_resolution, 0, 0, 0, 1, 1, 1);

    if (smesh::Env::read("SFEM_USE_SFC", false)) {
        auto sfc = smesh::SFC::create_from_env();
        sfc->reorder(*mesh);
    }

    FunctionSpace::PackedMesh::create(mesh, {}, true);

    auto fs = FunctionSpace::create(mesh, 1);

    auto op = create_op(fs, "Gradient", EXECUTION_SPACE_HOST);

    const double setup_t0 = MPI_Wtime();
    op->initialize();
    const double setup_t1 = MPI_Wtime();

    const ptrdiff_t n_in  = op->n_dofs_domain();
    const ptrdiff_t n_out = op->n_dofs_image();

    auto h_buf   = create_buffer<real_t>(n_in, EXECUTION_SPACE_HOST);
    auto out_buf = create_buffer<real_t>(n_out, EXECUTION_SPACE_HOST);

    real_t *const h   = h_buf->data();
    real_t *const out = out_buf->data();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n_in; i++) {
        h[i] = (real_t)((i % 97) * (1.0 / 97.0));
    }

    for (int i = 0; i < warmup; i++) {
        op->apply(nullptr, h, out);
    }

    device_synchronize();
    const double t0 = MPI_Wtime();

    for (int r = 0; r < repeat; r++) {
        op->apply(nullptr, h, out);
    }

    device_synchronize();
    const double t1 = MPI_Wtime();

    const double elapsed = (t1 - t0) / repeat;

    const double in_rate_m  = 1e-6 * (double)n_in / elapsed;
    const double out_rate_m = 1e-6 * (double)n_out / elapsed;
    const double bw_gbps    = ((double)(n_in + n_out) * (double)sizeof(real_t)) / elapsed / 1e9;

    volatile real_t sink = out[0];

    printf("op Gradient\n");
    printf("element_type %s\n", type_to_string(mesh->element_type(0)));
    printf("#elements %ld\n", (long)mesh->n_elements());
    printf("#nodes %ld\n", (long)mesh->n_nodes());
    printf("setup %g [s]\n", setup_t1 - setup_t0);
    printf("time %g [s]\n", elapsed);
    printf("rate_in %g [MDoF/s]\n", in_rate_m);
    printf("rate_out %g [MDoF/s]\n", out_rate_m);
    printf("bw %g [GB/s]\n", bw_gbps);
    printf("sink %g\n", (double)sink);

    return SFEM_SUCCESS;
}
