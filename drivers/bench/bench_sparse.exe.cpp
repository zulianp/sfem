#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <mpi.h>

#include "sfem_API.hpp"
#include "sfem_BSR.hpp"
#include "sfem_BSRSoA.hpp"
#include "sfem_CRS.hpp"
#include "sfem_Function.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh_reorder.hpp"
#include "smesh_path.hpp"

struct SparseBench {
    const char *name;
    ptrdiff_t   rows;
    ptrdiff_t   nnz;
    size_t      nbytes;
    double      assembly;
    double      apply;

    static const char *header() {
        return "Format        Assembly [s]    Apply [s]    Apply Rate [MDoF/s]    Apply BW [GB/s]    Rows        NNZ         "
               "Bytes\n"
               "----------    ------------    ---------    ---------------------    ---------------    --------    --------    "
               "--------\n";
    }

    void print() const {
        const double rate = apply > 0 ? 1e-6 * rows / apply : 0;
        const double bw   = apply > 0 ? 1e-9 * nbytes / apply : 0;
        printf("%-10s    %12.3e    %9.3e    %21.3f    %15.3f    %8td    %8td    %8zu\n",
               name,
               assembly,
               apply,
               rate,
               bw,
               rows,
               nnz,
               nbytes);
    }
};

template <class Kernel>
static double measure(const int repeat, Kernel kernel) {
    kernel();

    const double start = MPI_Wtime();
    for (int r = 0; r < repeat; ++r) {
        kernel();
    }

    return (MPI_Wtime() - start) / repeat;
}

// Convert a row-major AoS BSR value buffer into the SoA layout consumed by the BSRSoA operator:
// soa[d1][k * block_size + d2] = aos[k * block_size * block_size + d1 * block_size + d2]
static void bsr_aos_to_soa(const ptrdiff_t                   nnz,
                           const int                         block_size,
                           const real_t *const SFEM_RESTRICT aos,
                           real_t **const SFEM_RESTRICT      soa) {
    const int block_entries = block_size * block_size;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t k = 0; k < nnz; ++k) {
        const real_t *const SFEM_RESTRICT block = &aos[k * block_entries];
        for (int d1 = 0; d1 < block_size; ++d1) {
            real_t *const SFEM_RESTRICT row = soa[d1];
            for (int d2 = 0; d2 < block_size; ++d2) {
                row[k * block_size + d2] = block[d1 * block_size + d2];
            }
        }
    }
}

static int crs_supported(const smesh::ElemType element_type) {
    return element_type == smesh::TRI3 || element_type == smesh::TET4 || element_type == smesh::TET10 ||
           element_type == smesh::MACRO_TET4;
}

static int bsr_supported(const smesh::ElemType element_type) {
    return element_type == smesh::TET4 || element_type == smesh::HEX8 || sfem::is_semistructured_type(element_type);
}

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    auto          comm = context.communicator();

    if (comm->size() > 1) {
        SFEM_ERROR("Parallel execution not supported!\n");
    }

    const int SFEM_BASE_RESOLUTION = smesh::Env::read("SFEM_BASE_RESOLUTION", 50);
    const int SFEM_REPEAT          = smesh::Env::read("SFEM_REPEAT", 5);

    std::shared_ptr<sfem::Mesh> mesh;
    const std::string           mesh_path = smesh::Env::read_string("SFEM_MESH", "");

    if (!mesh_path.empty()) {
        mesh = sfem::Mesh::create_from_file(comm, smesh::Path(mesh_path.c_str()));
    } else {
        const auto element_type = smesh::type_from_string(smesh::Env::read_string("SFEM_ELEM_TYPE", "TET4").c_str());

        mesh = sfem::Mesh::create_cube(comm,
                                       static_cast<smesh::ElemType>(element_type),
                                       SFEM_BASE_RESOLUTION,
                                       SFEM_BASE_RESOLUTION,
                                       SFEM_BASE_RESOLUTION,
                                       0,
                                       0,
                                       0,
                                       1,
                                       1,
                                       1);
    }

    if (smesh::Env::read("SFEM_USE_SFC", false)) {
        auto sfc = smesh::SFC::create_from_env();
        sfc->reorder(*mesh);
    }

    const int             block_size    = mesh->spatial_dimension();
    const int             block_entries = block_size * block_size;
    const smesh::ElemType element_type  = mesh->element_type(0);

    auto space = sfem::FunctionSpace::create(mesh, block_size);
    auto f     = sfem::Function::create(space);
    auto op    = sfem::create_op(space, "LinearElasticity", sfem::EXECUTION_SPACE_HOST);
    op->initialize();
    f->add_operator(op);

    auto x = sfem::create_host_buffer<real_t>(space->n_dofs());

    // Input/output vectors for the operator application (SpMV) benchmark
    auto                        x_apply = sfem::create_host_buffer<real_t>(space->n_dofs());
    auto                        y       = sfem::create_host_buffer<real_t>(space->n_dofs());
    real_t *const SFEM_RESTRICT x_data  = x_apply->data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < space->n_dofs(); ++i) {
        x_data[i] = 1;
    }

    SparseBench benches[3];
    int         nbenches = 0;

    if (crs_supported(element_type)) {
        auto dof_graph = f->crs_graph();
        auto values    = sfem::create_host_buffer<real_t>(dof_graph->nnz());

        const size_t nbytes   = dof_graph->rowptr()->nbytes() + dof_graph->colidx()->nbytes() + values->nbytes();
        const double assembly = measure(SFEM_REPEAT, [&]() {
            memset(values->data(), 0, values->nbytes());
            f->hessian_crs(x->data(), dof_graph->rowptr()->data(), dof_graph->colidx()->data(), values->data());
        });

        auto crs = sfem::h_crs_spmv(
                dof_graph->n_nodes(), dof_graph->n_nodes(), dof_graph->rowptr(), dof_graph->colidx(), values, (real_t)0);

        const double apply = measure(SFEM_REPEAT, [&]() { crs->apply(x_apply->data(), y->data()); });

        benches[nbenches++] = {"CSR", space->n_dofs(), static_cast<ptrdiff_t>(dof_graph->nnz()), nbytes, assembly, apply};
    } else {
        fprintf(stderr, "[Warning] Skipping CSR for %s\n", type_to_string(element_type));
    }

    if (bsr_supported(element_type)) {
        auto node_graph = space->node_to_node_graph();
        auto values     = sfem::create_host_buffer<real_t>(node_graph->nnz() * block_entries);

        const size_t nbytes   = node_graph->rowptr()->nbytes() + node_graph->colidx()->nbytes() + values->nbytes();
        const double assembly = measure(SFEM_REPEAT, [&]() {
            memset(values->data(), 0, values->nbytes());
            f->hessian_bsr(x->data(), node_graph->rowptr()->data(), node_graph->colidx()->data(), values->data());
        });

        const ptrdiff_t scalar_nnz = static_cast<ptrdiff_t>(node_graph->nnz()) * block_entries;

        auto bsr = sfem::h_bsr_spmv(node_graph->n_nodes(),
                                    node_graph->n_nodes(),
                                    block_size,
                                    node_graph->rowptr(),
                                    node_graph->colidx(),
                                    values,
                                    (real_t)0);

        const double apply = measure(SFEM_REPEAT, [&]() { bsr->apply(x_apply->data(), y->data()); });

        benches[nbenches++] = {"BSR", space->n_dofs(), scalar_nnz, nbytes, assembly, apply};

        auto soa_values = sfem::create_host_buffer_fake_SoA<real_t>(block_size, node_graph->nnz() * block_size);

        const size_t soa_nbytes   = node_graph->rowptr()->nbytes() + node_graph->colidx()->nbytes() + soa_values->nbytes();
        const double soa_assembly = measure(SFEM_REPEAT, [&]() {
            memset(values->data(), 0, values->nbytes());
            f->hessian_bsr(x->data(), node_graph->rowptr()->data(), node_graph->colidx()->data(), values->data());
            bsr_aos_to_soa(static_cast<ptrdiff_t>(node_graph->nnz()), block_size, values->data(), soa_values->data());
        });

        auto bsr_soa = sfem::h_bsr_soa_spmv(node_graph->n_nodes(),
                                            node_graph->n_nodes(),
                                            block_size,
                                            node_graph->rowptr(),
                                            node_graph->colidx(),
                                            soa_values,
                                            (real_t)0);

        const double soa_apply = measure(SFEM_REPEAT, [&]() { bsr_soa->apply(x_apply->data(), y->data()); });

        benches[nbenches++] = {"BSR_SoA", space->n_dofs(), scalar_nnz, soa_nbytes, soa_assembly, soa_apply};
    } else {
        fprintf(stderr, "[Warning] Skipping BSR/BSR_SoA for %s\n", type_to_string(element_type));
    }

    printf("#elements %td\n", mesh->n_elements());
    printf("#nodes %td\n", mesh->n_nodes());
    printf("#dofs %td\n", space->n_dofs());
    printf("#element_type %s\n", type_to_string(element_type));
    printf("%s", SparseBench::header());
    for (int i = 0; i < nbenches; ++i) {
        benches[i].print();
    }

    return SFEM_SUCCESS;
}
