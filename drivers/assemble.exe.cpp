#include "sfem_API.hpp"

int assemble(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    const double tick = smesh::time_seconds();

    if (comm->size() != 1) {
        SFEM_ERROR("Parallel execution not supported!\n");
    }

    if (argc != 3) {
        SFEM_ERROR("usage: %s <folder> <output>\n", argv[0]);
    }

    const char *output_folder    = argv[2];
    const char *SFEM_OPERATOR    = "Laplacian";
    int         SFEM_BLOCK_SIZE  = 1;
    int         SFEM_EXPORT_FP32 = 0;

    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);
    SFEM_READ_ENV(SFEM_EXPORT_FP32, atoi);

    auto es = sfem::EXECUTION_SPACE_HOST;

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    auto        m      = sfem::Mesh::create_from_file(comm, smesh::Path(folder));
    auto        fs     = sfem::FunctionSpace::create(m, SFEM_BLOCK_SIZE);
    auto        f      = sfem::Function::create(fs);

    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    f->add_operator(op);

    auto dbc = sfem::DirichletConditions::create_from_env(fs);
    f->add_constraint(dbc);

    auto nbc = sfem::NeumannConditions::create_from_env(fs);
    f->add_operator(nbc);

    ///////////////////////////////////////////////////////////////////////////////
    // Zero solution vector
    ///////////////////////////////////////////////////////////////////////////////

    auto x = sfem::create_buffer<real_t>(m->n_nodes(), es);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS matrix
    ///////////////////////////////////////////////////////////////////////////////

    auto      crs_graph = f->crs_graph();
    ptrdiff_t nnz       = crs_graph->nnz();
    auto      values    = sfem::create_buffer<real_t>(nnz, es);
    f->hessian_crs(x->data(), crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data());

    ///////////////////////////////////////////////////////////////////////////////
    // RHS
    ///////////////////////////////////////////////////////////////////////////////

    auto rhs = sfem::create_buffer<real_t>(m->n_nodes(), es);
    f->gradient(x->data(), rhs->data());

    auto blas = sfem::blas<real_t>(es);

    // Move to RHS
    blas->scal(rhs->size(), -1, rhs->data());

    ///////////////////////////////////////////////////////////////////////////////
    // Write CRS matrix and rhs vector
    ///////////////////////////////////////////////////////////////////////////////

    smesh::create_directory(output_folder);

    if (SFEM_EXPORT_FP32) {
        auto crs = sfem::h_crs_spmv<count_t, idx_t, float>(
                m->n_nodes(), m->n_nodes(), crs_graph->rowptr(), crs_graph->colidx(), sfem::astype<float>(values), (float)1);
        crs->to_file(smesh::Path(output_folder));
    } else {
        auto crs = sfem::h_crs_spmv<count_t, idx_t, real_t>(
                m->n_nodes(), m->n_nodes(), crs_graph->rowptr(), crs_graph->colidx(), values, (real_t)1);
        crs->to_file(smesh::Path(output_folder));
    }

    {
        const smesh::Path rhs_path = smesh::Path(output_folder) / "rhs.raw";
        if (SFEM_EXPORT_FP32) {
            sfem::astype<float>(rhs)->to_file(rhs_path);
        } else {
            rhs->to_file(rhs_path);
        }
    }

    ptrdiff_t nelements = m->n_elements();
    ptrdiff_t nnodes    = m->n_nodes();

    const double tock = smesh::time_seconds();

    if (!comm->rank()) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld\n", (long)nelements, (long)nnodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }
    return 0;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return assemble(ctx->communicator(), argc, argv);
}
