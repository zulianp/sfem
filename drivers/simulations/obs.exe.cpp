#include <memory>

#include "sfem_test.h"

#include "sfem_Function.hpp"

#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "matrixio_array.h"

#include "sfem_API.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_ssmgc.hpp"

int solve_obstacle_problem(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    
    if (argc != 6) {
        fprintf(stderr, "usage: %s <mesh> <sdf> <dirichlet_conditions> <contact_boundary> <output>\n", argv[0]);
        return SFEM_FAILURE;
    }

    const char *mesh_path             = argv[1];
    const char *sdf_path              = argv[2];
    const char *dirichlet_path        = argv[3];
    const char *contact_boundary_path = argv[4];
    std::string output_path           = argv[5];

    int SFEM_ELEMENT_REFINE_LEVEL = 2;

    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);
    SFEM_TEST_ASSERT(SFEM_ELEMENT_REFINE_LEVEL > 1);

    const char *SFEM_OPERATOR = "LinearElasticity";
    SFEM_READ_ENV(SFEM_OPERATOR, );

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;
    {
        const char *SFEM_EXECUTION_SPACE{nullptr};
        SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
        if (SFEM_EXECUTION_SPACE) {
            es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
        }
    }

    auto mesh = sfem::Mesh::create_from_file(comm, mesh_path);
    const int block_size = mesh->spatial_dimension();
    auto fs = sfem::FunctionSpace::create(mesh, block_size);

    fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    fs->semi_structured_mesh().apply_hierarchical_renumbering();

// FIXME
#ifdef SFEM_ENABLE_CUDA
    {
        auto elements = fs->device_elements();
        if (!elements) {
            elements = create_device_elements(fs, fs->element_type());
            fs->set_device_elements(elements);
        }
    }
#endif

    auto dirichlet_conditions = sfem::DirichletConditions::create_from_file(fs, dirichlet_path);

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    op->initialize();
    f->add_operator(op);
    f->add_constraint(dirichlet_conditions);

    auto sdf              = sfem::Grid<geom_t>::create_from_file(comm, sdf_path);
    auto contact_boundary = sfem::Sideset::create_from_file(comm, contact_boundary_path);
    auto contact_conds    = sfem::ContactConditions::create(fs, sdf, {contact_boundary}, es);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            x     = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs   = sfem::create_buffer<real_t>(ndofs, es);
    auto            gap   = sfem::create_buffer<real_t>(ndofs, es);

    f->apply_constraints(rhs->data());
    f->apply_constraints(x->data());

    int SFEM_USE_SPMG = 1;
    SFEM_READ_ENV(SFEM_USE_SPMG, atoi);

    if (SFEM_USE_SPMG) {
        std::shared_ptr<sfem::Input> in;
        const char                  *SFEM_SSMGC_YAML{nullptr};
        SFEM_READ_ENV(SFEM_SSMGC_YAML, );

        if (SFEM_SSMGC_YAML) {
            in = sfem::YAMLNoIndent::create_from_file(SFEM_SSMGC_YAML);
        }

        auto solver = sfem::create_ssmgc(f, contact_conds, in);
        solver->apply(rhs->data(), x->data());
    } else {
        auto solver = sfem::create_shifted_penalty(f, contact_conds, nullptr);
        solver->apply(rhs->data(), x->data());
    }

    // Output to disk
    sfem::create_directory(output_path.c_str());

    fs->mesh_ptr()->write((output_path + "/coarse_mesh").c_str());
    fs->semi_structured_mesh().export_as_standard((output_path + "/mesh").c_str());

    auto out = f->output();

    out->set_output_dir((output_path + "/out").c_str());
    out->enable_AoS_to_SoA(true);

    out->write("rhs",  sfem::to_host(rhs)->data());
    out->write("disp", sfem::to_host(x)->data());

    if (es != sfem::EXECUTION_SPACE_DEVICE) {
        // FIXME

        contact_conds->update(x->data());
        contact_conds->signed_distance_for_mesh_viz(x->data(), gap->data());
        out->write("gap", gap->data());

        auto blas = sfem::blas<real_t>(es);
        blas->zeros(rhs->size(), rhs->data());
        f->gradient(x->data(), rhs->data());

        blas->zeros(x->size(), x->data());
        contact_conds->full_apply_boundary_mass_inverse(rhs->data(), x->data());
        out->write("contact_stress", x->data());
    }

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_obstacle_problem(ctx->communicator(), argc, argv);
}
