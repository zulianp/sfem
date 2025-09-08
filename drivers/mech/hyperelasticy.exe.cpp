#include <memory>

#include "sfem_test.h"

#include "sfem_Function.hpp"

#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "matrixio_array.h"

#include "sfem_API.hpp"
#include "sfem_DirichletConditions.hpp"


#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_ssmgc.hpp"

int solve_hyperelasticity(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    
    if (argc != 4) {
        fprintf(stderr, "usage: %s <mesh> <dirichlet_conditions> <output>\n", argv[0]);
        return SFEM_FAILURE;
    }

    const char *mesh_path             = argv[1];
    const char *dirichlet_path        = argv[2];
    std::string output_path           = argv[3];

    // int SFEM_ELEMENT_REFINE_LEVEL = 2;

    // SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);
    // SFEM_TEST_ASSERT(SFEM_ELEMENT_REFINE_LEVEL > 1);

    const char *SFEM_OPERATOR = "NeoHookeanOgden";
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
    auto dirichlet_conditions = sfem::DirichletConditions::create_from_file(fs, dirichlet_path);

    

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

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    op->initialize();
    f->add_operator(op);
    f->add_constraint(dirichlet_conditions);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            displacement     = sfem::create_buffer<real_t>(ndofs, es);
    auto            increment     = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs   = sfem::create_buffer<real_t>(ndofs, es);

    f->apply_constraints(rhs->data());
    f->apply_constraints(displacement->data());

    auto linear_op = sfem::create_linear_operator("MF", f, displacement, es);
    auto cg = sfem::create_cg<real_t>(linear_op, es);
    cg->verbose = true;
    cg->set_max_it(1000);
    cg->set_op(linear_op);
    cg->set_rtol(1e-8);

    cg->apply(rhs->data(), increment->data());
    sfem::blas<real_t>(es)->axpy(ndofs, 1, increment->data(), displacement->data());

    // TODO: Newton iteration

   
    // Output to disk
    sfem::create_directory(output_path.c_str());

    fs->mesh_ptr()->write((output_path + "/coarse_mesh").c_str());
    // fs->semi_structured_mesh().export_as_standard((output_path + "/mesh").c_str());

    auto out = f->output();

    out->set_output_dir((output_path + "/out").c_str());
    out->enable_AoS_to_SoA(true);

    out->write("rhs",  sfem::to_host(rhs)->data());
    out->write("disp", sfem::to_host(displacement)->data());


    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_hyperelasticity(ctx->communicator(), argc, argv);
}
