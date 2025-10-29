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
#include "sfem_Env.hpp"
#include "sfem_P1toP2.hpp"
#include "sfem_Packed.hpp"
#include "sfem_SFC.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_ssmgc.hpp"

struct RotateYZ {
    std::shared_ptr<sfem::FunctionSpace>  space;
    int                                   steps;
    real_t                                angle;
    std::shared_ptr<sfem::Sideset>        sideset;
    sfem::SharedBuffer<idx_t>             nodeset;
    std::shared_ptr<sfem::Buffer<real_t>> uy;
    std::shared_ptr<sfem::Buffer<real_t>> uz;
    sfem::ExecutionSpace                  execution_space;
    real_t                                rcenter[3] = {0, 0, 0};

    RotateYZ(  //
            const std::shared_ptr<sfem::FunctionSpace>  &space,
            const int                                    steps,
            const real_t                                 angle,
            const std::shared_ptr<sfem::Sideset>        &sideset,
            const sfem::SharedBuffer<idx_t>             &nodeset,
            const std::shared_ptr<sfem::Buffer<real_t>> &uy,
            const std::shared_ptr<sfem::Buffer<real_t>> &uz,
            const sfem::ExecutionSpace                   execution_space)
        : space(space),
          steps(steps),
          angle(angle),
          sideset(sideset),
          nodeset(nodeset),
          uy(uy),
          uz(uz),
          execution_space(execution_space) {}

    std::shared_ptr<sfem::Constraint> create_constraint() {
        sfem::DirichletConditions::Condition yrot{
                .sidesets = {sideset}, .nodeset = nodeset, .values = uy, .value = 0, .component = 1};
        sfem::DirichletConditions::Condition zrot{
                .sidesets = {sideset}, .nodeset = nodeset, .values = uz, .value = 0, .component = 2};
        auto conds = sfem::create_dirichlet_conditions(space, {yrot, zrot}, execution_space);
        return conds;
    }

    void update(int step) {
        auto   points        = space->points()->data();
        real_t current_angle = step * angle / steps;
        printf("%d) current_angle = %g\n", step, current_angle);
        real_t mat[4] = {
                cos(current_angle),
                -sin(current_angle),
                sin(current_angle),
                cos(current_angle),
        };

        for (ptrdiff_t i = 0; i < nodeset->size(); i++) {
            const ptrdiff_t dof = nodeset->data()[i];

            geom_t ypos = points[1][dof] - rcenter[1];
            geom_t zpos = points[2][dof] - rcenter[2];

            geom_t ypos_rot = mat[0] * ypos + mat[1] * zpos;
            geom_t zpos_rot = mat[2] * ypos + mat[3] * zpos;

            uy->data()[i] = ypos_rot - ypos;
            uz->data()[i] = zpos_rot - zpos;
        }
    }

    static std::shared_ptr<RotateYZ> create(const std::shared_ptr<sfem::FunctionSpace> &space,
                                            const std::shared_ptr<sfem::Sideset>       &sideset,
                                            const int                                   steps,
                                            const real_t                                angle,
                                            const sfem::ExecutionSpace                  execution_space) {
        auto nodeset = sfem::create_nodeset_from_sideset(space, sideset);

        auto uy  = sfem::create_buffer<real_t>(nodeset->size(), sfem::EXECUTION_SPACE_HOST);
        auto uz  = sfem::create_buffer<real_t>(nodeset->size(), sfem::EXECUTION_SPACE_HOST);
        auto ret = std::make_shared<RotateYZ>(space, steps, angle, sideset, nodeset, uy, uz, execution_space);
        return ret;
    }

    static std::shared_ptr<RotateYZ> create_from_env(const std::shared_ptr<sfem::FunctionSpace> &space,
                                                     const sfem::ExecutionSpace                  execution_space) {
        const real_t      angle        = sfem::Env::read("SFEM_ROTATE_ANGLE", 0.0);
        const std::string sideset_path = sfem::Env::read_string("SFEM_ROTATE_SIDESET", "");
        const int         steps        = sfem::Env::read("SFEM_ROTATE_STEPS", 10);

        if (!sideset_path.empty()) {
            printf("Rotating sideset %s with angle %g\n", sideset_path.c_str(), angle);
            auto sideset    = sfem::Sideset::create_from_file(space->mesh_ptr()->comm(), sideset_path.c_str());
            auto ret        = RotateYZ::create(space, sideset, steps, angle, execution_space);
            ret->rcenter[0] = sfem::Env::read("SFEM_ROTATE_RCENTER_X", 0.0);
            ret->rcenter[1] = sfem::Env::read("SFEM_ROTATE_RCENTER_Y", 0.0);
            ret->rcenter[2] = sfem::Env::read("SFEM_ROTATE_RCENTER_Z", 0.0);
            return ret;
        }
        return nullptr;
    }
};

int solve_hyperelasticity(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    SFEM_TRACE_SCOPE("solve_hyperelasticity");

    if (argc != 4) {
        fprintf(stderr, "usage: %s <mesh> <dirichlet_conditions> <output>\n", argv[0]);
        return SFEM_FAILURE;
    }

    if (comm->size() > 1) {
        SFEM_ERROR("MPI runtimes are not supported!\n");
    }

    const char *mesh_path      = argv[1];
    const char *dirichlet_path = argv[2];
    std::string output_path    = argv[3];

    int         SFEM_ELEMENT_REFINE_LEVEL = sfem::Env::read("SFEM_ELEMENT_REFINE_LEVEL", 0);
    const char *SFEM_OPERATOR             = "NeoHookeanOgden";
    SFEM_READ_ENV(SFEM_OPERATOR, );

    const bool   SFEM_VERBOSE     = sfem::Env::read("SFEM_VERBOSE", 0);
    const real_t SFEM_LSOLVE_RTOL = sfem::Env::read("SFEM_LSOLVE_RTOL", 1e-3);
    const real_t SFEM_NL_TOL      = sfem::Env::read("SFEM_NL_TOL", 1e-9);

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;
    {
        const char *SFEM_EXECUTION_SPACE{nullptr};
        SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
        if (SFEM_EXECUTION_SPACE) {
            es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
        }
    }

    auto mesh = sfem::Mesh::create_from_file(comm, mesh_path);

    if (sfem::Env::read("SFEM_PROMOTE_TO_P2", false)) {
        mesh = sfem::convert_p1_mesh_to_p2(mesh);
    }

    // FIXME SFC should also sort the BCs
    // if (sfem::Env::read("SFEM_USE_SFC", false)) {
    //     auto sfc = sfem::SFC::create_from_env();
    //     sfc->reorder(*mesh);
    // }

    std::shared_ptr<sfem::FunctionSpace::PackedMesh> packed_mesh;
    if (sfem::Env::read("SFEM_USE_PACKED_MESH", false)) {
        packed_mesh = sfem::FunctionSpace::PackedMesh::create(mesh, {}, true);
    }

    const int                            block_size = mesh->spatial_dimension();
    std::shared_ptr<sfem::FunctionSpace> fs;
    if (packed_mesh) {
        fs = sfem::FunctionSpace::create(packed_mesh, block_size);
    } else {
        fs = sfem::FunctionSpace::create(mesh, block_size);
    }

    if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
        fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    }

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

    auto rotate_conds = RotateYZ::create_from_env(fs, es);
    if (rotate_conds) {
        f->add_constraint(rotate_conds->create_constraint());
    }

    const ptrdiff_t ndofs        = fs->n_dofs();
    auto            displacement = sfem::create_buffer<real_t>(ndofs, es);
    auto            increment    = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs          = sfem::create_buffer<real_t>(ndofs, es);

    const std::string SFEM_OP_TYPE = sfem::Env::read_string("SFEM_OP_TYPE", "MF");
    auto linear_op = sfem::create_linear_operator(SFEM_OP_TYPE.c_str(), f, displacement, es);
    auto cg        = sfem::create_cg<real_t>(linear_op, es);
    cg->verbose    = SFEM_VERBOSE;
    cg->set_max_it(20000);
    cg->set_op(linear_op);
    cg->set_rtol(SFEM_LSOLVE_RTOL);
    cg->set_atol(1e-11);

    std::function<void(const real_t *const)> update_preconditioner = [](const real_t *const disp) {};

    if (sfem::Env::read("SFEM_USE_PRECONDITIONER", false)) {
        if (fs->element_type() == HEX8) {
            auto diag                = sfem::create_buffer<real_t>(ndofs, es);
            auto sj                  = sfem::create_shiftable_jacobi(diag, es);
            sj->relaxation_parameter = 1. / fs->block_size();
            cg->set_preconditioner_op(sj);
            update_preconditioner = [=](const real_t *const disp) {
                f->hessian_diag(disp, diag->data());
                sj->set_diag(diag);
            };
        }
    }

    // Newton iteration
    int    nl_max_it          = sfem::Env::read("SFEM_NL_MAX_IT", 30);
    real_t alpha              = sfem::Env::read("SFEM_NL_ALPHA", 1.0);
    bool   enable_line_search = sfem::Env::read("SFEM_ENABLE_LINE_SEARCH", true);
    auto   blas               = sfem::blas<real_t>(es);

    printf("Solving hyperelasticity: #%ld dofs\n", (long)fs->n_dofs());

    // Output to disk
    sfem::create_directory(output_path.c_str());
    if (fs->has_semi_structured_mesh()) {
        fs->semi_structured_mesh().export_as_standard((output_path + "/mesh").c_str());
        fs->mesh_ptr()->write((output_path + "/coarse_mesh").c_str());
    } else {
        fs->mesh_ptr()->write((output_path + "/mesh").c_str());
    }

    auto out = f->output();
    sfem::create_directory((output_path + "/out").c_str());
    out->set_output_dir((output_path + "/out").c_str());
    out->enable_AoS_to_SoA(true);

    if (sfem::Env::read("SFEM_USE_GRADIENT_DESCENT", false)) {
        for (int i = 0; i < nl_max_it; i++) {
            blas->zeros(ndofs, rhs->data());
            f->gradient(displacement->data(), rhs->data());

            const real_t gnorm = blas->norm2(ndofs, rhs->data());
            printf("%d) gnorm = %g\n", i, gnorm);

            if (gnorm < SFEM_NL_TOL) break;

            blas->axpy(ndofs, -alpha, rhs->data(), displacement->data());
        }

        out->write("rhs", sfem::to_host(rhs)->data());
        out->write("disp", sfem::to_host(displacement)->data());
    } else {
        real_t energy         = 0;
        real_t selected_alpha = 0;
        f->value(displacement->data(), &energy);

        // Newton solver with line search
        printf("%-10s %-5s %-14s %-14s %-14s\n", "Iteration", "CG", "gnorm", "energy", "alpha");
        printf("-------------------------------------------------------------\n");

        int steps = rotate_conds ? rotate_conds->steps : 1;
        if (rotate_conds) {
            out->write_time_step("rhs", 0, sfem::to_host(rhs)->data());
            out->write_time_step("disp", 0, sfem::to_host(displacement)->data());
        }

        int       last_iterations         = 0;
        ptrdiff_t total_linear_iterations = 0;
        for (int step = 1; step <= steps; step++) {
            if (rotate_conds) {
                rotate_conds->update(step);
            }

            for (int i = 0; i < nl_max_it; i++) {
                f->update(displacement->data());
                update_preconditioner(displacement->data());
                blas->zeros(ndofs, rhs->data());
                f->gradient(displacement->data(), rhs->data());

                const real_t gnorm = blas->norm2(ndofs, rhs->data());
                printf("%-10d %-5d %-14.4e %-14.4e %-14.4f\n", i, last_iterations, gnorm, energy, -selected_alpha);

                if (gnorm < SFEM_NL_TOL) break;

                if (SFEM_OP_TYPE != "MF") {
                    linear_op = sfem::create_linear_operator(SFEM_OP_TYPE.c_str(), f, displacement, es);
                    cg->set_op(linear_op);
                }

                blas->zeros(ndofs, increment->data());
                f->copy_constrained_dofs(rhs->data(), increment->data());
                cg->apply(rhs->data(), increment->data());
                last_iterations = cg->iterations();
                total_linear_iterations += last_iterations;

                if (enable_line_search) {
                    std::vector<real_t> alphas{-2 * alpha,
                                               -alpha,
                                               -(real_t)0.9 * alpha,
                                               -2 * alpha / 3,
                                               -alpha / 2,
                                               -alpha / 4,
                                               -alpha / 8,
                                               -alpha / 32,
                                               -alpha / 128};
                    std::vector<real_t> energies(alphas.size(), 0);

                    f->value_steps(displacement->data(), increment->data(), alphas.size(), alphas.data(), energies.data());
                    const int min_energy_index =
                            std::distance(energies.begin(), std::min_element(energies.begin(), energies.end()));
                    selected_alpha = alphas[min_energy_index];
                    energy         = energies[min_energy_index];
                    blas->axpy(ndofs, selected_alpha, increment->data(), displacement->data());
                } else {
                    selected_alpha = -alpha;
                    blas->axpy(ndofs, selected_alpha, increment->data(), displacement->data());
                    f->value(displacement->data(), &energy);
                }
            }

            if (rotate_conds) {
                out->write_time_step("rhs", step, sfem::to_host(rhs)->data());
                out->write_time_step("disp", step, sfem::to_host(displacement)->data());
            } else {
                out->write("rhs", sfem::to_host(rhs)->data());
                out->write("disp", sfem::to_host(displacement)->data());
            }
        }

        printf("Total linear iterations: %ld\n", (long)total_linear_iterations);
    }

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_hyperelasticity(ctx->communicator(), argc, argv);
}
