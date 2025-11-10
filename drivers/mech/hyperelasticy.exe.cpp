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

#include "sfem_NeoHookeanOgdenActiveStrainPacked.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_ssmgc.hpp"

static void fill_active_strain_Fa(const std::shared_ptr<sfem::Mesh> &mesh,
                                  real_t *const                      Fa_aos,
                                  const geom_t *const                center,
                                  const geom_t                       radius,
                                  const real_t                       a11_final,
                                  const int                          step,
                                  const int                          total_steps) {
    const ptrdiff_t nelements = mesh->n_elements();
    const int       nxe       = mesh->n_nodes_per_element();
    auto            elements  = mesh->elements()->data();
    auto            points    = mesh->points()->data();

    const geom_t r2 = radius * radius;
    const double s  = (total_steps > 0) ? (double)step / (double)total_steps : 1.0;
    const real_t a11_ramped = (real_t)(1.0 - (1.0 - (double)a11_final) * s);

    for (ptrdiff_t e = 0; e < nelements; ++e) {
        geom_t cx = 0, cy = 0, cz = 0;
        for (int v = 0; v < nxe; ++v) {
            const idx_t i = elements[v][e];
            cx += points[0][i];
            cy += points[1][i];
            cz += points[2][i];
        }
        cx /= (geom_t)nxe;
        cy /= (geom_t)nxe;
        cz /= (geom_t)nxe;

        const geom_t dx_c = cx - center[0];
        const geom_t dy_c = cy - center[1];
        const geom_t dz_c = cz - center[2];
        const geom_t d2   = dx_c * dx_c + dy_c * dy_c + dz_c * dz_c;

        const bool   inside = (d2 <= r2);
        const real_t a11    = inside ? a11_ramped : (real_t)1;
        const real_t a22    = (real_t)1;
        const real_t a33    = (real_t)1;

        const real_t detA = a11 * a22 * a33;

        const ptrdiff_t base = 9 * e;
        Fa_aos[base + 0]     = a11/detA;
        Fa_aos[base + 1]     = 0;
        Fa_aos[base + 2]     = 0;
        Fa_aos[base + 3]     = 0;
        Fa_aos[base + 4]     = a22/detA;
        Fa_aos[base + 5]     = 0;
        Fa_aos[base + 6]     = 0;
        Fa_aos[base + 7]     = 0;
        Fa_aos[base + 8]     = a33/detA;

        // const ptrdiff_t base = 9 * e;
        // Fa_aos[base + 0]     = a11;
        // Fa_aos[base + 1]     = 0;
        // Fa_aos[base + 2]     = 0;
        // Fa_aos[base + 3]     = 0;
        // Fa_aos[base + 4]     = a22;
        // Fa_aos[base + 5]     = 0;
        // Fa_aos[base + 6]     = 0;
        // Fa_aos[base + 7]     = 0;
        // Fa_aos[base + 8]     = a33;
    }
}

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


    // Active strain setup marker
    const bool use_active_strain =
            (
                // std::string(SFEM_OPERATOR) == "NeoHookeanOgdenActiveStrain" ||
             std::string(SFEM_OPERATOR) == "NeoHookeanOgdenActiveStrainPacked");

    const real_t SFEM_ACTIVE_STRAIN_XX  = sfem::Env::read("SFEM_ACTIVE_STRAIN_XX", 0.5);

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

    // Generate basic Fa if active strain operator is requested
    std::shared_ptr<sfem::Buffer<real_t>> Fa_storage;
    if (use_active_strain) {
        auto as_op = std::dynamic_pointer_cast<sfem::NeoHookeanOgdenActiveStrainPacked>(op);
        if (as_op) {
            // Bounding box and sphere parameters
            auto bbox      = mesh->compute_bounding_box();
            auto bb_min    = bbox.first->data();
            auto bb_max    = bbox.second->data();
            geom_t center[3] = {(geom_t)0, (geom_t)0, (geom_t)0};
            for (int d = 0; d < mesh->spatial_dimension(); d++) {
                center[d] = (bb_min[d] + bb_max[d]) * (geom_t)0.5;
            }
            const geom_t dx     = bb_max[0] - bb_min[0];
            const geom_t dy     = bb_max[1] - bb_min[1];
            const geom_t dz     = bb_max[2] - bb_min[2];
            const geom_t span   = std::min(dx, std::min(dy, dz));
            geom_t       radius = sfem::Env::read("SFEM_ACTIVE_STRAIN_RADIUS", (double)(0.25 * span));
            const geom_t r2     = radius * radius;

            const ptrdiff_t nelements = mesh->n_elements();
            const int       nxe       = mesh->n_nodes_per_element();
            auto            elements  = mesh->elements()->data();
            auto            points    = mesh->points()->data();

            // AoS layout: 9 values per element (row-major 3x3)
            Fa_storage = sfem::create_host_buffer<real_t>(9 * nelements);
            auto Fa    = Fa_storage->data();

            (void)elements;
            (void)points;
            fill_active_strain_Fa(mesh, Fa, center, radius, SFEM_ACTIVE_STRAIN_XX, 1, 1);

            // One global AoS buffer used for all blocks (single-block meshes recommended)
            as_op->set_active_strain_global(Fa_storage->data(), 9);
            
        } else {
            fprintf(stderr,
                    "[Warning] Active strain requested but operator is not NeoHookeanOgdenActiveStrainPacked; ignoring Fa\n");
        }
    }

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

    // Write the diagonal components of Fa to file (nodal averages, split by component)
    // if (use_active_strain && Fa_storage) {
    //     const ptrdiff_t nnodes    = mesh->n_nodes();
    //     const ptrdiff_t nelements = mesh->n_elements();
    //     const int       nxe       = mesh->n_nodes_per_element();
    //     auto            elements  = mesh->elements()->data();
    //     auto            Fa        = Fa_storage->data();

    //     auto sum11  = sfem::create_host_buffer<real_t>(nnodes);
    //     auto sum22  = sfem::create_host_buffer<real_t>(nnodes);
    //     auto sum33  = sfem::create_host_buffer<real_t>(nnodes);
    //     auto counts = sfem::create_host_buffer<ptrdiff_t>(nnodes);

    //     auto d_sum11  = sum11->data();
    //     auto d_sum22  = sum22->data();
    //     auto d_sum33  = sum33->data();
    //     auto d_counts = counts->data();

    //     for (ptrdiff_t e = 0; e < nelements; ++e) {
    //         const ptrdiff_t base = 9 * e;
    //         const real_t    a11  = Fa[base + 0];
    //         const real_t    a22  = Fa[base + 4];
    //         const real_t    a33  = Fa[base + 8];
    //         for (int v = 0; v < nxe; ++v) {
    //             const idx_t node = elements[v][e];
    //             d_sum11[node] += a11;
    //             d_sum22[node] += a22;
    //             d_sum33[node] += a33;
    //             d_counts[node] += 1;
    //         }
    //     }

    //     // Build AoS buffer (nnodes * 3) and write; AoS_to_SoA will split per component
    //     auto fa_diag = sfem::create_host_buffer<real_t>(fs->n_dofs());
    //     auto d_diag  = fa_diag->data();
    //     for (ptrdiff_t i = 0; i < nnodes; ++i) {
    //         const real_t inv = d_counts[i] > 0 ? (real_t)(1.0 / (double)d_counts[i]) : 0;
    //         const real_t v11 = d_sum11[i] * inv;
    //         const real_t v22 = d_sum22[i] * inv;
    //         const real_t v33 = d_sum33[i] * inv;
    //         const ptrdiff_t base = i * 3;
    //         d_diag[base + 0]     = v11;
    //         d_diag[base + 1]     = v22;
    //         d_diag[base + 2]     = v33;
    //     }

    //     out->write("Fa_diag", fa_diag->data());
    // }

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

            // Update active strain field per step (incremental loading)
            if (use_active_strain) {
                auto as_op_step = std::dynamic_pointer_cast<sfem::NeoHookeanOgdenActiveStrainPacked>(op);
                if (as_op_step && Fa_storage) {
                    auto bbox_step = mesh->compute_bounding_box();
                    auto bb_min_s  = bbox_step.first->data();
                    auto bb_max_s  = bbox_step.second->data();
                    geom_t center_s[3] = {(geom_t)0, (geom_t)0, (geom_t)0};
                    for (int d = 0; d < mesh->spatial_dimension(); d++) {
                        center_s[d] = (bb_min_s[d] + bb_max_s[d]) * (geom_t)0.5;
                    }
                    const geom_t dx_s   = bb_max_s[0] - bb_min_s[0];
                    const geom_t dy_s   = bb_max_s[1] - bb_min_s[1];
                    const geom_t dz_s   = bb_max_s[2] - bb_min_s[2];
                    const geom_t span_s = std::min(dx_s, std::min(dy_s, dz_s));
                    geom_t       radius_s =
                            sfem::Env::read("SFEM_ACTIVE_STRAIN_RADIUS", (double)(0.25 * span_s));

                    fill_active_strain_Fa(mesh, Fa_storage->data(), center_s, radius_s, SFEM_ACTIVE_STRAIN_XX, step, steps);
                    as_op_step->set_active_strain_global(Fa_storage->data(), 9);
                }
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
