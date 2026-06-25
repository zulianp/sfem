#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "hex8_inline_cpu.hpp"
#include "kernel_diagnostics.hpp"
#include "sfem_API.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_Function.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh.hpp"

extern "C" int generated_neohookean_ogden_hex8_hex8_gradient_affine_mesh_soa(ptrdiff_t,
                                                                             ptrdiff_t,
                                                                             idx_t **,
                                                                             const real_t *,
                                                                             const real_t *,
                                                                             const real_t *,
                                                                             const real_t *,
                                                                             const real_t *,
                                                                             const real_t *,
                                                                             const real_t *,
                                                                             const real_t *,
                                                                             const real_t *,
                                                                             const real_t *,
                                                                             real_t,
                                                                             real_t,
                                                                             ptrdiff_t,
                                                                             const real_t *,
                                                                             const real_t *,
                                                                             const real_t *,
                                                                             ptrdiff_t,
                                                                             real_t *,
                                                                             real_t *,
                                                                             real_t *);

extern "C" int generated_neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa(ptrdiff_t,
                                                                                    ptrdiff_t,
                                                                                    idx_t **,
                                                                                    geom_t **,
                                                                                    real_t,
                                                                                    real_t,
                                                                                    ptrdiff_t,
                                                                                    const real_t *,
                                                                                    const real_t *,
                                                                                    const real_t *,
                                                                                    ptrdiff_t,
                                                                                    real_t *,
                                                                                    real_t *,
                                                                                    real_t *);

extern "C" int generated_neohookean_ogden_hex8_hex8_apply_affine_mesh_soa(ptrdiff_t,
                                                                          ptrdiff_t,
                                                                          idx_t **,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          real_t,
                                                                          real_t,
                                                                          ptrdiff_t,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          ptrdiff_t,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          const real_t *,
                                                                          ptrdiff_t,
                                                                          real_t *,
                                                                          real_t *,
                                                                          real_t *);

extern "C" int generated_neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa(ptrdiff_t,
                                                                                 ptrdiff_t,
                                                                                 idx_t **,
                                                                                 geom_t **,
                                                                                 real_t,
                                                                                 real_t,
                                                                                 ptrdiff_t,
                                                                                 const real_t *,
                                                                                 const real_t *,
                                                                                 const real_t *,
                                                                                 ptrdiff_t,
                                                                                 const real_t *,
                                                                                 const real_t *,
                                                                                 const real_t *,
                                                                                 ptrdiff_t,
                                                                                 real_t *,
                                                                                 real_t *,
                                                                                 real_t *);

extern "C" void generated_neohookean_ogden_hex8_hex8_gradient_affine_mesh_soa_print_rate(
        double, ptrdiff_t, ptrdiff_t, int);
extern "C" void generated_neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa_print_rate(
        double, ptrdiff_t, ptrdiff_t, int);
extern "C" void generated_neohookean_ogden_hex8_hex8_apply_affine_mesh_soa_print_rate(
        double, ptrdiff_t, ptrdiff_t, int);
extern "C" void generated_neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa_print_rate(
        double, ptrdiff_t, ptrdiff_t, int);

namespace {

    struct BoundaryNodes {
        sfem::SharedBuffer<idx_t> left;
        sfem::SharedBuffer<idx_t> right;
    };

    BoundaryNodes create_x_boundary_nodes(const std::shared_ptr<sfem::Mesh> &mesh) {
        const ptrdiff_t nnodes = mesh->n_nodes();
        geom_t **const  points = mesh->points()->data();

        geom_t xmin = points[0][0];
        geom_t xmax = points[0][0];

#pragma omp parallel for reduction(min : xmin) reduction(max : xmax)
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            xmin = std::min(xmin, points[0][i]);
            xmax = std::max(xmax, points[0][i]);
        }

        const geom_t tolerance = std::max<geom_t>((xmax - xmin) * 1e-8, 1e-12);
        ptrdiff_t    nleft     = 0;
        ptrdiff_t    nright    = 0;

#pragma omp parallel for reduction(+ : nleft, nright)
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            nleft += std::abs(points[0][i] - xmin) <= tolerance;
            nright += std::abs(points[0][i] - xmax) <= tolerance;
        }

        auto left  = sfem::create_host_buffer<idx_t>(nleft);
        auto right = sfem::create_host_buffer<idx_t>(nright);

        ptrdiff_t left_offset  = 0;
        ptrdiff_t right_offset = 0;
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            const geom_t x = points[0][i];
            if (std::abs(x - xmin) <= tolerance) {
                left->data()[left_offset++] = i;
            }

            if (std::abs(x - xmax) <= tolerance) {
                right->data()[right_offset++] = i;
            }
        }

        return {left, right};
    }

    void print_rate(const char *name, const double elapsed, const ptrdiff_t nelements, const ptrdiff_t ndofs, const int repeat) {
        const double seconds_per_call = elapsed / repeat;
        const double melements_per_s  = 1e-6 * static_cast<double>(nelements) / seconds_per_call;
        const double mdofs_per_s      = 1e-6 * static_cast<double>(ndofs) / seconds_per_call;

        printf("%-72s %12.6e %16.3f %13.3f %10s %13s\n",
               name,
               seconds_per_call,
               melements_per_s,
               mdofs_per_s,
               "-",
               "-");
    }

    void compute_hex8_affine_geometry(const std::shared_ptr<sfem::Mesh> &mesh,
                                      real_t **const                     adjugate,
                                      real_t *const                      determinant) {
        const ptrdiff_t nelements = mesh->n_elements();
        idx_t **const   elements  = mesh->elements(0)->data();
        geom_t **const  points    = mesh->points()->data();

#pragma omp parallel for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            real_t x[8];
            real_t y[8];
            real_t z[8];

            for (int v = 0; v < 8; ++v) {
                const idx_t node = elements[v][e];
                x[v]             = points[0][node];
                y[v]             = points[1][node];
                z[v]             = points[2][node];
            }

            real_t local_adjugate[9];
            hex8_adjugate_and_det(x, y, z, 0.5, 0.5, 0.5, local_adjugate, &determinant[e]);
            for (int d = 0; d < 9; ++d) {
                adjugate[d][e] = local_adjugate[d];
            }
        }
    }

}  // namespace

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    auto          comm = context.communicator();

    if (comm->size() != 1) {
        SFEM_ERROR("bench_hyperelasticity.exe supports one MPI rank\n");
    }

    const int         resolution         = smesh::Env::read("SFEM_BASE_RESOLUTION", 16);
    const int         warmup             = smesh::Env::read("SFEM_WARMUP", 3);
    const int         repeat             = smesh::Env::read("SFEM_REPEAT", 10);
    const int         nl_max_it          = smesh::Env::read("SFEM_NL_MAX_IT", 10);
    const int         linear_max_it      = smesh::Env::read("SFEM_LSOLVE_MAX_IT", 500);
    const real_t      linear_rtol        = smesh::Env::read("SFEM_LSOLVE_RTOL", 1e-6);
    const real_t      nonlinear_tol      = smesh::Env::read("SFEM_NL_TOL", 1e-9);
    const real_t      displacement_value = smesh::Env::read("SFEM_DISPLACEMENT", 0.05);
    const real_t      damping            = smesh::Env::read("SFEM_NL_ALPHA", 1.0);
    const real_t      mu                 = smesh::Env::read("SFEM_SHEAR_MODULUS", 1.0);
    const real_t      lambda             = smesh::Env::read("SFEM_FIRST_LAME_PARAMETER", 1.0);
    const std::string operator_name      = smesh::Env::read_string("SFEM_OPERATOR", "NeoHookeanOgden");
    const std::string codegen_geometry   = smesh::Env::read_string("SFEM_CODEGEN_GEOMETRY", "affine");

    const auto element_type = smesh::type_from_string(smesh::Env::read_string("SFEM_ELEM_TYPE", "HEX8").c_str());
    auto       mesh         = sfem::Mesh::create_cube(
            comm, static_cast<smesh::ElemType>(element_type), resolution, resolution, resolution, 0, 0, 0, 1, 1, 1);

    const int block_size = mesh->spatial_dimension();
    if (block_size != 3) {
        SFEM_ERROR("bench_hyperelasticity.exe requires a three-dimensional mesh\n");
    }
    if (mesh->element_type(0) != smesh::HEX8) {
        SFEM_ERROR("generated solve path currently requires SFEM_ELEM_TYPE=HEX8\n");
    }
    if (codegen_geometry != "affine" && codegen_geometry != "isoparametric") {
        SFEM_ERROR("SFEM_CODEGEN_GEOMETRY must be affine or isoparametric\n");
    }

    auto fs = sfem::FunctionSpace::create(mesh, block_size);
    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
    op->initialize();
    f->add_operator(op);

    const BoundaryNodes                               boundary = create_x_boundary_nodes(mesh);
    std::vector<sfem::DirichletConditions::Condition> conditions;
    conditions.reserve(4);
    for (int component = 0; component < block_size; ++component) {
        conditions.push_back({.nodeset = boundary.left, .value = 0, .component = component});
    }
    conditions.push_back({.nodeset = boundary.right, .value = displacement_value, .component = 0});
    f->add_constraint(sfem::DirichletConditions::create(fs, conditions));

    const ptrdiff_t nelements = mesh->n_elements();
    const ptrdiff_t ndofs     = fs->n_dofs();
    auto            x         = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            rhs       = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            increment = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            trial     = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            output    = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto            blas      = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    blas->zeros(ndofs, x->data());
    f->apply_constraints(x->data());

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        trial->data()[i] = static_cast<real_t>((i % 97) + 1) / 97;
    }
    f->apply_zero_constraints(trial->data());

    f->update(x->data());
    auto baseline_linear_op = sfem::create_linear_operator("MF", f, x, sfem::EXECUTION_SPACE_HOST);

    for (int i = 0; i < warmup; ++i) {
        blas->zeros(ndofs, rhs->data());
        f->gradient(x->data(), rhs->data());
        blas->zeros(ndofs, output->data());
        baseline_linear_op->apply(trial->data(), output->data());
    }

    sfem::device_synchronize();
    blas->zeros(ndofs, rhs->data());
    const double gradient_t0 = MPI_Wtime();
    for (int i = 0; i < repeat; ++i) {
        f->gradient(x->data(), rhs->data());
    }
    sfem::device_synchronize();
    const double gradient_t1 = MPI_Wtime();

    blas->zeros(ndofs, output->data());
    const double apply_t0 = MPI_Wtime();
    for (int i = 0; i < repeat; ++i) {
        baseline_linear_op->apply(trial->data(), output->data());
    }
    sfem::device_synchronize();
    const double apply_t1 = MPI_Wtime();

    double generated_affine_gradient_elapsed = 0;
    double generated_affine_apply_elapsed    = 0;
    double generated_iso_gradient_elapsed    = 0;
    double generated_iso_apply_elapsed       = 0;

    auto adjugate    = sfem::create_host_buffer<real_t>(9, nelements);
    auto determinant = sfem::create_host_buffer<real_t>(nelements);
    compute_hex8_affine_geometry(mesh, adjugate->data(), determinant->data());

    {
        idx_t **const  elements = mesh->elements(0)->data();
        geom_t **const points   = mesh->points()->data();
        const real_t  *ux       = &x->data()[0];
        const real_t  *uy       = &x->data()[1];
        const real_t  *uz       = &x->data()[2];
        const real_t  *hx       = &trial->data()[0];
        const real_t  *hy       = &trial->data()[1];
        const real_t  *hz       = &trial->data()[2];
        real_t        *outx     = &output->data()[0];
        real_t        *outy     = &output->data()[1];
        real_t        *outz     = &output->data()[2];

        for (int i = 0; i < warmup; ++i) {
            blas->zeros(ndofs, output->data());
            generated_neohookean_ogden_hex8_hex8_gradient_affine_mesh_soa(nelements,
                                                                          mesh->n_nodes(),
                                                                          elements,
                                                                          adjugate->data()[0],
                                                                          adjugate->data()[1],
                                                                          adjugate->data()[2],
                                                                          adjugate->data()[3],
                                                                          adjugate->data()[4],
                                                                          adjugate->data()[5],
                                                                          adjugate->data()[6],
                                                                          adjugate->data()[7],
                                                                          adjugate->data()[8],
                                                                          determinant->data(),
                                                                          mu,
                                                                          lambda,
                                                                          3,
                                                                          ux,
                                                                          uy,
                                                                          uz,
                                                                          3,
                                                                          outx,
                                                                          outy,
                                                                          outz);
            blas->zeros(ndofs, output->data());
            generated_neohookean_ogden_hex8_hex8_apply_affine_mesh_soa(nelements,
                                                                       mesh->n_nodes(),
                                                                       elements,
                                                                       adjugate->data()[0],
                                                                       adjugate->data()[1],
                                                                       adjugate->data()[2],
                                                                       adjugate->data()[3],
                                                                       adjugate->data()[4],
                                                                       adjugate->data()[5],
                                                                       adjugate->data()[6],
                                                                       adjugate->data()[7],
                                                                       adjugate->data()[8],
                                                                       determinant->data(),
                                                                       mu,
                                                                       lambda,
                                                                       3,
                                                                       ux,
                                                                       uy,
                                                                       uz,
                                                                       3,
                                                                       hx,
                                                                       hy,
                                                                       hz,
                                                                       3,
                                                                       outx,
                                                                       outy,
                                                                       outz);
            blas->zeros(ndofs, output->data());
            generated_neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa(
                    nelements, mesh->n_nodes(), elements, points, mu, lambda, 3, ux, uy, uz, 3, outx, outy, outz);
            blas->zeros(ndofs, output->data());
            generated_neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa(
                    nelements, mesh->n_nodes(), elements, points, mu, lambda, 3, ux, uy, uz, 3, hx, hy, hz, 3, outx, outy, outz);
        }

        blas->zeros(ndofs, output->data());
        double t0 = MPI_Wtime();
        for (int i = 0; i < repeat; ++i) {
            generated_neohookean_ogden_hex8_hex8_gradient_affine_mesh_soa(nelements,
                                                                          mesh->n_nodes(),
                                                                          elements,
                                                                          adjugate->data()[0],
                                                                          adjugate->data()[1],
                                                                          adjugate->data()[2],
                                                                          adjugate->data()[3],
                                                                          adjugate->data()[4],
                                                                          adjugate->data()[5],
                                                                          adjugate->data()[6],
                                                                          adjugate->data()[7],
                                                                          adjugate->data()[8],
                                                                          determinant->data(),
                                                                          mu,
                                                                          lambda,
                                                                          3,
                                                                          ux,
                                                                          uy,
                                                                          uz,
                                                                          3,
                                                                          outx,
                                                                          outy,
                                                                          outz);
        }
        generated_affine_gradient_elapsed = MPI_Wtime() - t0;

        blas->zeros(ndofs, output->data());
        t0 = MPI_Wtime();
        for (int i = 0; i < repeat; ++i) {
            generated_neohookean_ogden_hex8_hex8_apply_affine_mesh_soa(nelements,
                                                                       mesh->n_nodes(),
                                                                       elements,
                                                                       adjugate->data()[0],
                                                                       adjugate->data()[1],
                                                                       adjugate->data()[2],
                                                                       adjugate->data()[3],
                                                                       adjugate->data()[4],
                                                                       adjugate->data()[5],
                                                                       adjugate->data()[6],
                                                                       adjugate->data()[7],
                                                                       adjugate->data()[8],
                                                                       determinant->data(),
                                                                       mu,
                                                                       lambda,
                                                                       3,
                                                                       ux,
                                                                       uy,
                                                                       uz,
                                                                       3,
                                                                       hx,
                                                                       hy,
                                                                       hz,
                                                                       3,
                                                                       outx,
                                                                       outy,
                                                                       outz);
        }
        generated_affine_apply_elapsed = MPI_Wtime() - t0;

        blas->zeros(ndofs, output->data());
        t0 = MPI_Wtime();
        for (int i = 0; i < repeat; ++i) {
            generated_neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa(
                    nelements, mesh->n_nodes(), elements, points, mu, lambda, 3, ux, uy, uz, 3, outx, outy, outz);
        }
        generated_iso_gradient_elapsed = MPI_Wtime() - t0;

        blas->zeros(ndofs, output->data());
        t0 = MPI_Wtime();
        for (int i = 0; i < repeat; ++i) {
            generated_neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa(
                    nelements, mesh->n_nodes(), elements, points, mu, lambda, 3, ux, uy, uz, 3, hx, hy, hz, 3, outx, outy, outz);
        }
        generated_iso_apply_elapsed = MPI_Wtime() - t0;
    }

    printf("operator %s\n", operator_name.c_str());
    printf("element_type %s\n", type_to_string(mesh->element_type(0)));
    printf("#elements %ld\n", static_cast<long>(nelements));
    printf("#nodes %ld\n", static_cast<long>(mesh->n_nodes()));
    printf("#dofs %ld\n", static_cast<long>(ndofs));
    printf("#left_nodes %ld\n", static_cast<long>(boundary.left->size()));
    printf("#right_nodes %ld\n", static_cast<long>(boundary.right->size()));
    printf("\n%-72s %12s %16s %13s %10s %13s\n",
           "Operation",
           "Time [s]",
           "Rate [MElem/s]",
           "Rate [MDOF/s]",
           "AI",
           "Rate [GFLOP/s]");
    printf("----------------------------------------------------------------------------------------------------------------------------------------------\n");
    print_rate("gradient", gradient_t1 - gradient_t0, nelements, ndofs, repeat);
    print_rate("hessian_apply", apply_t1 - apply_t0, nelements, ndofs, repeat);
    generated_neohookean_ogden_hex8_hex8_gradient_affine_mesh_soa_print_rate(
            generated_affine_gradient_elapsed, nelements, ndofs, repeat);
    generated_neohookean_ogden_hex8_hex8_apply_affine_mesh_soa_print_rate(
            generated_affine_apply_elapsed, nelements, ndofs, repeat);
    generated_neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa_print_rate(
            generated_iso_gradient_elapsed, nelements, ndofs, repeat);
    generated_neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa_print_rate(
            generated_iso_apply_elapsed, nelements, ndofs, repeat);

    idx_t **const  elements           = mesh->elements(0)->data();
    geom_t **const points             = mesh->points()->data();
    const bool     use_isoparametric  = codegen_geometry == "isoparametric";
    const auto     generated_gradient = [&](const real_t *const state, real_t *const out) {
        blas->zeros(ndofs, out);
        if (use_isoparametric) {
            generated_neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa(nelements,
                                                                                 mesh->n_nodes(),
                                                                                 elements,
                                                                                 points,
                                                                                 mu,
                                                                                 lambda,
                                                                                 3,
                                                                                 &state[0],
                                                                                 &state[1],
                                                                                 &state[2],
                                                                                 3,
                                                                                 &out[0],
                                                                                 &out[1],
                                                                                 &out[2]);
        } else {
            generated_neohookean_ogden_hex8_hex8_gradient_affine_mesh_soa(nelements,
                                                                          mesh->n_nodes(),
                                                                          elements,
                                                                          adjugate->data()[0],
                                                                          adjugate->data()[1],
                                                                          adjugate->data()[2],
                                                                          adjugate->data()[3],
                                                                          adjugate->data()[4],
                                                                          adjugate->data()[5],
                                                                          adjugate->data()[6],
                                                                          adjugate->data()[7],
                                                                          adjugate->data()[8],
                                                                          determinant->data(),
                                                                          mu,
                                                                          lambda,
                                                                          3,
                                                                          &state[0],
                                                                          &state[1],
                                                                          &state[2],
                                                                          3,
                                                                          &out[0],
                                                                          &out[1],
                                                                          &out[2]);
        }
        f->constraints_gradient(state, out);
    };

    auto generated_linear_op = sfem::make_op<real_t>(
            ndofs,
            ndofs,
            [&](const real_t *const direction, real_t *const out) {
                blas->zeros(ndofs, out);
                if (use_isoparametric) {
                    generated_neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa(nelements,
                                                                                      mesh->n_nodes(),
                                                                                      elements,
                                                                                      points,
                                                                                      mu,
                                                                                      lambda,
                                                                                      3,
                                                                                      &x->data()[0],
                                                                                      &x->data()[1],
                                                                                      &x->data()[2],
                                                                                      3,
                                                                                      &direction[0],
                                                                                      &direction[1],
                                                                                      &direction[2],
                                                                                      3,
                                                                                      &out[0],
                                                                                      &out[1],
                                                                                      &out[2]);
                } else {
                    generated_neohookean_ogden_hex8_hex8_apply_affine_mesh_soa(nelements,
                                                                               mesh->n_nodes(),
                                                                               elements,
                                                                               adjugate->data()[0],
                                                                               adjugate->data()[1],
                                                                               adjugate->data()[2],
                                                                               adjugate->data()[3],
                                                                               adjugate->data()[4],
                                                                               adjugate->data()[5],
                                                                               adjugate->data()[6],
                                                                               adjugate->data()[7],
                                                                               adjugate->data()[8],
                                                                               determinant->data(),
                                                                               mu,
                                                                               lambda,
                                                                               3,
                                                                               &x->data()[0],
                                                                               &x->data()[1],
                                                                               &x->data()[2],
                                                                               3,
                                                                               &direction[0],
                                                                               &direction[1],
                                                                               &direction[2],
                                                                               3,
                                                                               &out[0],
                                                                               &out[1],
                                                                               &out[2]);
                }
                f->copy_constrained_dofs(direction, out);
            },
            sfem::EXECUTION_SPACE_HOST);

    auto cg = sfem::create_cg<real_t>(generated_linear_op, sfem::EXECUTION_SPACE_HOST);
    cg->set_max_it(linear_max_it);
    cg->set_rtol(linear_rtol);
    cg->set_atol(1e-12);
    cg->verbose = false;

    printf("\n%-10s %-8s %-14s %-12s %-14s\n", "Newton", "CG", "Residual", "Time [s]", "Rate [MDOF/s]");
    printf("-----------------------------------------------------------------\n");
    printf("solve_operator generated_neohookean_ogden_hex8_%s\n", codegen_geometry.c_str());

    int       completed_newton = 0;
    ptrdiff_t total_cg_it      = 0;
    double    solve_t0         = MPI_Wtime();
    for (int i = 0; i < nl_max_it; ++i) {
        const double iteration_t0 = MPI_Wtime();
        generated_gradient(x->data(), rhs->data());

        const real_t residual = blas->norm2(ndofs, rhs->data());
        if (residual < nonlinear_tol) {
            printf("%-10d %-8d %-14.4e %-12.4e %-14.3f\n", i, 0, residual, 0.0, 0.0);
            completed_newton = i;
            break;
        }

        blas->zeros(ndofs, increment->data());
        f->copy_constrained_dofs(rhs->data(), increment->data());
        cg->apply(rhs->data(), increment->data());

        const int cg_it = cg->iterations();
        total_cg_it += cg_it;
        blas->axpy(ndofs, -damping, increment->data(), x->data());

        const double iteration_time = MPI_Wtime() - iteration_t0;
        const double rate_m         = 1e-6 * static_cast<double>(ndofs) / iteration_time;
        printf("%-10d %-8d %-14.4e %-12.4e %-14.3f\n", i, cg_it, residual, iteration_time, rate_m);
        completed_newton = i + 1;
    }
    const double solve_time = MPI_Wtime() - solve_t0;

    generated_gradient(x->data(), rhs->data());
    const real_t final_residual = blas->norm2(ndofs, rhs->data());

    printf("\nnewton_iterations %d\n", completed_newton);
    printf("cg_iterations %ld\n", static_cast<long>(total_cg_it));
    printf("solve_time %g [s]\n", solve_time);
    printf("solve_rate %g [MDOF/s]\n", 1e-6 * static_cast<double>(ndofs) / solve_time);
    printf("final_residual %g\n", static_cast<double>(final_residual));

    return std::isfinite(final_residual) ? SFEM_SUCCESS : SFEM_FAILURE;
}
