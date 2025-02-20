#include "sfem_API.hpp"

#include "adj_table.h"
#include "div.h"
#include "sfem_macros.h"
#include "sortreduce.h"

#include "sfem_API.hpp"

void compute_pseudo_normals(enum ElemType                element_type,
                            const ptrdiff_t              n_elements,
                            idx_t **const SFEM_RESTRICT  elements,
                            geom_t **const SFEM_RESTRICT points,
                            real_t **const SFEM_RESTRICT normals) {
    // TODO: pseudo normal computation using surface
    assert(element_type == EDGE2 && "IMPLEMENT OTHER CASES");

    for (ptrdiff_t i = 0; i < n_elements; i++) {
        const idx_t i0 = elements[0][i];
        const idx_t i1 = elements[1][i];

        const geom_t ux = points[0][i1] - points[0][i0];
        const geom_t uy = points[1][i1] - points[1][i0];

        const geom_t len = sqrt(ux * ux + uy * uy);
        assert(len != 0);

        const geom_t nx = uy / len;
        const geom_t ny = -ux / len;

        normals[0][i0] = -nx;
        normals[1][i0] = -ny;

        normals[0][i1] = -nx;
        normals[1][i1] = -ny;

        assert(nx * nx + ny * ny > 0);
    }
}

static SFEM_INLINE void tri3_div_points(const scalar_t                      px0,
                                        const scalar_t                      px1,
                                        const scalar_t                      px2,
                                        const scalar_t                      py0,
                                        const scalar_t                      py1,
                                        const scalar_t                      py2,
                                        const scalar_t *const SFEM_RESTRICT ux,
                                        const scalar_t *const SFEM_RESTRICT uy,
                                        scalar_t *const SFEM_RESTRICT       element_vector) {
    const scalar_t x0 = -px0 + px1;
    const scalar_t x1 = -py0 + py2;
    const scalar_t x2 = px0 - px2;
    const scalar_t x3 = py0 - py1;
    const scalar_t x4 = x0 * x1 - x2 * x3;
    const scalar_t x5 = 1.0 / x4;
    const scalar_t x6 = (1.0 / 6.0) * x5;
    const scalar_t x7 = ux[0] * x6;
    const scalar_t x8 = uy[0] * x6;
    const scalar_t x9 = x4 * ((1.0 / 6.0) * ux[1] * x1 * x5 + (1.0 / 6.0) * ux[2] * x3 * x5 + (1.0 / 6.0) * uy[1] * x2 * x5 +
                              (1.0 / 6.0) * uy[2] * x0 * x5 - x0 * x8 - x1 * x7 - x2 * x8 - x3 * x7);
    element_vector[0] = x9;
    element_vector[1] = x9;
    element_vector[2] = x9;
}

int tri3_div_apply(const ptrdiff_t     nelements,
                   const ptrdiff_t     nnodes,
                   idx_t **const       elems,
                   geom_t **const      xyz,
                   const real_t *const ux,
                   const real_t *const uy,
                   real_t *const       values) {
    SFEM_UNUSED(nnodes);

    idx_t  ev[3];
    real_t element_vector[3];
    real_t element_ux[3];
    real_t element_uy[3];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 3; ++v) {
            element_ux[v] = ux[ev[v]];
        }

        for (int v = 0; v < 3; ++v) {
            element_uy[v] = uy[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        tri3_div_points(
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                // Data
                element_ux,
                element_uy,
                // Output
                element_vector);

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            values[dof_i] += element_vector[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // if (argc != 4) {
    //     if (!rank) {
    //         fprintf(stderr, "usage: %s <trisurf> <mesh> <output_folder>\n", argv[0]);
    //     }

    //     return EXIT_FAILURE;
    // }

    // auto surf = argv[1];

    // // auto        trisurf       = sfem::Mesh::create_from_file(comm, argv[1]);
    // auto        mesh          = sfem::Mesh::create_from_file(comm, argv[2]);
    // const char *output_folder = argv[3];

    // int SFEM_ENABLE_ORACLE=1;
    // SFEM_READ_ENV(SFEM_ENABLE_ORACLE, atoi);

    std::string output_folder = "test_approxsdf";
    sfem::create_directory(output_folder.c_str());

    auto mesh = sfem::Mesh::create_tri3_square(comm, 80, 80, 0, 0, 1, 1);
    // auto mesh = sfem::Mesh::create_hex8_cube(comm, 40, 40, 40, 0, 0, 0, 1, 1, 1);

    mesh->write((output_folder + "/mesh").c_str());

    auto fs = sfem::FunctionSpace::create(mesh, 1);

    const ptrdiff_t nnodes = mesh->n_nodes();
    auto            points = mesh->points();
    const int       dim    = mesh->spatial_dimension();

    auto p = points->data();

    auto distance = sfem::create_host_buffer<real_t>(nnodes);
    auto normals  = sfem::create_host_buffer<real_t>(dim, nnodes);
    auto allnodes = sfem::create_host_buffer<idx_t>(nnodes);

    // auto oracle = sfem::create_host_buffer<real_t>(nnodes);

    auto d = distance->data();
    auto n = normals->data();

    ptrdiff_t nconstraints = 0;
    auto      idx          = allnodes->data();

    // Include mesh surface
    {
        ptrdiff_t      n_surf_elements = 0;
        element_idx_t *parent          = 0;
        int16_t       *side_idx        = 0;

        if (extract_skin_sideset(mesh->n_elements(),
                                 mesh->n_nodes(),
                                 mesh->element_type(),
                                 mesh->elements()->data(),
                                 &n_surf_elements,
                                 &parent,
                                 &side_idx) != SFEM_SUCCESS) {
            SFEM_ERROR("Failed to extract skin!\n");
        }

        auto sideset = std::make_shared<sfem::Sideset>(
                comm, sfem::manage_host_buffer(n_surf_elements, parent), sfem::manage_host_buffer(n_surf_elements, side_idx));

        const auto st    = side_type(mesh->element_type());
        const int  nnxs  = elem_num_nodes(st);
        auto       sides = sfem::create_host_buffer<idx_t>(nnxs, sideset->parent()->size());

        if (extract_surface_from_sideset(fs->element_type(),
                                         mesh->elements()->data(),
                                         sideset->parent()->size(),
                                         sideset->parent()->data(),
                                         sideset->lfi()->data(),
                                         sides->data()) != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to extract surface from sideset!\n");
        }

        compute_pseudo_normals(st, sides->extent(1), sides->data(), p, normals->data());

        auto            boundary_nodes = create_nodeset_from_sideset(fs, sideset);
        auto            bn             = boundary_nodes->data();
        const ptrdiff_t nbnodes        = boundary_nodes->size();

        for (ptrdiff_t i = 0; i < nbnodes; i++) {
            idx[i + nconstraints] = bn[i];
            d[i + nconstraints]   = 0;
        }

        nconstraints += boundary_nodes->size();
    }

    const geom_t c[3]   = {0.5, 0.5, 0.5};
    const geom_t radius = 0.2;

    {
        // Toy setup to be replaced with mesh
        const geom_t dist_tol = 0.015;
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            geom_t pdist = 0;
            geom_t vec[3];

            for (int d = 0; d < dim; d++) {
                auto dx = p[d][i] - c[d];
                pdist += dx * dx;
                vec[d] = dx;
            }

            pdist = radius - sqrt(pdist);

            // Avoid square-root
            if (fabs(pdist) < dist_tol) {
                d[i] = pdist;

                auto neg = signbit(pdist);

                geom_t norm_vec = 0;
                for (int d = 0; d < dim; d++) {
                    auto dx = vec[d];
                    norm_vec += dx * dx;
                }

                norm_vec = sqrt(norm_vec);
                norm_vec = neg ? -norm_vec : norm_vec;
                for (int d = 0; d < dim; d++) {
                    n[d][i] = vec[d] / norm_vec;
                }

                idx[nconstraints++] = i;
            }
        }
    }

    distance->to_file((output_folder + "/input_distance.raw").c_str());
    normals->to_files((output_folder + "/input_normals.%d.raw").c_str());

    auto es      = sfem::EXECUTION_SPACE_HOST;
    auto nodeset = sfem::view(allnodes, 0, nconstraints);

    auto op = sfem::create_op(fs, "Laplacian", es);
    op->initialize();

    auto f = sfem::Function::create(fs);
    f->add_operator(op);

    sfem::DirichletConditions::Condition prescribed_normal{.nodeset = nodeset, .value = 0, .component = 0};
    auto                                 conds = sfem::create_dirichlet_conditions(fs, {prescribed_normal}, es);
    f->add_constraint(conds);

    auto linear_op     = sfem::create_linear_operator("CRS", f, nullptr, es);
    auto linear_solver = sfem::create_cg<real_t>(linear_op, es);

    auto blas = sfem::blas<real_t>(es);

    auto rhs        = sfem::create_host_buffer<real_t>(fs->n_dofs());
    auto correction = sfem::create_host_buffer<real_t>(fs->n_dofs());

    for (int d = 0; d < dim; d++) {
        auto normal_comp = sfem::sub(normals, d);
        blas->zeros(rhs->size(), rhs->data());

        linear_op->apply(normal_comp->data(), rhs->data());
        blas->zeros(correction->size(), correction->data());

        f->apply_constraints(rhs->data());
        linear_solver->apply(rhs->data(), correction->data());
        blas->axpy(correction->size(), -1, correction->data(), normal_comp->data());
    }

    for (ptrdiff_t i = 0; i < nnodes; i++) {
        geom_t norm_vec = 0;
        for (int d = 0; d < dim; d++) {
            auto dx = n[d][i];
            norm_vec += dx * dx;
        }

        norm_vec = sqrt(norm_vec);

        for (int d = 0; d < dim; d++) {
            assert(norm_vec != 0);
            n[d][i] /= norm_vec;
        }
    }

    auto div = sfem::create_host_buffer<real_t>(fs->n_dofs());

    // TODO: divergence of normal field for different element types
    tri3_div_apply(
            mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), n[0], n[1], div->data());

    // solve potential for the distance
    blas->zeros(rhs->size(), rhs->data());
    linear_op->apply(distance->data(), rhs->data());
    blas->axpy(rhs->size(), -1, div->data(), rhs->data());

    f->apply_constraints(rhs->data());

    blas->zeros(correction->size(), correction->data());
    linear_solver->apply(rhs->data(), correction->data());
    blas->axpy(correction->size(), -1, correction->data(), distance->data());

    // TODO: visualize divergence
    // ...
    distance->to_file((output_folder + "/distance.raw").c_str());
    normals->to_files((output_folder + "/normals.%d.raw").c_str());
    return MPI_Finalize();
}
