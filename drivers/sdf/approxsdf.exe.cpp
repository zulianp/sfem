#include "sfem_API.hpp"

#include "sortreduce.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

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

    std::string output_folder = "test_approxsdf";

    auto mesh = sfem::Mesh::create_tri3_square(comm, 40, 40, 0, 0, 1, 1);
    // auto mesh = sfem::Mesh::create_hex8_cube(comm, 40, 40, 40, 0, 0, 0, 1, 1, 1);

    const ptrdiff_t nnodes = mesh->n_nodes();
    auto            points = mesh->points();
    const int       dim    = mesh->spatial_dimension();

    const geom_t c[3]   = {0.5, 0.5, 0.5};
    const geom_t radius = 0.2;

    auto p = points->data();

    auto distance = sfem::create_host_buffer<real_t>(nnodes);
    auto normals  = sfem::create_host_buffer<real_t>(dim, nnodes);
    auto allnodes = sfem::create_host_buffer<idx_t>(nnodes);

    auto d = distance->data();
    auto n = normals->data();

    ptrdiff_t nconstraints = 0;
    auto      idx          = allnodes->data();

    {
        // Toy setup to be replaced with mesh
        const geom_t dist_tol = 0.02;
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            geom_t pdist = 0;
            geom_t vec[3];

            for (int d = 0; d < dim; d++) {
                auto dx = p[d][i] - c[d];
                pdist += dx * dx;
                vec[d] = dx;
            }

            pdist = sqrt(pdist) - radius;

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

    auto es      = sfem::EXECUTION_SPACE_HOST;
    auto fs      = sfem::FunctionSpace::create(mesh, 1);
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

    // TODO: divergence of normal field
    // TODO: visualize divergence
    // TODO: solve potential for the distance


    // Output
    sfem::create_directory(output_folder.c_str());
    mesh->write((output_folder + "/mesh").c_str());
    distance->to_file((output_folder + "/distance.raw").c_str());
    normals->to_files((output_folder + "/normals.%d.raw").c_str());
    return MPI_Finalize();
}
