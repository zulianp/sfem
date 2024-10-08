
#include "isolver_lsolve.h"

extern "C" {
#include "crs_graph.h"
#include "dirichlet.h"

#include "laplacian.h"
#include "matrixio_array.h"
#include "neumann.h"
#include "read_mesh.h"
#include "sfem_mesh.h"
#include "sfem_mesh_write.h"
}

#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>

template<typename T>
void get(YAML::Node &node, const std::string &str, T &val)
{
    if(node[str]) {
        val = node[str].as<T>();
    }
}

typedef struct {
    ptrdiff_t rows;
    ptrdiff_t cols;
    count_t *rowptr;
    idx_t *colidx;
    real_t *values;
} crs_matrix_t;

void crs_matrix_destroy(crs_matrix_t *const matrix) {
    matrix->rows = 0;
    matrix->cols = 0;
    free(matrix->rowptr);
    free(matrix->colidx);
    free(matrix->values);
}

void create_crs_matrix(const mesh_t *const mesh, crs_matrix_t *const matrix, int block_size) {
    ptrdiff_t nnz = 0;
    count_t *rowptr = 0;
    idx_t *colidx = 0;
    real_t *values = 0;

    build_crs_graph(mesh->nelements, mesh->nnodes, mesh->elements, &rowptr, &colidx);

    nnz = rowptr[mesh->nnodes];
    values = (real_t *)malloc(nnz * sizeof(real_t));
    memset(values, 0, nnz * block_size * sizeof(real_t));

    matrix->rows = mesh->nnodes;
    matrix->cols = mesh->nnodes;

    matrix->rowptr = rowptr;
    matrix->colidx = colidx;
    matrix->values = values;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_rank(comm, &size);

    if (argc != 2) {
        if (!rank) {
            fprintf(stderr, "usage: %s <config.yaml>\n", argv[0]);
        }

        MPI_Finalize();
        return 1;
    }

    YAML::Node config = YAML::LoadFile(argv[1]);

    mesh_t mesh;
    {
        auto mesh_path = config["mesh"]["path"].as<std::string>();
        if (mesh_read(comm, mesh_path.c_str(), &mesh)) {
            return EXIT_FAILURE;
        }
    }

    std::vector<real_t> rhs(mesh.nnodes, 0);

    if (config["forcing_function"]) {
        auto &&node = config["forcing_function"];
        if (node["type"].as<std::string>() == "file") {
            auto path_forcing_function = config["forcing_function"]["path"].as<std::string>();

            // FIXME parallel
            array_read(comm, path_forcing_function.c_str(), SFEM_MPI_REAL_T, rhs.data(), mesh.nnodes, mesh.nnodes);
        } else if (node["type"].as<std::string>() == "value") {
            std::fill(rhs.begin(), rhs.end(), node["value"].as<real_t>());
        } else {
            assert(false);
        }
    }

    if (config["neumann"]) {
        auto &&node = config["neumann"];
        if (node["type"].as<std::string>() == "value") {
            auto neumann_path = node["idx"].as<std::string>();
            auto neumann_value = node["value"].as<real_t>();

            const idx_t *neumann_faces = 0;
            ptrdiff_t n_neumann_faces_local, n_neumann_faces;

            array_create_from_file(comm,
                                   neumann_path.c_str(),
                                   SFEM_MPI_IDX_T,
                                   (void **)&neumann_faces,
                                   &n_neumann_faces_local,
                                   &n_neumann_faces);

            n_neumann_faces_local /= 3;
            n_neumann_faces /= 3;

            surface_forcing_function(mesh.element_type, n_neumann_faces_local, neumann_faces, mesh.points, neumann_value, rhs.data());
        }
    }

    crs_matrix_t matrix;
    create_crs_matrix(&mesh, &matrix, 1);

    laplacian_crs(
        mesh.element_type,
        mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, matrix.rowptr, matrix.colidx, matrix.values);

    std::vector<real_t> x(mesh.nnodes, 0);

    if (config["dirichlet"]) {
        auto &&node = config["dirichlet"];
        auto dirichlet_path = node["idx"].as<std::string>();
        auto dirichlet_value = node["value"].as<real_t>();


        const idx_t *dirichlet_nodes = 0;
        ptrdiff_t n_dirichlet_nodes_local, n_dirichlet_nodes;
        array_create_from_file(comm,
                               dirichlet_path.c_str(),
                               SFEM_MPI_IDX_T,
                               (void **)&dirichlet_nodes,
                               &n_dirichlet_nodes_local,
                               &n_dirichlet_nodes);


        if(!rank) {
            printf("[dirichlet] idx: %s\n", dirichlet_path.c_str());
            printf("[dirichlet] value: %g\n", (double)dirichlet_value);
            printf("[dirichlet] n_dirichlet_nodes: %ld\n", (long)n_dirichlet_nodes);
        }

        constraint_nodes_to_value(n_dirichlet_nodes, dirichlet_nodes, dirichlet_value, rhs.data());
        constraint_nodes_to_value(n_dirichlet_nodes, dirichlet_nodes, dirichlet_value, x.data());

        crs_constraint_nodes_to_identity(
            n_dirichlet_nodes, dirichlet_nodes, 1.0, matrix.rowptr, matrix.colidx, matrix.values);

        free((void *)dirichlet_nodes);
    } else {
        if(!rank) {
            fprintf(stderr, "Dirichlet nodes are required for well-posedness!\n");
        }

        MPI_Abort(comm, -1);
    }

    if (config["aux_out"]) {
        auto &&node = config["aux_out"];
        if (node["rhs"]) {
            auto rhs_out_path = node["rhs"].as<std::string>();
            mesh_write_nodal_field(&mesh, rhs_out_path.c_str(), SFEM_MPI_REAL_T, (void *)rhs.data());
        }
    }

    auto output_path = config["out"].as<std::string>();
    
    {
        double tick = MPI_Wtime();
        // Solve linear system
        isolver_lsolve_t lsolve;
        lsolve.comm = comm;

        isolver_lsolve_init(&lsolve);

        int max_it = 1000;
        real_t atol = 1e-14;
        real_t rtol = 1e-14;
        real_t stol = 1e-19;

        if(config["linear_solver"]) {
            auto &&node = config["linear_solver"];
            get(node, "atol", atol);
            get(node, "rtol", rtol);
            get(node, "stol", stol);
            get(node, "max_it", max_it);
        }

        isolver_lsolve_set_max_iterations(&lsolve, max_it);
        isolver_lsolve_set_atol(&lsolve, atol);
        isolver_lsolve_set_rtol(&lsolve, rtol);
        isolver_lsolve_set_stol(&lsolve, stol);
        isolver_lsolve_set_verbosity(&lsolve, 1);

        isolver_lsolve_update_crs(&lsolve, mesh.nnodes, mesh.nnodes, matrix.rowptr, matrix.colidx, matrix.values);
        isolver_lsolve_apply(&lsolve, rhs.data(), x.data());

        isolver_lsolve_destroy(&lsolve);

        double tock = MPI_Wtime();

        if (!rank) {
            printf("ssolve.cpp: Linear solver time %g\n", tock - tick);
        }
    }

    mesh_write_nodal_field(&mesh, output_path.c_str(), SFEM_MPI_REAL_T, (void *)x.data());

    {
        // clean-up
        crs_matrix_destroy(&matrix);
        mesh_destroy(&mesh);
    }

    return MPI_Finalize();
}
