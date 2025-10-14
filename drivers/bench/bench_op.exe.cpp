#include <memory>
#include "sfem_Function.hpp"

#include "matrixio_array.h"
#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "sfem_API.hpp"
#include "sfem_Env.hpp"
#include "sfem_SFC.hpp"
#include "sfem_P1toP2.hpp"
#include "sfem_Packed.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#include <vector>

typedef struct OpDesc {
    std::string name;
    std::string type;
    int         block_size;
    int         rows, cols;
    double      elapsed;
    double      throughput;
    double      bandwidth;
    double      setup;

    template <class Op, class X, class Y>
    void measure(Op op, X x, Y y, int repeat) {
        // Warm-up
        op->apply(x, y);

        sfem::device_synchronize();

        double start = MPI_Wtime();

        for (int r = 0; r < repeat; r++) {
            op->apply(x, y);
        }

        sfem::device_synchronize();
        double stop = MPI_Wtime();
        rows        = op->rows();
        cols        = op->cols();
        elapsed     = (stop - start) / repeat;
        throughput  = 1e-6 * op->rows() / elapsed;
        bandwidth   = 1e-6 * (op->rows() + op->cols()) / elapsed;
    }

    static const char *header() {
        return "Operation          Type           Time [s]    Rate [MDOF/s]    BW [MDOF/s]    Setup [s]    Dimensions\n"
               "----------------   ------------   ---------   -------------    -----------    ---------    ------------\n";
    }

    void print(std::ostream &os) {
        char buf[256];
        snprintf(buf,
                 sizeof(buf),
                 "%-16s   %-12s   %9.3e   %13.3f    %11.3f    %9.3e    (%d, %d)\n",
                 name.c_str(),
                 type.c_str(),
                 elapsed,
                 throughput,
                 bandwidth,
                 setup,
                 rows,
                 cols);
        os << buf;
    }

} OpDesc_t;

void add_matrix_free_scalar_ops(enum ElemType                   element_type,
                                const bool                      semi_structured,
                                const enum sfem::ExecutionSpace es,
                                std::vector<OpDesc_t>          &ops) {
    ops.push_back({.name = "Laplacian", .type = MATRIX_FREE, .block_size = 1});

    if (semi_structured) {
        ops.push_back({.name = "em:Laplacian", .type = MATRIX_FREE, .block_size = 1});
    } else {
        ops.push_back({.name = "PackedLaplacian", .type = MATRIX_FREE, .block_size = 1});
    }

    if (element_type == HEX8 && !semi_structured) {
        ops.push_back({.name = "Mass", .type = MATRIX_FREE, .block_size = 1});
    }
}

void add_matrix_based_scalar_ops(enum ElemType                   element_type,
                                 const bool                      semi_structured,
                                 const enum sfem::ExecutionSpace es,
                                 std::vector<OpDesc_t>          &ops) {
    if (!semi_structured) {
        ops.push_back({.name = "Laplacian", .type = CRS, .block_size = 1});
        ops.push_back({.name = "Laplacian", .type = SPLITCRS, .block_size = 1});
        ops.push_back({.name = "Laplacian", .type = ALIGNEDCRS, .block_size = 1});
        ops.push_back({.name = "Laplacian", .type = SPLITDACRS, .block_size = 1});
        ops.push_back({.name = "Laplacian", .type = SELL, .block_size = 1});
    }
}

void add_matrix_free_vector_ops(const int                       dim,
                                enum ElemType                   element_type,
                                const bool                      semi_structured,
                                const enum sfem::ExecutionSpace es,
                                std::vector<OpDesc_t>          &ops) {
    ops.push_back({.name = "LinearElasticity", .type = MATRIX_FREE, .block_size = dim});

    if ((element_type == TET4 && !semi_structured) || element_type == HEX8) {
        ops.push_back({.name = "NeoHookeanOgden", .type = MATRIX_FREE, .block_size = dim});
    }
}

void add_matrix_based_vector_ops(const int                       dim,
                                 enum ElemType                   element_type,
                                 const bool                      semi_structured,
                                 const enum sfem::ExecutionSpace es,
                                 std::vector<OpDesc_t>          &ops) {
    if(element_type == TET10 || semi_structured) return;
    ops.push_back({.name = "LinearElasticity", .type = BSR, .block_size = dim});

    if (element_type == HEX8) {
        ops.push_back({.name = "LinearElasticity", .type = BSR_SYM, .block_size = dim});  // FIXME
    }
}

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    {
        auto comm = context.communicator();

        if (comm->size() > 1) {
            SFEM_ERROR("Parallel execution not supported!\n");
        }

        int SFEM_BASE_RESOLUTION      = sfem::Env::read("SFEM_BASE_RESOLUTION", 50);
        int SFEM_ELEMENT_REFINE_LEVEL = sfem::Env::read("SFEM_ELEMENT_REFINE_LEVEL", 0);
        int SFEM_REPEAT               = sfem::Env::read("SFEM_REPEAT", 5);

        auto es = sfem::EXECUTION_SPACE_HOST;

        const char *SFEM_EXECUTION_SPACE{nullptr};
        SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
        if (SFEM_EXECUTION_SPACE) {
            es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
        }

        std::string                 path = sfem::Env::read_string("SFEM_MESH", "");
        std::shared_ptr<sfem::Mesh> m;

        if (!path.empty()) {
            m = sfem::Mesh::create_from_file(comm, path.c_str());
        } else {
            auto SFEM_ELEM_TYPE = type_from_string(sfem::Env::read_string("SFEM_ELEM_TYPE", "HEX8").c_str());

            m = sfem::Mesh::create_cube(
                    comm, SFEM_ELEM_TYPE, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, 0, 0, 0, 1, 1, 1);
        }

        if(sfem::Env::read("SFEM_PROMOTE_TO_P2", false)) {
            m = sfem::convert_p1_mesh_to_p2(m);
        }

        if (sfem::Env::read("SFEM_USE_SFC", false)) {
            auto sfc = sfem::SFC::create_from_env();
            sfc->reorder(*m);
        }

        printf("element_type %s\n", type_to_string(m->element_type()));

        std::vector<OpDesc_t>                     ops;
        std::shared_ptr<sfem::SemiStructuredMesh> ssmesh;
        std::shared_ptr<sfem::FunctionSpace::PackedMesh> packed_mesh;

        if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
            ssmesh = sfem::SemiStructuredMesh::create(m, SFEM_ELEMENT_REFINE_LEVEL);
        } else {
            packed_mesh = sfem::FunctionSpace::PackedMesh::create(m, {}, true);
        }

        double start = MPI_Wtime();
        m->node_to_node_graph();
        double stop = MPI_Wtime();

        printf("CRS Graph creation %g [s]\n", (stop - start));

        int dim = m->spatial_dimension();
        add_matrix_free_scalar_ops(m->element_type(), SFEM_ELEMENT_REFINE_LEVEL > 1, es, ops);
        add_matrix_based_scalar_ops(m->element_type(), SFEM_ELEMENT_REFINE_LEVEL > 1, es, ops);

        add_matrix_free_vector_ops(dim, m->element_type(), SFEM_ELEMENT_REFINE_LEVEL > 1, es, ops);

        // Limited by memory
        if (m->n_nodes() * dim < 10100100) {
            add_matrix_based_vector_ops(dim, m->element_type(), SFEM_ELEMENT_REFINE_LEVEL > 1, es, ops);
        }

        for (auto &op_desc : ops) {
            std::shared_ptr<sfem::FunctionSpace> fs;
            if (ssmesh) {
                fs = sfem::FunctionSpace::create(ssmesh, op_desc.block_size);
            } else if(packed_mesh) {
                fs = sfem::FunctionSpace::create(packed_mesh, op_desc.block_size);
            } else {
                fs = sfem::FunctionSpace::create(m, op_desc.block_size);
            }

            auto f = sfem::Function::create(fs);

            auto op = sfem::create_op(fs, op_desc.name.c_str(), es);
            op->initialize();
            f->add_operator(op);

            auto x      = sfem::create_buffer<real_t>(fs->n_dofs(), es);
            auto input  = sfem::create_buffer<real_t>(fs->n_dofs(), es);
            auto output = sfem::create_buffer<real_t>(fs->n_dofs(), es);

            double start = MPI_Wtime();
            f->update(x->data());
            auto   linear_op = sfem::create_linear_operator(op_desc.type, f, x, es);
            double stop      = MPI_Wtime();
            op_desc.setup    = stop - start;

            op_desc.measure(linear_op, input->data(), output->data(), 5);
        }

        std::cout << "#elements " << m->n_elements() << "\n";
        std::cout << "#nodes " << m->n_nodes() << "\n";
        std::cout << OpDesc_t::header();
        for (auto &op_desc : ops) {
            op_desc.print(std::cout);
        }
    }

    return SFEM_SUCCESS;
}
