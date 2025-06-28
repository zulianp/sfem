#include <memory>
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

#include <vector>

typedef struct OpDesc {
    std::string name;
    std::string type;
    int         block_size;
    int         rows, cols;
    double      elapsed;
    double      throughput;
    double      bandwidth;

    template <class Op, class X, class Y>
    void measure(Op op, X x, Y y, int repeat) {
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
        return "Operation          Type      Time [s]    Rate [MDOF/s]    BW [MDOF/s]    Dimensions\n"
               "----------------   -------   ---------   -------------    -----------    ------------\n";
    }

    void print(std::ostream &os) {
        char buf[256];
        snprintf(buf,
                 sizeof(buf),
                 "%-16s   %-7s   %9.3e   %13.3f    %11.3f    (%d, %d)\n",
                 name.c_str(),
                 type.c_str(),
                 elapsed,
                 throughput,
                 bandwidth,
                 rows,
                 cols);
        os << buf;
    }

} OpDesc_t;

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    {
        auto comm = context.comm();

        int SFEM_BASE_RESOLUTION = 50;
        SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

        int SFEM_ELEMENT_REFINE_LEVEL = 0;
        SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

        sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;

        const char *SFEM_EXECUTION_SPACE{nullptr};
        SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
        if (SFEM_EXECUTION_SPACE) {
            es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
        }

        auto m = sfem::Mesh::create_hex8_cube(
                comm, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, 0, 0, 0, 1, 1, 1);

        std::vector<OpDesc_t> ops({{.name = "Laplacian", .type = "MF", .block_size = 1},
                                   {.name = "LinearElasticity", .type = "MF", .block_size = 3},
                                   {.name = "LinearElasticity", .type = "BSR", .block_size = 3}});

        if (SFEM_ELEMENT_REFINE_LEVEL > 1) {
            ops.push_back({.name = "em:Laplacian", .type = "MF", .block_size = 1});
        } else {
            ops.push_back({.name = "Laplacian", .type = "CRS", .block_size = 1});
            ops.push_back({.name = "Mass", .type = "MF", .block_size = 1});
            ops.push_back({.name = "LumpedMass", .type = "MF", .block_size = 1});
        }

        for (auto &op_desc : ops) {
            // FIXME: It should be possible to construct by passing semistructured mesh
            auto fs = sfem::FunctionSpace::create(m, op_desc.block_size);

            if (SFEM_ELEMENT_REFINE_LEVEL > 0) {
                fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
            }

            auto f = sfem::Function::create(fs);

            auto op = sfem::create_op(fs, op_desc.name.c_str(), es);
            op->initialize();
            f->add_operator(op);

            auto x         = sfem::create_buffer<real_t>(fs->n_dofs(), es);
            auto input     = sfem::create_buffer<real_t>(fs->n_dofs(), es);
            auto output    = sfem::create_buffer<real_t>(fs->n_dofs(), es);
            auto linear_op = sfem::create_linear_operator(op_desc.type, f, x, es);

            op_desc.measure(linear_op, input->data(), output->data(), 5);
        }

        std::cout << OpDesc_t::header();
        for (auto &op_desc : ops) {
            op_desc.print(std::cout);
        }
    }

    return SFEM_SUCCESS;
}
