#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_GeneratedNeoHookeanOgden.hpp"
#include "sfem_GeneratedNeoHookeanOgden_element_api.hpp"

#include "smesh_env.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef SFEM_ELEMENT_ASSEMBLY_BATCH_SIZE
#define SFEM_ELEMENT_ASSEMBLY_BATCH_SIZE 16
#endif

namespace {

    static constexpr int ELEMENT_BATCH_SIZE = SFEM_ELEMENT_ASSEMBLY_BATCH_SIZE;

    struct Timing {
        double best{std::numeric_limits<double>::max()};
        double total{0};
    };

    static void print_usage(const char *const argv0) {
        std::printf(
                "usage: %s\n"
                "Environment:\n"
                "  SFEM_MESH=<folder>             optional input mesh folder\n"
                "  SFEM_ELEM_TYPE=HEX8|TET4|...   generated cube element type when SFEM_MESH is unset\n"
                "  SFEM_BASE_RESOLUTION=<n>       cube resolution, default 8\n"
                "  SFEM_REPEAT=<n>                timed repeats, default 10\n"
                "  SFEM_VERIFY=0|1                scatter dense local matrices to BSR and compare, default 1\n"
                "  SFEM_MU=<value>                material mu, default 1\n"
                "  SFEM_LMBDA=<value>             material lambda, default 1\n"
                "Compile-time:\n"
                "  SFEM_ELEMENT_ASSEMBLY_BATCH_SIZE=<n>  batched dense element API vector size, default 16\n",
                argv0);
    }

    template <typename Function>
    static Timing time_repeated(const int repeat, Function &&fn) {
        Timing ret;
        for (int r = 0; r < repeat; ++r) {
            const double start = smesh::time_seconds();
            fn();
            const double elapsed = smesh::time_seconds() - start;
            ret.best             = std::min(ret.best, elapsed);
            ret.total += elapsed;
        }
        return ret;
    }

    static void fill_state(const sfem::Mesh &mesh, const int dim, real_t *const SFEM_RESTRICT state) {
        const geom_t *const *const points = const_cast<const geom_t *const *>(mesh.points()->data());
        const ptrdiff_t            nnodes = mesh.n_nodes();
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            const real_t x = dim > 0 ? real_t(points[0][node]) : real_t(0);
            const real_t y = dim > 1 ? real_t(points[1][node]) : real_t(0);
            const real_t z = dim > 2 ? real_t(points[2][node]) : real_t(0);
            if (dim > 0) state[node * dim + 0] = real_t(0.010) * x + real_t(0.002) * y;
            if (dim > 1) state[node * dim + 1] = real_t(0.003) * x + real_t(0.012) * y + real_t(0.001) * z;
            if (dim > 2) state[node * dim + 2] = real_t(0.002) * x + real_t(0.004) * y + real_t(0.011) * z;
        }
    }

    static int fill_api_shape_to_mesh_shape(const smesh::ElemType    element_type,
                                            const int                nshape,
                                            int *const SFEM_RESTRICT shape_map) {
        for (int i = 0; i < nshape; ++i) {
            shape_map[i] = i;
        }

        return SFEM_SUCCESS;
    }

    static void assign_const_streams(const int            nstreams,
                                     const ptrdiff_t      nelements,
                                     const real_t *const  storage,
                                     const real_t **const streams) {
        for (int stream = 0; stream < nstreams; ++stream) {
            streams[stream] = storage + ptrdiff_t(stream) * nelements;
        }
    }

    static void assign_mutable_streams(const int       nstreams,
                                       const ptrdiff_t nelements,
                                       real_t *const   storage,
                                       real_t **const  streams) {
        for (int stream = 0; stream < nstreams; ++stream) {
            streams[stream] = storage + ptrdiff_t(stream) * nelements;
        }
    }

    template <int BATCH_SIZE>
    struct ElementThreadContext {
        std::vector<real_t>         coords;
        std::vector<real_t>         state;
        std::vector<real_t>         matrix_soa;
        std::vector<const real_t *> coord_streams;
        std::vector<const real_t *> state_streams;
        std::vector<real_t *>       matrix_streams;

        void initialize(const int ndofs) {
            coords.resize(size_t(ndofs) * size_t(BATCH_SIZE));
            state.resize(size_t(ndofs) * size_t(BATCH_SIZE));
            matrix_soa.resize(size_t(ndofs) * size_t(ndofs) * size_t(BATCH_SIZE));
            coord_streams.resize(size_t(ndofs));
            state_streams.resize(size_t(ndofs));
            matrix_streams.resize(size_t(ndofs) * size_t(ndofs));
            assign_const_streams(ndofs, BATCH_SIZE, coords.data(), coord_streams.data());
            assign_const_streams(ndofs, BATCH_SIZE, state.data(), state_streams.data());
            assign_mutable_streams(ndofs * ndofs, BATCH_SIZE, matrix_soa.data(), matrix_streams.data());
        }
    };

    template <int BATCH_SIZE>
    struct BatchedElementAssemblyContext {
        smesh::ElemType                               element_type;
        const sfem::Mesh                             *mesh;
        int                                           dim;
        int                                           nshape;
        int                                           ndofs;
        ptrdiff_t                                     nelements;
        real_t                                        lmbda;
        real_t                                        mu;
        const int                                    *shape_map;
        const real_t                                 *state;
        real_t                                       *element_matrix_aos;
        std::vector<ElementThreadContext<BATCH_SIZE>> thread_contexts;
    };

    template <int BATCH_SIZE>
    static int assemble_element_batch(BatchedElementAssemblyContext<BATCH_SIZE> &ctx,
                                      const ptrdiff_t                            batch_begin,
                                      const int                                  batch_nelems,
                                      ElementThreadContext<BATCH_SIZE>          &thread_ctx) {
        if (ctx.dim == 2) {
            return sfem::codegen::neohookean_ogden_hessian_2d_element_soa<real_t, BATCH_SIZE>(ctx.element_type,
                                                                                              batch_nelems,
                                                                                              thread_ctx.coord_streams.data(),
                                                                                              ctx.lmbda,
                                                                                              ctx.mu,
                                                                                              thread_ctx.state_streams.data(),
                                                                                              thread_ctx.matrix_streams.data());
        }
        if (ctx.dim == 3) {
            return sfem::codegen::neohookean_ogden_hessian_3d_element_soa<real_t, BATCH_SIZE>(ctx.element_type,
                                                                                              batch_nelems,
                                                                                              thread_ctx.coord_streams.data(),
                                                                                              ctx.lmbda,
                                                                                              ctx.mu,
                                                                                              thread_ctx.state_streams.data(),
                                                                                              thread_ctx.matrix_streams.data());
        }
        (void)batch_begin;
        return SFEM_FAILURE;
    }

    template <int BATCH_SIZE>
    static void gather_batch(const BatchedElementAssemblyContext<BATCH_SIZE> &ctx,
                             const ptrdiff_t                                  batch_begin,
                             const int                                        batch_nelems,
                             ElementThreadContext<BATCH_SIZE>                &thread_ctx) {
        idx_t **const SFEM_RESTRICT elements = ctx.mesh->elements(0)->data();
        const geom_t *const *const  points   = const_cast<const geom_t *const *>(ctx.mesh->points()->data());

        for (int stream = 0; stream < ctx.ndofs; ++stream) {
            real_t *const SFEM_RESTRICT coord_stream = thread_ctx.coords.data() + ptrdiff_t(stream) * BATCH_SIZE;
            real_t *const SFEM_RESTRICT state_stream = thread_ctx.state.data() + ptrdiff_t(stream) * BATCH_SIZE;
            const int                   shape        = stream / ctx.dim;
            const int                   comp         = stream - shape * ctx.dim;
            const int                   mesh_shape   = ctx.shape_map[shape];
#pragma omp simd
            for (int lane = 0; lane < batch_nelems; ++lane) {
                const idx_t node   = elements[mesh_shape][batch_begin + lane];
                coord_stream[lane] = real_t(points[comp][node]);
                state_stream[lane] = ctx.state[node * ctx.dim + comp];
            }
        }
    }

    template <int BATCH_SIZE>
    static void store_batch_soa_to_aos(const BatchedElementAssemblyContext<BATCH_SIZE> &ctx,
                                       const ptrdiff_t                                  batch_begin,
                                       const int                                        batch_nelems,
                                       const ElementThreadContext<BATCH_SIZE>          &thread_ctx) {
        const int matrix_entries = ctx.ndofs * ctx.ndofs;

        for (int row = 0; row < ctx.ndofs; ++row) {
            for (int col = 0; col < ctx.ndofs; ++col) {
                const real_t *const SFEM_RESTRICT matrix_stream =
                        thread_ctx.matrix_soa.data() + ptrdiff_t(row * ctx.ndofs + col) * BATCH_SIZE;
#pragma omp simd
                for (int lane = 0; lane < batch_nelems; ++lane) {
                    ctx.element_matrix_aos[(batch_begin + lane) * matrix_entries + row * ctx.ndofs + col] = matrix_stream[lane];
                }
            }
        }
    }

    template <int BATCH_SIZE>
    static int assemble_batched_element_matrices(BatchedElementAssemblyContext<BATCH_SIZE> &ctx) {
        int failed = 0;
#pragma omp parallel reduction(| : failed)
        {
#ifdef _OPENMP
            const int thread_id = omp_get_thread_num();
#else
            const int thread_id = 0;
#endif
            ElementThreadContext<BATCH_SIZE> &thread_ctx = ctx.thread_contexts[size_t(thread_id)];

#pragma omp for schedule(static)
            for (ptrdiff_t batch_begin = 0; batch_begin < ctx.nelements; batch_begin += BATCH_SIZE) {
                const int batch_nelems = int(std::min<ptrdiff_t>(BATCH_SIZE, ctx.nelements - batch_begin));
                gather_batch(ctx, batch_begin, batch_nelems, thread_ctx);
                failed |= (assemble_element_batch(ctx, batch_begin, batch_nelems, thread_ctx) != SFEM_SUCCESS);
                store_batch_soa_to_aos(ctx, batch_begin, batch_nelems, thread_ctx);
            }
        }

        return failed ? SFEM_FAILURE : SFEM_SUCCESS;
    }

    struct BSRAssemblyContext {
        sfem::Function *function;
        const real_t   *state;
        const count_t  *rowptr;
        const idx_t    *colidx;
        real_t         *values;
        size_t          nvalues;
    };

    static int find_col(const count_t                    row_begin,
                        const count_t                    row_end,
                        const idx_t                      target,
                        const idx_t *const SFEM_RESTRICT colidx,
                        count_t *const SFEM_RESTRICT     out) {
        for (count_t k = row_begin; k < row_end; ++k) {
            if (colidx[k] == target) {
                *out = k;
                return SFEM_SUCCESS;
            }
        }
        return SFEM_FAILURE;
    }

    static int scatter_dense_element_matrices_to_bsr(const sfem::Mesh                 &mesh,
                                                     const int                         dim,
                                                     const int                         nshape,
                                                     const int *const SFEM_RESTRICT    shape_map,
                                                     const real_t *const SFEM_RESTRICT element_matrix_aos,
                                                     const count_t *const              rowptr,
                                                     const idx_t *const                colidx,
                                                     real_t *const                     values) {
        const ptrdiff_t             nelements      = mesh.n_elements(0);
        idx_t **const SFEM_RESTRICT elements       = mesh.elements(0)->data();
        const int                   ndofs          = dim * nshape;
        const int                   matrix_entries = ndofs * ndofs;
        std::vector<count_t>        entries(size_t(nshape) * size_t(nshape));

        for (ptrdiff_t e = 0; e < nelements; ++e) {
            const real_t *const SFEM_RESTRICT element_matrix = element_matrix_aos + e * matrix_entries;
            for (int i = 0; i < nshape; ++i) {
                const idx_t   row_node  = elements[shape_map[i]][e];
                const count_t row_begin = rowptr[row_node];
                const count_t row_end   = rowptr[row_node + 1];
                for (int j = 0; j < nshape; ++j) {
                    count_t     entry    = 0;
                    const idx_t col_node = elements[shape_map[j]][e];
                    if (find_col(row_begin, row_end, col_node, colidx, &entry) != SFEM_SUCCESS) {
                        SFEM_ERROR("missing BSR graph entry (%ld, %ld)\n", (long)row_node, (long)col_node);
                        return SFEM_FAILURE;
                    }
                    entries[size_t(i) * size_t(nshape) + size_t(j)] = entry;
                }
            }

            for (int i = 0; i < nshape; ++i) {
                for (int j = 0; j < nshape; ++j) {
                    real_t *const block = values + entries[size_t(i) * size_t(nshape) + size_t(j)] * dim * dim;
                    for (int bi = 0; bi < dim; ++bi) {
                        const int dense_row = i * dim + bi;
                        for (int bj = 0; bj < dim; ++bj) {
                            const int dense_col = j * dim + bj;
                            block[bi * dim + bj] += element_matrix[dense_row * ndofs + dense_col];
                        }
                    }
                }
            }
        }

        return SFEM_SUCCESS;
    }

    static int compare_buffers(const real_t *const a,
                               const real_t *const b,
                               const size_t        n,
                               real_t *const       max_abs,
                               real_t *const       max_rel) {
        real_t abs_err = 0;
        real_t rel_err = 0;
        for (size_t i = 0; i < n; ++i) {
            const real_t diff  = std::abs(a[i] - b[i]);
            const real_t denom = std::max(std::abs(a[i]), std::abs(b[i]));
            abs_err            = std::max(abs_err, diff);
            if (denom > real_t(0)) {
                rel_err = std::max(rel_err, diff / denom);
            }
        }
        *max_abs = abs_err;
        *max_rel = rel_err;
        return SFEM_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    auto          comm = context.communicator();

    if (argc > 1 && (!std::strcmp(argv[1], "-h") || !std::strcmp(argv[1], "--help"))) {
        print_usage(argv[0]);
        return SFEM_SUCCESS;
    }

    if (comm->size() != 1) {
        SFEM_ERROR("neohookean_assemble supports one MPI rank\n");
        return SFEM_FAILURE;
    }

    const int    repeat          = smesh::Env::read("SFEM_REPEAT", 10);
    const int    verify          = smesh::Env::read("SFEM_VERIFY", 1);
    const int    base_resolution = smesh::Env::read("SFEM_BASE_RESOLUTION", 8);
    const real_t mu              = smesh::Env::read<real_t>("SFEM_MU", real_t(1));
    const real_t lmbda           = smesh::Env::read<real_t>("SFEM_LMBDA", real_t(1));

    std::shared_ptr<sfem::Mesh> mesh;
    const std::string           mesh_path = smesh::Env::read_string("SFEM_MESH", "");
    if (!mesh_path.empty()) {
        mesh = sfem::Mesh::create_from_file(comm, smesh::Path(mesh_path.c_str()));
    } else {
        const smesh::ElemType element_type = smesh::type_from_string(smesh::Env::read_string("SFEM_ELEM_TYPE", "HEX8").c_str());
        if (element_type == smesh::TRI3 || element_type == smesh::TRI6 || element_type == smesh::QUAD4 ||
            element_type == smesh::PROTEUS_QUAD4) {
            mesh = sfem::Mesh::create_square(comm, element_type, base_resolution, base_resolution, 0, 0, 1, 1);
        } else {
            mesh = sfem::Mesh::create_cube(
                    comm, element_type, base_resolution, base_resolution, base_resolution, 0, 0, 0, 1, 1, 1);
        }
    }

    if (!mesh || mesh->n_blocks() != 1) {
        SFEM_ERROR("neohookean_assemble requires a single-block mesh\n");
        return SFEM_FAILURE;
    }

    const int             dim                      = mesh->spatial_dimension();
    const smesh::ElemType element_type             = mesh->element_type(0);
    const int             nshape                   = mesh->n_nodes_per_element(0);
    const int             ndofs_per_element        = dim * nshape;
    const ptrdiff_t       nnodes                   = mesh->n_nodes();
    const ptrdiff_t       nelements                = mesh->n_elements(0);
    const size_t          n_element_matrix_entries = size_t(ndofs_per_element) * size_t(ndofs_per_element) * size_t(nelements);

    if (dim != 2 && dim != 3) {
        SFEM_ERROR("neohookean_assemble supports 2D/3D generated NeoHookean elements\n");
        return SFEM_FAILURE;
    }

    auto space    = sfem::FunctionSpace::create(mesh, dim);
    auto function = sfem::Function::create(space);
    auto op       = sfem::create_op(space, "GeneratedNeoHookeanOgden", sfem::EXECUTION_SPACE_HOST);
    if (!op) {
        SFEM_ERROR("Could not create GeneratedNeoHookeanOgden\n");
        return SFEM_FAILURE;
    }
    op->initialize();
    if (auto *const generated = dynamic_cast<sfem::GeneratedNeoHookeanOgden *>(op.get())) {
        generated->set_value_in_block("default", "mu", mu);
        generated->set_value_in_block("default", "lmbda", lmbda);
    }
    function->add_operator(op);

    auto            graph       = space->node_to_node_graph();
    const ptrdiff_t bsr_nnz     = graph->nnz();
    const size_t    bsr_entries = size_t(bsr_nnz) * size_t(dim) * size_t(dim);

    std::vector<real_t> state(size_t(nnodes) * size_t(dim), real_t(0));
    fill_state(*mesh, dim, state.data());

    std::vector<int> shape_map;
    shape_map.resize(static_cast<size_t>(nshape));
    if (fill_api_shape_to_mesh_shape(element_type, nshape, shape_map.data()) != SFEM_SUCCESS) {
        SFEM_ERROR("Unsupported element shape ordering for %s\n", smesh::type_to_string(element_type));
        return SFEM_FAILURE;
    }

    std::vector<real_t> element_matrix_aos(n_element_matrix_entries, real_t(0));

    std::vector<real_t> bsr_values(bsr_entries, real_t(0));
    std::vector<real_t> dense_scattered_bsr;

#ifdef _OPENMP
    const int n_threads = omp_get_max_threads();
#else
    const int n_threads = 1;
#endif

    BatchedElementAssemblyContext<ELEMENT_BATCH_SIZE> element_ctx;
    element_ctx.element_type       = element_type;
    element_ctx.mesh               = mesh.get();
    element_ctx.dim                = dim;
    element_ctx.nshape             = nshape;
    element_ctx.ndofs              = ndofs_per_element;
    element_ctx.nelements          = nelements;
    element_ctx.lmbda              = lmbda;
    element_ctx.mu                 = mu;
    element_ctx.shape_map          = shape_map.data();
    element_ctx.state              = state.data();
    element_ctx.element_matrix_aos = element_matrix_aos.data();
    element_ctx.thread_contexts.resize(size_t(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        element_ctx.thread_contexts[size_t(t)].initialize(ndofs_per_element);
    }

    BSRAssemblyContext bsr_ctx{
            function.get(), state.data(), graph->rowptr()->data(), graph->colidx()->data(), bsr_values.data(), bsr_values.size()};

    auto assemble_from_elemental_matrices = [&]() -> int { return assemble_batched_element_matrices(element_ctx); };
    auto assemble_bsr                     = [&]() -> int {
        std::fill(bsr_ctx.values, bsr_ctx.values + bsr_ctx.nvalues, real_t(0));
        return bsr_ctx.function->hessian_bsr(bsr_ctx.state, bsr_ctx.rowptr, bsr_ctx.colidx, bsr_ctx.values);
    };

    if (assemble_from_elemental_matrices() != SFEM_SUCCESS) {
        SFEM_ERROR("Generated dense element API does not support element type %s\n", smesh::type_to_string(element_type));
        return SFEM_FAILURE;
    }
    if (assemble_bsr() != SFEM_SUCCESS) {
        SFEM_ERROR("current BSR Hessian assembly failed\n");
        return SFEM_FAILURE;
    }

    real_t max_abs = 0;
    real_t max_rel = 0;
    if (verify) {
        dense_scattered_bsr.assign(bsr_entries, real_t(0));
        if (scatter_dense_element_matrices_to_bsr(*mesh,
                                                  dim,
                                                  nshape,
                                                  shape_map.data(),
                                                  element_matrix_aos.data(),
                                                  graph->rowptr()->data(),
                                                  graph->colidx()->data(),
                                                  dense_scattered_bsr.data()) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        compare_buffers(bsr_values.data(), dense_scattered_bsr.data(), bsr_entries, &max_abs, &max_rel);
    }

    const Timing element_timing = time_repeated(repeat, [&]() {
        if (assemble_from_elemental_matrices() != SFEM_SUCCESS) {
            SFEM_ERROR("neohookean dense element Hessian assembly failed\n");
        }
    });
    const Timing bsr_timing     = time_repeated(repeat, [&]() {
        if (assemble_bsr() != SFEM_SUCCESS) {
            SFEM_ERROR("current BSR Hessian assembly failed\n");
        }
    });

    const double avg_element          = element_timing.total / double(repeat);
    const double avg_bsr              = bsr_timing.total / double(repeat);
    const double matrix_mbytes        = double(n_element_matrix_entries * sizeof(real_t)) / 1e6;
    const double bsr_mbytes           = double(bsr_entries * sizeof(real_t)) / 1e6;
    const double element_melems_per_s = 1e-6 * double(nelements) / avg_element;
    const double bsr_melems_per_s     = 1e-6 * double(nelements) / avg_bsr;
    const double speedup_vs_bsr       = avg_element > 0 ? avg_bsr / avg_element : 0;

    std::printf("\n");
    std::printf("+---------------------------------------------------------------+\n");
    std::printf("| NeoHookean Element Assembly Benchmark                         |\n");
    std::printf("+------------------------+--------------------------------------+\n");
    std::printf("| %-22s | %-36s |\n", "element type", smesh::type_to_string(element_type));
    std::printf("| %-22s | %36d |\n", "dimension", dim);
    std::printf("| %-22s | %36ld |\n", "nodes", (long)nnodes);
    std::printf("| %-22s | %36ld |\n", "elements", (long)nelements);
    std::printf("| %-22s | %36d |\n", "element dofs", ndofs_per_element);
    std::printf("| %-22s | %36d |\n", "batch size", ELEMENT_BATCH_SIZE);
    std::printf("| %-22s | %36d |\n", "threads", n_threads);
    std::printf("| %-22s | %36d |\n", "repeat", repeat);
    std::printf("+------------------------+--------------------------------------+\n");

    std::printf("\n");
    std::printf("+-----------------------+----------------+---------------+\n");
    std::printf("| Storage               | Entries/Blocks | Size [MB]     |\n");
    std::printf("+-----------------------+----------------+---------------+\n");
    std::printf("| %-21s | %14zu | %13.3f |\n", "dense element AoS", n_element_matrix_entries, matrix_mbytes);
    std::printf("| %-21s | %14ld | %13.3f |\n", "current BSR", (long)bsr_nnz, bsr_mbytes);
    std::printf("+-----------------------+----------------+---------------+\n");

    if (verify) {
        std::printf("\n");
        std::printf("+----------------+---------------------+\n");
        std::printf("| Verification   | Error               |\n");
        std::printf("+----------------+---------------------+\n");
        std::printf("| %-14s | %19.6e |\n", "max abs", double(max_abs));
        std::printf("| %-14s | %19.6e |\n", "max rel", double(max_rel));
        std::printf("+----------------+---------------------+\n");
    }

    std::printf("\n");
    std::printf("+-----------------------+--------------+--------------+--------------+------------+\n");
    std::printf("| Kernel                | Best [s]     | Avg [s]      | Melem/s      | Size [MB]  |\n");
    std::printf("+-----------------------+--------------+--------------+--------------+------------+\n");
    std::printf("| %-21s | %12.6e | %12.6e | %12.6f | %10.3f |\n",
                "element dense Hessian",
                element_timing.best,
                avg_element,
                element_melems_per_s,
                matrix_mbytes);
    std::printf("| %-21s | %12.6e | %12.6e | %12.6f | %10.3f |\n",
                "current BSR Hessian",
                bsr_timing.best,
                avg_bsr,
                bsr_melems_per_s,
                bsr_mbytes);
    std::printf("+-----------------------+--------------+--------------+--------------+------------+\n");
    std::printf("| %-21s | %12s | %12.6f | %12s | %10s |\n", "BSR/dense avg ratio", "", speedup_vs_bsr, "", "");
    std::printf("+-----------------------+--------------+--------------+--------------+------------+\n");
    return SFEM_SUCCESS;
}
