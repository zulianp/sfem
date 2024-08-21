#ifndef SFEM_API_HPP
#define SFEM_API_HPP

#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_mesh.h"

#include "sfem_Function.hpp"
#include "sfem_cg.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "cu_tet4_prolongation_restriction.h"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_crs_SpMV.hpp"
#include "sfem_cuda_solver.hpp"
#endif

#include "sfem_prolongation_restriction.h"

namespace sfem {

    template <typename T>
    std::shared_ptr<Buffer<T>> create_buffer(const std::ptrdiff_t n, const MemorySpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == MEMORY_SPACE_DEVICE) return sfem::d_buffer<T>(n);
#endif  // SFEM_ENABLE_CUDA
        return sfem::h_buffer<T>(n);
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> create_buffer(const std::ptrdiff_t n, const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) return sfem::d_buffer<T>(n);
#endif  // SFEM_ENABLE_CUDA
        return sfem::h_buffer<T>(n);
    }

    std::shared_ptr<Op> create_op(const std::shared_ptr<FunctionSpace> &space,
                                  const char *name,
                                  const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) return sfem::Factory::create_op_gpu(space, name);
#endif  // SFEM_ENABLE_CUDA
        return sfem::Factory::create_op(space, name);
    }

    template <typename T>
    std::shared_ptr<ConjugateGradient<T>> create_cg(const std::shared_ptr<Operator<T>> &op,
                                                    const ExecutionSpace es) {
        std::shared_ptr<ConjugateGradient<T>> cg;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            cg = sfem::d_cg<T>();
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            cg = sfem::h_cg<T>();
        }

        cg->set_n_dofs(op->rows());
        cg->set_op(op);
        return cg;
    }

    template <typename T>
    std::shared_ptr<BiCGStab<T>> create_bcgs(const std::shared_ptr<Operator<T>> &op,
                                             const ExecutionSpace es) {
        std::shared_ptr<BiCGStab<T>> bcgs;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            bcgs = sfem::d_bcgs<T>();
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            bcgs = sfem::h_bcgs<T>();
        }

        bcgs->set_n_dofs(op->rows());
        bcgs->set_op(op);
        return bcgs;
    }

    template <typename T>
    std::shared_ptr<Chebyshev3<T>> create_cheb3(const std::shared_ptr<Operator<T>> &op,
                                                const ExecutionSpace es) {
        std::shared_ptr<Chebyshev3<T>> cheb;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            cheb = sfem::d_cheb3<T>(op);
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            cheb = sfem::h_cheb3<T>(op);
        }

        return cheb;
    }

    template <typename T>
    std::shared_ptr<Multigrid<T>> create_mg(const ExecutionSpace es) {
        std::shared_ptr<Multigrid<T>> mg;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            mg = sfem::d_mg<T>();

        } else
#endif  // SFEM_ENABLE_CUDA
        {
            mg = sfem::h_mg<T>();
        }

        return mg;
    }

    std::shared_ptr<Constraint> create_dirichlet_conditions_from_env(
            const std::shared_ptr<FunctionSpace> &space,
            const ExecutionSpace es) {
        auto conds = sfem::DirichletConditions::create_from_env(space);

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            return sfem::to_device(conds);
        }
#endif  // SFEM_ENABLE_CUDA

        return conds;
    }

    std::shared_ptr<Buffer<idx_t>> create_edge_idx(CRSGraph &crs_graph) {
        const ptrdiff_t rows = crs_graph.n_nodes();
        auto p2_vertices = h_buffer<idx_t>(crs_graph.nnz());

        build_p1_to_p2_edge_map(
                rows, crs_graph.rowptr()->data(), crs_graph.colidx()->data(), p2_vertices->data());

        return p2_vertices;
    }

    std::shared_ptr<CRSGraph> create_derefined_crs_graph(FunctionSpace &space) {
        auto et = (enum ElemType)space.element_type();
        auto coarse_et = macro_base_elem(et);
        auto crs_graph = space.mesh().create_node_to_node_graph(coarse_et);
        return crs_graph;
    }

    std::shared_ptr<Operator<real_t>> create_hierarchical_restriction(
            const ptrdiff_t n_fine_nodes,
            const int block_size,
            const std::shared_ptr<CRSGraph> &crs_graph,
            const std::shared_ptr<Buffer<idx_t>> &edges,
            const ExecutionSpace es) {
        const ptrdiff_t n_coarse_nodes = crs_graph->n_nodes();

        ptrdiff_t rows = n_coarse_nodes * block_size;
        ptrdiff_t cols = n_fine_nodes * block_size;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            auto d_edges = to_device(edges);
            auto d_crs_graph = to_device(crs_graph);

            return std::make_shared<LambdaOperator<real_t>>(
                    rows,
                    cols,
                    [=](const real_t *const from, real_t *const to) {
                        // FIXME make it generic for all elements!
                        cu_macrotet4_to_tet4_restriction(n_coarse_nodes,
                                                         d_crs_graph->rowptr()->data(),
                                                         d_crs_graph->colidx()->data(),
                                                         d_edges->data(),
                                                         block_size,
                                                         SFEM_REAL_DEFAULT,
                                                         from,
                                                         SFEM_REAL_DEFAULT,
                                                         to,
                                                         SFEM_DEFAULT_STREAM);
                    },
                    EXECUTION_SPACE_DEVICE);
        }
#endif  // SFEM_ENABLE_CUDA

        return std::make_shared<LambdaOperator<real_t>>(
                rows,
                cols,
                [=](const real_t *const from, real_t *const to) {
                    ::hierarchical_restriction_with_edge_map(n_coarse_nodes,
                                                             crs_graph->rowptr()->data(),
                                                             crs_graph->colidx()->data(),
                                                             edges->data(),
                                                             block_size,
                                                             from,
                                                             to);
                },
                EXECUTION_SPACE_HOST);
    }

    std::shared_ptr<Operator<real_t>> create_hierarchical_prolongation(
        const std::shared_ptr<Function> &function,
            const std::shared_ptr<CRSGraph> &crs_graph,
            const std::shared_ptr<Buffer<idx_t>> &edges,
            const ExecutionSpace es) {

        const ptrdiff_t n_fine_nodes = function->space()->mesh().n_nodes();
        int block_size = function->space()->block_size();
        const ptrdiff_t n_coarse_nodes = crs_graph->n_nodes();

        ptrdiff_t rows = n_fine_nodes * block_size;
        ptrdiff_t cols = n_coarse_nodes * block_size;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            auto d_edges = to_device(edges);
            auto d_crs_graph = to_device(crs_graph);

            return std::make_shared<LambdaOperator<real_t>>(
                    rows,
                    cols,
                    [=](const real_t *const from, real_t *const to) {
                        // FIXME make it generic for all elements!
                        cu_tet4_to_macrotet4_prolongation(n_coarse_nodes,
                                                          d_crs_graph->rowptr()->data(),
                                                          d_crs_graph->colidx()->data(),
                                                          d_edges->data(),
                                                          block_size,
                                                          SFEM_REAL_DEFAULT,
                                                          from,
                                                          SFEM_REAL_DEFAULT,
                                                          to,
                                                          SFEM_DEFAULT_STREAM);

                        function->apply_zero_constraints(to);
                    },
                    EXECUTION_SPACE_DEVICE);
        }
#endif  // SFEM_ENABLE_CUDA

        return std::make_shared<LambdaOperator<real_t>>(
                rows,
                cols,
                [=](const real_t *const from, real_t *const to) {
                    ::hierarchical_prolongation_with_edge_map(n_coarse_nodes,
                                                              crs_graph->rowptr()->data(),
                                                              crs_graph->colidx()->data(),
                                                              edges->data(),
                                                              block_size,
                                                              from,
                                                              to);

                    function->apply_zero_constraints(to);
                },
                EXECUTION_SPACE_HOST);
    }

    template <typename T>
    std::shared_ptr<Operator<T>> create_inverse_diagonal_scaling(
            const std::shared_ptr<Buffer<T>> &diag,
            const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            auto d_diag = to_device(diag);
            return sfem::make_op<T>(
                    d_diag->size(),
                    d_diag->size(),
                    [=](const T *const x, T *const y) {
                        auto d = d_diag->data();
                        // FIXME (only supports real_t)
                        d_ediv(d_diag->size(), x, d, y);
                    },
                    EXECUTION_SPACE_DEVICE);
        }
#endif  // SFEM_ENABLE_CUDA

        return sfem::make_op<T>(
                diag->size(),
                diag->size(),
                [=](const T *const x, T *const y) {
                    auto d = diag->data();

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < diag->size(); ++i) {
                        y[i] = x[i] / d[i];
                    }
                },
                EXECUTION_SPACE_HOST);
    }

    std::shared_ptr<Operator<real_t>> make_linear_op(const std::shared_ptr<Function> &f) {
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) { f->apply(nullptr, x, y); },
                f->execution_space());
    }

    auto crs_hessian(sfem::Function &f,
                     const std::shared_ptr<CRSGraph> &crs_graph,
                     const sfem::ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            auto d_crs_graph = sfem::to_device(crs_graph);
            auto values = sfem::create_buffer<real_t>(d_crs_graph->nnz(), es);

            f.hessian_crs(nullptr,
                          d_crs_graph->rowptr()->data(),
                          d_crs_graph->colidx()->data(),
                          values->data());

            return sfem::d_crs_spmv(d_crs_graph->n_nodes(),
                                    d_crs_graph->n_nodes(),
                                    d_crs_graph->rowptr(),
                                    d_crs_graph->colidx(),
                                    values,
                                    (real_t)1);
        }
#endif
        auto values = sfem::h_buffer<real_t>(crs_graph->nnz());

        f.hessian_crs(
                nullptr, crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data());

        // Owns the pointers
        return sfem::h_crs_spmv(crs_graph->n_nodes(),
                                crs_graph->n_nodes(),
                                crs_graph->rowptr(),
                                crs_graph->colidx(),
                                values,
                                (real_t)1);
    }

    real_t residual(sfem::Operator<real_t> &op,
                    const real_t *const rhs,
                    const real_t *const x,
                    real_t *const r) {
#ifdef SFEM_ENABLE_CUDA
        if (op.execution_space() == sfem::EXECUTION_SPACE_DEVICE) {
            d_memset(r, 0, op.rows() * sizeof(real_t));
            op.apply(x, r);
            d_axpby(op.rows(), 1, rhs, -1, r);
            return d_nrm2(op.rows(), r);
        } else
#endif
        {
            memset(r, 0, op.rows() * sizeof(real_t));
            op.apply(x, r);

            real_t ret = 0;
#pragma omp parallel for reduction(+ : ret)
            for (ptrdiff_t i = 0; i < op.rows(); i++) {
                r[i] = rhs[i] - r[i];
                ret += r[i] * r[i];
            }
            return sqrt(ret);
        }
    }

}  // namespace sfem

#endif  // SFEM_API_HPP
