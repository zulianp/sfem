#ifndef SFEM_API_HPP
#define SFEM_API_HPP

#include "sfem_Buffer.hpp"
#include "sfem_base.h"

#include "sfem_Function.hpp"
#include "sfem_cg.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

namespace sfem {

    template <typename T>
    std::shared_ptr<Buffer<T>> create_buffer(const std::ptrdiff_t n, const MemorySpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == MEMORY_SPACE_DEVICE) return sfem::d_buffer<T>(n);
#endif //SFEM_ENABLE_CUDA
        return sfem::h_buffer<T>(n);
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> create_buffer(const std::ptrdiff_t n, const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) return sfem::d_buffer<T>(n);
#endif //SFEM_ENABLE_CUDA
        return sfem::h_buffer<T>(n);
    }

    std::shared_ptr<Op> create_op(const std::shared_ptr<FunctionSpace> &space,
                                  const char *name,
                                  const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) return sfem::Factory::create_op_gpu(space, name);
#endif //SFEM_ENABLE_CUDA
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
#endif //SFEM_ENABLE_CUDA
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
#endif //SFEM_ENABLE_CUDA
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
#endif //SFEM_ENABLE_CUDA
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
            // mg = sfem::d_mg<T>();
            assert(false && "IMPLEMENT ME");
        } else
#endif //SFEM_ENABLE_CUDA
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

    std::shared_ptr<Operator<real_t>> create_hierarchical_restriction(
            const std::shared_ptr<Function> &fine_function,
            const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            assert(false && "IMPLEMENT ME");
        }
#endif  // SFEM_ENABLE_CUDA
        return fine_function->hierarchical_restriction();
    }

    std::shared_ptr<Operator<real_t>> create_hierarchical_prolongation(
            const std::shared_ptr<Function> &fine_function,
            const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            assert(false && "IMPLEMENT ME");
        }
#endif  // SFEM_ENABLE_CUDA
        return fine_function->hierarchical_prolongation();
    }

    template <typename T>
    std::shared_ptr<Operator<T>> create_inverse_diagonal_scaling(
            const std::shared_ptr<Buffer<T>> &diag,
            const ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            return sfem::make_op<T>(diag->size(), diag->size(), [=](const T *const x, T *const y) {
                auto d = diag->data();
                // FIXME (only supports real_t)
                d_ediv(diag->size(), x, d, y);
            });
        }
#endif  // SFEM_ENABLE_CUDA

        return sfem::make_op<T>(diag->size(), diag->size(), [=](const T *const x, T *const y) {
            auto d = diag->data();

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < diag->size(); ++i) {
                y[i] = x[i] / d[i];
            }
        });
    }

    std::shared_ptr<Operator<real_t>> make_linear_op(const std::shared_ptr<Function> &f) {
        return sfem::make_op<real_t>(
                f->space()->n_dofs(),
                f->space()->n_dofs(),
                [=](const real_t *const x, real_t *const y) { f->apply(nullptr, x, y); });
    }

}  // namespace sfem

#endif  // SFEM_API_HPP
