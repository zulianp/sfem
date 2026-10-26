#include "sfem_cg.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_solver.hpp"
#endif


namespace sfem {

    template <typename T>
    std::shared_ptr<ConjugateGradient<T>> ConjugateGradient<T>::create(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        SFEM_UNUSED(es);
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

    template std::shared_ptr<ConjugateGradient<real_t>> ConjugateGradient<real_t>::create(const std::shared_ptr<Operator<real_t>> &,
                                                                                          const ExecutionSpace);

}  // namespace sfem
