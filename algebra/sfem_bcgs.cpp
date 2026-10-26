#include "sfem_bcgs.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_solver_cuda.hpp"
#endif


namespace sfem {

    template <typename T>
    std::shared_ptr<BiCGStab<T>> BiCGStab<T>::create(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        SFEM_UNUSED(es);
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

    template std::shared_ptr<BiCGStab<real_t>> BiCGStab<real_t>::create(const std::shared_ptr<Operator<real_t>> &, const ExecutionSpace);

}  // namespace sfem
