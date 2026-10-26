#include "sfem_Chebyshev3.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_solver_cuda.hpp"
#endif


namespace sfem {

    template <typename T>
    std::shared_ptr<Chebyshev3<T>> Chebyshev3<T>::create(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        SFEM_UNUSED(es);
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

    template std::shared_ptr<Chebyshev3<real_t>> Chebyshev3<real_t>::create(const std::shared_ptr<Operator<real_t>> &, const ExecutionSpace);

}  // namespace sfem
