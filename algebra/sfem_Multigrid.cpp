#include "sfem_Multigrid.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_solver.hpp"
#endif


namespace sfem {

    template <typename T>
    std::shared_ptr<Multigrid<T>> Multigrid<T>::create(const ExecutionSpace es) {
        SFEM_UNUSED(es);
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

    template std::shared_ptr<Multigrid<real_t>> Multigrid<real_t>::create(const ExecutionSpace);

}  // namespace sfem
