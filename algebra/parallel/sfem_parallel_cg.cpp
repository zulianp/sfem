#include "sfem_parallel_cg.hpp"


namespace sfem {

    template <typename T>
    std::shared_ptr<ParallelConjugateGradient<T>> ParallelConjugateGradient<T>::create(const std::shared_ptr<ParallelOperator<T>> &op) {
        if (!op) {
            SFEM_ERROR("create_parallel_cg: null operator\n");
            return nullptr;
        }

        if (op->execution_space() != EXECUTION_SPACE_HOST) {
            SFEM_ERROR("create_parallel_cg: only EXECUTION_SPACE_HOST is supported currently\n");
            return nullptr;
        }

        auto cg = h_parallel_cg<T>();
        cg->set_n_dofs(op->rows());
        cg->set_op(op);
        return cg;
    }

    template std::shared_ptr<ParallelConjugateGradient<real_t>> ParallelConjugateGradient<real_t>::create(
            const std::shared_ptr<ParallelOperator<real_t>> &);

}  // namespace sfem
