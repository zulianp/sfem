#ifndef SFEM_CUDA_SHIFTABLE_JACOBI_HPP
#define SFEM_CUDA_SHIFTABLE_JACOBI_HPP

#include "sfem_execution_context.hpp"
#include "sfem_openmp_ShiftableJacobi.hpp"

namespace sfem {
    template <typename HP, typename LP = HP>
    struct ShiftableBlockSymJacobi_CUDA {
        /// The passes on the device, ordered on `context`'s stream.
        static int build(const int                                   dim,
                         struct ShiftableBlockSymJacobi_Tpl<HP, LP>& tpl,
                         const ExecutionContext&                     context = ExecutionContext::device());
    };
    
}  // namespace sfem

#endif  // SFEM_CUDA_SHIFTABLE_JACOBI_HPP
