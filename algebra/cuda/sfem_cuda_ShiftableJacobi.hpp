#ifndef SFEM_CUDA_SHIFTABLE_JACOBI_HPP
#define SFEM_CUDA_SHIFTABLE_JACOBI_HPP

#include "sfem_openmp_ShiftableJacobi.hpp"

namespace sfem {
    template <typename HP, typename LP = HP>
    struct ShiftableBlockSymJacobi_CUDA {
        static int build(const int dim, struct ShiftableBlockSymJacobi_Tpl<HP, LP>& tpl);
    };
    
}  // namespace sfem

#endif  // SFEM_CUDA_SHIFTABLE_JACOBI_HPP
