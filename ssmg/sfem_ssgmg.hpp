#ifndef SFEM_SSGMG_HPP
#define SFEM_SSGMG_HPP

#include "sfem_Multigrid.hpp"
#include "sfem_Function.hpp"
#include "sfem_Input.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_blas.hpp"
#endif

namespace sfem {
    std::shared_ptr<Multigrid<real_t>> create_ssgmg(const std::shared_ptr<Function> &f,
                                                    const enum ExecutionSpace        es,
                                                    const std::shared_ptr<Input>    &in = nullptr);
}  // namespace sfem

#endif  // SFEM_SSGMG_HPP
