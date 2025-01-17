#ifndef SFEM_SSMGC_HPP
#define SFEM_SSMGC_HPP

#include "sfem_Buffer.hpp"
#include "sfem_ContactConditions.hpp"
#include "sfem_Function.hpp"
#include "sfem_Input.hpp"
#include "sfem_ShiftedPenaltyMultigrid.hpp"

#include <memory>

namespace sfem {
    std::shared_ptr<ShiftedPenaltyMultigrid<real_t>> create_ssmgc(const std::shared_ptr<Function>         &f,
                                                                  const std::shared_ptr<ContactConditions> contact_conds,
                                                                  const enum ExecutionSpace                es,
                                                                  std::shared_ptr<Input>                  &in);
}

#endif  // SFEM_SSMGC_HPP
