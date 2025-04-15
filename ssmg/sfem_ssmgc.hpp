#ifndef SFEM_SSMGC_HPP
#define SFEM_SSMGC_HPP

#include "sfem_Buffer.hpp"
#include "sfem_ContactConditions.hpp"
#include "sfem_Function.hpp"
#include "sfem_Input.hpp"
#include "sfem_ShiftedPenaltyMultigrid.hpp"
#include "sfem_ShiftedPenalty.hpp"

#include <memory>

namespace sfem {

    std::shared_ptr<ShiftedPenalty<real_t>> create_shifted_penalty(
            const std::shared_ptr<Function>         &f,
            const std::shared_ptr<ContactConditions> contact_conds,
            const std::shared_ptr<Input>            &in);

    std::shared_ptr<ShiftedPenaltyMultigrid<real_t>> create_ssmgc(const std::shared_ptr<Function>         &f,
                                                                  const std::shared_ptr<ContactConditions> contact_conds,
                                                                  const std::shared_ptr<Input>            &in);
}  // namespace sfem

#endif  // SFEM_SSMGC_HPP
