#ifndef SFEM_SSMGC_HPP
#define SFEM_SSMGC_HPP

#include "sfem_Buffer.hpp"
#include "sfem_ContactConditions.hpp"
#include "sfem_Function.hpp"
#include "sfem_Input.hpp"
#include "sfem_ShiftedPenalty.hpp"
#include "sfem_ShiftedPenaltyMultigrid.hpp"

#include <memory>

namespace sfem {

    std::shared_ptr<ShiftedPenalty<real_t>> create_shifted_penalty(const std::shared_ptr<Function>         &f,
                                                                   const std::shared_ptr<ContactConditions> contact_conds,
                                                                   const std::shared_ptr<Input>            &in);

    template <typename T>
    class SSMGC final : public Operator<T> {
    public:
        ~SSMGC();
        SSMGC();

        static std::shared_ptr<SSMGC> create(const std::shared_ptr<Function>         &f,
                                             const std::shared_ptr<ContactConditions> contact_conds,
                                             const std::shared_ptr<Input>            &in);

        int            apply(const T *const rhs, T *const x) override;
        std::ptrdiff_t rows() const override;
        std::ptrdiff_t cols() const override;
        ExecutionSpace execution_space() const override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    std::shared_ptr<SSMGC<real_t>> create_ssmgc(const std::shared_ptr<Function>         &f,
                                                const std::shared_ptr<ContactConditions> contact_conds,
                                                const std::shared_ptr<Input>            &in);

}  // namespace sfem

#endif  // SFEM_SSMGC_HPP
