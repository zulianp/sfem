#ifndef SFEM_SSMGC_HPP
#define SFEM_SSMGC_HPP

#include "sfem_aliases.hpp"
#include "sfem_ContactConditions.hpp"
#include "sfem_ShiftedPenalty.hpp"
#include "sfem_ShiftedPenaltyMultigrid.hpp"
#include "sfem_Function.hpp"
#include "sfem_Input.hpp"

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
        int            update(const T *const disp);
        /// Outer (shifted-penalty) iterations of the last apply.
        int            iterations() const;
        /// An absolute tolerance for the residual, apart from the penetration's atol (0: atol).
        void           set_residual_atol(const T val);
        /// The constraints' curvature, a packed symmetric block per contact node (6 in 3D; null:
        /// none, and nothing is allocated for it): added to the fine operator, its smoother and,
        /// restricted, the coarse penalties. apply then solves (K + C) x = rhs under the contact.
        void           set_curvature(const SharedBuffer<T> &blocks);
        /// y += C x, C of set_curvature.
        void           apply_curvature(const T *const x, T *const y);
        std::ptrdiff_t rows() const override;
        std::ptrdiff_t cols() const override;
        ExecutionContext execution_context() const override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    std::shared_ptr<SSMGC<real_t>> create_ssmgc(const std::shared_ptr<Function>         &f,
                                                const std::shared_ptr<ContactConditions> contact_conds,
                                                const std::shared_ptr<Input>            &in);

}  // namespace sfem

#endif  // SFEM_SSMGC_HPP
