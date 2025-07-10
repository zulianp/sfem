#pragma once
#include "sfem_Op.hpp"

namespace sfem {
    class CVFEMMass final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType                  element_type { INVALID };
        const char                    *name() const override { return "CVFEMMass"; }
        inline bool                    is_linear() const override { return true; }
        static std::unique_ptr<Op>     create(const std::shared_ptr<FunctionSpace> &space);
        int                            initialize(const std::vector<std::string> &block_names = {}) override;
        CVFEMMass(const std::shared_ptr<FunctionSpace> &space);
        int                 hessian_diag(const real_t *const, real_t *const values) override;
        int                 hessian_crs(const real_t *const  x,
                                        const count_t *const rowptr,
                                        const idx_t *const   colidx,
                                        real_t *const        values) override;
        int                 gradient(const real_t *const x, real_t *const out) override;
        int                 apply(const real_t *const x, const real_t *const h, real_t *const out) override;
        int                 value(const real_t *x, real_t *const out) override;
        int                 report(const real_t *const) override;
        std::shared_ptr<Op> clone() const override;
        ~CVFEMMass() override;
    };
}  // namespace sfem