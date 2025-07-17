#pragma once
#include "sfem_Op.hpp"

namespace sfem {
    class CVFEMUpwindConvection final : public Op {
    public:
        std::shared_ptr<FunctionSpace>  space;
        std::shared_ptr<Buffer<real_t>> vel[3];
        enum ElemType                   element_type { INVALID };
        const char                     *name() const override { return "CVFEMUpwindConvection"; }
        inline bool                     is_linear() const override { return true; }
        void set_field(const char *name, const std::shared_ptr<Buffer<real_t>> &v, const int component) override;
        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);
        int                        initialize(const std::vector<std::string> &block_names = {}) override;
        CVFEMUpwindConvection(const std::shared_ptr<FunctionSpace> &space);
        ~CVFEMUpwindConvection();
        int                 hessian_crs(const real_t *const  x,
                                        const count_t *const rowptr,
                                        const idx_t *const   colidx,
                                        real_t *const        values) override;
        int                 gradient(const real_t *const x, real_t *const out) override;
        int                 apply(const real_t *const x, const real_t *const h, real_t *const out) override;
        int                 value(const real_t *x, real_t *const out) override;
        int                 report(const real_t *const) override;
        std::shared_ptr<Op> clone() const override;
    };
}  // namespace sfem