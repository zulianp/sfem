#pragma once
#include "sfem_Op.hpp"

namespace sfem {
class SemiStructuredLaplacian : public Op {
public:
    std::shared_ptr<FunctionSpace> space;
    enum ElemType element_type { INVALID };
    bool use_affine_approximation{true};
    bool use_stencil{true};
    std::shared_ptr<Buffer<jacobian_t>> fff;
    long calls{0};
    double total_time{0};
    ~SemiStructuredLaplacian();
    static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);
    std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override;
    std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override;
    const char *name() const override;
    inline bool is_linear() const override { return true; }
    int initialize() override;
    SemiStructuredLaplacian(const std::shared_ptr<FunctionSpace> &space);
    int hessian_crs(const real_t *const x, const count_t *const rowptr, const idx_t *const colidx, real_t *const values) override;
    int hessian_diag(const real_t *const, real_t *const out) override;
    int gradient(const real_t *const x, real_t *const out) override;
    int apply(const real_t *const x, const real_t *const h, real_t *const out) override;
    int value(const real_t *x, real_t *const out) override;
    int report(const real_t *const) override;
    std::shared_ptr<Op> clone() const override;
};
} // namespace sfem 