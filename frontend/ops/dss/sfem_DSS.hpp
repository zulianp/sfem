#pragma once

#include "sfem_Op.hpp"

namespace sfem {

class DSS final : public Op {
public:
    explicit DSS(const std::shared_ptr<FunctionSpace> &space);
    ~DSS() override;

    static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

    const char *name() const override;
    bool is_linear() const override;
    bool energy_or_potential_based() const override;
    ptrdiff_t n_dofs_domain() const override;
    ptrdiff_t n_dofs_image() const override;

    int hessian_crs(const real_t *const x,
                    const count_t *const rowptr,
                    const idx_t *const colidx,
                    real_t *const values) override;
    int hessian_diag(const real_t *const x, real_t *const values) override;
    int gradient(const real_t *const x, real_t *const out) override;
    int apply(const real_t *const x, const real_t *const h, real_t *const out) override;
    int value(const real_t *const x, real_t *const out) override;
    std::shared_ptr<Op> clone() const override;

private:
    std::shared_ptr<FunctionSpace> space_;
};

}  // namespace sfem
