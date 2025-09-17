#ifndef SFEM_CONSTRAINT_HPP
#define SFEM_CONSTRAINT_HPP

#include <memory>
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_mask.h"
#include "sfem_FunctionSpace.hpp"

namespace sfem {

class Constraint {
public:
    virtual ~Constraint() = default;
    virtual int value(const real_t *const /*x*/, real_t *const /*out*/) { SFEM_ERROR("IMEPLEMENT ME"); return SFEM_FAILURE; }
    virtual int apply(real_t *const x) = 0;
    virtual int apply_value(const real_t value, real_t *const x) = 0;
    virtual int apply_zero(real_t *const x);
    virtual int gradient(const real_t *const x, real_t *const g) = 0;
    virtual int copy_constrained_dofs(const real_t *const src, real_t *const dest) = 0;
    virtual int mask(mask_t *mask) = 0;

    virtual int hessian_crs(const real_t *const  x,
                            const count_t *const rowptr,
                            const idx_t *const   colidx,
                            real_t *const        values) = 0;

    virtual int hessian_bsr(const real_t *const /*x*/,
                            const count_t *const /*rowptr*/,
                            const idx_t *const /*colidx*/,
                            real_t *const /*values*/) {
        assert(false);
        return SFEM_FAILURE;
    }

    virtual std::shared_ptr<Constraint> derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
                                                 const bool as_zero) const = 0;
    virtual std::shared_ptr<Constraint> lor() const = 0;
};

} // namespace sfem

#endif // SFEM_CONSTRAINT_HPP 