#include "sfem_DSS.hpp"

#include "sfem_dss.h"
#include "sfem_logger.hpp"

#include <cstdio>

namespace sfem {

DSS::DSS(const std::shared_ptr<FunctionSpace> &space) : space_(space) {}

DSS::~DSS() = default;

std::unique_ptr<Op> DSS::create(const std::shared_ptr<FunctionSpace> &space) {
    return std::make_unique<DSS>(space);
}

const char *DSS::name() const { return "DSS"; }

bool DSS::is_linear() const { return true; }

bool DSS::energy_or_potential_based() const { return false; }

ptrdiff_t DSS::n_dofs_domain() const { return space_->n_dofs(); }

ptrdiff_t DSS::n_dofs_image() const { return space_->n_dofs(); }

int DSS::hessian_crs(const real_t *const,
                     const count_t *const,
                     const idx_t *const,
                     real_t *const) {
    SFEM_ERROR("[Error] DSS::hessian_crs NOT IMPLEMENTED!\n");
    return SFEM_FAILURE;
}

int DSS::hessian_diag(const real_t *const, real_t *const) {
    SFEM_ERROR("[Error] DSS::hessian_diag NOT IMPLEMENTED!\n");
    return SFEM_FAILURE;
}

int DSS::gradient(const real_t *const, real_t *const) {
    SFEM_ERROR("[Error] DSS::gradient NOT IMPLEMENTED!\n");
    return SFEM_FAILURE;
}

int DSS::apply(const real_t *const, const real_t *const h, real_t *const out) {
    SFEM_TRACE_SCOPE("DSS::apply");

    if (space_->mesh_ptr()->n_blocks() != 1) {
        std::fprintf(stderr,
                     "DSS::apply: expected exactly one mesh block, got %zu\n",
                     space_->mesh_ptr()->n_blocks());
        return SFEM_FAILURE;
    }

    if constexpr (sizeof(real_t) == sizeof(float)) {
        return sfem_dss_apply_real32(space_->n_dofs(),
                                     reinterpret_cast<const float *>(h),
                                     reinterpret_cast<float *>(out));
    }

    if constexpr (sizeof(real_t) == sizeof(double)) {
        return sfem_dss_apply_real64(space_->n_dofs(),
                                     reinterpret_cast<const double *>(h),
                                     reinterpret_cast<double *>(out));
    }

    SFEM_ERROR("[Error] DSS::apply unsupported real_t size %zu\n", sizeof(real_t));
    return SFEM_FAILURE;
}

int DSS::value(const real_t *const, real_t *const) {
    SFEM_ERROR("[Error] DSS::value NOT IMPLEMENTED!\n");
    return SFEM_FAILURE;
}

std::shared_ptr<Op> DSS::clone() const {
    auto ret = std::make_shared<DSS>(space_);
    return ret;
}

}  // namespace sfem
