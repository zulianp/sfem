#include "sfem_CVFEMMass.hpp"
#include <mpi.h>
#include "cvfem_operators.h"
#include "sfem_Mesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

namespace sfem {

    std::unique_ptr<Op> CVFEMMass::create(const std::shared_ptr<FunctionSpace> &space) {
        assert(1 == space->block_size());

        auto ret          = std::make_unique<CVFEMMass>(space);
        ret->element_type = (enum ElemType)space->element_type();
        return ret;
    }

    int CVFEMMass::initialize(const std::vector<std::string> &block_names) { return SFEM_SUCCESS; }

    CVFEMMass::CVFEMMass(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    int CVFEMMass::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        auto mesh = space->mesh_ptr();
        cvfem_cv_volumes(
                element_type, mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), values);

        return SFEM_SUCCESS;
    }

    int CVFEMMass::hessian_crs(const real_t *const  x,
                               const count_t *const rowptr,
                               const idx_t *const   colidx,
                               real_t *const        values) {
        assert(0);
        return SFEM_FAILURE;
    }

    int CVFEMMass::gradient(const real_t *const x, real_t *const out) {
        assert(0);
        return SFEM_FAILURE;
    }

    int CVFEMMass::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        assert(0);
        return SFEM_FAILURE;
    }

    int CVFEMMass::value(const real_t *x, real_t *const out) {
        assert(0);
        return SFEM_FAILURE;
    }

    int CVFEMMass::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> CVFEMMass::clone() const {
        auto ret = std::make_shared<CVFEMMass>(space);
        *ret     = *this;
        return ret;
    }

    CVFEMMass::~CVFEMMass() = default;

}  // namespace sfem