#include "sfem_VectorLaplacian.hpp"
#include "sfem_Tracer.hpp"

#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"

#include "hex8_vector_laplacian.h"
#include "hex8_fff.h"

#include "sfem_Mesh.hpp"
#include "sfem_FunctionSpace.hpp"

namespace sfem {

    std::unique_ptr<Op> VectorLaplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("VectorLaplacian::create");

        assert(1 != space->block_size());
        auto ret          = std::make_unique<VectorLaplacian>(space);
        ret->element_type = (enum ElemType)space->element_type();

        int SFEM_VECTOR_LAPLACIAN_FFF = 1;
        SFEM_READ_ENV(SFEM_VECTOR_LAPLACIAN_FFF, atoi);

        if (SFEM_VECTOR_LAPLACIAN_FFF) {
            ret->fff = create_host_buffer<jacobian_t>(space->mesh_ptr()->n_elements() * 6);

            if (SFEM_SUCCESS != hex8_fff_fill(space->mesh_ptr()->n_elements(),
                                              space->mesh_ptr()->elements()->data(),
                                              space->mesh_ptr()->points()->data(),
                                              ret->fff->data())) {
                SFEM_ERROR("Unable to create fff");
            }
        }

        return ret;
    }

    std::shared_ptr<Op> VectorLaplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret          = std::make_shared<VectorLaplacian>(space);
        ret->element_type = macro_type_variant(element_type);
        return ret;
    }

    std::shared_ptr<Op> VectorLaplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret          = std::make_shared<VectorLaplacian>(space);
        ret->element_type = macro_base_elem(element_type);
        return ret;
    }

    VectorLaplacian::VectorLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
    
    VectorLaplacian::~VectorLaplacian() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("VectorLaplacian::apply called %ld times. Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    int VectorLaplacian::hessian_crs(const real_t *const  x,
                                     const count_t *const rowptr,
                                     const idx_t *const   colidx,
                                     real_t *const        values) {
        SFEM_TRACE_SCOPE("VectorLaplacian::hessian_crs");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::hessian_crs_sym(const real_t *const  x,
                                         const count_t *const rowptr,
                                         const idx_t *const   colidx,
                                         real_t *const        diag_values,
                                         real_t *const        off_diag_values) {
        SFEM_TRACE_SCOPE("VectorLaplacian::hessian_crs_sym");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("VectorLaplacian::hessian_diag");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("VectorLaplacian::gradient");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("VectorLaplacian::apply");

        double tick = MPI_Wtime();

        const int             block_size = space->block_size();
        std::vector<real_t *> vec_in(block_size), vec_out(block_size);

        // AoS
        for (int d = 0; d < block_size; d++) {
            vec_in[d]  = const_cast<real_t *>(&h[d]);
            vec_out[d] = &out[d];
        }

        auto mesh = space->mesh_ptr();
        int  err;
        if (this->fff) {
            err = affine_hex8_vector_laplacian_apply_fff(
                    // element_type,
                    mesh->n_elements(),
                    mesh->elements()->data(),
                    this->fff->data(),
                    block_size,
                    block_size,
                    vec_in.data(),
                    vec_out.data());
        } else {
            err = affine_hex8_vector_laplacian_apply(
                    // element_type,
                    mesh->n_elements(),
                    mesh->n_nodes(),
                    mesh->elements()->data(),
                    mesh->points()->data(),
                    block_size,
                    block_size,
                    vec_in.data(),
                    vec_out.data());
        }

        double tock = MPI_Wtime();
        total_time += (tock - tick);
        calls++;
        return err;
    }

    int VectorLaplacian::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("VectorLaplacian::value");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    // std::shared_ptr<Op> VectorLaplacian::clone() const {
    //     auto ret = std::make_shared<VectorLaplacian>(space);
    //     *ret     = *this;
    //     return ret;
    // }

} // namespace sfem 