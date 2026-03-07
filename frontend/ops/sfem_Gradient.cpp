#include "sfem_Gradient.hpp"
#include "sfem_Tracer.hpp"

#include "crs_graph.h"
#include "sfem_macros.hpp"

#include "sfem_Buffer.hpp"
#include "tet4_patch_gradient.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_FunctionSpace.hpp"
#include "smesh_mesh.hpp"

#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"

namespace sfem {

    class Gradient::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        std::shared_ptr<MultiDomainOp> domains;
        SharedBuffer<count_t>          n2e_ptr;
        SharedBuffer<element_idx_t>    n2e_idx;
        ptrdiff_t                      max_indicence;

#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {
#if SFEM_PRINT_THROUGHPUT
            op_profiler = std::make_unique<OpTracer>(space, "Gradient::apply");
#endif
        }
        ~Impl() {}

        void print_info() { domains->print_info(); }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    ptrdiff_t Gradient::n_dofs_domain() const { return impl_->space->n_dofs(); }

    ptrdiff_t Gradient::n_dofs_image() const { return impl_->space->n_dofs() * impl_->space->mesh_ptr()->spatial_dimension(); }

    int Gradient::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("Gradient::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        //

        auto mesh = impl_->space->mesh_ptr();

        if (impl_->space->block_size() != 1) {
            SFEM_ERROR("Gradient operator only supports single block size");
            return SFEM_FAILURE;
        }

        count_t       *d_n2e_ptr;
        element_idx_t *d_n2e_idx;
        smesh::create_n2e(mesh->n_elements(),
                  mesh->n_nodes(),
                  mesh->n_nodes_per_element(0),
                  mesh->elements(0)->data(),
                  &d_n2e_ptr,
                  &d_n2e_idx);

        impl_->n2e_ptr = manage_host_buffer<count_t>(mesh->n_nodes() + 1, d_n2e_ptr);
        impl_->n2e_idx = manage_host_buffer<element_idx_t>(d_n2e_ptr[mesh->n_nodes()], d_n2e_idx);

        ptrdiff_t max_indicence = 0;
        ptrdiff_t nnodes        = impl_->space->mesh_ptr()->n_nodes();

        for (ptrdiff_t i = 0; i < nnodes; i++) {
            max_indicence = MAX(d_n2e_ptr[i + 1] - d_n2e_ptr[i], max_indicence);
        }

        impl_->max_indicence = max_indicence;

        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> Gradient::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Gradient::create");

        assert(1 == space->block_size());

        auto ret = std::make_unique<Gradient>(space);
        return ret;
    }

    std::shared_ptr<Op> Gradient::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<Gradient>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> Gradient::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<Gradient>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        return ret;
    }

    Gradient::Gradient(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {}

    Gradient::~Gradient() = default;

    int Gradient::hessian_crs(const real_t *const  x,
                              const count_t *const rowptr,
                              const idx_t *const   colidx,
                              real_t *const        values) {
        SFEM_TRACE_SCOPE("Gradient::hessian_crs");

        //     auto mesh  = impl_->space->mesh_ptr();
        //     auto graph = impl_->space->dof_to_dof_graph();
        //     int  err   = SFEM_SUCCESS;

        //     impl_->iterate([&](const OpDomain &domain) {
        //         return laplacian_crs(domain.element_type,
        //                              domain.block->n_elements(),
        //                              mesh->n_nodes(),
        //                              domain.block->elements()->data(),
        //                              mesh->points()->data(),
        //                              graph->rowptr()->data(),
        //                              graph->colidx()->data(),
        //                              values);
        //     });

        //     return err;
        return SFEM_FAILURE;
    }

    int Gradient::hessian_crs_sym(const real_t *const  x,
                                  const count_t *const rowptr,
                                  const idx_t *const   colidx,
                                  real_t *const        diag_values,
                                  real_t *const        off_diag_values) {
        SFEM_TRACE_SCOPE("Gradient::hessian_crs_sym");

        // auto mesh = impl_->space->mesh_ptr();
        // int  err  = SFEM_SUCCESS;

        // impl_->iterate([&](const OpDomain &domain) {
        //     return laplacian_crs_sym(domain.element_type,
        //                              domain.block->n_elements(),
        //                              mesh->n_nodes(),
        //                              domain.block->elements()->data(),
        //                              mesh->points()->data(),
        //                              rowptr,
        //                              colidx,
        //                              diag_values,
        //                              off_diag_values);
        // });

        // return err;

        return SFEM_FAILURE;
    }

    int Gradient::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("Gradient::hessian_diag");

        // auto mesh = impl_->space->mesh_ptr();
        // int  err  = SFEM_SUCCESS;

        // impl_->iterate([&](const OpDomain &domain) {
        //     return laplacian_diag(domain.element_type,
        //                           domain.block->n_elements(),
        //                           mesh->n_nodes(),
        //                           domain.block->elements()->data(),
        //                           mesh->points()->data(),
        //                           values);
        // });

        // return err;

        return SFEM_FAILURE;
    }

    int Gradient::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Gradient::gradient");

        //     auto mesh = impl_->space->mesh_ptr();
        //     return impl_->iterate([&](const OpDomain &domain) {
        //         return laplacian_assemble_gradient(domain.element_type,
        //                                            domain.block->n_elements(),
        //                                            mesh->n_nodes(),
        //                                            domain.block->elements()->data(),
        //                                            mesh->points()->data(),
        //                                            x,
        //                                            out);
        //     });

        return SFEM_FAILURE;
    }

    int Gradient::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Gradient::apply");
        SFEM_OP_CAPTURE();

        auto mesh = impl_->space->mesh_ptr();
        // return impl_->iterate([&](const OpDomain &domain) {
        // if (domain.user_data) {
        //     auto fff = std::static_pointer_cast<Buffer<jacobian_t>>(domain.user_data);
        //     return laplacian_apply_opt(
        //             domain.element_type, domain.block->n_elements(), domain.block->elements()->data(), fff->data(), h, out);
        // }

        // return laplacian_apply(domain.element_type,
        //                        domain.block->n_elements(),
        //                        mesh->n_nodes(),
        //                        domain.block->elements()->data(),
        //                        mesh->points()->data(),
        //                        h,
        //                        out);
        // });

        auto block   = mesh->block(0);
        auto n2e_ptr = impl_->n2e_ptr;
        auto n2e_idx = impl_->n2e_idx;

        return tet4_patch_gradient(block->n_elements(),
                                   block->elements()->data(),
                                   mesh->n_nodes(),
                                   impl_->max_indicence,
                                   n2e_ptr->data(),
                                   n2e_idx->data(),
                                   mesh->points()->data(),
                                   h,
                                   3,
                                   &out[0],
                                   &out[1],
                                   &out[2]);
    }

    int Gradient::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("Gradient::value");

        // auto mesh = impl_->space->mesh_ptr();
        // return impl_->iterate([&](const OpDomain &domain) {
        //     return laplacian_assemble_value(domain.element_type,
        //                                     domain.block->n_elements(),
        //                                     mesh->n_nodes(),
        //                                     domain.block->elements()->data(),
        //                                     mesh->points()->data(),
        //                                     x,
        //                                     out);
        // });

        return SFEM_FAILURE;
    }

    int Gradient::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> Gradient::clone() const {
        SFEM_ERROR("IMPLEMENT ME!\n");
        return nullptr;
    }

    void Gradient::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void Gradient::override_element_types(const std::vector<smesh::ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }
}  // namespace sfem
