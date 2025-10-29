#include "sfem_Laplacian.hpp"
#include "sfem_Tracer.hpp"

<<<<<<< HEAD
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"
=======
#include "hex8_fff.h"
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_mesh.h"
#include "tet4_fff.h"
>>>>>>> origin/main

#include "laplacian.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"

#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"

namespace sfem {

    class Laplacian::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;  ///< Function space for the operator
        std::shared_ptr<MultiDomainOp> domains;
#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {
#if SFEM_PRINT_THROUGHPUT
            op_profiler = std::make_unique<OpTracer>(space, "Laplacian::apply");
#endif
        }
        ~Impl() {}

        void print_info() { domains->print_info(); }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    int Laplacian::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("Laplacian::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
<<<<<<< HEAD
=======

        const int dim  = impl_->space->mesh_ptr()->spatial_dimension();
        auto      mesh = impl_->space->mesh_ptr();

        for (auto &n2d : impl_->domains->domains()) {
            auto &domain       = n2d.second;
            auto element_type = domain.element_type;
            auto block        = domain.block;
            auto fff          = create_host_buffer<jacobian_t>(block->n_elements() * 6);
            
            if (element_type == HEX8 || element_type == SSHEX8) {
                hex8_fff_fill(block->n_elements(), block->elements()->data(), mesh->points()->data(), fff->data());
            } else {
                tet4_fff_fill(block->n_elements(), block->elements()->data(), mesh->points()->data(), fff->data());
            }

            domain.user_data = std::static_pointer_cast<void>(fff);
        }

>>>>>>> origin/main
        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> Laplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Laplacian::create");

        assert(1 == space->block_size());

        auto ret = std::make_unique<Laplacian>(space);
        return ret;
    }

    std::shared_ptr<Op> Laplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<Laplacian>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> Laplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::make_shared<Laplacian>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        return ret;
    }

    Laplacian::Laplacian(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {}

    Laplacian::~Laplacian() = default;

    int Laplacian::hessian_crs(const real_t *const  x,
                               const count_t *const rowptr,
                               const idx_t *const   colidx,
                               real_t *const        values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_crs");

        auto mesh  = impl_->space->mesh_ptr();
        auto graph = impl_->space->dof_to_dof_graph();
        int  err   = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            return laplacian_crs(domain.element_type,
                                 domain.block->n_elements(),
                                 mesh->n_nodes(),
                                 domain.block->elements()->data(),
                                 mesh->points()->data(),
                                 graph->rowptr()->data(),
                                 graph->colidx()->data(),
                                 values);
        });

        return err;
    }

    int Laplacian::hessian_crs_sym(const real_t *const  x,
                                   const count_t *const rowptr,
                                   const idx_t *const   colidx,
                                   real_t *const        diag_values,
                                   real_t *const        off_diag_values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_crs_sym");

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            return laplacian_crs_sym(domain.element_type,
                                     domain.block->n_elements(),
                                     mesh->n_nodes(),
                                     domain.block->elements()->data(),
                                     mesh->points()->data(),
                                     rowptr,
                                     colidx,
                                     diag_values,
                                     off_diag_values);
        });

        return err;
    }

    int Laplacian::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;

        impl_->iterate([&](const OpDomain &domain) {
            return laplacian_diag(domain.element_type,
                                  domain.block->n_elements(),
                                  mesh->n_nodes(),
                                  domain.block->elements()->data(),
                                  mesh->points()->data(),
                                  values);
        });

        return err;
    }

    int Laplacian::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::gradient");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            return laplacian_assemble_gradient(domain.element_type,
                                               domain.block->n_elements(),
                                               mesh->n_nodes(),
                                               domain.block->elements()->data(),
                                               mesh->points()->data(),
                                               x,
                                               out);
        });
    }

    int Laplacian::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::apply");
        SFEM_OP_CAPTURE();

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
<<<<<<< HEAD
=======
            if (domain.user_data) {
                auto fff = std::static_pointer_cast<Buffer<jacobian_t>>(domain.user_data);
                return laplacian_apply_opt(
                        domain.element_type, domain.block->n_elements(), domain.block->elements()->data(), fff->data(), h, out);
            }
            
>>>>>>> origin/main
            return laplacian_apply(domain.element_type,
                                   domain.block->n_elements(),
                                   mesh->n_nodes(),
                                   domain.block->elements()->data(),
                                   mesh->points()->data(),
                                   h,
                                   out);
        });
    }

    int Laplacian::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("Laplacian::value");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            return laplacian_assemble_value(domain.element_type,
                                            domain.block->n_elements(),
                                            mesh->n_nodes(),
                                            domain.block->elements()->data(),
                                            mesh->points()->data(),
                                            x,
                                            out);
        });
    }

    int Laplacian::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> Laplacian::clone() const {
        SFEM_ERROR("IMPLEMENT ME!\n");
        return nullptr;
    }

    void Laplacian::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void Laplacian::override_element_types(const std::vector<enum ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }
}  // namespace sfem
