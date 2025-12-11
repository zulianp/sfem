#include "sfem_MooneyRivlinVisco.hpp"

#include "mooney_rivlin_visco.h"
#include "sfem_defs.h"
#include "sfem_logger.h"
#include "sfem_macros.h"
#include "sfem_mesh.h"

#include "sfem_CRSGraph.hpp"
#include "sfem_Env.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"
#include "sfem_Buffer.hpp"
#include "sfem_API.hpp" // Added for blas

#include <math.h>
#include <mpi.h>
#include <vector>

namespace sfem {

    class MooneyRivlinVisco::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        
        // Material parameters
        real_t C10{1}, K{1}, C01{1}, dt{0.1};
        
        // Prony Series parameters
        int num_prony_terms{0};
        std::vector<real_t> prony_g;
        std::vector<real_t> prony_tau;
        
        // Use flexible hessian (loop-based) vs fixed (unrolled)
        bool use_flexible_hessian{false};
        
        // History buffer: managed internally
        // Stores [S_dev_n (6), H_1^n (6), H_2^n (6), ...] per quadrature point
        std::shared_ptr<Buffer<real_t>> history_buffer;
        std::shared_ptr<Buffer<real_t>> new_history_buffer; // Temp buffer for update

        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
        ~Impl();
        
        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
        
        void allocate_history_buffers() {
            // Calculate total quadrature points across all elements
            auto mesh = space->mesh_ptr();
            // FIXME: Assuming HEX8 for now (8 nodes, 27 or 8 QPs depending on order)
            // For now we assume standard HEX8 with 27 QPs (3x3x3) or 8 QPs (2x2x2)
            // We need to ask the element type about its quadrature count.
            // This is a simplification. A robust impl would query quadrature size.
            
            ptrdiff_t total_elements = 0;
            iterate([&](const OpDomain &domain) -> int {
                total_elements += domain.block->n_elements();
                return SFEM_SUCCESS;
            });

            // Assuming HEX8 with 27 QPs (line_q3) for higher accuracy or 8 (line_q2)
            // From hex8_mooney_rivlin_visco.c, it uses line_q2 (2 points per edge) -> 8 QPs total
            const int n_qp = 8; 
            
            // History size per QP: 6 (S_dev) + num_prony * 6 (H_i)
            const ptrdiff_t history_per_qp = 6 + num_prony_terms * 6;
            const ptrdiff_t total_size = total_elements * n_qp * history_per_qp;
            
            history_buffer = create_buffer<real_t>(total_size, sfem::EXECUTION_SPACE_HOST);
            new_history_buffer = create_buffer<real_t>(total_size, sfem::EXECUTION_SPACE_HOST);
            
            // Initialize with zeros
            auto blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
            blas->zeros(total_size, history_buffer->data());
            blas->zeros(total_size, new_history_buffer->data());
        }
        
        void swap_history_buffers() {
            std::swap(history_buffer, new_history_buffer);
        }
    };

    std::unique_ptr<Op> MooneyRivlinVisco::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::create");
        auto ret = std::make_unique<MooneyRivlinVisco>(space);
        // Read env vars or defaults
        ret->impl_->C10 = sfem::Env::read("SFEM_MOONEY_RIVLIN_C10", ret->impl_->C10);
        ret->impl_->K   = sfem::Env::read("SFEM_MOONEY_RIVLIN_K", ret->impl_->K);
        ret->impl_->C01 = sfem::Env::read("SFEM_MOONEY_RIVLIN_C01", ret->impl_->C01);
        ret->impl_->dt  = sfem::Env::read("SFEM_DT", ret->impl_->dt);
        ret->impl_->use_flexible_hessian = sfem::Env::read("SFEM_USE_FLEXIBLE_HESSIAN", 0) != 0;
        return ret;
    }

    MooneyRivlinVisco::MooneyRivlinVisco(const std::shared_ptr<FunctionSpace> &space) 
        : impl_(std::make_unique<Impl>(space)) {}

    MooneyRivlinVisco::~MooneyRivlinVisco() = default;

    int MooneyRivlinVisco::initialize(const std::vector<std::string> &block_names) {
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        return SFEM_SUCCESS;
    }
    
    void MooneyRivlinVisco::initialize_history() {
        impl_->allocate_history_buffers();
    }

    int MooneyRivlinVisco::hessian_crs(const real_t *const x,
                                       const count_t *const rowptr,
                                       const idx_t *const colidx,
                                       real_t *const values) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::hessian_crs");
        // Currently delegating to BSR if structure matches, or we might need general implementation
        // Since the C backend is BSR, we should use BSR assembly generally.
        // If the user requests CRS values, we assume they want BSR packed into CRS array 
        // (which is valid for block solvers) OR we implement a true CRS assembly.
        // For now, let's reuse BSR logic as it's the primary path for 3D elasticity.
        return hessian_bsr(x, rowptr, colidx, values);
    }

    int MooneyRivlinVisco::hessian_bsr(const real_t *const x,
                                       const count_t *const rowptr,
                                       const idx_t *const colidx,
                                       real_t *const values) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::hessian_bsr");
        auto mesh = impl_->space->mesh_ptr();
        
        if(!impl_->history_buffer) {
            SFEM_ERROR("History buffer not initialized! Call initialize_history() first.");
            return SFEM_FAILURE;
        }

        ptrdiff_t history_offset = 0;
        // We assume line_q2 (8 QPs) for HEX8. 
        const int n_qp = 8; 
        const ptrdiff_t history_per_qp = 6 + impl_->num_prony_terms * 6;
        // history_stride is the history size per ELEMENT (not cumulative offset!)
        const ptrdiff_t history_stride = n_qp * history_per_qp;

        return impl_->iterate([&](const OpDomain &domain) -> int {
            const ptrdiff_t nelements = domain.block->n_elements();
            
            int ret;
            if (impl_->use_flexible_hessian) {
                // Use loop-based flexible version (supports arbitrary Prony terms)
                ret = mooney_rivlin_visco_bsr_flexible(
                    domain.element_type,
                    nelements,
                    mesh->n_nodes(),
                    domain.block->elements()->data(),
                    mesh->points()->data(),
                    impl_->C10,
                    impl_->K,
                    impl_->C01,
                    impl_->dt,
                    impl_->num_prony_terms,
                    impl_->prony_g.data(),
                    impl_->prony_tau.data(),
                    history_stride,
                    impl_->history_buffer->data() + history_offset,
                    3, &x[0], &x[1], &x[2],
                    rowptr, colidx, values);
            } else {
                // Use unrolled fixed version (hardcoded for 10 Prony terms)
                ret = mooney_rivlin_visco_bsr(
                    domain.element_type,
                    nelements,
                    mesh->n_nodes(),
                    domain.block->elements()->data(),
                    mesh->points()->data(),
                    impl_->C10,
                    impl_->K,
                    impl_->C01,
                    impl_->dt,
                    impl_->num_prony_terms,
                    impl_->prony_g.data(),
                    impl_->prony_tau.data(),
                    history_stride,
                    impl_->history_buffer->data() + history_offset,
                    3, &x[0], &x[1], &x[2],
                    rowptr, colidx, values);
            }
                
            // Advance history offset for next block
            history_offset += nelements * history_stride;
            return ret;
        });
    }

    int MooneyRivlinVisco::hessian_diag(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::hessian_diag");
        auto mesh = impl_->space->mesh_ptr();
        
        if(!impl_->history_buffer) {
            SFEM_ERROR("History buffer not initialized! Call initialize_history() first.");
            return SFEM_FAILURE;
        }

        ptrdiff_t history_offset = 0;
        const int n_qp = 8; 
        const ptrdiff_t history_per_qp = 6 + impl_->num_prony_terms * 6;
        const ptrdiff_t history_stride = n_qp * history_per_qp;

        return impl_->iterate([&](const OpDomain &domain) -> int {
            const ptrdiff_t nelements = domain.block->n_elements();
            
            int ret = mooney_rivlin_visco_hessian_diag_aos(
                domain.element_type,
                nelements,
                mesh->n_nodes(),
                domain.block->elements()->data(),
                mesh->points()->data(),
                impl_->C10,
                impl_->K,
                impl_->C01,
                impl_->dt,
                impl_->num_prony_terms,
                impl_->prony_g.data(),
                impl_->prony_tau.data(),
                history_stride,
                impl_->history_buffer->data() + history_offset,
                x, out);
                
            history_offset += nelements * history_stride;
            return ret;
        });
    }

    int MooneyRivlinVisco::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::gradient");
        auto mesh = impl_->space->mesh_ptr();
        
        if(!impl_->history_buffer) {
            SFEM_ERROR("History buffer not initialized! Call initialize_history() first.");
            return SFEM_FAILURE;
        }

        ptrdiff_t history_offset = 0;
        const int n_qp = 8; 
        const ptrdiff_t history_per_qp = 6 + impl_->num_prony_terms * 6;
        const ptrdiff_t history_stride = n_qp * history_per_qp;

        return impl_->iterate([&](const OpDomain &domain) -> int {
            const ptrdiff_t nelements = domain.block->n_elements();
            
            int ret = mooney_rivlin_visco_gradient_aos(
                domain.element_type,
                nelements,
                mesh->n_nodes(),
                domain.block->elements()->data(),
                mesh->points()->data(),
                impl_->C10,
                impl_->K,
                impl_->C01,
                impl_->dt,
                impl_->num_prony_terms,
                impl_->prony_g.data(),
                impl_->prony_tau.data(),
                history_stride,
                impl_->history_buffer->data() + history_offset,
                x, out);
                
            history_offset += nelements * history_stride;
            return ret;
        });
    }

    int MooneyRivlinVisco::update_history(const real_t *const x) {
        SFEM_TRACE_SCOPE("MooneyRivlinVisco::update_history");
        auto mesh = impl_->space->mesh_ptr();
        
        if(!impl_->history_buffer || !impl_->new_history_buffer) {
            SFEM_ERROR("History buffers not initialized!");
            return SFEM_FAILURE;
        }

        ptrdiff_t history_offset = 0;
        const int n_qp = 8; 
        const ptrdiff_t history_per_qp = 6 + impl_->num_prony_terms * 6;
        const ptrdiff_t history_stride = n_qp * history_per_qp;

        int ret = impl_->iterate([&](const OpDomain &domain) -> int {
            const ptrdiff_t nelements = domain.block->n_elements();
            
            int r = mooney_rivlin_visco_update_history_aos(
                domain.element_type,
                nelements,
                mesh->n_nodes(),
                domain.block->elements()->data(),
                mesh->points()->data(),
                impl_->C10,
                impl_->K,
                impl_->C01,
                impl_->dt,
                impl_->num_prony_terms,
                impl_->prony_g.data(),
                impl_->prony_tau.data(),
                history_stride,
                impl_->history_buffer->data() + history_offset,
                impl_->new_history_buffer->data() + history_offset, // Output to new buffer
                x);
                
            history_offset += nelements * history_stride;
            return r;
        });
        
        if(ret == SFEM_SUCCESS) {
            // Swap buffers so new becomes current
            impl_->swap_history_buffers();
        }
        return ret;
    }

    void MooneyRivlinVisco::set_C10(const real_t val) { impl_->C10 = val; }
    void MooneyRivlinVisco::set_K(const real_t val) { impl_->K = val; }
    void MooneyRivlinVisco::set_C01(const real_t val) { impl_->C01 = val; }
    void MooneyRivlinVisco::set_dt(const real_t val) { impl_->dt = val; }
    
    void MooneyRivlinVisco::set_prony_terms(const int n, const real_t *g, const real_t *tau) {
        impl_->num_prony_terms = n;
        impl_->prony_g.assign(g, g + n);
        impl_->prony_tau.assign(tau, tau + n);
        
        // Reallocate history because size per QP changed
        // WARNING: This clears previous history!
        if (impl_->history_buffer) {
            initialize_history(); 
        }
    }
    
    void MooneyRivlinVisco::set_use_flexible_hessian(bool use_flexible) {
        impl_->use_flexible_hessian = use_flexible;
    }

    MooneyRivlinVisco::Impl::~Impl() {}

}  // namespace sfem

