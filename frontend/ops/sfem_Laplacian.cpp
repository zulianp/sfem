#include "sfem_Laplacian.hpp"

#include "laplacian.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"
#include "smesh_glob.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"
#include "smesh_spaces.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace sfem {

    namespace {

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh, const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); i++) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("Laplacian: mesh block pointer not found in mesh.blocks()");
            return 0;
        }

        /// Range into this domain's SoA only. Serial ALL/OWNED is the whole block;
        /// never a mesh-concatenated [0, n_elements()) that overruns later blocks.
        ElementRange domain_element_range(const smesh::Mesh &mesh, const OpDomain &domain, const ElementScope scope) {
            if (!mesh_is_distributed(mesh)) {
                if (scope == ElementScope::SHARED_AND_AURA) {
                    return {0, 0};
                }
                return {0, domain.block->n_elements()};
            }
            const smesh::block_idx_t b = block_id_for_domain(mesh, *domain.block);
            return element_range(mesh, b, scope);
        }

        ElementRange clamp_to_block(const ElementRange range, const ptrdiff_t n_block) {
            const ptrdiff_t begin = std::max(range.begin, ptrdiff_t(0));
            const ptrdiff_t end   = std::min(range.end, n_block);
            if (begin >= end) {
                return {0, 0};
            }
            return {begin, end};
        }

        static real_t domain_diffusion(const OpDomain &domain) {
            if (!domain.parameters) {
                return real_t(1);
            }
            return domain.parameters->get_real_value("k", real_t(1));
        }

        static void laplacian_seed_diffusion(MultiDomainOp &m, const real_t k) {
            for (auto &kv : m.domains()) {
                kv.second.parameters->set_value("k", k);
            }
        }

        static void laplacian_copy_diffusion(const MultiDomainOp &from, MultiDomainOp &to) {
            for (const auto &kv : from.domains()) {
                auto it = to.domains().find(kv.first);
                if (it == to.domains().end()) {
                    continue;
                }
                it->second.parameters->set_value("k", kv.second.parameters->get_real_value("k", real_t(1)));
            }
        }

        static bool laplacian_domain_uses_fff(const smesh::ElemType element_type) {
            if (smesh::is_semistructured_type(element_type)) {
                return smesh::is_hex_ss_family(element_type);
            }

            return element_type == smesh::TET4 || element_type == smesh::TET10 || element_type == smesh::HEX8;
        }

        /// SoA view of [begin, begin+ne). begin==0 is the original columns (SS HEX nxe can be huge).
        idx_t **element_soa_view(idx_t **const elems,
                                 const int     nxe,
                                 const ptrdiff_t begin,
                                 idx_t        *stack_view[32],
                                 std::vector<idx_t *> &heap_view) {
            if (begin == 0) {
                return elems;
            }
            if (nxe <= 32) {
                for (int v = 0; v < nxe; ++v) {
                    stack_view[v] = elems[v] + begin;
                }
                return stack_view;
            }
            heap_view.resize((size_t)nxe);
            for (int v = 0; v < nxe; ++v) {
                heap_view[v] = elems[v] + begin;
            }
            return heap_view.data();
        }

        int laplacian_dispatch_domain_vector(const OpDomain     &domain,
                                             smesh::Mesh        &mesh,
                                             const real_t *const u,
                                             real_t *const       out,
                                             const ElementRange  range_in) {
            const ElementRange range = clamp_to_block(range_in, domain.block->n_elements());
            if (range.empty()) {
                return SFEM_SUCCESS;
            }

            const ptrdiff_t         ne    = range.size();
            idx_t **const           elems = domain.block->elements()->data();
            const int               nxe   = elem_num_nodes(domain.element_type);
            idx_t                  *stack_view[32];
            std::vector<idx_t *>    heap_view;
            idx_t **const           view  = element_soa_view(elems, nxe, range.begin, stack_view, heap_view);
            const real_t            k     = domain_diffusion(domain);

            if (domain.user_data) {
                auto                fff      = std::static_pointer_cast<smesh::FFF>(domain.user_data);
                constexpr ptrdiff_t fff_size = 6;
                const jacobian_t   *fff_in   = fff->fff_AoS()->data() + range.begin * fff_size;
                if (k == real_t(1)) {
                    return laplacian_apply_opt(domain.element_type, ne, view, fff_in, u, out);
                }

                const ptrdiff_t nfff   = fff_size * ne;
                jacobian_t     *scaled = (jacobian_t *)malloc((size_t)nfff * sizeof(jacobian_t));
                for (ptrdiff_t i = 0; i < nfff; ++i) {
                    scaled[i] = static_cast<jacobian_t>(k * fff_in[i]);
                }
                const int err = laplacian_apply_opt(domain.element_type, ne, view, scaled, u, out);
                free(scaled);
                return err;
            }

            if (k == real_t(1)) {
                return laplacian_apply(domain.element_type, ne, mesh.n_nodes(), view, mesh.points()->data(), u, out);
            }

            const ptrdiff_t n   = mesh.n_nodes();
            real_t         *tmp = (real_t *)calloc((size_t)n, sizeof(real_t));
            const int       err = laplacian_apply(domain.element_type, ne, n, view, mesh.points()->data(), u, tmp);
            for (ptrdiff_t i = 0; i < n; ++i) {
                out[i] += k * tmp[i];
            }
            free(tmp);
            return err;
        }

        int laplacian_value_domain_range(const OpDomain     &domain,
                                         smesh::Mesh        &mesh,
                                         const real_t *const x,
                                         real_t *const       out,
                                         const ElementRange  range_in) {
            const ElementRange range = clamp_to_block(range_in, domain.block->n_elements());
            if (range.empty()) {
                return SFEM_SUCCESS;
            }
            const ptrdiff_t         ne    = range.size();
            idx_t **const           elems = domain.block->elements()->data();
            const int               nxe   = elem_num_nodes(domain.element_type);
            idx_t                  *stack_view[32];
            std::vector<idx_t *>    heap_view;
            idx_t **const           view  = element_soa_view(elems, nxe, range.begin, stack_view, heap_view);
            const real_t            k     = domain_diffusion(domain);
            if (k == real_t(1)) {
                return laplacian_assemble_value(domain.element_type, ne, mesh.n_nodes(), view, mesh.points()->data(), x, out);
            }

            real_t    acc = 0;
            const int err =
                    laplacian_assemble_value(domain.element_type, ne, mesh.n_nodes(), view, mesh.points()->data(), x, &acc);
            *out += k * acc;
            return err;
        }

    }  // namespace

    class Laplacian::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif
        explicit Impl(const std::shared_ptr<FunctionSpace> &sp) : space(sp) {
#if SFEM_PRINT_THROUGHPUT
            const std::string op_name = std::string("Laplacian[") + sfem::type_to_string(sp->element_type()) + "]::apply";
            op_profiler               = std::make_unique<OpTracer>(space, op_name);
#endif
        }

        void print_info() { domains->print_info(); }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    inline ptrdiff_t Laplacian::n_dofs_domain() const { return impl_->space->n_dofs(); }

    inline ptrdiff_t Laplacian::n_dofs_image() const { return impl_->space->n_dofs(); }

    int Laplacian::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("Laplacian::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        laplacian_seed_diffusion(*impl_->domains, real_t(1));

        auto mesh = impl_->space->mesh_ptr();

        for (auto &n2d : impl_->domains->domains()) {
            OpDomain &domain = n2d.second;
            auto      block  = domain.block;

            if (!laplacian_domain_uses_fff(domain.element_type)) {
                domain.user_data = nullptr;
                continue;
            }

            const smesh::block_idx_t block_id = block_id_for_domain(*mesh, *block);
            auto                     fff      = smesh::FFF::create_AoS(mesh, smesh::MEMORY_SPACE_HOST, block_id);
            if (!fff) {
                return SFEM_FAILURE;
            }

            domain.user_data = std::static_pointer_cast<void>(fff);
        }

        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> Laplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Laplacian::create");

        assert(1 == space->block_size());

        return std::make_unique<Laplacian>(space);
    }

    std::shared_ptr<Op> Laplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type())) {
            SMESH_ERROR("Laplacian::lor_op NOT IMPLEMENTED for semi-structured mesh!\n");
            return nullptr;
        }
        auto ret            = std::make_shared<Laplacian>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        laplacian_copy_diffusion(*impl_->domains, *ret->impl_->domains);
        return ret;
    }

    std::shared_ptr<Op> Laplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("Laplacian::derefine_op");

        if (space->has_semi_structured_mesh() && is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<Laplacian>(space);
            ret->initialize({});
            laplacian_copy_diffusion(*impl_->domains, *ret->impl_->domains);
            return ret;
        }

        // SS hierarchy bottom: coarse space is standard (e.g. HEX8). MultiDomainOp::derefine_op maps
        // element types with macro_base_elem and aborts on HEX8 — match old SemiStructuredLaplacian.
        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type()) &&
            !is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<Laplacian>(space);
            ret->initialize({});
            laplacian_copy_diffusion(*impl_->domains, *ret->impl_->domains);
            return ret;
        }

        auto ret            = std::make_shared<Laplacian>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        laplacian_copy_diffusion(*impl_->domains, *ret->impl_->domains);
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

        return impl_->iterate([&](const OpDomain &domain) {
            return laplacian_crs(domain.element_type,
                                 domain.block->n_elements(),
                                 mesh->n_nodes(),
                                 domain.block->elements()->data(),
                                 mesh->points()->data(),
                                 graph->rowptr()->data(),
                                 graph->colidx()->data(),
                                 values);
        });
    }

    int Laplacian::hessian_crs_sym(const real_t *const  x,
                                   const count_t *const rowptr,
                                   const idx_t *const   colidx,
                                   real_t *const        diag_values,
                                   real_t *const        off_diag_values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_crs_sym");

        auto mesh = impl_->space->mesh_ptr();

        return impl_->iterate([&](const OpDomain &domain) {
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
    }

    int Laplacian::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("Laplacian::hessian_diag");

        auto mesh = impl_->space->mesh_ptr();
        const ptrdiff_t n = mesh->n_nodes();

        return impl_->iterate([&](const OpDomain &domain) {
            const real_t k = domain_diffusion(domain);
            if (k == real_t(1)) {
                return laplacian_diag(domain.element_type,
                                      domain.block->n_elements(),
                                      n,
                                      domain.block->elements()->data(),
                                      mesh->points()->data(),
                                      values);
            }

            real_t   *tmp = (real_t *)calloc((size_t)n, sizeof(real_t));
            const int err = laplacian_diag(domain.element_type,
                                           domain.block->n_elements(),
                                           n,
                                           domain.block->elements()->data(),
                                           mesh->points()->data(),
                                           tmp);
            for (ptrdiff_t i = 0; i < n; ++i) {
                values[i] += k * tmp[i];
            }
            free(tmp);
            return err;
        });
    }

    int Laplacian::gradient(const real_t *const x, real_t *const out) {
        return gradient(x, out, ElementScope::ALL);
    }

    int Laplacian::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        return apply(x, h, out, ElementScope::ALL);
    }

    int Laplacian::value(const real_t *x, real_t *const out) { return value(x, out, ElementScope::ALL); }

    int Laplacian::gradient(const real_t *const x, real_t *const out, const ElementScope scope) {
        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            return laplacian_dispatch_domain_vector(domain, *mesh, x, out, domain_element_range(*mesh, domain, scope));
        });
    }

    int Laplacian::apply(const real_t *const x, const real_t *const h, real_t *const out, const ElementScope scope) {
        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            return laplacian_dispatch_domain_vector(domain, *mesh, h, out, domain_element_range(*mesh, domain, scope));
        });
    }

    int Laplacian::value(const real_t *x, real_t *const out, const ElementScope scope) {
        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            return laplacian_value_domain_range(domain, *mesh, x, out, domain_element_range(*mesh, domain, scope));
        });
    }

    int Laplacian::apply_scope_flat_range(const real_t *const /*x*/,
                                          const real_t *const h,
                                          real_t *const       out,
                                          const ElementScope  scope,
                                          const ptrdiff_t     flat_begin,
                                          const ptrdiff_t     flat_end) {
        SFEM_TRACE_SCOPE("Laplacian::apply_scope_flat_range");
        SFEM_OP_CAPTURE();

        if (flat_end <= flat_begin) {
            return SFEM_SUCCESS;
        }

        auto mesh = impl_->space->mesh_ptr();
        int  err  = SFEM_SUCCESS;
        for (const auto &slice : flat_block_element_chunks(*mesh, scope, flat_begin, flat_end)) {
            const smesh::block_idx_t block = slice.block;
            const ElementRange         range = slice.range;
            if (impl_->iterate([&](const OpDomain &domain) {
                    if (block_id_for_domain(*mesh, *domain.block) != block) {
                        return SFEM_SUCCESS;
                    }
                    return laplacian_dispatch_domain_vector(domain, *mesh, h, out, range);
                }) != SFEM_SUCCESS) {
                err = SFEM_FAILURE;
            }
        }
        return err;
    }

    int Laplacian::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> Laplacian::clone() const {
        auto ret            = std::make_shared<Laplacian>(impl_->space);
        ret->impl_->domains = impl_->domains;
        return ret;
    }

    void Laplacian::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void Laplacian::override_element_types(const std::vector<smesh::ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

    void Laplacian::set_option(const std::string & /*name*/, bool /*val*/) {}

}  // namespace sfem
