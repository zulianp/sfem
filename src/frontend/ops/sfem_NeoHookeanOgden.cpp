#include "sfem_NeoHookeanOgden.hpp"

#include "neohookean_ogden.hpp"
#include "sfem_team.h"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"
#include "sfem_macros.hpp"
#include "smesh_mesh.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_env.hpp"

#include "smesh_glob.hpp"

// FIXME
#include "hex8_neohookean_ogden.hpp"
#include "hex8_partial_assembly_neohookean_inline.hpp"
#include "tet4_neohookean_ogden.hpp"
#include "tet4_partial_assembly_neohookean_inline.hpp"

#include "sfem_ElasticityAssemblyData.hpp"

#include <math.h>
#include <mpi.h>

namespace sfem {

    namespace {

        void neo_seed_material(MultiDomainOp &m, const real_t mu, const real_t lambda) {
            for (auto &d : m.domains()) {
                d.second.parameters->set_value("mu", mu);
                d.second.parameters->set_value("lambda", lambda);
            }
        }

        void neo_copy_material(const MultiDomainOp &from, MultiDomainOp &to) {
            for (const auto &kv : from.domains()) {
                auto it = to.domains().find(kv.first);
                if (it == to.domains().end()) {
                    continue;
                }
                it->second.parameters->set_value("mu", kv.second.parameters->require_real_value("mu"));
                it->second.parameters->set_value("lambda", kv.second.parameters->require_real_value("lambda"));
            }
        }

        int neohookean_partial_assembly_metric_cols(const smesh::ElemType element_type) {
            if (is_semistructured_type(element_type)) {
                return HEX8_S_IKMN_SIZE;
            }
            switch (element_type) {
                case smesh::TET4:
                    return TET4_S_IKMN_SIZE;
                case smesh::TET10:
                case smesh::HEX8:
                    return HEX8_S_IKMN_SIZE;
                default:
                    return TET4_S_IKMN_SIZE;
            }
        }

        /// The element columns of `range`: the stored ones from element 0, the interface ones made at
        /// initialize from the first SHARED_AND_AURA element; nullptr for any other start.
        idx_t **columns_from(const ElementRange range, const SharedBuffer<idx_t *> &stored, std::vector<idx_t *> &interface,
                             const ptrdiff_t interface_begin) {
            if (range.begin == 0) return stored->data();
            if (range.begin == interface_begin && !interface.empty()) return interface.data();
            SFEM_ERROR("NeoHookeanOgden: no element columns from element %ld\n", (long)range.begin);
            return nullptr;
        }

        /// out += H(x) h over the elements of `range` in the domain's block: by the threads of the
        /// caller's region (TEAM) or opening its own.
        template <bool TEAM>
        int neo_apply_domain(const OpDomain     &domain,
                             const ptrdiff_t     n_nodes,
                             geom_t **const      points,
                             const real_t        mu,
                             const real_t        lambda,
                             const real_t *const x,
                             const real_t *const h,
                             real_t *const       out,
                             const ElementRange  range) {
            if (range.empty()) return SFEM_SUCCESS;

            auto *const     ua = static_cast<struct ElasticityAssemblyData *>(domain.user_data.get());
            const ptrdiff_t ne = range.size();

            if (ua->partial_assembly_buffer) {
                idx_t **const   elements = columns_from(range, ua->elements, ua->interface_elements, ua->interface_begin);
                const ptrdiff_t offset   = range.begin * neohookean_partial_assembly_metric_cols(domain.element_type);
                if (ua->use_compression) {
                    return SFEM_TEAM_KERNEL(TEAM, neohookean_ogden_compressed_partial_assembly_apply)(
                            domain.element_type,
                            ne,
                            ua->elements_stride,
                            elements,
                            ua->partial_assembly_compressed->data() + offset,
                            ua->compression_scaling->data() + range.begin,
                            3,
                            &h[0],
                            &h[1],
                            &h[2],
                            3,
                            &out[0],
                            &out[1],
                            &out[2]);
                }

                return SFEM_TEAM_KERNEL(TEAM, neohookean_ogden_partial_assembly_apply)(domain.element_type,
                                                                                       ne,
                                                                                       ua->elements_stride,
                                                                                       elements,
                                                                                       ua->partial_assembly_buffer->data() + offset,
                                                                                       3,
                                                                                       &h[0],
                                                                                       &h[1],
                                                                                       &h[2],
                                                                                       3,
                                                                                       &out[0],
                                                                                       &out[1],
                                                                                       &out[2]);
            }

            return SFEM_TEAM_KERNEL(TEAM, neohookean_ogden_apply_aos)(
                    domain.element_type,
                    ne,
                    n_nodes,
                    columns_from(range, domain.block->elements(), ua->interface_block_elements, ua->interface_begin),
                    points,
                    mu,
                    lambda,
                    x,
                    h,
                    out);
        }

    }  // namespace

    class NeoHookeanOgden::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        real_t                         mu{1}, lambda{1};
        bool                           use_affine_approximation{true};

        /// The mesh's points and their number, taken by team_prepare for the team product.
        geom_t  **points{nullptr};
        ptrdiff_t n_nodes{0};

        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    ptrdiff_t NeoHookeanOgden::n_dofs_domain() const { return impl_->space->n_dofs(); }

    ptrdiff_t NeoHookeanOgden::n_dofs_image() const { return impl_->space->n_dofs(); }

    std::unique_ptr<Op> NeoHookeanOgden::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::create");

        assert(space->mesh()->spatial_dimension() == space->block_size());
        auto ret           = std::make_unique<NeoHookeanOgden>(space);
        ret->impl_->mu     = smesh::Env::read("SFEM_SHEAR_MODULUS", ret->impl_->mu);
        ret->impl_->lambda = smesh::Env::read("SFEM_FIRST_LAME_PARAMETER", ret->impl_->lambda);

        int SFEM_HEX8_ASSUME_AFFINE = ret->impl_->use_affine_approximation ? 1 : 0;
        SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);
        ret->impl_->use_affine_approximation = SFEM_HEX8_ASSUME_AFFINE;

        ret->initialize({});
        return ret;
    }

    NeoHookeanOgden::NeoHookeanOgden(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>(space)) {}

    int NeoHookeanOgden::hessian_crs(const real_t *const  x,
                                     const count_t *const /*rowptr*/,
                                     const idx_t *const   /*colidx*/,
                                     real_t *const        values) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::hessian_crs");

        auto mesh  = impl_->space->mesh();
        auto graph = impl_->space->node_to_node_graph();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            return neohookean_ogden_hessian_aos(domain.element_type,
                                                domain.block->n_elements(),
                                                mesh->n_nodes(),
                                                domain.block->elements()->data(),
                                                mesh->points()->data(),
                                                domain.parameters->get_real_value("mu", impl_->mu),
                                                domain.parameters->get_real_value("lambda", impl_->lambda),
                                                x,
                                                graph->rowptr()->data(),
                                                graph->colidx()->data(),
                                                values);
        });
    }

    int NeoHookeanOgden::hessian_diag(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::hessian_diag");

        auto mesh = impl_->space->mesh();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            return neohookean_ogden_diag_aos(domain.element_type,
                                             domain.block->n_elements(),
                                             1,
                                             mesh->n_nodes(),
                                             domain.block->elements()->data(),
                                             mesh->points()->data(),
                                             domain.parameters->get_real_value("mu", impl_->mu),
                                             domain.parameters->get_real_value("lambda", impl_->lambda),
                                             x,
                                             out);

            // return neohookean_ogden_partial_assembly_diag(
            //     domain.element_type,
            //     mesh->n_elements(),
            //     1,
            //     domain.block->elements()->data(),
            //     mesh->points()->data(),
            //     this->impl_->mu,
            //     this->impl_->lambda,
            //     3,
            //     &x[0],
            //     &x[1],
            //     &x[2],
            //     3,
            //     &out[0],
            //     &out[1],
            //     &out[2]
            //  );
        });
    }

    int NeoHookeanOgden::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::gradient");

        auto mesh = impl_->space->mesh();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            return neohookean_ogden_gradient_aos(domain.element_type,
                                                 domain.block->n_elements(),
                                                 mesh->n_nodes(),
                                                 domain.block->elements()->data(),
                                                 mesh->points()->data(),
                                                 domain.parameters->get_real_value("mu", impl_->mu),
                                                 domain.parameters->get_real_value("lambda", impl_->lambda),
                                                 x,
                                                 out);
        });
    }

    int NeoHookeanOgden::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::apply");
        if (!team_supported()) return apply(x, h, out, ElementScope::ALL);
        return op_team_region(*this, impl_->space->n_dofs(), [&](const Team &team) {
            return team_apply(team, x, h, out, ElementScope::ALL);
        });
    }

    int NeoHookeanOgden::apply(const real_t *const x, const real_t *const h, real_t *const out, const ElementScope scope) {
        auto mesh = impl_->space->mesh();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            return neo_apply_domain<false>(domain,
                                           mesh->n_nodes(),
                                           mesh->points()->data(),
                                           domain.parameters->get_real_value("mu", impl_->mu),
                                           domain.parameters->get_real_value("lambda", impl_->lambda),
                                           x,
                                           h,
                                           out,
                                           element_range(*mesh, *domain.block, scope));
        });
    }

    // The kernels with a team form: the partial assembly (stored or compressed, the latter not on
    // semi-structured hexahedra) and, without it, the TET4 product.
    bool NeoHookeanOgden::team_supported() const {
        for (const auto &kv : impl_->domains->domains()) {
            const OpDomain &domain = kv.second;
            auto            ua     = std::static_pointer_cast<struct ElasticityAssemblyData>(domain.user_data);
            if (ua->partial_assembly_buffer) {
                if (ua->use_compression && is_semistructured_type(domain.element_type)) return false;
            } else if (domain.element_type != smesh::TET4) {
                return false;
            }
        }
        return true;
    }

    int NeoHookeanOgden::team_apply(const Team &team,
                                    const real_t *const x,
                                    const real_t *const h,
                                    real_t *const       out,
                                    const ElementScope  scope) {
        (void)team;
        smesh::Mesh &mesh = *impl_->space->mesh();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            return neo_apply_domain<true>(domain,
                                          impl_->n_nodes,
                                          impl_->points,
                                          domain.parameters->get_real_value("mu", impl_->mu),
                                          domain.parameters->get_real_value("lambda", impl_->lambda),
                                          x,
                                          h,
                                          out,
                                          element_range(mesh, *domain.block, scope));
        });
    }

    int NeoHookeanOgden::team_prepare() {
        impl_->points  = impl_->space->mesh()->points()->data();
        impl_->n_nodes = impl_->space->mesh()->n_nodes();
        return SFEM_SUCCESS;
    }

    int NeoHookeanOgden::initialize(const std::vector<std::string> &block_names) {
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        neo_seed_material(*impl_->domains, impl_->mu, impl_->lambda);

        bool use_partial_assembly = smesh::Env::read("SFEM_USE_PARTIAL_ASSEMBLY", false);
        bool use_compression      = smesh::Env::read("SFEM_USE_COMPRESSION", false);
        bool use_AoS              = smesh::Env::read("SFEM_NEOHOOKEAN_OGDEN_USE_AOS", false);

        for (auto &domain : impl_->domains->domains()) {
            auto ua                  = std::make_shared<struct ElasticityAssemblyData>();
            ua->use_partial_assembly = use_partial_assembly || domain.second.element_type == smesh::HEX8 ||
                                       domain.second.element_type == smesh::TET10 ||
                                       is_semistructured_type(domain.second.element_type);
            ua->use_compression     = use_compression;
            ua->use_AoS             = use_AoS;
            ua->elements            = domain.second.block->elements();
            ua->elements_stride     = 1;
            domain.second.user_data = ua;

            if (use_AoS) {
                auto nxe            = domain.second.block->n_nodes_per_element();
                ua->elements        = convert_host_buffer_to_fake_SoA(nxe, soa_to_aos(1, nxe, domain.second.block->elements()));
                ua->elements_stride = nxe;
            }

            auto mesh = impl_->space->mesh();
            if (mesh_is_distributed(*mesh)) {
                const int     nxe   = domain.second.block->n_nodes_per_element();
                idx_t **const own   = domain.second.block->elements()->data();
                idx_t **const store = ua->elements->data();
                ua->interface_begin = element_range(*mesh, *domain.second.block, ElementScope::SHARED_AND_AURA).begin;
                ua->interface_elements.resize(nxe);
                ua->interface_block_elements.resize(nxe);
                for (int v = 0; v < nxe; ++v) {
                    ua->interface_elements[v]       = store[v] + ua->interface_begin * ua->elements_stride;
                    ua->interface_block_elements[v] = own[v] + ua->interface_begin;
                }
            }
        }

        return SFEM_SUCCESS;
    }

    int NeoHookeanOgden::update(const real_t *const u) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::update");

        auto mesh = impl_->space->mesh();

        for (auto &domain : impl_->domains->domains()) {
            auto assembly_data = std::static_pointer_cast<struct ElasticityAssemblyData>(domain.second.user_data);
            if (!assembly_data->use_partial_assembly) continue;

            auto element_type = domain.second.element_type;

            const int pa_cols = neohookean_partial_assembly_metric_cols(element_type);
            if (!assembly_data->partial_assembly_buffer) {
                assembly_data->partial_assembly_buffer =
                        sfem::create_host_buffer<metric_tensor_t>(domain.second.block->n_elements() * (ptrdiff_t)pa_cols);
            }

            int err = neohookean_ogden_hessian_partial_assembly(domain.second.element_type,
                                                                domain.second.block->n_elements(),
                                                                assembly_data->elements_stride,
                                                                assembly_data->elements->data(),
                                                                mesh->points()->data(),
                                                                domain.second.parameters->get_real_value("mu", impl_->mu),
                                                                domain.second.parameters->get_real_value("lambda", impl_->lambda),
                                                                3,
                                                                &u[0],
                                                                &u[1],
                                                                &u[2],
                                                                assembly_data->partial_assembly_buffer->data());
            if (err != SFEM_SUCCESS) return err;
            assembly_data->compress_partial_assembly(domain.second);
        }

        return SFEM_SUCCESS;
    }

    int NeoHookeanOgden::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::value");

        auto mesh = impl_->space->mesh();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto ua = std::static_pointer_cast<ElasticityAssemblyData>(domain.user_data);

            return neohookean_ogden_objective_aos(domain.element_type,
                                                  domain.block->n_elements(),
                                                  ua->elements_stride,
                                                  mesh->n_nodes(),
                                                  ua->elements->data(),
                                                  mesh->points()->data(),
                                                  domain.parameters->get_real_value("mu", impl_->mu),
                                                  domain.parameters->get_real_value("lambda", impl_->lambda),
                                                  x,
                                                  false,
                                                  out);
        });
    }

    int NeoHookeanOgden::value_steps(const real_t       *x,
                                     const real_t       *h,
                                     const int           nsteps,
                                     const real_t *const steps,
                                     real_t *const       out) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::value_steps");

        auto mesh = impl_->space->mesh();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            auto ua = std::static_pointer_cast<ElasticityAssemblyData>(domain.user_data);
            return neohookean_ogden_objective_steps_aos(domain.element_type,
                                                        domain.block->n_elements(),
                                                        ua->elements_stride,
                                                        mesh->n_nodes(),
                                                        ua->elements->data(),
                                                        mesh->points()->data(),
                                                        domain.parameters->get_real_value("mu", impl_->mu),
                                                        domain.parameters->get_real_value("lambda", impl_->lambda),
                                                        x,
                                                        h,
                                                        nsteps,
                                                        steps,
                                                        out);
        });
    }

    int NeoHookeanOgden::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> NeoHookeanOgden::lor(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::lor");

        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type())) {
            SMESH_ERROR("NeoHookeanOgden::lor NOT IMPLEMENTED for semi-structured mesh!\n");
            return nullptr;
        }

        auto ret            = std::make_shared<NeoHookeanOgden>(space);
        ret->impl_->domains = impl_->domains->lor(space, {});
        neo_copy_material(*impl_->domains, *ret->impl_->domains);
        ret->impl_->mu                       = impl_->mu;
        ret->impl_->lambda                   = impl_->lambda;
        ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgden::derefine(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::derefine");

        if (space->has_semi_structured_mesh() && is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<NeoHookeanOgden>(space);
            ret->initialize({});
            neo_copy_material(*impl_->domains, *ret->impl_->domains);
            ret->impl_->mu                       = impl_->mu;
            ret->impl_->lambda                   = impl_->lambda;
            ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
            return ret;
        }

        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type()) &&
            !is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<NeoHookeanOgden>(space);
            ret->initialize({});
            neo_copy_material(*impl_->domains, *ret->impl_->domains);
            ret->impl_->mu                       = impl_->mu;
            ret->impl_->lambda                   = impl_->lambda;
            ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
            return ret;
        }

        auto ret            = std::make_shared<NeoHookeanOgden>(space);
        ret->impl_->domains = impl_->domains->derefine(space, {});
        neo_copy_material(*impl_->domains, *ret->impl_->domains);
        ret->impl_->mu                       = impl_->mu;
        ret->impl_->lambda                   = impl_->lambda;
        ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
        return ret;
    }

    std::shared_ptr<Op> NeoHookeanOgden::clone() const {
        SFEM_ERROR("IMPLEMENT ME!\n");
        auto ret = std::make_shared<NeoHookeanOgden>(impl_->space);
        return ret;
    }

    NeoHookeanOgden::~NeoHookeanOgden() = default;

    void NeoHookeanOgden::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void NeoHookeanOgden::override_element_types(const std::vector<smesh::ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

    void NeoHookeanOgden::set_mu(const real_t mu) {
        impl_->mu = mu;
        if (impl_->domains) {
            for (auto &d : impl_->domains->domains()) {
                d.second.parameters->set_value("mu", mu);
            }
        }
    }

    void NeoHookeanOgden::set_lambda(const real_t lambda) {
        impl_->lambda = lambda;
        if (impl_->domains) {
            for (auto &d : impl_->domains->domains()) {
                d.second.parameters->set_value("lambda", lambda);
            }
        }
    }

    void NeoHookeanOgden::set_option(const std::string &name, const bool val) {
        if (name == "ASSUME_AFFINE") {
            impl_->use_affine_approximation = val;
        }
    }

    int NeoHookeanOgden::hessian_bsr(const real_t *const  x,
                                     const count_t *const rowptr,
                                     const idx_t *const   colidx,
                                     real_t *const        values) {
        SFEM_TRACE_SCOPE("NeoHookeanOgden::hessian_bsr");

        auto mesh = impl_->space->mesh();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            return neohookean_ogden_bsr(domain.element_type,
                                        domain.block->n_elements(),
                                        1,
                                        domain.block->elements()->data(),
                                        mesh->points()->data(),
                                        domain.parameters->get_real_value("mu", impl_->mu),
                                        domain.parameters->get_real_value("lambda", impl_->lambda),
                                        3,
                                        &x[0],
                                        &x[1],
                                        &x[2],
                                        rowptr,
                                        colidx,
                                        values);
        });
        return SFEM_SUCCESS;
    }

    int NeoHookeanOgden::hessian_bcrs_sym(const real_t *const  x,
                                          const count_t *const rowptr,
                                          const idx_t *const   colidx,
                                          const ptrdiff_t      block_stride,
                                          real_t **const       diag_values,
                                          real_t **const       off_diag_values) {
        auto mesh = impl_->space->mesh();
        return impl_->iterate([&](const OpDomain &domain) -> int {
            return neohookean_ogden_bcrs_sym(domain.element_type,
                                             domain.block->n_elements(),
                                             1,
                                             domain.block->elements()->data(),
                                             mesh->points()->data(),
                                             domain.parameters->get_real_value("mu", impl_->mu),
                                             domain.parameters->get_real_value("lambda", impl_->lambda),
                                             3,
                                             &x[0],
                                             &x[1],
                                             &x[2],
                                             rowptr,
                                             colidx,
                                             block_stride,
                                             diag_values,
                                             off_diag_values);
        });
        SFEM_TRACE_SCOPE("NeoHookeanOgden::hessian_bcrs_sym");
        return SFEM_SUCCESS;
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> NeoHookeanOgden::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                         const ryml::ConstNodeRef             &node) {
        std::shared_ptr<NeoHookeanOgden> op(static_cast<NeoHookeanOgden *>(create(space).release()));
        if (op->initialize({}) != SFEM_SUCCESS) return nullptr;
        if (node.has_child("mu")) {
            real_t value = 0;
            node["mu"].load(&value);
            op->set_mu(value);
        }
        if (node.has_child("lambda")) {
            real_t value = 0;
            node["lambda"].load(&value);
            op->set_lambda(value);
        }
        return op;
    }
#endif

}  // namespace sfem
