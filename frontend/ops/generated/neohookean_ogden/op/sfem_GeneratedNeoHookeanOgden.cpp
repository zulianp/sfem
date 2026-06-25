#include "sfem_GeneratedNeoHookeanOgden.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_Parameters.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstring>
#include <memory>

extern "C" {
int generated_neohookean_ogden_tri3_tri3_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, real_t *);
int generated_neohookean_ogden_tri3_tri3_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int generated_neohookean_ogden_tri3_tri3_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, real_t *);
int generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int generated_neohookean_ogden_quad4_quad4_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, real_t *);
int generated_neohookean_ogden_quad4_quad4_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int generated_neohookean_ogden_quad4_quad4_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *);
int generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int generated_neohookean_ogden_hex8_hex8_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int generated_neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int generated_neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, real_t *);
int generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
int generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa(ptrdiff_t, ptrdiff_t, idx_t **, const geom_t *const *, const real_t mu, const real_t lmbda, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, const real_t *, const real_t *, const real_t *, ptrdiff_t, real_t *, real_t *, real_t *);
}

namespace sfem {
    namespace {
        void seed_parameters(Parameters &parameters) {
            parameters.set_value("mu", 1);
            parameters.set_value("lmbda", 1);
        }
    }  // namespace

    class GeneratedNeoHookeanOgden::Impl {
    public:
        explicit Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::unique_ptr<real_t[]> element_values;
        ptrdiff_t element_capacity{0};
    };

    std::unique_ptr<Op> GeneratedNeoHookeanOgden::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != space->mesh_ptr()->spatial_dimension()) {
            SFEM_ERROR("GeneratedNeoHookeanOgden requires block_size=spatial_dimension\n");
            return nullptr;
        }
        auto op = std::make_unique<GeneratedNeoHookeanOgden>(space);
        op->initialize();
        return op;
    }

    GeneratedNeoHookeanOgden::GeneratedNeoHookeanOgden(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}
    GeneratedNeoHookeanOgden::~GeneratedNeoHookeanOgden() = default;

    ptrdiff_t GeneratedNeoHookeanOgden::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t GeneratedNeoHookeanOgden::n_dofs_image() const { return impl_->space->n_dofs(); }

    int GeneratedNeoHookeanOgden::initialize(const std::vector<std::string> &block_names) {
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);
        for (auto &entry : impl_->domains->domains()) {
            seed_parameters(*entry.second.parameters);
            impl_->element_capacity =
                    std::max(impl_->element_capacity, entry.second.block->n_elements());
        }
        impl_->element_values.reset(new real_t[impl_->element_capacity]);
        return SFEM_SUCCESS;
    }

    int GeneratedNeoHookeanOgden::gradient(const real_t *const x, real_t *const out) {
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3:
                    return generated_neohookean_ogden_tri3_tri3_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                case smesh::TRI6:
                    return generated_neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                case smesh::QUAD4:
                    return generated_neohookean_ogden_quad4_quad4_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, out + 0, out + 1);
                case smesh::TET4:
                    return generated_neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::TET10:
                    return generated_neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::HEX8:
                    return generated_neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX27:
                    return generated_neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, out + 0, out + 1, out + 2);
                default:
                    SFEM_ERROR("GeneratedNeoHookeanOgden does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedNeoHookeanOgden::apply(const real_t *const x,
                      const real_t *const h,
                      real_t *const out) {
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        return impl_->domains->iterate([&](const OpDomain &domain) {
            switch (domain.element_type) {
                case smesh::TRI3:
                    return generated_neohookean_ogden_tri3_tri3_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, h + 0, h + 1, 2, out + 0, out + 1);
                case smesh::TRI6:
                    return generated_neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, h + 0, h + 1, 2, out + 0, out + 1);
                case smesh::QUAD4:
                    return generated_neohookean_ogden_quad4_quad4_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, 2, h + 0, h + 1, 2, out + 0, out + 1);
                case smesh::TET4:
                    return generated_neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                case smesh::TET10:
                    return generated_neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                case smesh::HEX8:
                    return generated_neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                case smesh::PROTEUS_HEX27:
                    return generated_neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa(domain.block->n_elements(), mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, 3, h + 0, h + 1, h + 2, 3, out + 0, out + 1, out + 2);
                default:
                    SFEM_ERROR("GeneratedNeoHookeanOgden does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
        });
    }

    int GeneratedNeoHookeanOgden::value(const real_t *x, real_t *const out) {
        auto mesh = impl_->space->mesh_ptr();
        auto points = const_cast<const geom_t *const *>(mesh->points()->data());
        *out = 0;
        return impl_->domains->iterate([&](const OpDomain &domain) {
            const ptrdiff_t nelements = domain.block->n_elements();
            std::fill(impl_->element_values.get(),
                      impl_->element_values.get() + nelements,
                      0);
            int status = SFEM_FAILURE;
            switch (domain.element_type) {
                case smesh::TRI3:
                    status = generated_neohookean_ogden_tri3_tri3_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, impl_->element_values.get());
                    break;
                case smesh::TRI6:
                    status = generated_neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, impl_->element_values.get());
                    break;
                case smesh::QUAD4:
                    status = generated_neohookean_ogden_quad4_quad4_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 2, x + 0, x + 1, impl_->element_values.get());
                    break;
                case smesh::TET4:
                    status = generated_neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::TET10:
                    status = generated_neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::HEX8:
                    status = generated_neohookean_ogden_hex8_hex8_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                case smesh::PROTEUS_HEX27:
                    status = generated_neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa(nelements, mesh->n_nodes(), domain.block->elements()->data(), points, domain.parameters->require_real_value("mu"), domain.parameters->require_real_value("lmbda"), 3, x + 0, x + 1, x + 2, impl_->element_values.get());
                    break;
                default:
                    SFEM_ERROR("GeneratedNeoHookeanOgden does not support element type %d\n",
                               domain.element_type);
                    return SFEM_FAILURE;
            }
            if (status != SFEM_SUCCESS) return status;
            real_t sum = 0;
#pragma omp simd reduction(+ : sum)
            for (ptrdiff_t element = 0; element < nelements; ++element) {
                sum += impl_->element_values[element];
            }
            *out += sum;
            return SFEM_SUCCESS;
        });
    }

    int GeneratedNeoHookeanOgden::hessian_crs(const real_t *const,
                            const count_t *const,
                            const idx_t *const,
                            real_t *const) {
        return SFEM_FAILURE;
    }

    void GeneratedNeoHookeanOgden::set_value_in_block(const std::string &block_name,
                                    const std::string &var_name,
                                    const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }
}  // namespace sfem
