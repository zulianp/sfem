#include "sfem_PlugInOp.hpp"

#include "sfem_Tracer.hpp"
#include "sfem_logger.h"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"

#include <dlfcn.h>
#include <cassert>
#include <cctype>
#include <map>

namespace sfem {

    namespace {
        inline std::string to_lower(const std::string &s) {
            std::string r = s;
            for (auto &c : r) c = std::tolower(c);
            return r;
        }

        inline const char *elem_suffix(enum ElemType t) {
            switch (t) {
                case HEX8: return "hex8";
                case TET4: return "tet4";
                default: return nullptr;
            }
        }

        inline std::string default_plugin_dir() {
            const char *env = std::getenv("SFEM_PLUGIN_DIR");
            if (env && env[0]) return env;
            // Fallback: relative path used by codegen
            return std::string("operators/generated");
        }

        inline void *open_library(const std::string &opname) {
            std::string dir = default_plugin_dir();
#if defined(__APPLE__)
            std::string dylib = dir + "/lib" + opname + ".dylib";
            void       *h     = dlopen(dylib.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (h) return h;
#endif
            std::string so = dir + "/lib" + opname + ".so";
            return dlopen(so.c_str(), RTLD_LAZY | RTLD_LOCAL);
        }
    }  // namespace

    // Function pointer signatures expected from plug-ins (SoA layout with stride)
    using grad_fn = int (*)(const ptrdiff_t, const ptrdiff_t, idx_t **, geom_t **, const real_t, const real_t,
                            const ptrdiff_t, const real_t *const, const real_t *const, const real_t *const,
                            const ptrdiff_t, real_t *const, real_t *const, real_t *const);

    using apply_fn = int (*)(const ptrdiff_t, const ptrdiff_t, idx_t **, geom_t **, const real_t, const real_t,
                             const ptrdiff_t, const real_t *const, const real_t *const, const real_t *const,
                             const ptrdiff_t, const real_t *const, const real_t *const, const real_t *const,
                             const ptrdiff_t, real_t *const, real_t *const, real_t *const);

    using update_fn = int (*)(const real_t *const);

    class PlugInOp::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
        std::string                    opname;
        void                          *handle{nullptr};

        // Parameters (common defaults)
        real_t mu{1};
        real_t lambda{1};

        std::map<enum ElemType, grad_fn>  grad;
        std::map<enum ElemType, apply_fn> apply;
        std::map<enum ElemType, update_fn> update;

        Impl(const std::shared_ptr<FunctionSpace> &s, const std::string &n) : space(s), opname(n) {}

        ~Impl() {
            if (handle) dlclose(handle);
        }

        int ensure_loaded(enum ElemType t) {
            if (!handle) {
                handle = open_library(opname);
                if (!handle) {
                    SFEM_ERROR("[PlugInOp] Unable to open plugin '%s' in %s: %s\n",
                               opname.c_str(), default_plugin_dir().c_str(), dlerror());
                    return SFEM_FAILURE;
                }
            }

            const char *tag = elem_suffix(t);
            if (!tag) {
                SFEM_ERROR("[PlugInOp] Unsupported element type for plugin '%s'\n", opname.c_str());
                return SFEM_FAILURE;
            }

            if (!grad.count(t)) {
                std::string sym = opname + "_" + tag + "_gradient";
                void       *fn  = dlsym(handle, sym.c_str());
                if (!fn) {
                    SFEM_ERROR("[PlugInOp] Missing symbol %s in plugin '%s'\n", sym.c_str(), opname.c_str());
                    return SFEM_FAILURE;
                }
                grad[t] = reinterpret_cast<grad_fn>(fn);
            }

            if (!apply.count(t)) {
                std::string sym = opname + "_" + tag + "_apply";
                void       *fn  = dlsym(handle, sym.c_str());
                if (!fn) {
                    SFEM_ERROR("[PlugInOp] Missing symbol %s in plugin '%s'\n", sym.c_str(), opname.c_str());
                    return SFEM_FAILURE;
                }
                apply[t] = reinterpret_cast<apply_fn>(fn);
            }

            if (!update.count(t)) {
                std::string sym = opname + "_" + tag + "_update";
                void       *fn  = dlsym(handle, sym.c_str());
                if (!fn) {
                    SFEM_ERROR("[PlugInOp] Missing symbol %s in plugin '%s'\n", sym.c_str(), opname.c_str());
                    return SFEM_FAILURE;
                }
                update[t] = reinterpret_cast<update_fn>(fn);
            }

            return SFEM_SUCCESS;
        }
    };

    std::unique_ptr<Op> PlugInOp::create(const std::shared_ptr<FunctionSpace> &space, const std::string &opname) {
        return std::unique_ptr<Op>(new PlugInOp(space, opname));
    }

    PlugInOp::PlugInOp(const std::shared_ptr<FunctionSpace> &space, const std::string &opname)
        : impl_(std::make_unique<Impl>(space, opname)), name_(std::string("PlugInOp:") + opname) {}

    PlugInOp::~PlugInOp() = default;

    int PlugInOp::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("PlugInOp::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        real_t SFEM_SHEAR_MODULUS        = 1;
        real_t SFEM_FIRST_LAME_PARAMETER = 1;
        SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
        SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);
        impl_->mu     = SFEM_SHEAR_MODULUS;
        impl_->lambda = SFEM_FIRST_LAME_PARAMETER;
        return SFEM_SUCCESS;
    }

    // Not provided by plug-in in this initial version
    int PlugInOp::hessian_crs(const real_t *const, const count_t *const, const idx_t *const, real_t *const) {
        SFEM_ERROR("PlugInOp::hessian_crs not implemented");
        return SFEM_FAILURE;
    }
    int PlugInOp::hessian_bsr(const real_t *const, const count_t *const, const idx_t *const, real_t *const) {
        SFEM_ERROR("PlugInOp::hessian_bsr not implemented");
        return SFEM_FAILURE;
    }
    int PlugInOp::hessian_bcrs_sym(const real_t *const, const count_t *const, const idx_t *const, const ptrdiff_t, real_t **const, real_t **const) {
        SFEM_ERROR("PlugInOp::hessian_bcrs_sym not implemented");
        return SFEM_FAILURE;
    }
    int PlugInOp::hessian_block_diag_sym(const real_t *const, real_t *const) {
        SFEM_ERROR("PlugInOp::hessian_block_diag_sym not implemented");
        return SFEM_FAILURE;
    }
    int PlugInOp::hessian_block_diag_sym_soa(const real_t *const, real_t **const) {
        SFEM_ERROR("PlugInOp::hessian_block_diag_sym_soa not implemented");
        return SFEM_FAILURE;
    }
    int PlugInOp::hessian_diag(const real_t *const, real_t *const) {
        SFEM_ERROR("PlugInOp::hessian_diag not implemented");
        return SFEM_FAILURE;
    }
    int PlugInOp::update(const real_t *const x) {
        SFEM_TRACE_SCOPE("PlugInOp::update");

        auto mesh = impl_->space->mesh_ptr();
        return impl_->domains->iterate([&](const OpDomain &domain) {
            auto block        = domain.block;
            auto element_type = domain.element_type;
            if(impl_->ensure_loaded(element_type) != SFEM_SUCCESS) return SFEM_FAILURE;
            return impl_->update[element_type](x);
        });
    }

    int PlugInOp::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("PlugInOp::gradient");

        auto mesh        = impl_->space->mesh_ptr();
        int  block_size  = impl_->space->block_size();
        if (block_size != 3) {
            SFEM_ERROR("PlugInOp expects block_size=3 (vector field)\n");
            return SFEM_FAILURE;
        }

        return impl_->domains->iterate([&](const OpDomain &domain) {
            auto block        = domain.block;
            auto lambda       = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu           = domain.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.element_type;

            if (impl_->ensure_loaded(element_type) != SFEM_SUCCESS) return SFEM_FAILURE;

            auto fn = impl_->grad[element_type];
            const real_t *ux = &x[0];
            const real_t *uy = &x[1];
            const real_t *uz = &x[2];
            real_t       *ox = &out[0];
            real_t       *oy = &out[1];
            real_t       *oz = &out[2];

            return fn(block->n_elements(),
                      mesh->n_nodes(),
                      block->elements()->data(),
                      mesh->points()->data(),
                      mu,
                      lambda,
                      block_size,
                      ux,
                      uy,
                      uz,
                      block_size,
                      ox,
                      oy,
                      oz);
        });
    }

    int PlugInOp::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("PlugInOp::apply");

        auto mesh        = impl_->space->mesh_ptr();
        int  block_size  = impl_->space->block_size();
        if (block_size != 3) {
            SFEM_ERROR("PlugInOp expects block_size=3 (vector field)\n");
            return SFEM_FAILURE;
        }

        return impl_->domains->iterate([&](const OpDomain &domain) {
            auto block        = domain.block;
            auto lambda       = domain.parameters->get_real_value("lambda", impl_->lambda);
            auto mu           = domain.parameters->get_real_value("mu", impl_->mu);
            auto element_type = domain.element_type;

            if (impl_->ensure_loaded(element_type) != SFEM_SUCCESS) return SFEM_FAILURE;

            auto fn = impl_->apply[element_type];

            // Optional state x (SoA view) — pass zeros if nullptr
            const real_t *ux = x ? &x[0] : nullptr;
            const real_t *uy = x ? &x[1] : nullptr;
            const real_t *uz = x ? &x[2] : nullptr;

            const real_t *hx = &h[0];
            const real_t *hy = &h[1];
            const real_t *hz = &h[2];

            real_t *ox = &out[0];
            real_t *oy = &out[1];
            real_t *oz = &out[2];

            return fn(block->n_elements(),
                      mesh->n_nodes(),
                      block->elements()->data(),
                      mesh->points()->data(),
                      mu,
                      lambda,
                      block_size,
                      ux,
                      uy,
                      uz,
                      block_size,
                      hx,
                      hy,
                      hz,
                      block_size,
                      ox,
                      oy,
                      oz);
        });
    }

    int PlugInOp::value(const real_t *x, real_t *const out) {
        SFEM_ERROR("PlugInOp::value not implemented");
        return SFEM_FAILURE;
    }

    std::shared_ptr<Op> PlugInOp::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::shared_ptr<PlugInOp>(new PlugInOp(space, impl_->opname));
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        ret->impl_->mu      = impl_->mu;
        ret->impl_->lambda  = impl_->lambda;
        return ret;
    }

    std::shared_ptr<Op> PlugInOp::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        auto ret            = std::shared_ptr<PlugInOp>(new PlugInOp(space, impl_->opname));
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        ret->impl_->mu      = impl_->mu;
        ret->impl_->lambda  = impl_->lambda;
        return ret;
    }

    void PlugInOp::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void PlugInOp::override_element_types(const std::vector<enum ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

}  // namespace sfem

