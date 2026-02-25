#include "sfem_OpFactory.hpp"

#include "sfem_LinearElasticity.hpp"
#include "sfem_Laplacian.hpp"
#include "sfem_Mass.hpp"
#include "sfem_VectorLaplacian.hpp"
#include "sfem_LumpedMass.hpp"
#include "sfem_SemiStructuredLinearElasticity.hpp"
#include "sfem_SemiStructuredLaplacian.hpp"
#include "sfem_SemiStructuredVectorLaplacian.hpp"
#include "sfem_SemiStructuredLumpedMass.hpp"
#include "sfem_SemiStructuredEMLaplacian.hpp"
#include "sfem_SpectralElementLaplacian.hpp"
#include "sfem_CVFEMMass.hpp"
#include "sfem_CVFEMUpwindConvection.hpp"
#include "sfem_NeoHookeanOgden.hpp"
#include "sfem_Hyperelasticity.hpp"
#include "sfem_SemiStructuredNeoHookeanOgden.hpp"
#include "sfem_PlugInOp.hpp"
#include "sfem_BoundaryMass.hpp"
#include "sfem_PackedLaplacian.hpp"
#include "sfem_NeoHookeanOgdenPacked.hpp"
#include "sfem_NeoHookeanOgdenActiveStrainPacked.hpp"
#include "sfem_MooneyRivlinActiveStrainPacked.hpp"
#include "sfem_MooneyRivlinVisco.hpp"
#include "sfem_SemiStructuredKelvinVoigtNewmark.hpp"
#include <map>

// Forward declarations for other operators that will be moved
namespace sfem {
    std::unique_ptr<Op> create_kelvin_voigt_newmark(const std::shared_ptr<FunctionSpace> &space);
}

namespace sfem {

    class Factory::Impl {
    public:
        std::map<std::string, FactoryFunction>         name_to_create;
        std::map<std::string, FactoryFunctionBoundary> name_to_create_boundary;
    };

    Factory::Factory() : impl_(std::make_unique<Impl>()) {}

    Factory::~Factory() = default;

    Factory &Factory::instance() {
        static Factory instance_;

        if (instance_.impl_->name_to_create.empty()) {
            instance_.private_register_op("KelvinVoigtNewmark", create_kelvin_voigt_newmark);
            instance_.private_register_op("ss:KelvinVoigtNewmark", SemiStructuredKelvinVoigtNewmark::create);
            instance_.private_register_op("LinearElasticity", LinearElasticity::create);
            instance_.private_register_op("ss:LinearElasticity", SemiStructuredLinearElasticity::create);
            instance_.private_register_op("Laplacian", Laplacian::create);
            instance_.private_register_op("VectorLaplacian", VectorLaplacian::create);
            instance_.private_register_op("ss:VectorLaplacian", SemiStructuredVectorLaplacian::create);
            instance_.private_register_op("ss:Laplacian", SemiStructuredLaplacian::create);
            instance_.private_register_op("ss:LumpedMass", SemiStructuredLumpedMass::create);
            instance_.private_register_op("ss:em:Laplacian", SemiStructuredEMLaplacian::create);
            instance_.private_register_op("ss:SpectralElementLaplacian", SpectralElementLaplacian::create);
            instance_.private_register_op("CVFEMUpwindConvection", CVFEMUpwindConvection::create);
            instance_.private_register_op("Mass", Mass::create);
            instance_.private_register_op("CVFEMMass", CVFEMMass::create);
            instance_.private_register_op("LumpedMass", LumpedMass::create);
            instance_.private_register_op("NeoHookeanOgden", NeoHookeanOgden::create);
            instance_.private_register_op("NeoHookeanOgdenPacked", NeoHookeanOgdenPacked::create);
            // instance_.private_register_op("NeoHookeanOgdenActiveStrain", NeoHookeanOgdenActiveStrainPacked::create);
            instance_.private_register_op("NeoHookeanOgdenActiveStrainPacked", NeoHookeanOgdenActiveStrainPacked::create);
            instance_.private_register_op("MooneyRivlin", MooneyRivlinActiveStrainPacked::create);
            instance_.private_register_op("MooneyRivlinActiveStrainPacked", MooneyRivlinActiveStrainPacked::create);
            instance_.private_register_op("MooneyRivlinVisco", MooneyRivlinVisco::create);
            instance_.private_register_op("Hyperelasticity", Hyperelasticity::create);
            instance_.private_register_op("ss:NeoHookeanOgden", SemiStructuredNeoHookeanOgden::create);
            instance_.private_register_op("PackedLaplacian", PackedLaplacian::create);
            instance_.impl_->name_to_create_boundary["BoundaryMass"] = BoundaryMass::create;
        }

        return instance_;
    }

    void Factory::private_register_op(const std::string &name, FactoryFunction factory_function) {
        impl_->name_to_create[name] = factory_function;
    }

    void Factory::register_op(const std::string &name, FactoryFunction factory_function) {
        instance().private_register_op(name, factory_function);
    }

    std::shared_ptr<Op> Factory::create_op_gpu(const std::shared_ptr<FunctionSpace> &space, const char *name) {
        return Factory::create_op(space, d_op_str(name).c_str());
    }

    std::shared_ptr<Op> Factory::create_op(const std::shared_ptr<FunctionSpace> &space, const char *name) {
        assert(instance().impl_);

        std::string m_name = name;

        if (space->has_semi_structured_mesh()) {
            m_name = "ss:" + m_name;
        }

        auto &ntc = instance().impl_->name_to_create;
        auto  it  = ntc.find(m_name);

        if (it == ntc.end()) {
            // Try dynamic plug-in: prefix "plugin:"
            const std::string prefix = "plugin:";
            if (m_name.rfind(prefix, 0) == 0) {
                std::string opname = m_name.substr(prefix.size());
                auto        uop    = PlugInOp::create(space, opname);
                if (!uop) return nullptr;
                return std::shared_ptr<Op>(uop.release());
            }

            std::cerr << "Unable to find op " << m_name << "\n";
            return nullptr;
        }

        return it->second(space);
    }

    std::shared_ptr<Op> Factory::create_boundary_op(const std::shared_ptr<FunctionSpace>   &space,
                                                    const std::shared_ptr<Buffer<idx_t *>> &boundary_elements,
                                                    const char                             *name) {
        assert(instance().impl_);

        auto &ntc = instance().impl_->name_to_create_boundary;
        auto  it  = ntc.find(name);

        if (it == ntc.end()) {
            std::cerr << "Unable to find op " << name << "\n";
            return nullptr;
        }

        return it->second(space, boundary_elements);
    }

    std::string d_op_str(const std::string &name) { return "gpu:" + name; }

} // namespace sfem 
