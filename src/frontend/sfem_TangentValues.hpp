#ifndef SFEM_TANGENT_VALUES_HPP
#define SFEM_TANGENT_VALUES_HPP

#include <string_view>

#include "sfem_base.hpp"
#include "smesh_types.hpp"

namespace sfem {

    /// The type an assembled tangent stores its values in: the state's real_t, or float when the
    /// tangent is assembled in single precision. SMESH_DEFAULT stands for real_t.
    inline smesh::PrimitiveType tangent_values_type(const smesh::PrimitiveType type) {
        return type == smesh::SMESH_DEFAULT ? smesh::TypeToEnum<real_t>::value() : type;
    }

    /// f(values) with `values` as the pointer type `type` names (tangent_values_type); fails on a
    /// type a tangent does not store.
    template <typename F>
    int with_tangent_values(const smesh::PrimitiveType type, void *const values, F &&f) {
        const smesh::PrimitiveType t = tangent_values_type(type);
        if (t == smesh::TypeToEnum<real_t>::value()) return f(static_cast<real_t *>(values));
#if !SFEM_REAL_T_IS_FLOAT32
        if (t == smesh::SMESH_FLOAT32) return f(static_cast<float *>(values));
#endif
        const std::string_view name = smesh::to_string(t);
        SFEM_ERROR("Tangent values of type %.*s are not supported (real_t or float32)\n", (int)name.size(), name.data());
        return SFEM_FAILURE;
    }

}  // namespace sfem

#endif  // SFEM_TANGENT_VALUES_HPP
