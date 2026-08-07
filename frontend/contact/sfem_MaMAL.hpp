#ifndef SFEM_MAMAL_HPP
#define SFEM_MAMAL_HPP

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_aliases.hpp"

#ifdef SFEM_ENABLE_YAML
#include <ryml.hpp>
#endif

#include <memory>

namespace sfem {
    /// Matrix-free Multigrid Augmented Lagrangian for contact problems
    class MaMAL {
    public:
        /// Use the create method instead of directly calling the constructor
        MaMAL(const std::shared_ptr<Function>& f);
        ~MaMAL();

#ifdef SFEM_ENABLE_YAML
        static std::shared_ptr<MaMAL> create(const std::shared_ptr<Function>& f, const ryml::ConstNodeRef& node);
#endif

        // USE only defaults
        static std::shared_ptr<MaMAL> create(const std::shared_ptr<Function>& f);

        int solve(const smesh::SharedBuffer<real_t>& x);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // SFEM_MAMAL_HPP