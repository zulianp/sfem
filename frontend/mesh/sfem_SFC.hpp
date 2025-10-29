#ifndef SFEM_SFC_HPP
#define SFEM_SFC_HPP

#include <memory>

#include "sfem_ForwardDeclarations.hpp"

namespace sfem {
    class SFC {
    public:
        SFC();
        ~SFC();

        int reorder(Mesh &mesh);

        static std::shared_ptr<SFC> create_from_env();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // SFEM_SFC_HPP