#ifndef SFEM_SELF_COLLISIONS_HPP
#define SFEM_SELF_COLLISIONS_HPP

#include "sfem_base.hpp"

#include "sfem_ForwardDeclarations.hpp"

#include <memory>

namespace sfem {

    class SelfCollisions {
    public:
        SelfCollisions();
        ~SelfCollisions();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // SFEM_SELF_COLLISIONS_HPP
