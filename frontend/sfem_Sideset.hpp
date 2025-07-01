#ifndef SFEM_SIDESET_HPP
#define SFEM_SIDESET_HPP

#include <mpi.h>
#include <cstddef>
#include <functional>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_ForwardDeclarations.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_base.h"
#include "sfem_defs.h"

namespace sfem {

    class Sideset final {
    public:
        int                             read(const std::shared_ptr<Communicator> &comm, const char *path);
        SharedBuffer<element_idx_t>     parent();
        SharedBuffer<int16_t>           lfi();
        static std::shared_ptr<Sideset> create_from_file(const std::shared_ptr<Communicator> &comm, const char *path);
        ptrdiff_t                       size() const;
        std::shared_ptr<Communicator>   comm() const;
        int                             write(const char *path) const;

        Sideset(const std::shared_ptr<Communicator> &comm,
                const SharedBuffer<element_idx_t>   &parent,
                const SharedBuffer<int16_t>         &lfi);
        Sideset();
        ~Sideset();

        static std::shared_ptr<Sideset> create(const std::shared_ptr<Communicator> &comm,
                                               const SharedBuffer<element_idx_t>   &parent,
                                               const SharedBuffer<int16_t>         &lfi);

        static std::shared_ptr<Sideset> create_from_selector(
                const std::shared_ptr<Mesh>                                         &mesh,
                const std::function<bool(const geom_t, const geom_t, const geom_t)> &selector);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    std::pair<enum ElemType, std::shared_ptr<Buffer<idx_t *>>> create_surface_from_sideset(
            const std::shared_ptr<FunctionSpace> &space,
            const std::shared_ptr<Sideset>       &sideset);

}  // namespace sfem

#endif  // SFEM_SIDESET_HPP