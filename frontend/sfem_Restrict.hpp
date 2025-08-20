#ifndef SFEM_RESTRICT_HPP
#define SFEM_RESTRICT_HPP

// C includes
#include "sfem_base.h"
#include "sfem_defs.h"

// C++ includes
#include "sfem_ForwardDeclarations.hpp"
#include "sfem_Operator.hpp"

#include <memory>

namespace sfem {

    template <typename T>
    class Restrict final : public Operator<T> {
    public:
        Restrict(const std::shared_ptr<FunctionSpace>& from, const std::shared_ptr<FunctionSpace>& to, const ExecutionSpace es);
        Restrict(const std::shared_ptr<FunctionSpace>& from,
                 const std::shared_ptr<FunctionSpace>& to,
                 const ExecutionSpace                  es,
                 const int                             block_size);

        static std::shared_ptr<Restrict> create(const std::shared_ptr<FunctionSpace>& from,
                                                const std::shared_ptr<FunctionSpace>& to,
                                                const ExecutionSpace                  es,
                                                const int                             block_size);

        ~Restrict();
        int            apply(const T* const x, T* const y) override;
        ptrdiff_t      rows() const override;
        ptrdiff_t      cols() const override;
        ExecutionSpace execution_space() const override;

        const SharedBuffer<uint16_t>& element_to_node_incidence_count() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    template <typename T>
    class SurfaceRestrict final : public Operator<T> {
    public:
        SurfaceRestrict(const int                     from_level,
                        const enum ElemType           from_elem_type,
                        const ptrdiff_t               from_n_nodes,
                        const SharedBuffer<idx_t*>&   from_sides,
                        const SharedBuffer<uint16_t>& from_count,
                        const int                     to_level,
                        const enum ElemType           to_elem_type,
                        const ptrdiff_t               to_n_nodes,
                        const SharedBuffer<idx_t*>&   to_sides,
                        const ExecutionSpace          es,
                        const int                     block_size);

        static std::shared_ptr<SurfaceRestrict<T>> create(const int                     from_level,
                                                          const enum ElemType           from_elem_type,
                                                          const ptrdiff_t               from_n_nodes,
                                                          const SharedBuffer<idx_t*>&   from_sides,
                                                          const SharedBuffer<uint16_t>& from_count,
                                                          const int                     to_level,
                                                          const enum ElemType           to_elem_type,
                                                          const ptrdiff_t               to_n_nodes,
                                                          const SharedBuffer<idx_t*>&   to_sides,
                                                          const ExecutionSpace          es,
                                                          const int                     block_size);

        ~SurfaceRestrict();

        int            apply(const T* const x, T* const y) override;
        ptrdiff_t      rows() const override;
        ptrdiff_t      cols() const override;
        ExecutionSpace execution_space() const override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif
