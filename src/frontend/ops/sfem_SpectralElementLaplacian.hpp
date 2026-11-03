#pragma once
#include "sfem_Op.hpp"

namespace sfem {
class SpectralElementLaplacian : public Op {
public:
    std::shared_ptr<FunctionSpace> space;
    smesh::ElemType element_type { smesh::INVALID };
    long calls{0};
    double total_time{0};
    ~SpectralElementLaplacian();
    static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);
    std::shared_ptr<Op> lor(const std::shared_ptr<FunctionSpace> &space) override;
    std::shared_ptr<Op> derefine(const std::shared_ptr<FunctionSpace> &space) override;
    const char *name() const override;
    //! the Dirichlet energy
    bool energy_or_potential_based() const override { return true; }
    inline bool is_linear() const override { return true; }
    inline ptrdiff_t n_dofs_domain() const override { return space->n_dofs(); }
    inline ptrdiff_t n_dofs_image() const override { return space->n_dofs(); }
    int initialize(const std::vector<std::string> &block_names = {}) override;
    SpectralElementLaplacian(const std::shared_ptr<FunctionSpace> &space);
    int hessian_crs(const real_t *const x, const count_t *const rowptr, const idx_t *const colidx, real_t *const values) override;
    int hessian_diag(const real_t *const, real_t *const out) override;
    int gradient(const real_t *const x, real_t *const out) override;
    int apply(const real_t *const x, const real_t *const h, real_t *const out) override;
    bool team_supported() const override;
    int team_apply(const Team &team,
                   const real_t *const x,
                   const real_t *const h,
                   real_t *const out,
                   const ElementScope scope) override;
    /// out += A h: by the threads of the caller's region (TEAM) or opening its own.
    template <bool TEAM>
    int apply_block(const real_t *const h, real_t *const out);
    /// Outside any region: the block's connectivity, the points and the level that apply_block reads.
    int team_prepare() override;
    BlockElements block_view;
    ptrdiff_t interior_start{0};
    geom_t **points{nullptr};
    int level{0};
    int value(const real_t *x, real_t *const out) override;
    std::shared_ptr<Op> clone() const override;
};
} // namespace sfem 