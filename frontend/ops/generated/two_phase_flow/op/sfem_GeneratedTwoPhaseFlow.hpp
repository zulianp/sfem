#pragma once

#include "sfem_Op.hpp"
#include "sfem_NeumannConditions.hpp"

namespace sfem {
  class GeneratedTwoPhaseFlow final : public Op {
  public:
    static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

    explicit GeneratedTwoPhaseFlow(const std::shared_ptr<FunctionSpace> &space);
    ~GeneratedTwoPhaseFlow() override;

    const char *name() const override { return "GeneratedTwoPhaseFlow"; }
    //! Where this `Op` runs, which the `Function` above it needs in order to
    //! allocate anything the `Op` will write.  `Op::execution_space` defaults
    //! to the host and the device wrapper did not override it, so
    //! `Function::execution_space` reported the host for a device `Op` and
    //! `Function::node_wise_merit` allocated its residual there -- which the
    //! device `gradient` then `atomicAdd`ed into.  `compute-sanitizer`:
    //! "Invalid __global__ atomic of size 8 bytes ... Address 0x3aa240f8 is
    //! out of bounds".
    ExecutionSpace execution_space() const override { return EXECUTION_SPACE_HOST; }
    bool is_linear() const override { return false; }
    ptrdiff_t n_dofs_domain() const override;
    ptrdiff_t n_dofs_image() const override;
    double flops_value() const override;
    double flops_gradient() const override;
    double flops_apply() const override;
    size_t memory_traffic_bytes_value() const override;
    size_t memory_traffic_bytes_gradient() const override;
    size_t memory_traffic_bytes_apply() const override;

    int initialize(const std::vector<std::string> &block_names = {}) override;
    int update(const real_t *const x) override;
    int update(const real_t *const previous, const real_t *const current) override;
    void set_field(const char *name,
                       const std::shared_ptr<Buffer<real_t>> &values,
                       int component) override;
    int gradient(const real_t *const x, real_t *const out) override;
    int apply(const real_t *const x,
                  const real_t *const h,
                  real_t *const out) override;
    int value(const real_t *x, real_t *const out) override;
    sfem::Op::ValueReduction value_reduction() const override;
    int hessian_crs(const real_t *const x,
            const count_t *const rowptr,
            const idx_t *const colidx,
            real_t *const values) override;
    int hessian_bsr(const real_t *const x,
            const count_t *const rowptr,
            const idx_t *const colidx,
            real_t *const values) override;

    void set_option(const std::string &name, bool val) override;
    void set_value_in_block(const std::string &block_name,
                const std::string &var_name,
                real_t value) override;
#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Op> create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                             const ryml::ConstNodeRef             &node) override;
#endif  // SFEM_ENABLE_RYAML

    //! The scalar type the kernels are asked for at run time.
    //!
    //! Mirrors GPULaplacian, which declares the same member with the same
    //! default and hands it to every kernel call.  SMESH_DEFAULT resolves
    //! to the build's real_t, so the default costs a caller nothing and is
    //! the common path rather than a fallback.  The Op interface itself is
    //! unchanged: its methods still take real_t*, which converts to void*
    //! at the call, exactly as gpu_laplacian_block_vector relies on.
    enum smesh::PrimitiveType real_type{smesh::SMESH_DEFAULT};

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };
}  // namespace sfem
