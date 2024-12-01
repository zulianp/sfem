#ifndef SFEM_CONTACT_CONDITIONS_HPP
#define SFEM_CONTACT_CONDITIONS_HPP

#include "sfem_Function.hpp"


namespace sfem {

	// This is for now just a copy and past of dirichlet conditions
	// it will change in the near future
	class AxisAlignedContactConditions final : public Constraint {
	public:
	    AxisAlignedContactConditions(const std::shared_ptr<FunctionSpace> &space);
	    ~AxisAlignedContactConditions();

	    std::shared_ptr<FunctionSpace> space();

	    static std::shared_ptr<AxisAlignedContactConditions> create_from_env(
	            const std::shared_ptr<FunctionSpace> &space);
	    int apply(real_t *const x) override;
	    int apply_value(const real_t value, real_t *const x) override;
	    int copy_constrained_dofs(const real_t *const src, real_t *const dest) override;
	    int mask(mask_t *mask) override;

	    int gradient(const real_t *const x, real_t *const g) override;

	    int hessian_crs(const real_t *const x,
	                    const count_t *const rowptr,
	                    const idx_t *const colidx,
	                    real_t *const values) override;

	    void add_condition(const ptrdiff_t local_size,
	                       const ptrdiff_t global_size,
	                       idx_t *const idx,
	                       const int component,
	                       real_t *const values);

	    void add_condition(const ptrdiff_t local_size,
	                       const ptrdiff_t global_size,
	                       idx_t *const idx,
	                       const int component,
	                       const real_t value);

	    int n_conditions() const;
	    void *impl_conditions();

	    std::shared_ptr<Constraint> derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
	                                         const bool as_zero) const override;
	    std::shared_ptr<Constraint> lor() const override;

	private:
	    class Impl;
	    std::unique_ptr<Impl> impl_;
	};

	class ContactConditions final : public Constraint {
	public:
	    ContactConditions(const std::shared_ptr<FunctionSpace> &space);
	    ~ContactConditions();

	    std::shared_ptr<FunctionSpace> space();

	    static std::shared_ptr<ContactConditions> create_from_env(
	            const std::shared_ptr<FunctionSpace> &space);

	    static std::shared_ptr<ContactConditions> create_from_file(
	            const std::shared_ptr<FunctionSpace> &space, const std::string &path);

	    /// $x \in R^{n}$
	    int apply(real_t *const x) override;

	    /// $x \in R^{n}$
	    int apply_value(const real_t value, real_t *const x) override;

	    /// $\text{src} \in R^{n}, \text{dest} \in R^{n}$
	    int copy_constrained_dofs(const real_t *const src, real_t *const dest) override;

	    /// $\text{mask} \in R^{n}$
	    int mask(mask_t *mask) override;


	    /// $x \in R^{n}, g \in R^{m}$
	    int gradient(const real_t *const x, real_t *const g) override;

	    int hessian_crs(const real_t *const x,
	                    const count_t *const rowptr,
	                    const idx_t *const colidx,
	                    real_t *const values) override;

	    std::shared_ptr<Constraint> derefine(const std::shared_ptr<FunctionSpace> &coarse_space,
	                                         const bool as_zero) const override;
	    std::shared_ptr<Constraint> lor() const override;

	    ptrdiff_t n_constrained_dofs() const;

	    int gradient_for_mesh_viz(const real_t *const x, real_t *const g) const;

	private:
	    class Impl;
	    std::unique_ptr<Impl> impl_;
	};
}

#endif //SFEM_CONTACT_CONDITIONS_HPP
