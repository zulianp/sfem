 #pragma once
 #include "sfem_Op.hpp"
 
 namespace sfem {
 

     class SemiStructuredNeoHookeanOgden : public Op {
     public:
         std::shared_ptr<FunctionSpace> space;                           ///< Function space for the operator
         enum ElemType                  element_type { INVALID };        ///< Element type
         real_t                         mu{1}, lambda{1};                ///< Lamé parameters
         bool                           use_affine_approximation{true};  ///< Use affine approximation for performance
         long                           calls{0};                        ///< Number of apply() calls for performance tracking
         double                         total_time{0};                   ///< Total time spent in apply() for performance tracking
 
         /**
          * @brief Destructor
          *
          * Prints performance statistics if SFEM_PRINT_THROUGHPUT is enabled.
          */
         ~SemiStructuredNeoHookeanOgden();
 
         /**
          * @brief Create a SemiStructuredLinearElasticity operator
          * @param space Function space
          * @return Unique pointer to the operator
          */
         static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);
 
         /**
          * @brief Set operator options
          * @param name Option name
          * @param val Option value
          *
          * Supported options:
          * - TODO
          */
         void set_option(const std::string &name, bool val) override;
 
         /**
          * @brief Create a clone of this operator
          * @return Shared pointer to the cloned operator
          */
         std::shared_ptr<Op> clone() const override;
 
         /**
          * @brief Create a level-of-refinement (LOR) version
          * @param space Function space for LOR operator
          * @return Shared pointer to LOR operator
          */
         std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override;
 
         /**
          * @brief Create a derefined version
          * @param space Function space for derefined operator
          * @return Shared pointer to derefined operator
          */
         std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override;
 
         const char *name() const override;
         inline bool is_linear() const override { return true; }
 
         /**
          * @brief Initialize the operator
          * @return SFEM_SUCCESS on success, SFEM_FAILURE on error
          */
         int initialize(const std::vector<std::string> &block_names = {}) override;
 
         /**
          * @brief Constructor
          * @param space Function space
          */
         SemiStructuredNeoHookeanOgden(const std::shared_ptr<FunctionSpace> &space);
 
         // Matrix assembly methods
         int hessian_crs(const real_t *const  x,
                         const count_t *const rowptr,
                         const idx_t *const   colidx,
                         real_t *const        values) override;
 
         int hessian_crs_sym(const real_t *const  x,
                             const count_t *const rowptr,
                             const idx_t *const   colidx,
                             real_t *const        diag_values,
                             real_t *const        off_diag_values) override;
 
         int hessian_bsr(const real_t *const  x,
                         const count_t *const rowptr,
                         const idx_t *const   colidx,
                         real_t *const        values) override;
 
         int hessian_bcrs_sym(const real_t *const  x,
                              const count_t *const rowidx,
                              const idx_t *const   colidx,
                              const ptrdiff_t      block_stride,
                              real_t **const       diag_values,
                              real_t **const       off_diag_values) override;
 
         virtual int hessian_block_diag_sym(const real_t *const x, real_t *const values) override;
 
         int hessian_diag(const real_t *const /*x*/, real_t *const values) override;
 
         // Vector operations
         int gradient(const real_t *const x, real_t *const out) override;
         int apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) override;
         int value(const real_t *x, real_t *const out) override;
 
         int report(const real_t *const) override;
     };
 
 }  // namespace sfem