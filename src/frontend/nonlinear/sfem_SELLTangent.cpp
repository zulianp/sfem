#include "sfem_SELLTangent.hpp"

#include "sfem_Function.hpp"
#include "sfem_LinearOperatorFactory.hpp"

namespace sfem {

    InexactOptions SELLTangent::options(const InexactOptions::Encoding encoding) {
        InexactOptions ret;
        ret.store    = InexactOptions::Store::BSR;
        ret.encoding = encoding;
        ret.layout   = InexactOptions::Layout::SELL;
        ret.validate();
        return ret;
    }

    std::shared_ptr<SELLTangent> SELLTangent::create(const std::shared_ptr<Function> &f,
                                                     const ExecutionSpace             es,
                                                     const InexactOptions::Encoding   encoding) {
        std::shared_ptr<SELLTangent> ret(new SELLTangent());
        ret->f_       = f;
        ret->es_      = es;
        ret->options_ = options(encoding);
        return ret;
    }

    int SELLTangent::assemble(const real_t *const x) {
        SFEM_TRACE_SCOPE("SELLTangent::assemble");
        if (!configured_) {
            // An operator that stores a tangent may join after creation: checked here, once.
            if (!f_->inexact_supported()) {
                fprintf(stderr, "SELLTangent: no operator here assembles a tangent of blocks (format SELL needs one)\n");
                return SFEM_FAILURE;
            }
            f_->set_inexact_options(options_);
            configured_ = true;
        }
        return f_->inexact_update(x);
    }

    std::shared_ptr<Operator<real_t>> SELLTangent::op(const SharedBuffer<real_t> &state) const {
        return create_linear_operator(op_type::INEXACT, f_, state, es_);
    }

    int SELLTangent::diagonal(const real_t *const x, real_t *const diag) const { return f_->inexact_diag(x, diag); }

    std::size_t SELLTangent::bytes() const { return f_->inexact_bytes(); }

    std::string SELLTangent::describe() const { return "SELL " + options_.describe(); }

}  // namespace sfem
