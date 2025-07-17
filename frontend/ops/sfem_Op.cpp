#include "sfem_Op.hpp"

namespace sfem {

    std::shared_ptr<Op> no_op() {
        return std::make_shared<NoOp>();
    }

} // namespace sfem 