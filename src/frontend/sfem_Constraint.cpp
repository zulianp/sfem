#include "sfem_Constraint.hpp"

namespace sfem {

int Constraint::apply_zero(real_t *const x) {
    return apply_value(0, x);
}

} // namespace sfem 