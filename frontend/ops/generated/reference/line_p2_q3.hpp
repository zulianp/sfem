#ifndef SFEM_CODEGEN_REFERENCE_LINE_P2_Q3_HPP
#define SFEM_CODEGEN_REFERENCE_LINE_P2_Q3_HPP

#include "../../reference/quad_line_q3.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct ref_line_p2_q3 {
  static const s_t *shape_1d() {
    static const s_t data[9] = {s_t(0.68729833462074175), s_t(0.39999999999999997), s_t(-0.087298334620741685), s_t(0), s_t(1), s_t(0), s_t(-0.087298334620741658), s_t(0.39999999999999991), s_t(0.68729833462074175)};
    return data;
  }
  static const s_t *grad_1d() {
    static const s_t data[9] = {s_t(-2.5491933384829668), s_t(3.0983866769659336), s_t(-0.54919333848296681), s_t(-1), s_t(0), s_t(1), s_t(0.54919333848296681), s_t(-3.0983866769659336), s_t(2.5491933384829668)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
