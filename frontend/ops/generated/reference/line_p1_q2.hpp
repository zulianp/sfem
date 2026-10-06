#ifndef SFEM_CODEGEN_REFERENCE_LINE_P1_Q2_HPP
#define SFEM_CODEGEN_REFERENCE_LINE_P1_Q2_HPP

#include "../../reference/quad_line_q2.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct ref_line_p1_q2 {
  static const s_t *shape_1d() {
    static const s_t data[4] = {s_t(0.78867513459481287), s_t(0.21132486540518708), s_t(0.21132486540518713), s_t(0.78867513459481287)};
    return data;
  }
  static const s_t *grad_1d() {
    static const s_t data[4] = {s_t(-1), s_t(1), s_t(-1), s_t(1)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
