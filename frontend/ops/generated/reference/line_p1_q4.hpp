#ifndef SFEM_CODEGEN_REFERENCE_LINE_P1_Q4_HPP
#define SFEM_CODEGEN_REFERENCE_LINE_P1_Q4_HPP

#include "../../reference/quad_line_q4.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct ref_line_p1_q4 {
  static const s_t *shape_1d() {
    static const s_t data[8] = {s_t(0.93056815579702623), s_t(0.069431844202973714), s_t(0.66999052179242813), s_t(0.33000947820757187), s_t(0.33000947820757187), s_t(0.66999052179242813), s_t(0.069431844202973769), s_t(0.93056815579702623)};
    return data;
  }
  static const s_t *grad_1d() {
    static const s_t data[8] = {s_t(-1), s_t(1), s_t(-1), s_t(1), s_t(-1), s_t(1), s_t(-1), s_t(1)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
