#ifndef SFEM_CODEGEN_REFERENCE_QUAD_LINE_Q2_HPP
#define SFEM_CODEGEN_REFERENCE_QUAD_LINE_Q2_HPP

namespace sfem {
namespace codegen {

template <typename s_t>
struct quad_line_q2 {
  static const s_t *q_weight_1d() {
    static const s_t data[2] = {s_t(0.5), s_t(0.5)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
