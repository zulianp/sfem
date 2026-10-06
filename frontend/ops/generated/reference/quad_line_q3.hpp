#ifndef SFEM_CODEGEN_REFERENCE_QUAD_LINE_Q3_HPP
#define SFEM_CODEGEN_REFERENCE_QUAD_LINE_Q3_HPP

namespace sfem {
namespace codegen {

template <typename s_t>
struct quad_line_q3 {
  static const s_t *q_weight_1d() {
    static const s_t data[3] = {s_t(0.27777777777777779), s_t(0.44444444444444442), s_t(0.27777777777777779)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
