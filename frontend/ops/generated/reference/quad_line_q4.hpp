#ifndef SFEM_CODEGEN_REFERENCE_QUAD_LINE_Q4_HPP
#define SFEM_CODEGEN_REFERENCE_QUAD_LINE_Q4_HPP

namespace sfem {
namespace codegen {

template <typename s_t>
struct quad_line_q4 {
  static const s_t *q_weight_1d() {
    static const s_t data[4] = {s_t(0.17392742256872692), s_t(0.3260725774312731), s_t(0.3260725774312731), s_t(0.17392742256872692)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
