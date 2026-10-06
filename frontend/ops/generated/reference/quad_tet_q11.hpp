#ifndef SFEM_CODEGEN_REFERENCE_QUAD_TET_Q11_HPP
#define SFEM_CODEGEN_REFERENCE_QUAD_TET_Q11_HPP

namespace sfem {
namespace codegen {

template <typename s_t>
struct quad_tet_q11 {
  static const s_t *q_weight() {
    static const s_t data[11] = {s_t(-0.013155555555555556), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
