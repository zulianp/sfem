#ifndef SFEM_CODEGEN_REFERENCE_TET4_Q1_HPP
#define SFEM_CODEGEN_REFERENCE_TET4_Q1_HPP

#include "../../reference/quad_tet_q1.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct ref_tet4_q1 {
  static const s_t *shape() {
    static const s_t data[4] = {s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[4] = {s_t(-1), s_t(1), s_t(0), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[4] = {s_t(-1), s_t(0), s_t(1), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_z() {
    static const s_t data[4] = {s_t(-1), s_t(0), s_t(0), s_t(1)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
