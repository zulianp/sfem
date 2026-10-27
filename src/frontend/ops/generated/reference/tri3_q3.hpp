#ifndef SFEM_CODEGEN_REFERENCE_TRI3_Q3_HPP
#define SFEM_CODEGEN_REFERENCE_TRI3_Q3_HPP

#include "quad_tri_q3.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct ref_tri3_q3 {
  static const s_t *shape() {
    static const s_t data[9] = {s_t(0.66666666666666674), s_t(0.16666666666666666), s_t(0.16666666666666666), s_t(0.16666666666666671), s_t(0.66666666666666663), s_t(0.16666666666666666), s_t(0.16666666666666674), s_t(0.16666666666666666), s_t(0.66666666666666663)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[9] = {s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[9] = {s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
