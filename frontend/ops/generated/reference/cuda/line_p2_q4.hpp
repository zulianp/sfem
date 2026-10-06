#ifndef SFEM_CODEGEN_REFERENCE_LINE_P2_Q4_HPP
#define SFEM_CODEGEN_REFERENCE_LINE_P2_Q4_HPP

#include "../../../reference/cuda/quad_line_q4.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct ref_line_p2_q4 {
  static __host__ __device__ __forceinline__ const s_t *shape_1d() {
    static const s_t data[12] = {s_t(0.80134602936993082), s_t(0.25844425285419081), s_t(-0.059790282224121687), s_t(0.22778407679095203), s_t(0.88441289000295209), s_t(-0.11219696679390417), s_t(-0.11219696679390401), s_t(0.88441289000295198), s_t(0.22778407679095214), s_t(-0.05979028222412186), s_t(0.25844425285419081), s_t(0.80134602936993082)};
    return data;
  }
  static __host__ __device__ __forceinline__ const s_t *grad_1d() {
    static const s_t data[12] = {s_t(-2.7222726231881049), s_t(3.4445452463762103), s_t(-0.72227262318810515), s_t(-1.6799620871697125), s_t(1.359924174339425), s_t(0.32003791283028749), s_t(-0.32003791283028749), s_t(-1.359924174339425), s_t(1.6799620871697125), s_t(0.72227262318810492), s_t(-3.4445452463762098), s_t(2.7222726231881049)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
