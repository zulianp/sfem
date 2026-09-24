#include <cstdio>
#include "tensor_product_kernels.hpp"
using namespace sfem::codegen;

template <int NQ, int NS, int ND, int NC>
static int run(const char *label) {
  constexpr int VS = 1;
  constexpr int NQ1 = integer_root(NQ, ND), NS1 = integer_root(NS, ND);
  double shape_1d[NQ1 * NS1], grad_1d[NQ1 * NS1];
  for (int i = 0; i < NQ1 * NS1; ++i) {
    shape_1d[i] = 0.25 + 0.125 * i;
    grad_1d[i] = -0.5 + 0.375 * i;
  }
  double data[NS * NC][VS];
  const double *streams[NS * NC];
  for (int s = 0; s < NS * NC; ++s) {
    data[s][0] = 0.1 * (s + 1) - 0.03 * (s % 5);
    streams[s] = &data[s][0];
  }
  int bad = 0;

  // gradient / test: the weak-form family
  double gb[NQ * ND * VS] = {0}, gs[NQ * ND] = {0};
  for (int c = 0; c < NC; ++c) {
    tensor_gradient<double, NQ, NS, VS, ND, NC>(1, shape_1d, grad_1d, streams, c, gb);
    tensor_gradient_scalar<double, NQ, NS, ND, NC>(shape_1d, grad_1d, streams, c, gs);
    for (int i = 0; i < NQ * ND; ++i) bad += (gb[i] != gs[i]);
  }
  double flux[NQ * ND * VS];
  for (int i = 0; i < NQ * ND; ++i) flux[i] = 0.05 * (i + 1) - 0.011 * (i % 7);
  double ob[NS * NC][VS] = {{0}}, os_[NS * NC][VS] = {{0}};
  double *obp[NS * NC], *osp[NS * NC];
  for (int s = 0; s < NS * NC; ++s) { obp[s] = &ob[s][0]; osp[s] = &os_[s][0]; }
  for (int c = 0; c < NC; ++c) {
    tensor_test<double, NQ, NS, VS, ND, NC>(1, shape_1d, grad_1d, flux, obp, c);
    tensor_test_scalar<double, NQ, NS, ND, NC>(shape_1d, grad_1d, flux, osp, c);
  }
  for (int s = 0; s < NS * NC; ++s) bad += (ob[s][0] != os_[s][0]);

  // evaluate / integrate: the residual family
  double vb[NC * NQ * VS] = {0}, vs[NC * NQ] = {0};
  double dgb[NC * NQ * ND * VS] = {0}, dgs[NC * NQ * ND] = {0};
  tensor_evaluate<double, NQ, NS, VS, ND, NC>(1, shape_1d, grad_1d, streams, vb, dgb);
  tensor_evaluate_scalar<double, NQ, NS, ND, NC>(shape_1d, grad_1d, streams, vs, dgs);
  for (int i = 0; i < NC * NQ; ++i) bad += (vb[i] != vs[i]);
  for (int i = 0; i < NC * NQ * ND; ++i) bad += (dgb[i] != dgs[i]);

  double vc[NC * NQ * VS], gc[NC * NQ * ND * VS];
  for (int i = 0; i < NC * NQ; ++i) vc[i] = 0.07 * (i + 2);
  for (int i = 0; i < NC * NQ * ND; ++i) gc[i] = -0.04 * (i + 1);
  double ib[NC * NS][VS] = {{0}}, is_[NC * NS][VS] = {{0}};
  double *ibp[NC * NS], *isp[NC * NS];
  for (int s = 0; s < NC * NS; ++s) { ibp[s] = &ib[s][0]; isp[s] = &is_[s][0]; }
  tensor_integrate<double, NQ, NS, VS, ND, NC>(1, shape_1d, grad_1d, vc, gc, ibp);
  tensor_integrate_scalar<double, NQ, NS, ND, NC>(shape_1d, grad_1d, vc, gc, isp);
  for (int s = 0; s < NC * NS; ++s) bad += (ib[s][0] != is_[s][0]);

  printf("%-10s %s\n", label, bad ? "DIFFER" : "identical");
  return bad;
}

int main() {
  int bad = 0;
  bad += run<4, 4, 2, 2>("quad4 d2");
  bad += run<8, 8, 3, 3>("hex8 d3");
  bad += run<27, 27, 3, 3>("hex27 d3");
  return bad != 0;
}
