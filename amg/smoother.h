#ifndef SMOOTHER_H
#define SMOOTHER_H

#include "sparse.h"

int l2_smoother(const SymmCOOMatrix *mat, real_t *smoother);
int l1_smoother(const SymmCOOMatrix *mat, real_t *smoother);

#endif  // SMOOTHER_H
