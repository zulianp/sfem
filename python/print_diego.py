#!/usr/bin/env python3

import numpy as np
A = np.zeros((4, 4))

colidx = np.fromfile("lhs.colindex.raw", np.int32)
rowptr = np.fromfile("lhs.rowindex.raw", np.int32)
values = np.fromfile("lhs.value.raw", np.float32)

for i in range(0, len(rowptr) -1):
	for k in range(rowptr[i], rowptr[i+1]):
		A[i, colidx[k]] = values[k]

print(A)
