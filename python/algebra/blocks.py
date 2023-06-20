import numpy as np
import scipy as sp
import sys
import os
import math
import matplotlib.pyplot as plt

idx_t = np.int32
real_t = np.float64

def read_block_vector(pattern, export_path=None):
	paths = glob.glob(pattern, recursive=False)

	rb = len(paths)

	data = None
	for r in range(0, rb):
		block = np.fromfile(paths[r], dtype=real_t)

		if r == 0:
			N = len(block)
			data = np.zeros(N*rb, drype=real_t)

		s = r*N
		e = (r+1)*N
		data[s:e] = block

	if export_path != None:
		data.tofile(export_path)
