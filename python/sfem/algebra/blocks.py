#!/usr/bin/env python3

import numpy as np
import scipy as sp
import sys
import os
import math
import matplotlib.pyplot as plt
import glob

try: idx_t
except NameError: 
    print('blocks.py: self contained mode')    
    idx_t = np.int32
    real_t = np.float64


def read_block_vector(pattern, export_path=None):
	paths = glob.glob(pattern, recursive=False)
	paths.sort()

	rb = len(paths)

	data = None
	for r in range(0, rb):
		block = np.fromfile(paths[r], dtype=real_t)

		if r == 0:
			N = len(block)
			data = np.zeros(N*rb, dtype=real_t)

		s = r*N
		e = (r+1)*N
		data[s:e] = block

	if export_path != None:
		data.tofile(export_path)


if __name__ == '__main__':
	argv = sys.argv
	if len(argv) != 3:
		print(f'usage {argv[0]} <pattern> <output>')
		exit(1)

	read_block_vector(argv[1], argv[2])
