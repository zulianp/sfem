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
    print('unblocks.py: self contained mode')    
    idx_t = np.int32
    real_t = np.float64

def split_block_vector(rb, path):
	bvec = np.fromfile(path, dtype=real_t)
	N = int(len(bvec)/rb)

	assert N * rb == len(bvec)

	fnameext = os.path.splitext(path)

	for r in range(0, rb):
		filename = f'{fnameext[0]}.{r}{fnameext[1]}'

		s = r * N
		e = (r + 1) * N
		block = bvec[s:e]
		block.tofile(filename)

if __name__ == '__main__':
	argv = sys.argv
	if len(argv) != 3:
		print(f'usage {argv[0]} <nblocks> <file>')
		exit(1)

	split_block_vector(int(argv[1]), argv[2])
