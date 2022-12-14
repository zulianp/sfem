#!/usr/bin/env python3

import numpy as np
import sys, getopt
import os
import matplotlib.pyplot as plt

def main(argv):
    if(len(argv) < 3):
        print(f"Usage {argv[0]} <left.fp32.raw> <right.fp32.raw> <out.fp32.raw>")
        return

    left = argv[1]
    right = argv[2]
    out = argv[3]

    left_dtype = np.float32
    right_dtype = np.float32
    out_dtype = np.float32

    x = np.fromfile(left, dtype=left_dtype)
    y = np.fromfile(right, dtype=right_dtype)

    if(len(x) != len(y)):
        exit(1)

    z = x.astype(out_dtype) - y.astype(out_dtype)
    z.tofile(out)
        
if __name__ == '__main__':
    main(sys.argv)
