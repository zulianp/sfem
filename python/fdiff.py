#!/usr/bin/env python3

import numpy as np
import sys, getopt
import os
import matplotlib.pyplot as plt

def main(argv):
    if(len(argv) < 3):
        print(f"Usage {argv[0]} <left.raw> <right.raw> [left_dtype=float64] [right_dtype=float64] [factor=1] [figurename]")
        return

    left = argv[1]
    right = argv[2]

    left_dtype = np.float64
    right_dtype = np.float64
    factor = 1

    if len(argv) > 3:
        left_dtype = argv[3]

    if len(argv) > 4:
        right_dtype = argv[4]

    if len(argv) > 5:
        factor = float(argv[5])
        # print(f"{argv[0]} {argv[1]} {argv[2]} {argv[3]} {argv[4]} {argv[5]}")

    if len(argv)> 6:
        figurename = argv[6]
    else:
        lstr=os.path.basename(left)
        rstr=os.path.basename(right)
        figurename = f"histo_{lstr}_{rstr}.png"

    x = np.fromfile(left, dtype=left_dtype)
    y = np.fromfile(right, dtype=right_dtype)

    if(len(x) != len(y)):
        exit(1)

    z = np.abs(x - factor*y.astype(left_dtype))

    sum_z = np.sum(z)
    print(f'diff\n - {left}\n - {right}\n > max: {np.max(z)}, sum={sum_z}')

    if(sum_z == 0):
        return


    # print(numpy.histogram(z, bins=10, range=None, normed=None, weights=None, density=None))
    _ = plt.hist(z, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram | l - r |")
    plt.savefig(figurename)
        
if __name__ == '__main__':
    main(sys.argv)
