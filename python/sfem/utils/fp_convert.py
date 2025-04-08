#!/usr/bin/env python3

import numpy as np
import sys, getopt
import pdb


def main(argv):
    argc = len(argv)
    if argc < 3:
        print(
            f"Usage {argv[0]} <input.raw> <output.raw> [dtype_input=float32] [dtype_output=float64]"
        )
        return

    path_input = argv[1]
    path_output = argv[2]
    dtype_input = np.float32
    dtype_output = np.float64

    if argc > 3:
        dtype_input = argv[3]

    if argc > 4:
        dtype_output = argv[4]

    # print(f"{argv[0]} {path_input} {path_output} {dtype_input} {dtype_output}")

    x = np.fromfile(path_input, dtype=dtype_input)
    x.astype(dtype_output).tofile(path_output)


if __name__ == "__main__":
    main(sys.argv)
