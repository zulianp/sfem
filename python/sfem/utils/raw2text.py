#!/usr/bin/env python3

import numpy as np
import sys


def main(argv):
    if len(argv) != 4:
        print(f"usage: {argv[0]} <data.raw> <dtype> <text.txt>")
        exit(1)

    arr = np.fromfile(argv[1], dtype=argv[2])
    text_file = open(argv[3], "w")

    for a in arr:
        n = text_file.write(f"{a}\n")

    text_file.close()


if __name__ == "__main__":
    main(sys.argv)
