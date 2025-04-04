import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from glob import glob


def main(argv):
    if len(argv) < 3:
        print(f"usage: {argv[0]} <prefix> <output_prefix>")
        exit()

    prefix_input = sys.argv[1]
    prefix_output = sys.argv[2]

    files = glob(prefix_input, recursive=False)
    files.sort()

    for f in files:
        a = np.fromfile(f, dtype=np.float64)
        print(a)
        name = os.path.basename(sys.argv[1])
        name = name.replace(".", "_")

        plt.figure().clear()
        plt.plot(a)

        plt.title("Profile")
        # plt.title('Left Title', loc='left')
        # plt.title('Right Title', loc='right')
        plt.savefig(f"{sys.argv[2]}{name}.png")


if __name__ == "__main__":
    main(sys.argv)
