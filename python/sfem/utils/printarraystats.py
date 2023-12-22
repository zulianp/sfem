#!/usr/bin/env python3

import numpy as np
import sys, getopt

def main(argv):
    path = "./"
    field  = None
    dtype = np.float64

    try:
        opts, args = getopt.getopt(
            argv[1:], "p:d:h",
            ["path=", "dtype=", "help"])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(f'{argv[0]} -d <path to array>')
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-d", "--dtype"):
            dtype = np.dtype(arg)

    data = np.fromfile(path, dtype=dtype)
    fp64data = data.astype(np.float64)
        
    print(f'min={np.min(fp64data)}')
    print(f'max={np.max(fp64data)}')
    print(f'min(abs)={np.min(np.abs(fp64data))}')
    print(f'max(abs)={np.max(np.abs(fp64data))}')
    print(f'sum={np.sum(fp64data)}')
    print(f'sum(abs)={np.sum(np.abs(fp64data))}')
        

if __name__ == '__main__':
    main(sys.argv)

