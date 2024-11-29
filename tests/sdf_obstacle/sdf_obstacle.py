#!/usr/bin/env python3

import pysfem as sfem
import numpy as np
from numpy import linalg
import sys, getopt, os

from sfem.sfem_config import *

import yaml

try:
    from yaml import SafeLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# -----------------------

def run(case):
	m = sfem.Mesh()
	m.read(case['mesh'])

	dim = m.spatial_dimension()
	fs = sfem.FunctionSpace(m, dim)
	cc = sfem.contact_conditions_from_file(fs, str(config['obstacle']))
	

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print(f'usage: {sys.argv[0]} <case>')
		exit(1)

	sfem.init()

	case_file = sys.argv[1]

	try:
	    opts, args = getopt.getopt(
	        sys.argv[2:], "ho:",
	        ["help","output="])

	except getopt.GetoptError as err:
	    print(err)
	    print(usage)
	    sys.exit(1)

	for opt, arg in opts:
	    if opt in ('-h', '--help'):
	        print(usage)
	        sys.exit()

	with open(case_file, 'r') as f:
	    config = list(yaml.load_all(f, Loader=Loader))[0]

	run(config)
	sfem.finalize()
