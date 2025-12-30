#!/usr/bin/env bash

set -e

python regression_prony.py --dataset qlv --fit-mr-first --n-tau 200 --tau-min 1e-8 --tau-max 1e8 --alpha 1e-3 --top-k 6 --verbose --plot