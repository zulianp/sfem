#!/usr/bin/env bash

set -e
set -x

SFEM_SHEAR_MODULUS=2e6 
SFEM_FIRST_LAME_PARAMETER=1e6 

SFEM_TRACE_FILE=fp64.csv SFEM_USE_PARTIAL_ASSEMBLY=0 SFEM_USE_COMPRESSION=0 ./hyperelasticity.sh
SFEM_TRACE_FILE=fp32.csv SFEM_USE_PARTIAL_ASSEMBLY=1 SFEM_USE_COMPRESSION=0 ./hyperelasticity.sh
SFEM_TRACE_FILE=fp16.csv SFEM_USE_PARTIAL_ASSEMBLY=1 SFEM_USE_COMPRESSION=1 ./hyperelasticity.sh

grep "NeoHookeanOgden::apply"  *.csv
grep "NeoHookeanOgden::update" *.csv