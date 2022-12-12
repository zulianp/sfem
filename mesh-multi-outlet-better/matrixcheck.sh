#!/usr/bin/env bash

set -e

#  Full

echo "-------------------"
echo "rowptr"
echo "-------------------"
../python/printarraystats.py -p ./rowptr.raw -d int32 

echo "-------------------"
echo "colidx"
echo "-------------------"
../python/printarraystats.py -p ./colidx.raw -d int32 

echo "-------------------"
echo "values"
echo "-------------------"
../python/printarraystats.py -p ./values.raw -d float64 


# Condensed

echo "-------------------"
echo "rowptr"
echo "-------------------"
../python/printarraystats.py -p ./condensed/rowptr.raw -d int32 


echo "-------------------"
echo "colidx"
echo "-------------------"
../python/printarraystats.py -p ./condensed/colidx.raw -d int32 

echo "-------------------"
echo "values"
echo "-------------------"
../python/printarraystats.py -p ./condensed/values.raw -d float64 
