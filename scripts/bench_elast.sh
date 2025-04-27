#!/usr/bin/env bash

set -e
# set -x

make -j8 sfem_GalerkinAssemblyTest 

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true

export SFEM_REPEAT=2
export SFEM_BLOCK_SIZE=3
export SFEM_ELEMENT_DEREFINE=0

export SFEM_ELEMENT_REFINE_LEVEL=4
res=(2 4 6 8 16 32 40)

# export SFEM_ELEMENT_REFINE_LEVEL=16
# res=(2 4 8)

rm -f log_bsr.md
rm -f log_mf.md

for r in "${res[@]}"
do
	echo "R=$r"
	export SFEM_BASE_RESOLUTION=$r
	export SFEM_OPERATOR=LinearElasticity 

	SFEM_FINE_OP_TYPE=BSR SFEM_COARSE_OP_TYPE=BSR  	./sfem_GalerkinAssemblyTest >> log_bsr.md
	SFEM_FINE_OP_TYPE=MF SFEM_COARSE_OP_TYPE=MF 	./sfem_GalerkinAssemblyTest >> log_mf.md
done

echo "----------------------------"
grep "RTP" log_bsr.md | head -1
echo "----------------------------"
grep "Coarse op" log_bsr.md | head -1
echo "----------------------------"
grep coarse_op log_bsr.md

echo "----------------------------"
grep "Coarse op" log_mf.md | head -1
echo "----------------------------"
grep coarse_op log_mf.md

echo "----------------------------"
grep "Fine op" log_bsr.md | head -1
echo "----------------------------"
grep fine_op log_bsr.md

echo "----------------------------"
grep "Fine op" log_mf.md | head -1
echo "----------------------------"
grep fine_op log_mf.md
