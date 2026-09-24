#!/usr/bin/env bash
# The A/B reference for the step-5 move, rebuilt as "every generated arrangement present".
#
# The pre-step-5 binary was destroyed by a build script that overwrote its own reference, and
# it cannot be reconstructed from this tree: the generated header, the stubs that stand in for
# the retired functions, and the CLI rejections all changed together, so reverting one of them
# alone does not compile.
#
# What CAN be built is -DCVFEM_ENABLE_SUBPAR=ON, which puts every retired arrangement back
# into the translation unit. That is a SUPERSET of the pre-step-5 state -- it also restores
# the four assembly arrangements retired before this session -- which makes the comparison
# conservative in the right direction: the surviving kernels are measured against a build
# carrying MORE code beside them than the true reference did, so a null result there is
# stronger evidence than a null result against the exact reference would have been.
#
# What it cannot tell apart is which of the two removals moved a number, if one did.
set -uo pipefail
cd /ritom/scratch/cscs/zulianp/sfem-coarseop/spikes/cvfem || exit 1
source scripts/alps_env.sh
export CVFEM_SFEM_INSTALL=/ritom/scratch/cscs/zulianp/installations/sfem-coarseop
export CVFEM_BUILD=$PWD/build_refsubpar
cvfem_configure -DCVFEM_ENABLE_SUBPAR=ON > /dev/null 2>&1
cvfem_uenv cmake --build "$CVFEM_BUILD" --target cvfem_hex8_ns_upwind_bench -j 32
echo "BUILD_RC=$?"
