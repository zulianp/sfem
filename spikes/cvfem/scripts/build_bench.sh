#!/usr/bin/env bash
cd /ritom/scratch/cscs/zulianp/sfem-coarseop/spikes/cvfem || exit 1
source scripts/alps_env.sh
export CVFEM_SFEM_INSTALL=/ritom/scratch/cscs/zulianp/installations/sfem-coarseop
export CVFEM_BUILD=$PWD/build
cvfem_touch
cvfem_uenv cmake --build "$CVFEM_BUILD" --target cvfem_hex8_ns_upwind_bench -j 32
rc=$?
# NO REFERENCE CAPTURE HERE ANY MORE.
#
# This script used to copy its output to perf/refbin/bench_prestep5, which is how the
# reference it existed to preserve was destroyed: rerun after the change it was meant to be
# the "before" of, it overwrote the before with the after. An A/B between two copies of one
# binary reports a clean pass, so that failure is invisible in the result.
#
# Capturing a reference is now a deliberate, separate act -- scripts/build_ref_subpar.sh
# builds one into its own tree -- because a reference is the only artifact here that cannot
# be regenerated from the current source, and nothing that runs routinely should touch it.
echo "BUILD_RC=$rc"
