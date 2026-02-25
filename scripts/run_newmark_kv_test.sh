#!/bin/bash

set -e

# switch to the directory of the script (build_release)
cd "$(dirname "$0")"


if ! make sfem_NewmarkKVTest -j12; then
    echo "Build failed, exiting..."
    exit 1
fi

rm -rf test_visualization* u_visualization.xmf u_visualization.h5
echo "Build and cleanup successful"

# Environment variables for Kelvin-Voigt Newmark test
export SFEM_BASE_RESOLUTION=10
export SFEM_ELEMENT_REFINE_LEVEL=0  # Set to >1 for semi-structured mesh
export SFEM_ENABLE_CONTACT=1        # Set to 1 to enable contact
export SFEM_NEUMANN_FORCE=-0.4     # Force on right boundary (negative = compression, same as MR)
export SFEM_NEWMARK_ENABLE_OUTPUT=1
export SFEM_ENABLE_OUTPUT=1
export SFEM_DENSITY=5
export SFEM_SHEAR_STIFFNESS_KV=35
export SFEM_DT=0.1
export SFEM_T=20
export SFEM_BULK_MODULUS=40
export SFEM_DAMPING_RATIO=0.2
# export SFEM_EXECUTION_SPACE=host

echo "Kelvin-Voigt Newmark Test Parameters:"
echo "  BASE_RESOLUTION = $SFEM_BASE_RESOLUTION"
echo "  ELEMENT_REFINE_LEVEL = $SFEM_ELEMENT_REFINE_LEVEL"
echo "  ENABLE_CONTACT = $SFEM_ENABLE_CONTACT"
echo "  NEUMANN_FORCE = $SFEM_NEUMANN_FORCE"
echo "  ENABLE_OUTPUT = $SFEM_NEWMARK_ENABLE_OUTPUT"



./sfem_NewmarkKVTest

mv test_newmark_kv test_visualization

python ../python/sfem/mesh/raw_to_db.py "test_visualization" "u_visualization.xmf" \
    --transient \
    --time_whole_txt="test_visualization/time.txt" \
    --point_data="test_visualization/disp.0.*.raw,test_visualization/disp.1.*.raw,test_visualization/disp.2.*.raw"

echo "Generated: u_visualization.xmf"


# # Start ParaView if available
echo "Starting ParaView..."
if [ -f "../../para_view_general_single.pvsm" ]; then
    /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview --state=../../para_view_general_single.pvsm &
    echo "Using ParaView state: ../../para_view_general_single.pvsm"
else
    echo "No state file found, opening files directly"
    /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview u_visualization.xmf &
fi


