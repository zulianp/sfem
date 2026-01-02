#!/bin/bash

set -e

# switch to the directory of the script (build_release)
cd "$(dirname "$0")"


if ! make -j12; then
    echo "Build failed, exiting..."
    exit 1
fi

rm -rf test_visualization* u_visualization.xmf u_visualization.h5
echo "Build and cleanup successful"

# make -j32
export SFEM_ENABLE_CONTACT=1
export SFEM_DENSITY=20.0    
export SFEM_C10=90
export SFEM_C01=70
export SFEM_BULK_MODULUS=10000.0
export SFEM_DT=0.1
export SFEM_T=15
export SFEM_NEUMANN_FORCE=-0.3
export SFEM_BASE_RESOLUTION=10
export SFEM_ENABLE_OUTPUT=1

echo "Material parameters:"
echo "  C10 = $SFEM_C10"
echo "  C01 = $SFEM_C01"
echo "  K = $SFEM_BULK_MODULUS"
echo "  DENSITY = $SFEM_DENSITY"
echo "  NEUMANN_FORCE = $SFEM_NEUMANN_FORCE"
echo "  DT = $SFEM_DT"
echo "  T = $SFEM_T"
echo "  BASE_RESOLUTION = $SFEM_BASE_RESOLUTION"
echo "  ENABLE_OUTPUT = $SFEM_ENABLE_OUTPUT"



./sfem_MooneyRivlinViscoTest

mv test_mooney_rivlin_visco test_visualization

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











