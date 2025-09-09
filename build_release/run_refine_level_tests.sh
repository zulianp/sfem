#!/bin/bash

set -e

# switch to the directory of the script (build_release)
cd "$(dirname "$0")"

REFINE_LEVELS=(32)

echo "Starting Kelvin-Voigt refine level comparison tests..."

export SFEM_SHEAR_MODULUS=2
export SFEM_FIRST_LAME_PARAMETER=1.6666666666666667
export SFEM_YOUNG_MODULUS=4
export SFEM_BULK_MODULUS=3
export SFEM_DT=0.1
export SFEM_DAMPING_RATIO=1
export SFEM_DENSITY=1
export SFEM_NEWMARK_ENABLE_OUTPUT=1

echo "Material parameters:"
echo "  SHEAR_MODULUS = $SFEM_SHEAR_MODULUS"
echo "  LAME_PARAMETER = $SFEM_FIRST_LAME_PARAMETER"
echo "  YOUNG_MODULUS = $SFEM_YOUNG_MODULUS"
echo "  BULK_MODULUS = $SFEM_BULK_MODULUS"
echo "  DT = $SFEM_DT"
echo "  DAMPING_RATIO = $SFEM_DAMPING_RATIO"

if ! make -j12; then
    echo "Build failed, exiting..."
    exit 1
fi

rm -rf test_newmark_kv_refine_* u_*.xmf u_*.h5
echo "Build and cleanup successful"

# Run Kelvin-Voigt tests with different refine levels
counter=1
for level in "${REFINE_LEVELS[@]}"; do
    echo "Running Kelvin-Voigt test with refine level = $level"
    
    SFEM_ELEMENT_REFINE_LEVEL=$level ./sfem_NewmarkKVTest && \
    mv test_newmark_kv "test_newmark_kv_refine_$level" && \
    
    python ../python/sfem/mesh/raw_to_db.py "test_newmark_kv_refine_$level" "u_${counter}.xmf" \
        --transient \
        --time_whole_txt="test_newmark_kv_refine_$level/time.txt" \
        --point_data="test_newmark_kv_refine_$level/disp.0.*.raw,test_newmark_kv_refine_$level/disp.1.*.raw,test_newmark_kv_refine_$level/disp.2.*.raw" && \
    
    echo "Generated: u_${counter}.xmf (refine level = $level)"
    ((counter++))
done

echo "All refine level tests completed!"
echo "Generated files:"
ls -la u_*.xmf

echo "Test directories:"
ls -d test_newmark_kv_refine_*

echo "Starting ParaView..."
if [ -f "../../paraview_setting_2x2_full_kv_with_mesh.pvsm" ]; then
    /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview --state=../../paraview_setting_2x2_full_kv_with_mesh.pvsm &
    echo "Using ParaView state: ../../paraview_setting_2x2_full_kv_with_mesh.pvsm"
else
    echo "No state file found, opening files directly"
    /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview u_*.xmf &
fi
