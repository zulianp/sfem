#!/bin/bash

set -e

# switch to the directory of the script (build_release)
cd "$(dirname "$0")"

REFINE_LEVELS=(1 2 3 4)

echo "Starting Linear Elasticity Newmark refine level comparison tests..."

export SFEM_SHEAR_MODULUS=2
export SFEM_FIRST_LAME_PARAMETER=1.6666666666666667
export SFEM_YOUNG_MODULUS=4.909090909090909
export SFEM_BULK_MODULUS=3
export SFEM_DT=0.1

echo "Material parameters:"
echo "  SHEAR_MODULUS = $SFEM_SHEAR_MODULUS"
echo "  LAME_PARAMETER = $SFEM_FIRST_LAME_PARAMETER"
echo "  YOUNG_MODULUS = $SFEM_YOUNG_MODULUS"
echo "  BULK_MODULUS = $SFEM_BULK_MODULUS"
echo "  DT = $SFEM_DT"

make -j12 && \
rm -rf test_newmark_refine_* u_*.xmf u_*.h5 && \
echo "Build and cleanup successful"

# Run Linear Elasticity Newmark tests with different refine levels
counter=1
for level in "${REFINE_LEVELS[@]}"; do
    echo "Running Linear Elasticity Newmark test with refine level = $level"
    
    SFEM_ELEMENT_REFINE_LEVEL=$level ./sfem_NewmarkTest && \
    mv test_newmark "test_newmark_refine_$level" && \
    
    python ../python/sfem/mesh/raw_to_db.py "test_newmark_refine_$level" "u_${counter}.xmf" \
        --transient \
        --time_whole_txt="test_newmark_refine_$level/time.txt" \
        --point_data="test_newmark_refine_$level/disp.0.*.raw,test_newmark_refine_$level/disp.1.*.raw,test_newmark_refine_$level/disp.2.*.raw" && \
    
    echo "Generated: u_${counter}.xmf (refine level = $level) - Linear Elasticity"
    ((counter++))
done

echo "All Linear Elasticity refine level tests completed!"
echo "Generated files:"
ls -la u_*.xmf

echo "Test directories:"
ls -d test_newmark_refine_*

echo "Starting ParaView..."
if [ -f "../../paraview_setting_2x2_full_kv_with_mesh.pvsm" ]; then
    echo "Note: ParaView state file is for KV tests, but will work for linear elasticity too"
    /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview --state=../../paraview_setting_2x2_full_kv_with_mesh.pvsm &
    echo "Using ParaView state: ../../paraview_setting_2x2_full_kv_with_mesh.pvsm"
else
    echo "No state file found, opening files directly"
    /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview u_*.xmf &
fi

echo ""
echo "Generated Linear Elasticity files: u_1.xmf to u_4.xmf (for refine levels 1-4)"
echo ""
echo "To compare with Kelvin-Voigt results later:"
echo "1. Save current Linear Elasticity results: mv u_*.xmf u_linear_*.xmf"
echo "2. Run KV tests: ./run_refine_level_tests.sh"  
echo "3. Compare both: /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview u_*.xmf u_linear_*.xmf"
