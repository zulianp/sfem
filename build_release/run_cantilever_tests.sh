#!/bin/bash

set -e

# switch to the directory of the script (build_release)
cd "$(dirname "$0")"

ETA_VALUES=(0 0.1 0.5 1)

echo "Starting Cantilever Kelvin-Voigt tests..."

export SFEM_SHEAR_MODULUS=34.246575342465754
export SFEM_FIRST_LAME_PARAMETER=393.835616438356164
export SFEM_YOUNG_MODULUS=68.493150684931507
export SFEM_BULK_MODULUS=416.6666666666667
export SFEM_DT=0.001

echo "Material parameters (E=100, nu=0.46, dt=1e-3):"
echo "  SHEAR_MODULUS = $SFEM_SHEAR_MODULUS"
echo "  LAME_PARAMETER = $SFEM_FIRST_LAME_PARAMETER"
echo "  YOUNG_MODULUS = $SFEM_YOUNG_MODULUS"
echo "  BULK_MODULUS = $SFEM_BULK_MODULUS"
echo "  DT = $SFEM_DT"

make -j12 && \
rm -rf test_newmark_cantilever_eta_* test_newmark_kv_cantilever u_eta_*.xmf u.xmf u_eta_*.h5 u.h5 && \
echo "Build and cleanup successful"

# Run Kelvin-Voigt cantilever tests
counter=1
for eta in "${ETA_VALUES[@]}"; do
    echo "Running Cantilever KV eta = $eta"
    
    SFEM_DAMPING_RATIO=$eta ./sfem_CantileverKVTest && \
    mv test_newmark_kv_cantilever "test_newmark_cantilever_eta_$eta" && \
    
    python ../python/sfem/mesh/raw_to_db.py "test_newmark_cantilever_eta_$eta" "u_eta_${counter}.xmf" \
        --transient \
        --time_whole_txt="test_newmark_cantilever_eta_$eta/time.txt" \
        --point_data="test_newmark_cantilever_eta_$eta/disp.0.*.raw,test_newmark_cantilever_eta_$eta/disp.1.*.raw,test_newmark_cantilever_eta_$eta/disp.2.*.raw" && \
    
    echo "Generated: u_eta_${counter}.xmf (eta = $eta)"
    ((counter++))
done

echo "All cantilever tests completed!"
echo "Generated files:"
ls -la u*.xmf

echo "Starting ParaView..."
if [ -f "../../paraview_setting_2x2_full_kv_with_mesh.pvsm" ]; then
    /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview --state=../../paraview_setting_2x2_full_kv_with_mesh.pvsm &
    echo "Using ParaView state: ../../paraview_setting_2x2_full_kv_with_mesh.pvsm"
else
    echo "No state file found, opening files directly"
    /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview u_eta_*.xmf &
fi
