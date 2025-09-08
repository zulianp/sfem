#!/bin/bash

set -e

# Usage: ./run_test_single.sh EXECUTABLE TEST_NAME MATERIAL_TYPE [ETA]
# Example: ./run_test_single.sh sfem_TimedepNeumannKVTest test_newmark_timedep_neumann kv 0.5
# Example: ./run_test_single.sh sfem_TimedepNeumannTest test_newmark_timedep_neumann linear

if [ $# -lt 3 ]; then
    echo "Usage: $0 EXECUTABLE TEST_NAME MATERIAL_TYPE [ETA]"
    echo "  EXECUTABLE: e.g., sfem_TimedepNeumannKVTest, sfem_CantileverKVTest"
    echo "  TEST_NAME: e.g., test_newmark_timedep_neumann, test_newmark_cantilever"
    echo "  MATERIAL_TYPE: linear or kv"
    echo "  ETA: damping ratio (required for kv material)"
    echo ""
    echo "Examples:"
    echo "  $0 sfem_TimedepNeumannKVTest test_newmark_timedep_neumann kv 0.5"
    echo "  $0 sfem_TimedepNeumannTest test_newmark_timedep_neumann linear"
    exit 1
fi

EXECUTABLE=$1
TEST_NAME=$2
MATERIAL_TYPE=$3
ETA=${4:-0}

# switch to the directory of the script
cd "$(dirname "$0")"

echo "Running single test: $EXECUTABLE"
echo "Test name: $TEST_NAME"
echo "Material type: $MATERIAL_TYPE"
if [ "$MATERIAL_TYPE" = "kv" ]; then
    echo "Eta (damping ratio): $ETA"
fi

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

# Build
make -j12

# Run test
if [ "$MATERIAL_TYPE" = "kv" ]; then
    export SFEM_DAMPING_RATIO=$ETA
    echo "Running with damping ratio: $SFEM_DAMPING_RATIO"
    ./$EXECUTABLE
else
    echo "Running linear elasticity test"
    ./$EXECUTABLE
fi

# Convert to visualization format
if [ "$MATERIAL_TYPE" = "kv" ]; then
    OUTPUT_DIR="${TEST_NAME}_kv"
    XMF_FILE="${TEST_NAME}_eta_${ETA}.xmf"
else
    OUTPUT_DIR="${TEST_NAME}"
    XMF_FILE="${TEST_NAME}_linear.xmf"
fi

if [ -d "$OUTPUT_DIR" ]; then
    python ../python/sfem/mesh/raw_to_db.py "$OUTPUT_DIR" u.xmf \
        --transient \
        --time_whole_txt="$OUTPUT_DIR/time.txt" \
        --point_data="$OUTPUT_DIR/disp.0.*.raw,$OUTPUT_DIR/disp.1.*.raw,$OUTPUT_DIR/disp.2.*.raw"
    
    echo "Generated: u.xmf"
    echo "Starting ParaView..."
    # /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview u.xmf
    if [ -f "../../neumann_single.pvsm" ]; then
        /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview --state=../../neumann_single.pvsm &
        echo "Using ParaView state: ../../neumann_single.pvsm"
    else
        echo "No state file found, opening files directly"
        /Applications/ParaView-5.13.3.app/Contents/MacOS/paraview u.xmf &
    fi
else
    echo "Warning: Output directory $OUTPUT_DIR not found"
    ls -la
fi
