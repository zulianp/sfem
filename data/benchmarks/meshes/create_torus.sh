#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Default parameters
MAJOR_RADIUS=5.0
MINOR_RADIUS=1.0
REFINEMENTS=0
OUTPUT="torus.vtk"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --major-radius)
            MAJOR_RADIUS="$2"
            shift 2
            ;;
        --minor-radius)
            MINOR_RADIUS="$2"
            shift 2
            ;;
        --refinements)
            REFINEMENTS="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --major-radius R    Major radius (default: 5.0)"
            echo "  --minor-radius r    Minor radius (default: 1.0)"
            echo "  --refinements N     Number of refinements (default: 0)"
            echo "  --output FILE       Output VTK file (default: torus.vtk)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Creating torus mesh"
echo "========================================="
echo "Major radius: $MAJOR_RADIUS"
echo "Minor radius: $MINOR_RADIUS"
echo "Refinements: $REFINEMENTS"
echo "Output: $OUTPUT"
echo "========================================="

# Create mesh using Python script
python3 $SCRIPTPATH/torus.py "$OUTPUT" \
    --major-radius=$MAJOR_RADIUS \
    --minor-radius=$MINOR_RADIUS \
    --refinements=$REFINEMENTS

echo "Mesh created: $OUTPUT"

