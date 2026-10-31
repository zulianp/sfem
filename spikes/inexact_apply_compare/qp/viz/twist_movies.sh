#!/bin/sh
# Movies and stills of the 90 degree twist for TET4, TET10, HEX8 and HEX27 (the Newton study's
# large-deformation case at a resolution that shows the elements), exact operator.
#   sh twist_movies.sh OUT_DIR
# Needs the spike builds build-spike-{TET4,TET10,HEX8,HEX27} in the repository root, the venv's
# python (numpy), ParaView's pvbatch and ffmpeg.  OUT_DIR gets the raw output, XDMF time series
# (case.xdmf, edges.xdmf; open in ParaView), the frames, twist_<element>.mp4/.png and the 2x2 composites.
set -e
OUT=$(cd "$1" 2>/dev/null || mkdir -p "$1"; cd "$1"; pwd)
HERE=$(cd "$(dirname "$0")"; pwd)
ROOT=$(cd "$HERE/../../../.."; pwd)
PY=${PY:-python3}
PV=${PV:-/Applications/ParaView-6.1.0.app/Contents/bin/pvbatch}
C=$ROOT/spikes/inexact_apply_compare/qp/cases
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
run() { "$ROOT/build-spike-$1/newton_qp" "$2" "$3" "$4" --set solver.linear.format=BSR --set output.path="$OUT/$5" --set output.export_freq=1 | grep "newton_qp: run"; }
"$ROOT/build/external/smesh/cube" TET4 8 8 8 0 0 0 1 1 1 "$OUT/tet4_8" > /dev/null
"$ROOT/build/external/smesh/cube" TET4 6 6 6 0 0 0 1 1 1 "$OUT/tet4_6" > /dev/null
run HEX8 "$C/hex8_twist.yaml" --set "mesh.box.n=[12,12,12]" hex8
run TET4 "$C/tet4_twist.yaml" --set "mesh.path=$OUT/tet4_8" tet4
run TET10 "$C/tet10_twist.yaml" --set "mesh.path=$OUT/tet4_6" tet10
run HEX27 "$C/hex27_twist.yaml" --set "mesh.box.n=[6,6,6]" hex27
cd "$OUT"
for e in hex8 tet4 tet10; do $PY "$HERE/make_xdmf.py" $e; done
$PY "$HERE/make_xdmf.py" hex27 --macro 6
"$PV" "$HERE/render.py" tet4 "TET4 - linear tetrahedra, 6144 elements" frames_tet4 | grep "^frames"
"$PV" "$HERE/render.py" tet10 "TET10 - quadratic tetrahedra, 2592 elements" frames_tet10 | grep "^frames"
"$PV" "$HERE/render.py" hex8 "HEX8 - trilinear hexahedra, 12 x 12 x 12 elements" frames_hex8 | grep "^frames"
"$PV" "$HERE/render.py" hex27 "HEX27 - triquadratic hexahedra, 6 x 6 x 6 elements" frames_hex27 | grep "^frames"
for e in tet4 tet10 hex8 hex27; do
  ffmpeg -loglevel error -y -framerate 6 -i frames_$e/frame_%03d.png -vf "tpad=stop_mode=clone:stop_duration=2" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 twist_$e.mp4
  cp frames_$e/frame_016.png twist_$e.png
done
ffmpeg -loglevel error -y -framerate 6 -i frames_tet4/frame_%03d.png -framerate 6 -i frames_tet10/frame_%03d.png \
  -framerate 6 -i frames_hex8/frame_%03d.png -framerate 6 -i frames_hex27/frame_%03d.png \
  -filter_complex "[0][1]hstack[a];[2][3]hstack[b];[a][b]vstack,scale=1920:-2,tpad=stop_mode=clone:stop_duration=2" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 twist_all_elements.mp4
ffmpeg -loglevel error -y -i frames_tet4/frame_016.png -i frames_tet10/frame_016.png -i frames_hex8/frame_016.png \
  -i frames_hex27/frame_016.png -filter_complex "[0][1]hstack[a];[2][3]hstack[b];[a][b]vstack,scale=1920:-2" \
  -frames:v 1 twist_all_elements.png
echo "written to $OUT"
