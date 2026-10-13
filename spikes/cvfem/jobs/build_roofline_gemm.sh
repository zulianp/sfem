#!/usr/bin/env bash
# Build against the explicitly selected library and retain the linkage evidence.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo 'usage: build_roofline_gemm.sh NEW_OUTPUT_DIRECTORY BLAS_LIBRARY [CXX]' >&2
    exit 2
fi
out=$1
library=$2
compiler=${3:-g++}
py=${CVFEM_PYTHON:-python3}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
source_file="$root/spikes/cvfem/src/drivers/cvfem_roofline_gemm.cpp"
[[ ! -e "$out" ]] || { echo 'build output directory already exists' >&2; exit 2; }
mkdir -p "$out"
args=(-std=c++17 -O3 -Wall -Wextra -Werror "$source_file")
if [[ $(uname -s) == Darwin && "$library" == Accelerate ]]; then
    [[ -z ${SDKROOT:-} ]] || args+=(-isysroot "$SDKROOT")
    args+=(-framework Accelerate)
else
    [[ -f "$library" ]] || { echo 'explicit BLAS library does not exist' >&2; exit 2; }
    library=$("$py" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$library")
    args+=("$library" "-Wl,-rpath,$(dirname "$library")" -ldl)
fi
binary="$out/cvfem_roofline_gemm"
args+=(-o "$binary")
"$compiler" --version > "$out/compiler.txt"
"$py" - "$out/build-command.json" "$compiler" "${args[@]}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:])+'\n')
PY
"$compiler" "${args[@]}" > "$out/build.log" 2>&1
if [[ $(uname -s) == Darwin ]]; then
    otool -L "$binary" > "$out/linkage.txt"
else
    ldd "$binary" > "$out/linkage.txt"
    ! grep -q 'not found' "$out/linkage.txt" || { echo 'DGEMM linked dependency missing' >&2; exit 2; }
fi
OMP_NUM_THREADS=72 OPENBLAS_NUM_THREADS=72 GOTO_NUM_THREADS=72 VECLIB_MAXIMUM_THREADS=72 \
    "$binary" preflight 128 72 > "$out/preflight.out" 2>&1
PYTHONPATH="$root/spikes/cvfem/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" - "$out" "$library" <<'PY'
import hashlib, json, platform, sys
from pathlib import Path
from roofline_gemm import library_identity, parse_output
out, library = Path(sys.argv[1]), sys.argv[2]
output = (out/'preflight.out').read_text()
identity = library_identity(output, 72, None if library == 'Accelerate' else Path(library),
                            production=platform.system() != 'Darwin')
parse_output(output, 'normal', 1, n=128, threads=72)
(out/'runtime-library.json').write_text(json.dumps(identity, indent=2)+'\n')
command = json.loads((out/'build-command.json').read_text())
source = next(Path(p) for p in command if p.endswith('/cvfem_roofline_gemm.cpp'))
files = [out/'cvfem_roofline_gemm', source, source.with_name('cvfem_perf_control.hpp')]
if library != 'Accelerate': files.append(Path(library))
(out/'build-inputs.sha256.json').write_text(json.dumps({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in files}, indent=2)+'\n')
PY
printf 'Checked DGEMM binary: %s\n' "$binary"
