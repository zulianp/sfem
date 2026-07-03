#!/usr/bin/env bash
#
# Compile SFEM under many CUDA CMake option combinations (SFEM_ENABLE_CUDA=ON).
# Sweeps optional CPU dependencies together with CUDA-specific dimensions:
# memory model (host/managed/unified) and line-info.
# Intended for CUDA-capable Linux workstations and CSCS Alps (GH200) nodes.
#
# Usage:
#   ./scripts/testing/test_CUDA_configs.sh              # quick sweep (default)
#   ./scripts/testing/test_CUDA_configs.sh --all        # full 2^N valid combinations
#   ./scripts/testing/test_CUDA_configs.sh --pairwise   # pairwise flag combinations
#   ./scripts/testing/test_CUDA_configs.sh --list       # print configs, do not build
#   ./scripts/testing/test_CUDA_configs.sh --config 42  # build one entry from --list
#
# Alps (CSCS) example:
#   cd $SCRATCH/sfem && SFEM_USE_UENV=1 JOBS=32 ./scripts/testing/test_CUDA_configs.sh --quick
#   sbatch --wrap='cd $SCRATCH/sfem && SFEM_USE_UENV=1 ./scripts/testing/test_CUDA_configs.sh --all'
#
# Environment (optional):
#   MODE=quick|pairwise|all     # sweep mode (default quick)
#   CONFIG_INDEX=42              # build only this index from the sweep list
#   BUILD_ROOT=...           # default: <repo>/build_cuda_configs
#   JOBS=8
#   RUN_TESTS=0|1            # run ctest after each successful build (default 0)
#   CMAKE_BUILD_TYPE=Release
#   MATRIXIO_DIR, METIS_DIR  # forwarded to CMake when set
#   SFEM_USE_UENV=auto|0|1   # Alps: wrap cmake/make in CSCS uenv (default auto)
#   UENV_IMAGE=prgenv-gnu/24.7:v3
#   OPENMP_DIR               # Homebrew libomp prefix on macOS (rarely relevant for CUDA)
#   SKIP_MISSING_DEPS=1      # skip configs needing unavailable externals (default 1)
#   ONLY_FLAGS=OPENMP,METIS  # limit which toggles enter the combination matrix
#   EXCLUDE_FLAGS=PYTHON     # never turn these ON in generated configs
#   CUDA_MEMORY_MODELS=host,managed,unified  # memory models to sweep (default host;
#                                            #   quick mode always covers all three on minimal)
#   CUDA_LINEINFO=0|1        # -DSFEM_ENABLE_CUDA_LINEINFO (default 0)
#   SFEM_CUDA_ARCH=90        # -DCMAKE_CUDA_ARCHITECTURES (default: CMake/nvcc auto-detect)
#   CUDACXX / CMAKE_CUDA_COMPILER  # override nvcc path
#

set -euo pipefail

usage() {
    sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'
}

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd -P)"

MODE="${MODE:-quick}"
SWEEP_MODE="${SWEEP_MODE:-}"
CONFIG_INDEX="${CONFIG_INDEX:-}"
BUILD_ROOT="${BUILD_ROOT:-${REPO_ROOT}/build_cuda_configs}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
RUN_TESTS="${RUN_TESTS:-0}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
SKIP_MISSING_DEPS="${SKIP_MISSING_DEPS:-1}"
SFEM_USE_UENV="${SFEM_USE_UENV:-auto}"
UENV_IMAGE="${UENV_IMAGE:-prgenv-gnu/24.7:v3}"
CUDA_LINEINFO="${CUDA_LINEINFO:-0}"
SFEM_CUDA_ARCH="${SFEM_CUDA_ARCH:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --all) SWEEP_MODE=all ;;
        --pairwise) SWEEP_MODE=pairwise ;;
        --quick) SWEEP_MODE=quick ;;
        --list) DRY_RUN=1 ;;
        --config)
            shift
            [[ $# -gt 0 ]] || die "--config requires an index"
            CONFIG_INDEX="$1"
            ;;
        --dry-run) DRY_RUN=1 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
    shift
done

if [[ -z "$SWEEP_MODE" ]]; then
    SWEEP_MODE="${MODE}"
    if [[ "$MODE" == config ]]; then
        SWEEP_MODE=quick
    fi
fi
MODE="$SWEEP_MODE"

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"
IS_MAC=0
IS_LINUX=0
IS_ALPS=0
IS_X86=0

case "$UNAME_S" in
    Darwin) IS_MAC=1 ;;
    Linux) IS_LINUX=1 ;;
esac

if [[ "$UNAME_M" == x86_64 || "$UNAME_M" == amd64 ]]; then
    IS_X86=1
fi

if [[ -n "${SLURM_CLUSTER_NAME:-}" && "${SLURM_CLUSTER_NAME}" == *daint* ]] \
    || [[ -d /capstor/scratch/cscs ]] \
    || [[ -n "${CSCS_SITE:-}" ]]; then
    IS_ALPS=1
fi

if [[ "$SFEM_USE_UENV" == auto ]]; then
    if [[ "$IS_ALPS" -eq 1 ]] && command -v srun >/dev/null 2>&1; then
        SFEM_USE_UENV=1
    else
        SFEM_USE_UENV=0
    fi
fi

# Optional CPU toggles swept alongside CUDA (SFEM_ENABLE_CUDA is always ON here).
# MPI OFF still injects MPI include/link flags for MatrixIO (<mpi.h>).
FLAG_NAMES=(
    AMG
    OPENMP
    MPI
    METIS
    PYTHON
    BLAS
    LAPACK
    RYAML
    CODEGEN
    SCCD
    SSDF
    RESAMPLING
    EXPLICIT_VECTORIZATION
    AVX2
    AVX512
    AVX512_SORT
    HXTSORT
)

flag_cmake_name() {
    case "$1" in
        AMG) echo SFEM_ENABLE_AMG ;;
        OPENMP) echo SFEM_ENABLE_OPENMP ;;
        MPI) echo SFEM_ENABLE_MPI ;;
        METIS) echo SFEM_ENABLE_METIS ;;
        PYTHON) echo SFEM_ENABLE_PYTHON ;;
        BLAS) echo SFEM_ENABLE_BLAS ;;
        LAPACK) echo SFEM_ENABLE_LAPACK ;;
        RYAML) echo SFEM_ENABLE_RYAML ;;
        CODEGEN) echo SFEM_ENABLE_CODEGEN ;;
        SCCD) echo SFEM_ENABLE_SCCD ;;
        SSDF) echo SFEM_ENABLE_SSDF ;;
        RESAMPLING) echo SFEM_ENABLE_RESAMPLING ;;
        EXPLICIT_VECTORIZATION) echo SFEM_ENABLE_EXPLICIT_VECTORIZATION ;;
        AVX2) echo SFEM_ENABLE_AVX2 ;;
        AVX512) echo SFEM_ENABLE_AVX512 ;;
        AVX512_SORT) echo SFEM_ENABLE_AVX512_SORT ;;
        HXTSORT) echo SFEM_ENABLE_HXTSORT ;;
        *) die "unknown flag alias: $1" ;;
    esac
}

# Apply ONLY_FLAGS / EXCLUDE_FLAGS filters
filter_flag_names() {
    local out=()
    local name
    local only="${ONLY_FLAGS:-}"
    local exclude="${EXCLUDE_FLAGS:-}"

    for name in "${FLAG_NAMES[@]}"; do
        if [[ -n "$only" ]]; then
            case ",${only}," in
                *,"${name}",*) ;;
                *) continue ;;
            esac
        fi
        if [[ -n "$exclude" ]]; then
            case ",${exclude}," in
                *,"${name}",*) continue ;;
            esac
        fi
        if [[ "$IS_X86" -eq 0 ]]; then
            case "$name" in
                AVX2|AVX512|AVX512_SORT) continue ;;
            esac
        fi
        out+=("$name")
    done
    ACTIVE_FLAGS=("${out[@]}")
}

filter_flag_names

# Memory models to sweep.
IFS=',' read -r -a MEM_MODELS <<< "${CUDA_MEMORY_MODELS:-host}"
for mm in "${MEM_MODELS[@]}"; do
    case "$mm" in
        host|managed|unified) ;;
        *) die "invalid CUDA memory model '${mm}' (expected host|managed|unified)" ;;
    esac
done

# ---------------------------------------------------------------------------
# External dependency probes
# ---------------------------------------------------------------------------

have_cuda() {
    [[ -n "${CUDACXX:-}" && -x "${CUDACXX}" ]] \
        || [[ -n "${CMAKE_CUDA_COMPILER:-}" && -x "${CMAKE_CUDA_COMPILER}" ]] \
        || command -v nvcc >/dev/null 2>&1 \
        || [[ -x /usr/local/cuda/bin/nvcc ]] \
        || { [[ "$SFEM_USE_UENV" == 1 ]]; }  # uenv provides nvcc at build time
}

have_matrixio() {
    [[ -n "${MATRIXIO_DIR:-}" && -f "${MATRIXIO_DIR}/matrixio_array.h" ]] \
        || [[ -f "${REPO_ROOT}/../matrix.io/matrixio_array.h" ]]
}

have_metis() {
    [[ -n "${METIS_DIR:-}" && -f "${METIS_DIR}/include/metis.h" ]] \
        || [[ -f /usr/include/metis.h ]] \
        || [[ -f /usr/local/include/metis.h ]] \
        || { command -v brew >/dev/null 2>&1 && brew --prefix metis >/dev/null 2>&1; }
}

have_python_bindings() {
    [[ -f "${REPO_ROOT}/venv/bin/python" || -n "${VIRTUAL_ENV:-}" ]] \
        && python3 -c "import nanobind" >/dev/null 2>&1
}

have_openmp() {
    if [[ -n "${OPENMP_DIR:-}" && -f "${OPENMP_DIR}/include/omp.h" ]]; then
        return 0
    fi
    if [[ "$IS_MAC" -eq 1 ]]; then
        [[ -f /opt/homebrew/opt/libomp/include/omp.h ]] \
            || [[ -f /usr/local/opt/libomp/include/omp.h ]]
    else
        [[ -f /usr/include/omp.h ]] \
            || [[ -f /usr/lib/x86_64-linux-gnu/openmp/include/omp.h ]] \
            || { command -v brew >/dev/null 2>&1 && brew --prefix libomp >/dev/null 2>&1; } \
            || { [[ "$SFEM_USE_UENV" == 1 ]]; }
    fi
}

have_mpi() {
    { command -v mpicc >/dev/null 2>&1 && command -v mpicxx >/dev/null 2>&1; } \
        || { [[ "$SFEM_USE_UENV" == 1 ]]; }
}

have_submodules() {
    [[ -f "${REPO_ROOT}/external/smesh/CMakeLists.txt" ]]
}

MISSING_NOTE=()

note_missing() {
    MISSING_NOTE+=("$1")
}

probe_externals() {
    MISSING_NOTE=()
    have_cuda || note_missing "CUDA toolkit (nvcc not found; set CUDACXX or use SFEM_USE_UENV=1)"
    have_matrixio || note_missing "MatrixIO (set MATRIXIO_DIR or clone ../matrix.io)"
    have_submodules || note_missing "git submodule update --init --recursive"
    if ! have_mpi; then
        note_missing "MPI compilers (mpicc/mpicxx)"
    fi
    if ! have_openmp; then
        note_missing "OpenMP (set OPENMP_DIR on macOS, e.g. brew --prefix libomp)"
    fi
    if ! have_metis; then
        note_missing "METIS (set METIS_DIR)"
    fi
    if ! have_python_bindings; then
        note_missing "Python nanobind (venv + pip install nanobind)"
    fi
}

probe_externals
if [[ ${#MISSING_NOTE[@]} -gt 0 ]]; then
    log "External dependency notes (configs needing these may be skipped):"
    for note in "${MISSING_NOTE[@]}"; do
        log "  - ${note}"
    done
fi

# ---------------------------------------------------------------------------
# Combination helpers
# ---------------------------------------------------------------------------

bit_is_set() {
    local mask=$1
    local bit=$2
    (( (mask & (1 << bit)) != 0 ))
}

mask_to_on_flags() {
    local mask=$1
    local i name
    MASK_ON=()
    for i in "${!ACTIVE_FLAGS[@]}"; do
        if bit_is_set "$mask" "$i"; then
            MASK_ON+=("${ACTIVE_FLAGS[$i]}")
        fi
    done
}

config_valid() {
    local mask=$1
    local i name
    local openmp=0 sccd=0 ssdf=0 hxt=0

    for i in "${!ACTIVE_FLAGS[@]}"; do
        name="${ACTIVE_FLAGS[$i]}"
        if ! bit_is_set "$mask" "$i"; then
            continue
        fi
        case "$name" in
            OPENMP) openmp=1 ;;
            SCCD) sccd=1 ;;
            SSDF) ssdf=1 ;;
            HXTSORT) hxt=1 ;;
        esac
    done

    if [[ "$hxt" -eq 1 && "$openmp" -eq 0 ]]; then
        return 1
    fi
    return 0
}

config_needs_skip() {
    local mask=$1
    local i name

    [[ "$SKIP_MISSING_DEPS" == 1 ]] || return 1

    for i in "${!ACTIVE_FLAGS[@]}"; do
        if ! bit_is_set "$mask" "$i"; then
            continue
        fi
        name="${ACTIVE_FLAGS[$i]}"
        case "$name" in
            METIS) have_metis || return 0 ;;
            PYTHON) have_python_bindings || return 0 ;;
            OPENMP|HXTSORT) have_openmp || return 0 ;;
            MPI) have_mpi || return 0 ;;
        esac
    done
    return 1
}

tolower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

mask_slug() {
    local mask=$1
    mask_to_on_flags "$mask"
    if [[ ${#MASK_ON[@]} -eq 0 ]]; then
        echo "cuda"
        return
    fi
    local slug="cuda+"
    local f
    for f in "${MASK_ON[@]}"; do
        slug+="$(tolower "$f")+"
    done
    echo "${slug%+}"
}

config_label() {
    local mask=$1
    mask_to_on_flags "$mask"
    if [[ ${#MASK_ON[@]} -eq 0 ]]; then
        echo "CUDA"
        return
    fi
    local out="CUDA+"
    local f
    for f in "${MASK_ON[@]}"; do
        out+="${f}+"
    done
    echo "${out%+}"
}

generate_masks_quick() {
    local n=${#ACTIVE_FLAGS[@]}
    local i mask
    GENERATED_MASKS=()

    # minimal: CUDA only, all optional toggles OFF
    GENERATED_MASKS+=(0)

    # defaults-ish: common GPU release flags
    local default_mask=0
    for i in "${!ACTIVE_FLAGS[@]}"; do
        case "${ACTIVE_FLAGS[$i]}" in
            AMG|MPI|RYAML|EXPLICIT_VECTORIZATION) default_mask=$((default_mask | (1 << i))) ;;
        esac
    done
    GENERATED_MASKS+=("$default_mask")

    # single-feature ON from minimal
    for i in $(seq 0 $((n - 1))); do
        mask=$((1 << i))
        if config_valid "$mask"; then
            GENERATED_MASKS+=("$mask")
        fi
    done

    # a few structured combos
    local combo_openmp_metis=0 combo_sccd_ssdf=0 combo_openmp_hxt=0
    for i in "${!ACTIVE_FLAGS[@]}"; do
        case "${ACTIVE_FLAGS[$i]}" in
            OPENMP) combo_openmp_metis=$((combo_openmp_metis | (1 << i))) ;;
            METIS) combo_openmp_metis=$((combo_openmp_metis | (1 << i))) ;;
            SCCD) combo_sccd_ssdf=$((combo_sccd_ssdf | (1 << i))) ;;
            SSDF) combo_sccd_ssdf=$((combo_sccd_ssdf | (1 << i))) ;;
            HXTSORT) combo_openmp_hxt=$((combo_openmp_hxt | (1 << i))) ;;
        esac
        if [[ "${ACTIVE_FLAGS[$i]}" == OPENMP ]]; then
            combo_openmp_hxt=$((combo_openmp_hxt | (1 << i)))
        fi
    done
    for mask in "$combo_openmp_metis" "$combo_sccd_ssdf" "$combo_openmp_hxt"; do
        if config_valid "$mask"; then
            GENERATED_MASKS+=("$mask")
        fi
    done
}

generate_masks_pairwise() {
    local n=${#ACTIVE_FLAGS[@]}
    local i j mask
    GENERATED_MASKS=(0)
    for i in $(seq 0 $((n - 1))); do
        for j in $(seq $((i + 1)) $((n - 1))); do
            mask=$(( (1 << i) | (1 << j) ))
            if config_valid "$mask"; then
                GENERATED_MASKS+=("$mask")
            fi
        done
    done
}

generate_masks_all() {
    local n=${#ACTIVE_FLAGS[@]}
    local max=$(( (1 << n) - 1 ))
    local mask
    GENERATED_MASKS=()
    for mask in $(seq 0 "$max"); do
        if config_valid "$mask"; then
            GENERATED_MASKS+=("$mask")
        fi
    done
}

dedupe_masks() {
    local sorted
    IFS=$'\n' sorted=($(printf '%s\n' "${GENERATED_MASKS[@]}" | sort -n | uniq))
    GENERATED_MASKS=("${sorted[@]}")
}

case "$MODE" in
    quick) generate_masks_quick ;;
    pairwise) generate_masks_pairwise ;;
    all) generate_masks_all ;;
    *) die "unknown MODE=${MODE}" ;;
esac

dedupe_masks

# ---------------------------------------------------------------------------
# Expand (mask) -> (mask, memory-model) jobs
# ---------------------------------------------------------------------------
# In quick mode the minimal config (mask 0) always exercises all three memory
# models so those code paths get covered even with the default CUDA_MEMORY_MODELS.
# Every other config uses the requested memory models (default: host only).

JOB_MASKS=()
JOB_MEMS=()

add_job() {
    JOB_MASKS+=("$1")
    JOB_MEMS+=("$2")
}

for mask in "${GENERATED_MASKS[@]}"; do
    if [[ "$MODE" == quick && "$mask" -eq 0 ]]; then
        add_job 0 host
        add_job 0 managed
        add_job 0 unified
        continue
    fi
    for mem in "${MEM_MODELS[@]}"; do
        add_job "$mask" "$mem"
    done
done

if [[ -n "$CONFIG_INDEX" ]]; then
    if [[ "$CONFIG_INDEX" -lt 0 || "$CONFIG_INDEX" -ge ${#JOB_MASKS[@]} ]]; then
        die "CONFIG_INDEX ${CONFIG_INDEX} out of range (0..$(( ${#JOB_MASKS[@]} - 1 )))"
    fi
    local_mask="${JOB_MASKS[$CONFIG_INDEX]}"
    local_mem="${JOB_MEMS[$CONFIG_INDEX]}"
    JOB_MASKS=("$local_mask")
    JOB_MEMS=("$local_mem")
fi

MEM_MODELS_DISPLAY="$(printf '%s,' "${MEM_MODELS[@]}")"
MEM_MODELS_DISPLAY="${MEM_MODELS_DISPLAY%,}"
log "Platform: ${UNAME_S}/${UNAME_M}  Alps=${IS_ALPS}  x86=${IS_X86}  uenv=${SFEM_USE_UENV}"
log "Mode=${MODE}  active toggles=${#ACTIVE_FLAGS[@]}  memory models=${MEM_MODELS_DISPLAY}  configs=${#JOB_MASKS[@]}"
log "CUDA arch=${SFEM_CUDA_ARCH:-auto}  lineinfo=${CUDA_LINEINFO}"
log "Build root: ${BUILD_ROOT}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    list_idx=0
    for j in "${!JOB_MASKS[@]}"; do
        mask="${JOB_MASKS[$j]}"
        mem="${JOB_MEMS[$j]}"
        if config_needs_skip "$mask"; then
            printf '%4d  SKIP  %-40s mem=%s\n' "$list_idx" "$(config_label "$mask")" "$mem"
        else
            printf '%4d  BUILD %-40s mem=%s\n' "$list_idx" "$(config_label "$mask")" "$mem"
        fi
        list_idx=$((list_idx + 1))
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# CMake / build helpers
# ---------------------------------------------------------------------------

run_cmd() {
    if [[ "$SFEM_USE_UENV" == 1 ]]; then
        local cmd=""
        local arg
        for arg in "$@"; do
            cmd+="$(printf '%q ' "$arg")"
        done
        srun -u --uenv="${UENV_IMAGE}" --view=default bash -lc "${cmd% }"
    else
        "$@"
    fi
}

detect_openmp_dir() {
    if [[ -n "${OPENMP_DIR:-}" ]]; then
        return 0
    fi
    if [[ "$IS_MAC" -eq 1 ]]; then
        if [[ -f /opt/homebrew/opt/libomp/include/omp.h ]]; then
            OPENMP_DIR=/opt/homebrew/opt/libomp
        elif [[ -f /usr/local/opt/libomp/include/omp.h ]]; then
            OPENMP_DIR=/usr/local/opt/libomp
        fi
    elif command -v brew >/dev/null 2>&1; then
        local prefix
        prefix="$(brew --prefix libomp 2>/dev/null || true)"
        if [[ -n "$prefix" && -f "${prefix}/include/omp.h" ]]; then
            OPENMP_DIR="$prefix"
        fi
    fi
}

detect_matrixio_dir() {
    if [[ -z "${MATRIXIO_DIR:-}" && -f "${REPO_ROOT}/../matrix.io/matrixio_array.h" ]]; then
        MATRIXIO_DIR="${REPO_ROOT}/../matrix.io"
    fi
}

detect_metis_dir() {
    if [[ -n "${METIS_DIR:-}" ]]; then
        return 0
    fi
    if command -v brew >/dev/null 2>&1; then
        local prefix
        prefix="$(brew --prefix metis 2>/dev/null || true)"
        if [[ -n "$prefix" && -f "${prefix}/include/metis.h" ]]; then
            METIS_DIR="$prefix"
        fi
    fi
}

mpi_showme_compile() {
    if ! command -v mpicc >/dev/null 2>&1; then
        return 0
    fi
    mpicc --showme:compile 2>/dev/null || true
}

mpi_showme_link() {
    if ! command -v mpicc >/dev/null 2>&1; then
        return 0
    fi
    mpicc --showme:link 2>/dev/null || true
}

mask_mpi_enabled() {
    local mask=$1
    local mpi_idx
    mpi_idx="$(index_of_flag MPI 2>/dev/null || true)"
    [[ -n "$mpi_idx" ]] && bit_is_set "$mask" "$mpi_idx"
}

cmake_args_for_mask() {
    local mask=$1
    local mem=$2
    local args=(
        -S "${REPO_ROOT}"
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        -DSFEM_ENABLE_CUDA=ON
        -DSFEM_ENABLE_TESTING=ON
        "-DSFEM_CUDA_MEMORY=${mem}"
    )
    local i name cmake_name

    if [[ "$CUDA_LINEINFO" == 1 ]]; then
        args+=(-DSFEM_ENABLE_CUDA_LINEINFO=ON)
    else
        args+=(-DSFEM_ENABLE_CUDA_LINEINFO=OFF)
    fi

    if [[ -n "$SFEM_CUDA_ARCH" ]]; then
        args+=("-DCMAKE_CUDA_ARCHITECTURES=${SFEM_CUDA_ARCH}")
    fi

    if [[ -n "${CUDACXX:-}" ]]; then
        args+=("-DCMAKE_CUDA_COMPILER=${CUDACXX}")
    elif [[ -n "${CMAKE_CUDA_COMPILER:-}" ]]; then
        args+=("-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")
    fi

    for i in "${!ACTIVE_FLAGS[@]}"; do
        name="${ACTIVE_FLAGS[$i]}"
        cmake_name="$(flag_cmake_name "$name")"
        if bit_is_set "$mask" "$i"; then
            args+=("-D${cmake_name}=ON")
        else
            args+=("-D${cmake_name}=OFF")
        fi
    done

    if mask_mpi_enabled "$mask"; then
        if have_mpi && command -v mpicc >/dev/null 2>&1; then
            args+=(
                -DCMAKE_C_COMPILER=mpicc
                -DCMAKE_CXX_COMPILER=mpicxx
            )
        fi
    else
        # MatrixIO always includes <mpi.h>; use serial compilers + MPI SDK flags.
        local mpi_cflags mpi_ldflags
        mpi_cflags="$(mpi_showme_compile)"
        mpi_ldflags="$(mpi_showme_link)"
        if [[ -n "$mpi_cflags" ]]; then
            args+=(
                "-DCMAKE_C_FLAGS=${mpi_cflags}"
                "-DCMAKE_CXX_FLAGS=${mpi_cflags}"
            )
        fi
        if [[ -n "$mpi_ldflags" ]]; then
            args+=(
                "-DCMAKE_EXE_LINKER_FLAGS=${mpi_ldflags}"
                "-DCMAKE_SHARED_LINKER_FLAGS=${mpi_ldflags}"
            )
        fi
    fi

    detect_matrixio_dir
    detect_metis_dir
    detect_openmp_dir

    if [[ -n "${MATRIXIO_DIR:-}" ]]; then
        args+=("-DMatrixIO_DIR=${MATRIXIO_DIR}")
    fi
    if [[ -n "${METIS_DIR:-}" ]]; then
        args+=("-DMETIS_DIR=${METIS_DIR}")
    fi
    if [[ -n "${OPENMP_DIR:-}" ]]; then
        args+=("-DOPENMP_DIR=${OPENMP_DIR}")
    fi

    if [[ -n "${EXTRA_CMAKE_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        args+=(${EXTRA_CMAKE_ARGS})
    fi

    printf '%s\n' "${args[@]}"
}

index_of_flag() {
    local want=$1
    local i
    for i in "${!ACTIVE_FLAGS[@]}"; do
        if [[ "${ACTIVE_FLAGS[$i]}" == "$want" ]]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

build_config() {
    local idx=$1
    local mask=$2
    local mem=$3
    local label slug build_dir log_file
    label="$(config_label "$mask")"
    slug="$(mask_slug "$mask")"
    build_dir="${BUILD_ROOT}/$(printf '%04d' "$idx")_${slug}_${mem}"
    log_file="${build_dir}/test_CUDA_configs.log"

    if config_needs_skip "$mask"; then
        log "SKIP [${idx}] ${label} mem=${mem} (missing external dependency)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    log "BUILD [${idx}] ${label} mem=${mem} -> ${build_dir}"
    mkdir -p "${build_dir}"

    local -a cmake_args=()
    while IFS= read -r line; do
        cmake_args+=("$line")
    done < <(cmake_args_for_mask "$mask" "$mem")

    {
        echo "=== $(date '+%Y-%m-%dT%H:%M:%S%z') config ${idx} mask=${mask} mem=${mem} label=${label} ==="
        printf 'cmake'
        local arg
        for arg in "${cmake_args[@]}"; do
            printf ' %q' "$arg"
        done
        printf ' -B %q\n' "${build_dir}"
    } >"${log_file}"

    if ! run_cmd cmake "${cmake_args[@]}" -B "${build_dir}" >>"${log_file}" 2>&1; then
        log "FAIL [${idx}] configure (${label} mem=${mem})"
        FAILED=$((FAILED + 1))
        return 0
    fi

    if ! run_cmd cmake --build "${build_dir}" -j "${JOBS}" >>"${log_file}" 2>&1; then
        log "FAIL [${idx}] compile (${label} mem=${mem})"
        FAILED=$((FAILED + 1))
        return 0
    fi

    if [[ "$RUN_TESTS" == 1 ]]; then
        if ! run_cmd ctest --test-dir "${build_dir}" --output-on-failure -j "${JOBS}" >>"${log_file}" 2>&1; then
            log "FAIL [${idx}] ctest (${label} mem=${mem})"
            FAILED=$((FAILED + 1))
            return 0
        fi
    fi

    log "PASS [${idx}] ${label} mem=${mem}"
    PASSED=$((PASSED + 1))
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

have_cuda || die "CUDA toolkit not found (set CUDACXX/CMAKE_CUDA_COMPILER, install nvcc, or SFEM_USE_UENV=1)"
have_matrixio || die "MatrixIO not found (clone ../matrix.io or set MATRIXIO_DIR)"
have_submodules || die "Run: git -C \"${REPO_ROOT}\" submodule update --init --recursive"
have_mpi || die "MPI SDK required (mpicc for MatrixIO headers even when SFEM_ENABLE_MPI=OFF)"

PASSED=0
FAILED=0
SKIPPED=0

for j in "${!JOB_MASKS[@]}"; do
    build_config "$j" "${JOB_MASKS[$j]}" "${JOB_MEMS[$j]}"
done

log "Done: passed=${PASSED} failed=${FAILED} skipped=${SKIPPED} total=${#JOB_MASKS[@]}"
[[ "$FAILED" -eq 0 ]]
