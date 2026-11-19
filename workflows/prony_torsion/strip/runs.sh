# shellcheck shell=bash
# The strip campaign's runs, shared by campaign.sbatch and collect.sbatch so the two stay in step.
#   REDUCED  the reduced history formats, each with the remainder form (<format>) and without it
#            (<format>_plain): float32, int16_scaled, float16_md, and the packed grid codes
#            qblock5/6/7 (quant_block at history_tolerance 1e-5, 1e-6, 1e-7), _sr with stochastic
#            rounding (qblock6 against qblock6_sr: the rounding alone)
#   S32      the runs repeated with float32 evaluated stresses (<run>_s32)
# Override either from the environment (REDUCED="int16_scaled qblock6_sr").
REDUCED=${REDUCED:-float32 int16_scaled float16_md qblock5_sr qblock6 qblock6_sr qblock7_sr}
S32=${S32:-full_a int16_scaled float16_md qblock6_sr qblock7_sr}
strip_runs() {  # the main campaign's runs; DIAG=1: the diagnostics build's (lost updates)
  local f runs
  if [ "${DIAG:-0}" = 1 ]; then
    runs="full_a"; for f in $REDUCED; do runs="$runs $f ${f}_plain"; done
  else
    runs="full_a full_b"; for f in $REDUCED; do runs="$runs $f ${f}_plain"; done
    for f in $S32; do runs="$runs ${f}_s32"; done
  fi
  echo "$runs"
}
