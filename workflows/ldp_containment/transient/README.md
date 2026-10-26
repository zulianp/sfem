# LDP valves through pump strokes

Two playground runs per case, from the case folders the containment uses (`env.sh`, `db/`):

1. `sim1.yaml`: the valves released out of their initial overlap into the cavity. EPDM (Prony
   Mooney-Rivlin) is relaxed throughout, because the steps (2.5e4 s) are far longer than its
   relaxation times. The valves end at rest under the outlet pressure.
2. `sim2.yaml`: four strokes from that state, dynamic (BDF2, variable steps). The material
   starts relaxed there. A vertical traction acts on each valve's chamber-side face: the
   pressure difference across the valve.

    run_transient.sh <case folder> [--sim1-only|--sim2-only] [--set key=value ...]

The work goes to `<case>/transient/`:
- `sim1/` and `sim2/` hold each run's fields, history and log;
- `sim1/whole/` is simulation 1's state on the whole mesh, which simulation 2 starts from;
- `sim1/released.e` (simulation 1's end) and `sim2/strokes.e` (simulation 2, every exported step with
  its time) are for ParaView, on the whole pump with zero on the diaphragm.

`fp70_membrane_pressure.csv` is the chamber pressure, from `pressure_table.py` applied to the
2023 FP70b FSI input `FP70_MembranePressureInputFile.h5`. It covers 0.18 s, which is four
strokes of 45 ms. The outlet pressure is 90,543 Pa and the suction side 0 Pa (FP70 FSI cases).
There is no FF200 data, so FF200 is driven by this table too.
