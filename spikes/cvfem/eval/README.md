# eval — the measurements the cleanup decided on

`subpar/README.md` says which variants were removed. These are the rows that say why.

| file | what it is |
|---|---|
| `cpu_saturated.csv` | one pass over the full cross-product at n=96, the CPU plateau |
| `cpu_repeats.csv` | the assembly cross-product again, three interleaved trials |
| `decide.py` | applies the survival rule to `cpu_repeats.csv` |

Run `python3 eval/decide.py eval/cpu_repeats.csv` to reproduce the decision table.

**Why two files.** The single pass produced results that could not be deleted on: one
cell 5x below its neighbours (`store` x `sympy` at 23.5 against ~71), and one that
reversed the ranking established at n=64 (`colored` x `sumfact` at 79.0 against 49.7).
Repeating with medians settled both — the 23.5 was noise, the 79.0 was real, and the
`colored` layout survived a plan that had scheduled it for removal.

**The rule.** A micro-kernel survives if it is fastest in at least one (layout, geometry)
cell by more than run-to-run noise, taken as ~4% on this machine. Below that margin it is
a tie and the simpler kernel wins. Reference implementations — `fd` (the correctness
oracle), `current` (the readable formulation the fused kernels are checked against), and
the CPU `split` (the host reference for the GPU split) — are exempt and stay regardless
of speed, because deleting them would delete the ability to verify the survivors.

**Size matters and is not monotone.** `store` is the best CPU assembly layout at n=64
(76.7 against `colored`'s 49.7) and second at n=96 (72.8 against 79.0). Neither dominates
everywhere, so both stayed. Numbers taken at one size are not a ranking.
