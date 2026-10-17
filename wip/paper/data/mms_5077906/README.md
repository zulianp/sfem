# Grace MMS follow-up at n=40

Job 5077906 ran on Daint node `nid006550` on 2026-10-17. It uses the identical
solver binary as job 5077627, with GCC 14.3, FP64 and eight configured threads.
Only the grid list and output directory differ in the recorded launch settings.
The source-manifest differences are limited to reporting, documentation and the
launch wrapper. Physical inputs, numerical sources and tolerances are unchanged.

All five final nonlinear targets are met. Unlimited reconstruction, two-node
clip, Venkatakrishnan with K=0 and Darwish all meet the fixed maximum velocity
error target 0.01 at n=40. First-order upwind remains above it; its exit code 1
and error are retained. The four passes resolve the coarse-grid target failure
for these reconstructed schemes on this affine manufactured solution without a
numerical fix or threshold change. They do not establish arbitrary-mesh accuracy.

The observed velocity-L2 rates from n=32 to n=40 are about 2.04 (unlimited),
2.07 (clip), 1.49 (Venkatakrishnan) and 1.36 (Darwish). First-order upwind gives
about 0.73. The limited schemes do not share a demonstrated second-order rate,
and this study does not isolate why their rates differ.

`raw/` preserves the 19 retrieved run files byte for byte. `RAW-SHA256.txt`
also covers the exact report-reader source. `verify.py` checks the earlier
`../mms_5077627` archive, this archive, source/binary/settings compatibility,
all five new verdicts and the combined twenty-case refinement table. Run
`python3 verify.py` here; no solver is executed. `verification.json` preserves
the computed combined data. Original summaries and failed coarse-grid exits
remain unchanged. The two allocations are not a timing comparison.

The scope remains steady, affine Farrell MMS on `[0,2]^3`, manufactured velocity
on all faces, zero-mean pressure, density 1 and dynamic viscosity 0.01. FGMRES
uses block Jacobi, restart 480, linear relative tolerance `1e-6` and final
nonlinear relative tolerance `1e-10`, with unfrozen convective corrections and
exact Rhie--Chow derivatives. This closes the bounded refinement assessment;
unstructured full-solver accuracy and general limiter-order guarantees remain
outside its evidence.
