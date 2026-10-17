# Grace storage-width phase measurements

Job 5039923 on nid006545, GNU 14.3.0, started 2026-10-12 at 23:21:50 UTC.
The experiment uses n=128, pack 1024, SFC ordering, FP64 state and one or 72
threads. Each phase has twelve paired samples with alternating order, three
warmups and five repeats. All samples are retained. Ranges describe sample
variation, not confidence intervals.

The graph contains 673407 entries and 561177 rows. Copies of its offsets and
indices use 4938340 bytes in uint32 instead of 9876680 bytes in ptrdiff_t.
The ten cached geometry arrays use 83886080 bytes in float instead of 167772160
bytes in double. Both wide arrays remain allocated in this probe. These sizes
are neither total-memory savings nor measured transported traffic.

The graph result agrees bitwise with the production reducer before and after
timing. Cube and sheared-coordinate geometry checks pass with maximum relative
conversion error 5.926008e-8; the timed dyadic cube converts exactly. This checks
storage conversion, not the effect of FP32 geometry on a full solve.

Run `python3 summarize.py` to reproduce `summary.json` from the raw timing logs.
Times are milliseconds per isolated phase. The ratio is wide/narrow.

| Threads | Phase | Wide mean | Narrow mean | Paired ratio median | Paired ratio range | Narrow faster |
|---:|---|---:|---:|---:|---|---:|
|1|graph reduction|4.008789|3.962497|1.014821|0.934290--1.090293|6/12|
|72|graph reduction|0.156381|0.128565|1.119311|0.806483--3.146848|7/12|
|1|geometry gather|7.302858|4.731181|1.580127|1.253340--1.697837|12/12|
|72|geometry gather|0.360485|0.197626|1.854902|1.432238--2.172991|12/12|

The graph comparison depends strongly on timing order. At one thread the
second arm wins every pair, regardless of width. At 72 threads the same pattern
holds in eleven pairs; sample 9 has an unusually long wide time and is retained.
The graph averages therefore do not establish a reliable benefit from narrowing.
No production index change is justified by this experiment alone.

FP32 geometry gathering is faster in all 24 pairs, with float-to-double conversion
included. This supports a candidate for an integrated experiment, not a complete
operator speedup. The gather escapes its values but performs no physics; graph
output clearing is outside both timers. Both phases use repeated warm buffers.
No operator bottleneck or solver acceleration follows from these results.

The archive includes source snapshots, compiler/build records, public gates,
the bundle manifest and its 4290 successful checks. The retrieved solver binary
matches its in-job hash. The storage probe's hash was measured after retrieval;
it had no independent in-job binary digest. See `provenance.json`. Local hashes
in `SHA256SUMS` protect the retained archive; binaries are kept outside the paper.
