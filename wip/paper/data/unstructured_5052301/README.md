# Grace unstructured operator result, job 5052301

The user ran the prepared public study on `nid006545` on 2026-10-14 with GCC
14.3.0. The CPU affinity was 0–71 on a Grace Neoverse-V2 socket. The complete
campaign passed with its original checks and all samples retained. This is a
first-order, steady interior Rhie–Chow residual/action experiment on two public
procedural, graded, non-affine HEX8 meshes. It is not a solver or spatial-error
study. Worker THP was disabled in both layouts.

The returned 4,352 source hashes exactly match the prepared source manifest.
All 12 case logs replay to the saved records and summary: four small-grid
numerical cases and eight timing cases, 384 retained timing samples and 144
required finite-difference comparisons. Maximum FD error is 5.96445273132e-8;
maximum layout discrepancy is 1.52424242424e-15. The minimum omitted-direction-
gradient error is 0.214875086531. The physical linear-gradient check's maximum
absolute error is 2.6645352591e-15. Both regression tests also pass at 1/72 threads.
The distorted-cell control reports zero after the edge-geometry correction and
0.0594551496738 with the old affine-edge shortcut.

## Runtime result

Ratios are mean atomic time divided by mean packed time. Values below one mean
packed is slower. The [full generated table](raw/study/summary.md) retains all
one-thread results and sample ranges; ranges are not confidence intervals.

| Mesh | Elements | Pack | Residual ratio at 72 threads | Jacobian ratio at 72 threads |
|---|---:|---:|---:|---:|
| medium | 74,664 | 1024 | 1.028 | 0.924 |
| medium | 74,664 | 256 | 1.537 | 1.392 |
| large | 265,636 | 1024 | 1.590 | 1.446 |
| large | 265,636 | 256 | 1.715 | 1.510 |

This extends the observed packed-layout benefit beyond the affine structured
cubes to these irregular-connectivity, non-affine fixtures. It also retains a
limit: with pack1024 on the medium mesh, the residual has only a 1.028x mean
benefit and the Jacobian takes about 8.3% longer than atomic. At pack256 the same
mesh benefits in both operators. Both tested pack sizes benefit on the larger
mesh. No claim of a universally fastest pack size follows.

The medium mesh has 73 packs at size1024. The current static split gives one of
72 threads two full packs; at size256 there are 292 packs and four or five per
thread. This is consistent with coarse work partitioning contributing to the
loss. Pack-local storage, cache behavior and other costs also change, so the
experiment does not isolate scheduling as the sole cause. No PMU or scatter
ablation was performed on these meshes.

## Setup and explicit storage ledger

Packing is measured once per case and excludes the preceding SFC reorder; both
reorder times are retained in the records. Payload below sums coordinate and
connectivity arrays, field/vector capacities, packed topology/map/reduction
arrays, ghost buffers and packed thread arenas as applicable. Packed retains its
original connectivity. Decimal MB are used. State backups, test directions,
shared reference outputs, transient validation buffers, allocator metadata,
stacks and other library/runtime storage are excluded. This is not peak RSS or a
complete solver memory footprint.

| Mesh | Threads | Pack | Packs | Packing ms | Atomic accounted MB | Packed accounted MB |
|---|---:|---:|---:|---:|---:|---:|
| medium | 1 | 256 | 292 | 16.134 | 16.735 | 22.943 |
| medium | 1 | 1024 | 73 | 12.956 | 16.735 | 21.215 |
| medium | 72 | 1024 | 73 | 13.003 | 16.735 | 46.122 |
| medium | 72 | 256 | 292 | 16.100 | 16.735 | 31.037 |
| large | 1 | 256 | 1038 | 60.877 | 58.163 | 81.256 |
| large | 1 | 1024 | 260 | 48.118 | 58.163 | 74.257 |
| large | 72 | 1024 | 260 | 48.143 | 58.163 | 99.689 |
| large | 72 | 256 | 1038 | 60.689 | 58.163 | 89.379 |

The ledger shows an explicit memory cost for the measured packed configuration;
it must not be described as a total-memory saving. At 72 threads and pack256,
accounted payload is about 1.85x atomic on the medium mesh and 1.54x on the large.
These ratios include the per-thread arena and depend on grid and pack size.

## Files and replay

`raw/` is the byte-preserved returned result directory, including the source
manifest, source patch, compiler/build information, binary hashes, commands,
mesh identities and every capture. `RAW-SHA256.txt` hashes these retained files.
`verification.json` records the independently replayed maxima and coverage.
The original mesh arrays remain in the prepared public bundle under
`public-unstructured/`; their per-file hashes and generator settings are recorded
in every case identity. The mesh generator and measurement protocol are documented
in `spikes/cvfem/docs/unstructured_study.md`.

From the sfem repository root, this replays the saved numerical/timing screens
and checks the generated summary without running a benchmark:

```sh
python3 - <<'PYCODE'
from pathlib import Path
import hashlib, json, sys
sys.path.insert(0, 'spikes/cvfem/scripts')
from unstructured_study import check, summary
p = Path('wip/paper/data/unstructured_5052301')
for line in (p/'RAW-SHA256.txt').read_text().splitlines():
    digest, name = line.split('  ', 1)
    assert hashlib.sha256((p/name).read_bytes()).hexdigest() == digest, name
data = json.loads((p/'raw/study/records.json').read_text())
expected = {(m,t,k,False) for m in ('medium','large') for t in (1,72) for k in (256,1024)}
expected |= {('small',t,k,True) for t in (1,72) for k in (64,256)}
assert len(data) == 12
assert {(r['mesh'],r['threads'],r['pack'],r['verify']) for r in data} == expected
for r in data:
    name = '{}-t{}-p{}-{}'.format(r['mesh'],r['threads'],r['pack'], 'check' if r['verify'] else 'timing')
    log = (p/'raw/study'/name/'run.log').read_text()
    assert check(log,r['identity'],r['threads'],r['pack'],12,12,r['verify']) == r['result']
assert summary(data) == (p/'raw/study/summary.md').read_text()
print('Saved unstructured campaign replay passed')
PYCODE
```

The source-hash comparison and actual mesh-file checks additionally used the
prepared local bundle during result review. Replaying this archive alone checks
the recorded evidence, not the executable or physical mesh arrays against a new
installation. The manuscript has not yet been changed by this result review.
