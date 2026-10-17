#!/usr/bin/env python3
"""Measure cached-state Jacobian actions and the matching lagged sparse product."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

import roofline_operator as residual
from roofline_calibration import counter_events, parse_perf, prepare_capture
from roofline_sweep_validation import spread, window_seconds


ARMS = (('jacobian', 'fo', 'atomic'), ('jacobian', 'fo', 'packed'),
        ('jacobian', 'darwish', 'atomic'), ('jacobian', 'darwish', 'packed'),
        ('lagged', 'fo', 'atomic'), ('lagged', 'fo', 'packed'), ('spmv', 'fo', 'packed'))


def groups(repeat_counts=residual.REPEAT_COUNTS):
    low, middle, high = residual.checked_repeat_counts(repeat_counts)
    for block, order in enumerate(itertools.permutations((low, middle, high))):
        kinds = [('empty-before', low)] + [('sweep', r) for r in order] + [('empty-after', low)]
        kinds.insert(0 if block % 2 == 0 else len(kinds), ('normal', middle))
        for operation, scheme, layout in (ARMS if block % 2 == 0 else ARMS[::-1]):
            yield [(block, operation, scheme, layout, kind, repeat) for kind, repeat in kinds]


def session_name(block, operation, scheme, layout):
    return '{}-{}-{}-b{}-reuse'.format(operation, scheme, layout, block)


class ActionSession(residual.OperatorSession):
    def __init__(self, topo, binary, env, out, name, operation, scheme, layout,
                 n=128, threads=72, launcher=True):
        if (operation, scheme, layout) not in ARMS:
            raise ValueError('unsupported action arm')
        self.operation = operation
        super().__init__(topo, binary, env, out, name, scheme, layout, n, threads, launcher)
        try:
            self.direction = direction_identity((out/(name+'.ready.out')).read_text(), n)
        except BaseException:
            self.close(abort=True)
            raise

    def worker_command(self, binary, elements, threads):
        command = super().worker_command(binary, elements, threads)
        command += ['--bsr-apply'] if self.operation == 'spmv' else ['--jac-action']
        if self.operation == 'lagged':
            command += ['--lagged-rc']
        return command


def direction_identity(output, n):
    rows = [line for line in output.splitlines() if line.startswith('ROOFLINE_DIRECTION ')]
    if len(rows) != 1:
        raise ValueError('missing coordinate direction identity')
    match = re.fullmatch(r'ROOFLINE_DIRECTION source=coordinates nodes=(\d+) xor=([0-9a-f]{16}) sum=([0-9a-f]{16})', rows[0])
    if match is None or int(match.group(1)) != (n+1)**3:
        raise ValueError('invalid coordinate direction identity')
    return dict(nodes=int(match.group(1)), xor=match.group(2), sum=match.group(3))


def parse_output(output, operation, scheme, layout, kind, repeat, n=128, threads=72):
    expected = 'ROOFLINE_ACTION operation={} direction=coordinate_fixed reconstruction={} PASS'.format(
        operation, 'per_apply' if operation == 'jacobian' else 'none')
    actions = [line for line in output.splitlines() if line.startswith('ROOFLINE_ACTION ')]
    if (operation, scheme, layout) not in ARMS or actions != [expected]:
        raise ValueError('action or reconstruction scope mismatch')
    return residual.parse_output(output, scheme, layout, kind, repeat, n, threads, warmup=30)


def analyze(records, repeat_counts=residual.REPEAT_COUNTS):
    repeat_counts = residual.checked_repeat_counts(repeat_counts)
    keys = ('block', 'operation', 'scheme', 'layout', 'kind', 'repeat')
    if [tuple(r[k] for k in keys) for r in records] != [case for group in groups(repeat_counts) for case in group]:
        raise ValueError('incomplete or reordered action campaign')
    sessions = {}
    for r in records:
        if r['direction'] != records[0]['direction']:
            raise ValueError('physical direction differs across action arms')
        name = session_name(r['block'], r['operation'], r['scheme'], r['layout'])
        group = sessions.setdefault(name, [])
        if (r['session'] != name or r['session_index'] != len(group) or
                (group and r['identity'] != group[0]['identity'])):
            raise ValueError('action worker or arrays changed within a block')
        group.append(r)
        required = ('seconds',) if r['kind'] == 'normal' else (
            'seconds', 'read_bytes', 'write_bytes', 'component_write_bytes', 'fp_ops', 'scf_seconds', 'core_seconds')
        if not all(math.isfinite(r[k]) and r[k] >= 0 for k in required) or r['seconds'] <= 0:
            raise ValueError('invalid action record')
        if r['kind'] == 'sweep' and r['read_bytes'] + r['write_bytes'] <= 0:
            raise ValueError('zero action traffic')
    lines = ['FP64 affine interior actions: n=128, pack=1024, RC on, perturb=0.017.',
             'State gradients cached; one coordinate-keyed direction shared by every layout. Exact actions reconstruct its gradients every apply.',
             'Lagged/SpMV omit reconstruction sensitivities; only their mutual comparison is like-for-like.',
             'No boundary/transient terms or basis-vector churn. Six blocks; 30 warmups per capture.',
             'PMU operations are speculative, all precisions. Empty controls are retained without subtraction.',
             'Ranges show sample variation, not confidence intervals.', '']
    if repeat_counts == residual.LONG_REPEAT_COUNTS:
        lines.append('Counter-window repeats: 30, 60, 90; normal timing paired with 60. Acceptance limits unchanged.')
    checks, normal_means = [], {}
    for arm in ARMS:
        group = [r for r in records if (r['operation'], r['scheme'], r['layout']) == arm]
        sweep = [r for r in group if r['kind'] == 'sweep']
        empty = [r for r in group if r['kind'].startswith('empty')]
        normal = [r for r in group if r['kind'] == 'normal']
        identity = all(r['write_bytes'] == r['component_write_bytes'] for r in sweep + empty)
        windows = all(.99 <= r[k] / r['seconds'] <= 1.10 for r in sweep for k in ('scf_seconds', 'core_seconds'))
        shortest = [r for r in sweep if r['repeat'] == repeat_counts[0]]
        empty_ok = all(r['read_bytes'] + r['write_bytes'] <= .01 * min(s['read_bytes'] + s['write_bytes'] for s in shortest)
                       and r['fp_ops'] <= .01 * min(s['fp_ops'] for s in shortest) for r in empty)
        direct = [{k: r[k] / r['repeat'] for k in ('read_bytes', 'write_bytes', 'fp_ops')} for r in sweep]
        slopes = []
        for block in range(6):
            by_repeat = {r['repeat']: r for r in sweep if r['block'] == block}
            for lo, hi in zip(repeat_counts, repeat_counts[1:]):
                slopes.append({k: (by_repeat[hi][k] - by_repeat[lo][k]) / (hi-lo) for k in direct[0]})
        stable = all(spread([s[k] for s in samples]) <= .10 for samples in (direct, slopes) for k in direct[0])
        stable = stable and all(spread([s['read_bytes'] + s['write_bytes'] for s in samples]) <= .10
                                for samples in (direct, slopes))
        overhead = [next(s['seconds'] for s in sweep if s['block'] == r['block'] and s['repeat'] == repeat_counts[1])
                    / r['seconds'] - 1 for r in normal]
        overhead_ok = abs(statistics.mean(overhead)) <= .10
        checks.extend((identity, windows, empty_ok, stable, overhead_ok))
        normal_ms = [1000 * r['seconds'] / r['repeat'] for r in normal]
        normal_means[arm] = statistics.mean(normal_ms)
        lines.append('{} {} {}: identity={}, windows={}, empty={}, count_stability={}, mean_timing_overhead={}'.format(
            *arm, identity, windows, empty_ok, stable, overhead_ok))
        if repeat_counts == residual.LONG_REPEAT_COUNTS:
            lines += residual.stability_details(direct, slopes)
        for name, values in (
                ('normal ms/apply', normal_ms), ('paired overhead percent', [100*x for x in overhead]),
                ('counted ms/apply', [1000*r['seconds']/r['repeat'] for r in sweep]),
                ('read bytes/apply', [r['read_bytes'] for r in direct]),
                ('write bytes/apply', [r['write_bytes'] for r in direct]),
                ('speculative FP ops/apply', [r['fp_ops'] for r in direct]),
                ('speculative FP ops/transported byte', [r['fp_ops']/(r['read_bytes']+r['write_bytes']) for r in sweep]),
                ('transported GB/s', [(r['read_bytes']+r['write_bytes'])/r['seconds']/1e9 for r in sweep]),
                ('speculative GFLOP/s', [r['fp_ops']/r['seconds']/1e9 for r in sweep]),
                ('SCF/loop', [r['scf_seconds']/r['seconds'] for r in sweep]),
                ('FP/loop', [r['core_seconds']/r['seconds'] for r in sweep])):
            lines.append('  {} mean={:.6f}, range=[{:.6f}, {:.6f}]'.format(
                name, statistics.mean(values), min(values), max(values)))
    for operation, scheme in (('jacobian', 'fo'), ('jacobian', 'darwish'), ('lagged', 'fo')):
        lines.append('{} {} normal atomic/packed time ratio={:.6f}'.format(operation, scheme,
            normal_means[(operation, scheme, 'atomic')]/normal_means[(operation, scheme, 'packed')]))
    lines.append('Equivalent lagged operator: normal SpMV/packed time ratio={:.6f}'.format(
        normal_means[('spmv', 'fo', 'packed')]/normal_means[('lagged', 'fo', 'packed')]))
    lines += ['', 'All captures retained. A pass supports these fixed-state applications, not a bottleneck or solver speedup.']
    return '\n'.join(lines)+'\n', all(checks)


def collect(binary, out, calibration, repeat_counts=residual.REPEAT_COUNTS):
    repeat_counts = residual.checked_repeat_counts(repeat_counts)
    control_topo, control_report = residual.calibration_reference(calibration)
    topo, binary, env = prepare_capture(binary, out, 128**3, 'action-reuse', ('cmem_wb_data', 'cmem_wr_data'))
    if topo != control_topo:
        raise ValueError('actions and references must share one allocation/topology')
    (out/'control-reference.txt').write_text(control_report)
    (out/'collection.json').write_text(json.dumps(dict(mode='action-reuse',
        repeat_counts=repeat_counts, warmup=30, normal_repeat=repeat_counts[1]), indent=2)+'\n')
    scf, core = counter_events(topo)
    scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data', 'cmem_wr_data')]
    records = []
    try:
        for group in groups(repeat_counts):
            block, operation, scheme, layout, unused_kind, unused_repeat = group[0]
            session = None
            try:
                session = ActionSession(topo, binary, env, out, session_name(block, operation, scheme, layout),
                                        operation, scheme, layout)
                for block, operation, scheme, layout, kind, repeat in group:
                    stem = out/'{}-{}-{}-{}-b{}-r{}'.format(operation, scheme, layout, kind, block, repeat)
                    index = session.index
                    output = session.capture(topo, binary, stem, kind, repeat)
                    row = dict(block=block, operation=operation, scheme=scheme, layout=layout, kind=kind,
                               repeat=repeat, session=session.name, session_index=index, identity=session.identity,
                               direction=session.direction)
                    row.update(parse_output(output, operation, scheme, layout, kind, repeat))
                    if kind != 'normal':
                        mt, ft = (stem.with_suffix(s).read_text() for s in ('.scf.perf', '.core.perf'))
                        m, f = parse_perf(mt, scf), parse_perf(ft, core)
                        row.update(read_bytes=32*m[scf[0]], write_bytes=m[scf[1]], fp_ops=sum(f.values()),
                                   component_write_bytes=32*m[scf[2]]+m[scf[3]],
                                   scf_seconds=window_seconds(mt, scf, 1), core_seconds=window_seconds(ft, core, 72))
                    records.append(row)
                    (out/'records.json').write_text(json.dumps(records, indent=2)+'\n')
                    print('captured '+stem.name, flush=True)
                session.close()
                session = None
            finally:
                if session is not None:
                    session.close(abort=True)
        report, ok = analyze(records, repeat_counts)
        (out/'summary.txt').write_text(control_report+'\n'+report)
        print(control_report+'\n'+report, end='')
        if not ok:
            raise ValueError('action campaign checks failed; retain all captures')
        print('ROOFLINE ACTION CAMPAIGN PASSED')
    finally:
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
                  if p.is_file() and p.name != 'files.sha256.json'}
        (out/'files.sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--long-windows', action='store_true', help='use fixed 30/60/90-apply windows')
    args = parser.parse_args()
    try:
        collect(args.binary, args.out.resolve(), args.calibration.resolve(),
                residual.LONG_REPEAT_COUNTS if args.long_windows else residual.REPEAT_COUNTS)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('ROOFLINE ACTION CAMPAIGN FAILED: '+str(error), file=sys.stderr)
        sys.exit(1)
