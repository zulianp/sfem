#!/usr/bin/env python3
"""Locate slow SpMV windows without accepting a new performance campaign."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import roofline_actions as actions
from roofline_calibration import counter_events, parse_perf, prepare_capture
from roofline_sweep_validation import window_seconds
from roofline_process_observation import snapshot, throttle_changes
import roofline_thp_study as thp

VARIANTS = ('normal', 'counted', 'normal-trace', 'counted-trace')
REPEAT = 60
WARMUP = 30
RESOURCE_FIELDS = ('user_seconds', 'system_seconds', 'minor_faults', 'major_faults',
                   'voluntary_switches', 'involuntary_switches')


def groups():
    for block in range(8):
        offset = block % len(VARIANTS)
        order = VARIANTS[offset:] + VARIANTS[:offset]
        yield [(block, index, variant) for index, variant in enumerate(order + order[::-1])]


def parse_trace(output, seconds, traced, repeat=REPEAT, resources=False):
    resource_rows = [line for line in output.splitlines() if line.startswith('ROOFLINE_RESOURCE ')]
    if resources and not traced:
        raise ValueError('resource sampling requires a timing trace')
    if resource_rows and not resources:
        raise ValueError('unexpected resource trace')
    rows = [line for line in output.splitlines() if line.startswith('ROOFLINE_TIMING ')]
    if not traced:
        if rows:
            raise ValueError('ordinary timing unexpectedly traced')
        return {}
    expected = [('warmup', i) for i in range(WARMUP)] + [('apply', i) for i in range(repeat)]
    if len(rows) != len(expected):
        raise ValueError('incomplete timing trace')
    result = dict(warmup_times=[], apply_times=[])
    for line, (phase, index) in zip(rows, expected):
        fields = dict(word.split('=') for word in line.split()[1:])
        value = float(fields['seconds'])
        if fields['phase'] != phase or int(fields['index']) != index or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid or reordered timing trace')
        result[phase+'_times'].append(value)
    if not math.isclose(sum(result['apply_times']), seconds, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError('per-apply timings do not cover the timed loop')
    if resources:
        if len(resource_rows) != len(expected):
            raise ValueError('incomplete resource trace')
        result.update(warmup_resources=[], apply_resources=[])
        for line, (phase, index) in zip(resource_rows, expected):
            fields = dict(word.split('=') for word in line.split()[1:])
            if set(fields) != set(RESOURCE_FIELDS) | {'phase', 'index'} or fields['phase'] != phase or int(fields['index']) != index:
                raise ValueError('invalid or reordered resource trace')
            row = {k: float(fields[k]) if k.endswith('_seconds') else int(fields[k]) for k in RESOURCE_FIELDS}
            if not all(math.isfinite(v) and v >= 0 for v in row.values()):
                raise ValueError('invalid resource delta: '+line)
            result[phase+'_resources'].append(row)
    return result


def analyze(records, resources=False):
    keys = ('block', 'index', 'variant')
    if [tuple(r[k] for k in keys) for r in records] != [r for group in groups() for r in group]:
        raise ValueError('incomplete or reordered SpMV timing diagnostic')
    workers = {}
    for r in records:
        if r['direction'] != records[0]['direction']:
            raise ValueError('direction changed')
        identity = workers.setdefault(r['block'], r['identity'])
        if r['identity'] != identity:
            raise ValueError('SpMV worker or arrays changed')
        if not math.isfinite(r['seconds']) or r['seconds'] <= 0:
            raise ValueError('invalid timing')
        traced = r['variant'].endswith('-trace')
        for phase, count in (('warmup', WARMUP), ('apply', REPEAT)):
            values = r.get(phase+'_times', [])
            if len(values) != (count if traced else 0) or not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError('invalid saved timing trace')
            usage = r.get(phase+'_resources', [])
            if len(usage) != (count if traced and resources else 0):
                raise ValueError('incomplete saved resource trace')
            for sample in usage:
                if set(sample) != set(RESOURCE_FIELDS) or not all(math.isfinite(v) and v >= 0 for v in sample.values()):
                    raise ValueError('invalid saved resource delta')
        if traced and not math.isclose(sum(r['apply_times']), r['seconds'], rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError('saved trace and loop duration differ')
        if r['variant'].startswith('counted'):
            fields = ('read_bytes', 'write_bytes', 'component_write_bytes', 'fp_ops', 'scf_seconds', 'core_seconds')
            if not all(math.isfinite(r[k]) and r[k] > 0 for k in fields):
                raise ValueError('invalid counted timing record')
            if r['write_bytes'] != r['component_write_bytes']:
                raise ValueError('write component identity failed')
        if resources:
            for stage in ('before', 'after'):
                observation = r['observations'][stage]
                if (observation['pid'] != r['identity']['pid'] or
                        not all(math.isfinite(observation[k]) and observation[k] >= 0 for k in ('monotonic_start', 'monotonic_end')) or
                        observation['monotonic_end'] < observation['monotonic_start']):
                    raise ValueError('invalid process observation')
            if r['observations']['after']['monotonic_start'] < r['observations']['before']['monotonic_end']:
                raise ValueError('reordered process observations')
    lines = ['SpMV timing diagnostic only; no campaign acceptance or paper speedup.',
             'Eight retained workers, 64 captures, 30 warmups and 60 applies per capture.',
             'Normal/counted and plain/traced variants rotate by worker, then reverse within each worker.',
             'Traces include timestamp/storage overhead and are checked against their own whole-loop time.',
             'Ordinary untraced captures remain separate. Every capture is retained.',
             'Counter events/scopes are unchanged; this diagnostic does not validate counter accuracy.',
             'Ranges show sample variation, not confidence intervals.', '']
    for variant in VARIANTS:
        rows = [r for r in records if r['variant'] == variant]
        values = [1000*r['seconds']/REPEAT for r in rows]
        lines.append('{} ms/apply mean={:.6f}, range=[{:.6f}, {:.6f}]'.format(
            variant, statistics.mean(values), min(values), max(values)))
    lines += ['', 'block index variant mean_ms first20_ms last20_ms max_ms warmup_first10_ms warmup_last10_ms']
    for r in records:
        if r['variant'].endswith('-trace'):
            times, warmup = r['apply_times'], r['warmup_times']
            values = [statistics.mean(times), statistics.mean(times[:20]), statistics.mean(times[-20:]),
                      max(times), statistics.mean(warmup[:10]), statistics.mean(warmup[-10:])]
            lines.append('{} {} {} {}'.format(r['block'], r['index'], r['variant'],
                ' '.join('{:.6f}'.format(1000*v) for v in values)))
        else:
            lines.append('{} {} {} {:.6f} - - - - -'.format(
                r['block'], r['index'], r['variant'], 1000*r['seconds']/REPEAT))
    lines += ['', 'Inspect complete traces and order effects before choosing another performance recipe.',
              'A slow apply or window does not by itself identify the operating-system or hardware cause.']
    if resources:
        lines += ['', 'Resource diagnostic: process CPU time sums all threads; getrusage overhead is included in traced intervals.',
                  'Plain captures have no per-apply resource calls. CPU/process observations bracket entire captures, including checks and warmup.',
                  'The 40 ms display threshold selects rows to print, not samples to retain or accept.',
                  'block capture phase apply wall_ms user_cpu_ms system_cpu_ms minor_faults major_faults voluntary_switches involuntary_switches']
        for r in records:
            for phase in ('warmup', 'apply'):
                for i, (wall, usage) in enumerate(zip(r.get(phase+'_times', []), r.get(phase+'_resources', []))):
                    if wall > .04:
                        lines.append('{} {} {} {} {:.6f} {:.6f} {:.6f} {} {} {} {}'.format(
                            r['block'], r['index'], phase, i, 1000*wall,
                            1000*usage['user_seconds'], 1000*usage['system_seconds'],
                            *(usage[k] for k in RESOURCE_FIELDS[2:])))
        lines += ['', 'CPU throttle-counter changes during entire captures (including checks and warmup). Ancestor groups can include other processes:']
        available = 0
        for r in records:
            for change in throttle_changes(r['observations']['before'], r['observations']['after']):
                available += 1
                if change['delta']:
                    lines.append('block={} capture={} path={} {} delta={}'.format(
                        r['block'], r['index'], change['path'], change['counter'], change['delta']))
        lines.append('{} comparable throttle counters; zero changes are retained in records. Missing files are recorded as unavailable.'.format(available))
        lines.append('Resource correlation alone does not identify a cause or validate performance.')
    return '\n'.join(lines)+'\n'


def collect(binary, out, resources=False, thp_study=False):
    resources = resources or thp_study
    topo, binary, env = prepare_capture(binary, out, 128**3, 'spmv-timing-diagnostic',
                                        ('cmem_wb_data', 'cmem_wr_data'))
    (out/'collection.json').write_text(json.dumps(dict(mode='spmv-timing-diagnostic',
        repeat=REPEAT, warmup=WARMUP, workers=16 if thp_study else 8,
        variants=VARIANTS, resources=resources, thp_study=thp_study), indent=2)+'\n')
    scf, core = counter_events(topo)
    scf += [topo['pmu']+'/'+s+'/' for s in ('cmem_wb_data', 'cmem_wr_data')]
    records = []
    try:
        work = thp.groups(groups()) if thp_study else ((None, group) for group in groups())
        for policy, group in work:
            name = 'spmv-timing-{}b{}-reuse'.format(policy+'-' if policy else '', group[0][0])
            factory = thp.ThpSession if thp_study else actions.ActionSession
            options = dict(policy=policy) if thp_study else {}
            worker = factory(topo, binary, env, out, name, 'spmv', 'fo', 'packed', **options)
            try:
                for block, index, variant in group:
                    stem = out/'{}b{}-i{}-{}'.format(policy+'-' if policy else '', block, index, variant)
                    kind = 'sweep' if variant.startswith('counted') else 'normal'
                    traced = variant.endswith('-trace')
                    observations = {}
                    if resources:
                        observations['before'] = snapshot(worker.identity['pid'])
                        stem.with_suffix('.before-system.json').write_text(json.dumps(observations['before'], indent=2)+'\n')
                        if thp_study:
                            thp.check_observation(observations['before'], policy)
                    options = dict(trace=traced)
                    if resources:
                        options['resources'] = traced
                    output = worker.capture(topo, binary, stem, kind, REPEAT, **options)
                    if resources:
                        observations['after'] = snapshot(worker.identity['pid'])
                        stem.with_suffix('.after-system.json').write_text(json.dumps(observations['after'], indent=2)+'\n')
                        if thp_study:
                            thp.check_observation(observations['after'], policy)
                    row = dict(block=block, index=index, variant=variant, identity=worker.identity,
                               direction=worker.direction)
                    row.update(actions.parse_output(output, 'spmv', 'fo', 'packed', kind, REPEAT))
                    row.update(parse_trace(output, row['seconds'], traced, resources=resources and traced))
                    if resources:
                        row['observations'] = observations
                    if thp_study:
                        row['policy'] = policy
                    if kind == 'sweep':
                        mt, ft = (stem.with_suffix(s).read_text() for s in ('.scf.perf', '.core.perf'))
                        m, f = parse_perf(mt, scf), parse_perf(ft, core)
                        row.update(read_bytes=32*m[scf[0]], write_bytes=m[scf[1]], fp_ops=sum(f.values()),
                            component_write_bytes=32*m[scf[2]]+m[scf[3]],
                            scf_seconds=window_seconds(mt, scf, 1), core_seconds=window_seconds(ft, core, 72))
                    records.append(row)
                    (out/'records.json').write_text(json.dumps(records, indent=2)+'\n')
                    print('captured '+stem.name, flush=True)
                worker.close()
            finally:
                worker.close(abort=True)
        report = thp.analyze(records, groups(), analyze) if thp_study else analyze(records, resources)
        (out/'summary.txt').write_text(report)
        print(report, end='')
        print('SPMV TIMING DIAGNOSTIC COLLECTED; performance campaign remains unvalidated')
    finally:
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
                  if p.is_file() and p.name != 'files.sha256.json'}
        (out/'files.sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resources', action='store_true', help='record per-apply usage and capture-wide process observations')
    parser.add_argument('--thp-study', action='store_true', help='compare default and disabled process THP policies with resource traces')
    args = parser.parse_args()
    try:
        collect(args.binary, args.out.resolve(), args.resources, args.thp_study)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('SPMV TIMING DIAGNOSTIC FAILED: '+str(error), file=sys.stderr)
        sys.exit(1)
