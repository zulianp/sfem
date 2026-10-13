#!/usr/bin/env python3
"""Collect a checked dense DGEMM reference, not a universal compute ceiling."""
import argparse
import ctypes
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys

from roofline_calibration import counter_events, parse_perf, prepare_capture
from roofline_operator import calibration_reference
from roofline_reuse_diagnostic import Session
from roofline_sweep_validation import control_timing, window_seconds, spread


def probe_library(path):
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    config = library.openblas_get_config
    config.argtypes, config.restype = [], ctypes.c_char_p
    threads = library.openblas_get_num_threads
    threads.argtypes, threads.restype = [], ctypes.c_int
    text, count = config(), threads()
    if not text or not text.startswith(b'OpenBLAS') or b'USE64BITINT' in text or count < 1:
        raise ValueError('unidentified or non-LP64 OpenBLAS configuration')
    return dict(config=text.decode('ascii'), threads=count)


def discover_library(root=Path('/user-environment'), override=None, probe=probe_library):
    if override:
        paths = [Path(override).resolve()]
    else:
        paths = sorted({p.resolve() for p in root.rglob('libopenblas*.so*') if p.is_file()})
    valid, reasons = [], []
    for path in paths:
        try:
            if not path.is_file():
                raise ValueError('not an existing library file')
            probe(path)
            valid.append(path)
        except (ValueError, OSError, AttributeError, UnicodeError) as error:
            reasons.append('{}: {}'.format(path, error))
    if len(valid) != 1:
        listed = ['{}: compatible LP64 OpenBLAS'.format(path) for path in valid]+reasons
        raise ValueError('Found {} usable OpenBLAS libraries. Candidates:\n{}\nSet BLAS_LIBRARY to one explicit LP64 shared-library path.'.format(
            len(valid), '\n'.join(listed) if listed else '(none under {})'.format(root)))
    return valid[0]


def groups():
    for block, order in enumerate(itertools.permutations((1, 2, 3))):
        captures = [('empty-before', 1)] + [('sweep', r) for r in order] + [('empty-after', 1)]
        captures.insert(0 if block % 2 == 0 else len(captures), ('normal', 2))
        yield [(block, kind, repeat) for kind, repeat in captures]


def cases():
    return [case for group in groups() for case in group]


def phase(kind):
    return 'normal' if kind == 'normal' else ('sweep' if kind == 'sweep' else 'sweep-empty')


def library_identity(output, threads, library=None, production=False):
    rows = [line for line in output.splitlines() if line.startswith('GEMM_IDENTITY ')]
    paths = [line[len('GEMM_LIBRARY '):] for line in output.splitlines() if line.startswith('GEMM_LIBRARY ')]
    if len(rows) != 1 or len(paths) != 1 or not paths[0]:
        raise ValueError('missing DGEMM runtime library identity')
    identity = dict(word.split('=') for word in rows[0].split()[1:])
    if identity.get('requested') != str(threads):
        raise ValueError('wrong DGEMM requested thread count')
    if identity.get('backend') == 'openblas':
        configs = [line[len('GEMM_CONFIG '):] for line in output.splitlines() if line.startswith('GEMM_CONFIG ')]
        if (identity.get('actual') != str(threads) or identity.get('thread_scope') != 'runtime' or
                len(configs) != 1 or not configs[0] or 'USE64BITINT' in configs[0]):
            raise ValueError('unverified or incompatible OpenBLAS runtime')
        identity['config'] = configs[0]
    elif identity.get('backend') == 'accelerate' and not production:
        if identity.get('actual') != 'unavailable' or identity.get('thread_scope') != 'requested_limit':
            raise ValueError('invalid Accelerate local thread scope')
    else:
        raise ValueError('Grace campaign requires runtime-identified OpenBLAS')
    identity['library'] = paths[0]
    if library is not None and Path(paths[0]).resolve() != library.resolve():
        raise ValueError('loaded DGEMM library {} differs from the explicit library {}'.format(
            Path(paths[0]).resolve(), library.resolve()))
    return identity


class GemmSession(Session):
    def __init__(self, topo, binary, env, out, name, n=8192, threads=72, launcher=True,
                 library=None, production=False):
        super().__init__(topo, binary, env, out, name, n, threads, launcher)
        try:
            self.library_identity = library_identity((out/(name+'.ready.out')).read_text(),
                                                     threads, library, production)
            metadata = out/(name+'.session.json')
            values = json.loads(metadata.read_text())
            values.update(warmup=3, lifetime='reuse', operation='dgemm', n=n,
                          library=self.library_identity)
            metadata.write_text(json.dumps(values, indent=2)+'\n')
        except BaseException:
            self.close(abort=True)
            raise

    def capture(self, topo, binary, stem, kind, repeat, monitor_prefix=None, timeout=180):
        command = None
        if kind != 'normal':
            scf, core = counter_events(topo)
            scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data', 'cmem_wr_data')]
            command = ['cat']
            for cpus, events, suffix, pair in (
                    (','.join(map(str, topo['cpus'])), core, '.core.perf', self.fds[4:]),
                    (str(topo['scf_cpu']), scf, '.scf.perf', self.fds[:4])):
                command = ['perf', 'stat', '--no-big-num', '--no-scale', '-x,', '-a', '-C', cpus,
                           '--delay=-1', '--control=fd:{},{}'.format(pair[0], pair[3]),
                           '-e', '{'+','.join(events)+'}', '-o', str(stem.with_suffix(suffix)), '--']+command
            if monitor_prefix is not None:
                command = list(monitor_prefix)+[str(self.fds[i]) for i in (0, 3, 4, 7)]
        return self.exchange(stem, '{} {}\n'.format(phase(kind), repeat), command, timeout)


def parse_output(output, kind, repeat, n=8192, threads=72):
    rows = [line for line in output.splitlines() if line.startswith('ROOFLINE_GEMM ')]
    if len(rows) != 1 or not rows[0].endswith(' PASS'):
        raise ValueError('missing checked DGEMM result')
    values = dict(word.split('=') for word in rows[0].split()[1:-1])
    expected = dict(window=phase(kind), n=str(n), repeats=str(repeat), warmup='3',
                    threads=str(threads), precision='fp64', layout='column_major', transa='N',
                    transb='N', alpha='1', beta='0', useful_flops=str(2*n**3))
    if any(values.get(k) != v for k, v in expected.items()):
        raise ValueError('DGEMM scope/work mismatch')
    seconds = float(values['seconds'])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('invalid DGEMM elapsed time')
    checks = [line for line in output.splitlines() if line.startswith('GEMM_CHECK ')]
    seen = set()
    if len(checks) != 2:
        raise ValueError('missing DGEMM before/after checks')
    for line in checks:
        if not line.endswith(' PASS'):
            raise ValueError('failed DGEMM correctness check')
        fields = dict(word.split('=') for word in line.split()[1:-1])
        error, samples = float(fields['relative']), int(fields['samples'])
        expected_samples = len({0, n*n-1, n-1, n*n-n, n*n//2} |
                               {((k*997+7)%n)+((k*619+3)%n)*n for k in range(16)})
        if (fields['stage'] not in ('before', 'after') or fields['stage'] in seen or
                fields['finite'] != str(n*n) or fields['tolerance'] != '1e-12' or
                samples != expected_samples or not math.isfinite(error) or not 0 <= error <= 1e-12):
            raise ValueError('invalid DGEMM numerical check')
        seen.add(fields['stage'])
    result = dict(seconds=seconds, useful_flops=2*n**3)
    if kind == 'normal':
        if 'PERF_PHASE' in output or 'PERF_CONTROL' in output:
            raise ValueError('normal DGEMM unexpectedly counted')
    else:
        selected = phase(kind)
        if output.splitlines().count('PERF_PHASE phase='+selected+' controllers=2 transitions=4 PASS') != 1:
            raise ValueError('DGEMM requires both acknowledged controllers')
        result['control_timing'] = control_timing(output, selected)
        if kind == 'sweep' and result['control_timing']['inner_seconds']+1e-9 < seconds:
            raise ValueError('DGEMM counters do not bracket the timed calls')
    return result


def analyze(records, n=8192):
    if [(r['block'], r['kind'], r['repeat']) for r in records] != cases():
        raise ValueError('incomplete, duplicate or reordered DGEMM campaign')
    for block in range(6):
        selected = [r for r in records if r['block'] == block]
        for index, row in enumerate(selected):
            if (row['session'] != 'gemm-b{}-reuse'.format(block) or row['session_index'] != index or
                    row['identity'] != selected[0]['identity'] or row['library'] != selected[0]['library']):
                raise ValueError('DGEMM worker, arrays or library changed')
    if any(r['library'] != records[0]['library'] for r in records):
        raise ValueError('DGEMM library changed across blocks')
    for row in records:
        required = ('seconds',) if row['kind'] == 'normal' else (
            'seconds', 'read_bytes', 'write_bytes', 'fp_ops', 'component_write_bytes', 'scf_seconds', 'core_seconds')
        if (row['useful_flops'] != 2*n**3 or row['seconds'] <= 0 or
                not all(math.isfinite(row[k]) and row[k] >= 0 for k in required)):
            raise ValueError('invalid DGEMM record')
    counted = [r for r in records if r['kind'] == 'sweep']
    ordinary = [r for r in records if r['kind'] == 'normal']
    empty = [r for r in records if r['kind'].startswith('empty')]
    shortest = [r for r in counted if r['repeat'] == 1]
    identity = all(r['component_write_bytes'] == r['write_bytes'] for r in counted+empty)
    windows = all(.99 <= r[key]/r['seconds'] <= 1.10 for r in counted for key in ('scf_seconds', 'core_seconds'))
    empty_ok = all(r['read_bytes']+r['write_bytes'] <= .01*min(s['read_bytes']+s['write_bytes'] for s in shortest)
                   and r['fp_ops'] <= .01*min(s['fp_ops'] for s in shortest) for r in empty)
    direct = [{k:r[k]/r['repeat'] for k in ('read_bytes', 'write_bytes', 'fp_ops')} for r in counted]
    slopes = []
    for block in range(6):
        by = {r['repeat']:r for r in counted if r['block'] == block}
        for lo, hi in ((1, 2), (2, 3)):
            slopes.append({k:(by[hi][k]-by[lo][k])/(hi-lo) for k in direct[0]})
    stable = all(spread([s[k] for s in samples]) <= .10 for samples in (direct, slopes) for k in direct[0])
    stable = stable and all(spread([s['read_bytes']+s['write_bytes'] for s in samples]) <= .10 for samples in (direct, slopes))
    overhead = [next(c['seconds'] for c in counted if c['block'] == r['block'] and c['repeat'] == 2)/r['seconds']-1 for r in ordinary]
    overhead_ok = abs(statistics.mean(overhead)) <= .10
    lines = ['Checked dense FP64 DGEMM reference: N={}, NN, column major, alpha=1, beta=0.'.format(n),
             'Six retained workers; all permutations of 1/2/3 calls, normal timing at 2 calls.',
             'Three warmups before every capture; independent sampled long-double dots and finite scan before/after.',
             'Initialization, validation and warmups lie outside acknowledged counter windows.',
             'Useful work is 2*N^3 per call; PMU operations are speculative and all-precision, reported separately.',
             'No sample deletion or background subtraction; ranges show variation, not confidence intervals.',
             'identity={}, windows={}, empty={}, count_stability={}, mean_timing_overhead={}'.format(identity, windows, empty_ok, stable, overhead_ok)]
    metrics = [('normal seconds/call', [r['seconds']/r['repeat'] for r in ordinary]),
               ('normal useful GFLOP/s', [r['repeat']*r['useful_flops']/r['seconds']/1e9 for r in ordinary]),
               ('counted seconds/call', [r['seconds']/r['repeat'] for r in counted]),
               ('counted useful GFLOP/s', [r['repeat']*r['useful_flops']/r['seconds']/1e9 for r in counted]),
               ('speculative PMU GFLOP/s', [r['fp_ops']/r['seconds']/1e9 for r in counted]),
               ('speculative PMU ops/useful work', [r['fp_ops']/(r['repeat']*r['useful_flops']) for r in counted]),
               ('transported bytes/call', [(r['read_bytes']+r['write_bytes'])/r['repeat'] for r in counted]),
               ('transported GB/s', [(r['read_bytes']+r['write_bytes'])/r['seconds']/1e9 for r in counted]),
               ('paired timing overhead percent', [100*x for x in overhead])]
    for label, values in metrics:
        lines.append('{} mean={:.6f}, range=[{:.6f}, {:.6f}]'.format(label, statistics.mean(values), min(values), max(values)))
    lines.append('This achieved DGEMM reference is not a strict hardware ceiling or a solver result.')
    return '\n'.join(lines)+'\n', all((identity, windows, empty_ok, stable, overhead_ok))


def collect(binary, out, calibration, library):
    if not library.is_file():
        raise ValueError('explicit BLAS library is missing')
    control_topo, control_report = calibration_reference(calibration)
    topo, binary, env = prepare_capture(binary, out, 8192, 'gemm-reuse', ('cmem_wb_data', 'cmem_wr_data'))
    records = []
    try:
        if topo != control_topo:
            raise ValueError('DGEMM and controls must share the same allocation/topology')
        env.update(OPENBLAS_NUM_THREADS='72', GOTO_NUM_THREADS='72')
        environment = json.loads((out/'environment.json').read_text())
        environment.update(OPENBLAS_NUM_THREADS='72', GOTO_NUM_THREADS='72', GEMM_WARMUP=3)
        (out/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
        (out/'control-reference.txt').write_text(control_report)
        (out/'blas-library.json').write_text(json.dumps(dict(path=str(library.resolve()),
            sha256=hashlib.sha256(library.read_bytes()).hexdigest()), indent=2)+'\n')
        sources = [Path(__file__), Path(__file__).with_name('roofline_reuse_diagnostic.py'),
                   Path(__file__).with_name('roofline_memory_policy.py'), Path(__file__).with_name('roofline_thp_exec.py'),
                   Path(__file__).with_name('roofline_calibration.py'), Path(__file__).with_name('roofline_sweep_validation.py'),
                   Path(__file__).with_name('roofline_operator.py'),
                   Path(__file__).parents[1]/'src/drivers/cvfem_roofline_gemm.cpp',
                   Path(__file__).parents[1]/'src/drivers/cvfem_perf_control.hpp']
        (out/'sources.sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                                         for p in sources}, indent=2)+'\n')
        scf, core = counter_events(topo)
        scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data', 'cmem_wr_data')]
        for group in groups():
            session = None
            try:
                session = GemmSession(topo, binary, env, out, 'gemm-b{}-reuse'.format(group[0][0]),
                                      library=library, production=True)
                for block, kind, repeat in group:
                    stem = out/'gemm-{}-b{}-r{}'.format(kind, block, repeat)
                    index = session.index
                    output = session.capture(topo, binary, stem, kind, repeat)
                    row = dict(block=block, kind=kind, repeat=repeat, session=session.name,
                               session_index=index, identity=session.identity, library=session.library_identity)
                    row.update(parse_output(output, kind, repeat))
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
        report, ok = analyze(records)
        (out/'summary.txt').write_text(control_report+'\n'+report)
        print(control_report+'\n'+report, end='')
        if not ok:
            raise ValueError('DGEMM campaign checks failed; retain every sample')
        print('ROOFLINE DGEMM CAMPAIGN PASSED')
    finally:
        hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
                  if p.is_file() and p.name != 'files.sha256.json'}
        (out/'files.sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discover-library', action='store_true')
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--blas-library', type=Path)
    args = parser.parse_args()
    try:
        if args.discover_library:
            print(discover_library(override=str(args.blas_library) if args.blas_library else os.environ.get('BLAS_LIBRARY')))
        else:
            if not all((args.binary, args.out, args.calibration, args.blas_library)):
                parser.error('collection requires --binary, --out, --calibration and --blas-library')
            collect(args.binary, args.out.resolve(), args.calibration.resolve(), args.blas_library.resolve())
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
