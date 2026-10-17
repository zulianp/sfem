#!/usr/bin/env python3
"""Run only inside the user's exclusive Grace allocation; preserve every counter capture."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys


def cpulist(value):
    result = set()
    for part in value.strip().split(','):
        limits = list(map(int, part.split('-')))
        if len(limits) == 1:
            result.add(limits[0])
        elif len(limits) == 2 and limits[0] <= limits[1]:
            result.update(range(limits[0], limits[1]+1))
        else:
            raise ValueError('invalid CPU list')
    return result


def parse_perf(text, events):
    found = {}
    for fields in csv.reader(text.splitlines()):
        if len(fields) < 3:
            continue
        event = fields[2].strip()
        if event not in events:
            continue
        if event in found or len(fields) < 5:
            raise ValueError('duplicate or incomplete event: '+event)
        count, runtime, coverage = float(fields[0]), float(fields[3]), float(fields[4])
        if not all(map(math.isfinite, (count, runtime, coverage))):
            raise ValueError('nonfinite counter: '+event)
        if count < 0 or runtime <= 0:
            raise ValueError('invalid counter count/runtime: '+event)
        if not 99.99 <= coverage <= 100.01:
            raise ValueError(f'counter {event} ran {coverage:.2f}% of enabled time; require 100% without multiplexing')
        found[event] = count
    if set(found) != set(events):
        raise ValueError('required counters missing: '+str(set(events)-set(found)))
    return found


def topology(extra_events=()):
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise ValueError('run this collector on Grace Linux')
    allowed = set(os.sched_getaffinity(0))
    if len(allowed) != 72 or os.environ.get('SLURM_CPUS_PER_TASK') != '72':
        raise ValueError('request and bind one 72-core socket')
    cpu_root = Path('/sys/devices/system/cpu')
    packages = {(cpu_root/f'cpu{c}/topology/physical_package_id').read_text().strip() for c in allowed}
    nodes = {p.name for c in allowed for p in (cpu_root/f'cpu{c}').glob('node[0-9]*')}
    if len(packages) != 1 or len(nodes) != 1:
        raise ValueError('CPU affinity crosses a package or NUMA node')
    pmus = []
    inventory = {}
    for pmu in Path('/sys/bus/event_source/devices').glob('nvidia_scf_pmu_*'):
        mask = (pmu/'cpumask').read_text().strip()
        inventory[pmu.name] = dict(cpumask=mask)
        if cpulist(mask) & allowed:
            pmus.append(pmu)
    if len(pmus) != 1:
        raise ValueError('cannot map exactly one SCF representative CPU into the bound socket')
    pmu = pmus[0]
    representatives = cpulist(inventory[pmu.name]['cpumask'])
    if len(representatives) != 1 or not representatives <= allowed:
        raise ValueError('SCF must expose one representative CPU in the bound socket')
    aliases = {}
    for event in ('cmem_rd_data', 'cmem_wr_total_bytes')+tuple(extra_events):
        aliases[event] = (pmu/'events'/event).read_text().strip()
    return dict(cpus=sorted(allowed), package=next(iter(packages)),
                node=int(next(iter(nodes))[4:]), pmu=pmu.name,
                scf_cpu=next(iter(representatives)), pmu_inventory=inventory,
                event_aliases=aliases, host=platform.node(), kernel=platform.release(),
                job=os.environ.get('SLURM_JOB_ID'))


def counter_events(topo):
    scf = [topo['pmu']+'/'+event+'/' for event in ('cmem_rd_data', 'cmem_wr_total_bytes')]
    # NVIDIA's raw aliases count SVE and non-SVE floating-point operations.
    return scf, ['r80c0', 'r80c1']


def capture_command(topo, binary, stem, mode, elements, repeat, scf_events=None):
    scf, core = counter_events(topo)
    if scf_events is not None:
        scf = scf_events
    cpus = ','.join(map(str, topo['cpus']))
    workload = ['sleep', '0.2'] if mode.startswith('idle-') else [str(binary), mode, str(elements), str(repeat), '72']
    command = ['numactl', '--physcpubind='+cpus, '--membind='+str(topo['node'])]+workload
    # SCF is shared by the socket. Opening it on every core duplicates the events.
    for scope, events, path in ((cpus, core, stem.with_suffix('.core.perf')),
                                (str(topo['scf_cpu']), scf, stem.with_suffix('.scf.perf'))):
        command = ['perf', 'stat', '--no-big-num', '--no-scale', '-x,', '-a', '-C', scope,
                   '-e', '{'+','.join(events)+'}', '-o', str(path), '--']+command
    return command


def capture_cases(diagnose_writes):
    if diagnose_writes:
        for block in range(4):
            yield block, 'idle-before', 0
            for mode in (('copy', 'triad') if block % 2 == 0 else ('triad', 'copy')):
                for repeat in ((10,40) if block % 2 == 0 else (40,10)):
                    yield block, mode, repeat
            yield block, 'idle-after', 0
    else:
        for block, order in enumerate(itertools.permutations((10, 40, 70))):
            for mode in (('copy', 'triad') if block % 2 == 0 else ('triad', 'copy')):
                for repeat in order:
                    yield block, mode, repeat


def write_diagnostic_report(records):
    actual = [(r['block'],r['mode'],r['repeat']) for r in records]
    expected = list(capture_cases(True))
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError('incomplete or duplicate write diagnostic captures')
    lines = ['Write-counter diagnostic only; this does not accept a roofline calibration.',
             'Same-group identity: total write bytes = 32 * write-back beats + other write bytes.',
             'Counts include startup, initialization, warmups, checks and teardown.',
             'Idle controls sleep 0.2 seconds with the same nested perf launch.',
             'Event order rotates once per block; commands preserve the order.',
             '', 'block mode repeats read_GB total_write_GB component_write_GB total_minus_components_GB']
    for r in records:
        lines.append(f'{r["block"]} {r["mode"]} {r["repeat"]} {r["read_bytes"]/1e9:.6f} '
                     f'{r["write_bytes"]/1e9:.6f} {r["component_write_bytes"]/1e9:.6f} '
                     f'{(r["write_bytes"]-r["component_write_bytes"])/1e9:+.6f}')
    lines += ['', 'Inspect identity differences, idle traffic and event-order dependence before choosing a new measurement.',
              'No counter was substituted or adjusted; the full calibration remains required.']
    return '\n'.join(lines)+'\n'


def analyze(records, elements):
    lines = ['Counter calibration: slopes across repeat counts 10, 40, 70.',
             'Ranges show sample variation, not confidence intervals.',
             'These are STREAM-style controls, not an official STREAM score.', '']
    checks = []
    for mode in ('copy', 'triad'):
        slopes = []
        for block in range(6):
            subset = {r['repeat']: r for r in records if r['mode'] == mode and r['block'] == block}
            if set(subset) != {10, 40, 70}:
                raise ValueError('incomplete calibration block')
            for lo, hi in ((10, 40), (40, 70)):
                a, b = subset[lo], subset[hi]
                slopes.append({key: (b[key]-a[key])/(hi-lo) for key in ('read_bytes', 'write_bytes', 'fp_ops')})
        totals = [s['read_bytes']+s['write_bytes'] for s in slopes]
        fp = [s['fp_ops'] for s in slopes]
        useful = elements*(16 if mode == 'copy' else 24)
        expected = elements*(0 if mode == 'copy' else 2)
        # Diagnostic tolerances: 2% of a triad's work and 10% traffic spread.
        fp_ok = max(abs(f-expected) for f in fp) <= 0.02*2*elements
        traffic_ok = min(totals) > 0 and (max(totals)-min(totals))/statistics.mean(totals) <= 0.10
        coverage_ok = 0.85 <= statistics.mean(totals)/useful <= 1.8
        checks += [fp_ok, traffic_ok, coverage_ok]
        lines += [f'{mode}: transported/useful bytes mean={statistics.mean(totals)/useful:.6f}, '
                  f'range=[{min(totals)/useful:.6f}, {max(totals)/useful:.6f}]',
                  f'  read bytes/sweep mean={statistics.mean(s["read_bytes"] for s in slopes):.3f}; '
                  f'write bytes/sweep mean={statistics.mean(s["write_bytes"] for s in slopes):.3f}',
                  f'  FP ops/sweep range=[{min(fp):.3f}, {max(fp):.3f}], expected={expected}',
                  f'  FLOP check={fp_ok}, traffic stability={traffic_ok}, traffic coverage={coverage_ok}']
    lines += ['', 'PMU events count speculative operations in all precisions.',
              'A pass supports this control configuration; operator measurements remain pending.']
    return '\n'.join(lines)+'\n', all(checks)


def prepare_capture(binary, out, elements, mode, extra_events=()):
    out.mkdir()  # Do not overwrite a previous measurement.
    topo = topology(extra_events)
    (out/'topology.json').write_text(json.dumps(topo, indent=2)+'\n')
    (out/'collection.json').write_text(json.dumps(dict(
        mode=mode, elements=elements), indent=2)+'\n')
    binary = binary.resolve()
    (out/'binary.sha256').write_text(hashlib.sha256(binary.read_bytes()).hexdigest()+'  '+str(binary)+'\n')
    for name, command in [('lscpu.txt', ['lscpu']), ('numactl.txt', ['numactl', '--hardware']),
                          ('perf-version.txt', ['perf', '--version'])]:
        (out/name).write_text(subprocess.check_output(command, universal_newlines=True))
    env = dict(os.environ, LC_ALL='C', OMP_NUM_THREADS='72', OMP_PROC_BIND='close',
               OMP_PLACES='cores', OMP_DYNAMIC='FALSE', CVFEM_CALIBRATION_WARMUP='3')
    (out/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if
        k.startswith('OMP_') or k in ('CVFEM_CALIBRATION_WARMUP','LC_ALL','SLURM_JOB_ID','SLURM_CPUS_PER_TASK','SLURM_CPU_BIND')}, indent=2)+'\n')
    return topo, binary, env


def control_seconds(text, mode, elements, repeat):
    expected = f'CALIBRATION mode={mode} elements={elements} repeats={repeat} threads=72 '
    rows = [s for s in text.splitlines() if s.startswith(expected) and s.endswith(' PASS')]
    if len(rows) != 1:
        raise ValueError('calibration output/team check missing')
    values = dict(word.split('=') for word in rows[0].split()[1:-1])
    seconds = float(values['seconds'])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('invalid control duration')
    if float(values['useful_bytes']) != elements*(16 if mode == 'copy' else 24) or float(values['flops']) != elements*(0 if mode == 'copy' else 2):
        raise ValueError('control work count mismatch')
    return seconds


def collect(binary, out, elements, diagnose_writes=False):
    extra_events = ('cmem_wb_data', 'cmem_wr_data') if diagnose_writes else ()
    topo, binary, env = prepare_capture(binary, out, elements,
        'write-diagnostic' if diagnose_writes else 'calibration', extra_events)
    scf, core = counter_events(topo)
    scf += [topo['pmu']+'/'+event+'/' for event in extra_events]
    records = []
    for block, mode, repeat in capture_cases(diagnose_writes):
        stem = out/f'{mode}-b{block}-r{repeat}'
        ordered_scf = scf[block:]+scf[:block] if diagnose_writes else scf
        command = capture_command(topo, binary, stem, mode, elements, repeat, ordered_scf)
        stem.with_suffix('.command.json').write_text(json.dumps(command)+'\n')
        with stem.with_suffix('.out').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        seconds = None
        if not mode.startswith('idle-'):
            seconds = control_seconds(stem.with_suffix('.out').read_text(), mode, elements, repeat)
        scf_counts = parse_perf(stem.with_suffix('.scf.perf').read_text(), scf)
        core_counts = parse_perf(stem.with_suffix('.core.perf').read_text(), core)
        record = dict(block=block, mode=mode, repeat=repeat, seconds=seconds,
                      read_bytes=32*scf_counts[scf[0]], write_bytes=scf_counts[scf[1]],
                      fp_ops=sum(core_counts.values()))
        if diagnose_writes:
            record.update(writeback_beats=scf_counts[scf[2]], other_write_bytes=scf_counts[scf[3]],
                          component_write_bytes=32*scf_counts[scf[2]]+scf_counts[scf[3]])
        records.append(record)
        print(f'captured {stem.name}', flush=True)
    (out/'records.json').write_text(json.dumps(records, indent=2)+'\n')
    report, ok = (write_diagnostic_report(records), None) if diagnose_writes else analyze(records, elements)
    (out/'summary.txt').write_text(report)
    print(report, end='')
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out/'files.sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    if diagnose_writes:
        print('WRITE DIAGNOSTIC COLLECTED; calibration remains unvalidated')
    elif not ok:
        raise ValueError('counter calibration checks failed; retain logs and inspect before operator runs')
    else:
        print('COUNTER CALIBRATION PASSED')


PHASES = ('empty', 'setup', 'warmup', 'sweep', 'check', 'teardown')


def phase_cases():
    for block in range(3):
        for mode in (('copy','triad') if block % 2 == 0 else ('triad','copy')):
            for phase in (PHASES if block % 2 == 0 else PHASES[::-1]):
                yield block, mode, phase


def phase_command(topo, binary, stem, mode, elements, phase, events, descriptors):
    control_read, control_write, ack_read, ack_write = descriptors
    cpus = ','.join(map(str,topo['cpus']))
    return ['perf','stat','--no-big-num','--no-scale','-x,','-a','-C',str(topo['scf_cpu']),
            '--delay=-1',f'--control=fd:{control_read},{ack_write}',
            '-e','{'+','.join(events)+'}','-o',str(stem.with_suffix('.scf.perf')),'--',
            'numactl','--physcpubind='+cpus,'--membind='+str(topo['node']),
            str(binary),mode,str(elements),'10','72',phase,str(control_write),str(ack_read)]


def phase_report(records):
    actual = [(r['block'],r['mode'],r['phase']) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != set(phase_cases()):
        raise ValueError('incomplete or duplicate phase captures')
    lines = ['Phase diagnostic only; this does not accept a roofline calibration.',
             'Three independent captures per phase/mode; ten repeated sweeps, three warmups.',
             'Ranges are sample variation. Independent phases are not an exact decomposition of one run.',
             'Empty measures control overhead. No baseline or counter adjustment is applied.',
             '', 'mode phase read_GB_range write_GB_range max_abs_identity_error_bytes']
    for mode in ('copy','triad'):
        for phase in PHASES:
            rows = [r for r in records if r['mode']==mode and r['phase']==phase]
            ranges = ['[{:.6f}, {:.6f}]'.format(min(r[k] for r in rows)/1e9,max(r[k] for r in rows)/1e9)
                      for k in ('read_bytes','write_bytes')]
            error = max(abs(r['write_bytes']-r['component_write_bytes']) for r in rows)
            lines.append(f'{mode} {phase} {ranges[0]} {ranges[1]} {error:.0f}')
    lines += ['', 'Inspect which phase contains excess writes before changing the calibration recipe.']
    return '\n'.join(lines)+'\n'


def diagnose_phases(binary, out, elements):
    extra = ('cmem_wb_data','cmem_wr_data')
    topo, binary, env = prepare_capture(binary,out,elements,'phase-diagnostic',extra)
    events = counter_events(topo)[0]+[topo['pmu']+'/'+s+'/' for s in extra]
    records = []
    for block, mode, phase in phase_cases():
        stem = out/f'{mode}-{phase}-b{block}'
        descriptors = []
        try:
            descriptors.extend(os.pipe())
            descriptors.extend(os.pipe())
            command = phase_command(topo,binary,stem,mode,elements,phase,events,descriptors)
            stem.with_suffix('.command.json').write_text(json.dumps(command)+'\n')
            with stem.with_suffix('.out').open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                               pass_fds=tuple(descriptors),timeout=60)
        finally:
            for descriptor in descriptors:
                os.close(descriptor)
        output = stem.with_suffix('.out').read_text()
        seconds = control_seconds(output,mode,elements,10)
        marker = f'PERF_PHASE phase={phase} transitions=2 PASS'
        if output.splitlines().count(marker) != 1:
            raise ValueError('both perf control acknowledgements must complete: '+phase)
        counts = parse_perf(stem.with_suffix('.scf.perf').read_text(),events)
        records.append(dict(block=block,mode=mode,phase=phase,repeat=10,seconds=seconds,
            read_bytes=32*counts[events[0]],write_bytes=counts[events[1]],
            writeback_beats=counts[events[2]],other_write_bytes=counts[events[3]],
            component_write_bytes=32*counts[events[2]]+counts[events[3]]))
        print(f'captured {stem.name}',flush=True)
    (out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    report = phase_report(records)
    (out/'summary.txt').write_text(report)
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out/'files.sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(report,end='')
    print('PHASE DIAGNOSTIC COLLECTED; calibration remains unvalidated')


def selftest():
    assert cpulist('0-2,5,7-8') == {0,1,2,5,7,8}
    topo = dict(cpus=list(range(72,144)), scf_cpu=72, pmu='nvidia_scf_pmu_1', node=1)
    command = capture_command(topo, Path('/control'), Path('/capture/copy-b0-r10'), 'copy', 80000000, 10)
    split = command.index('--')
    outer, rest = command[:split], command[split+1:]
    split = rest.index('--')
    inner, workload = rest[:split], rest[split+1:]
    scf, core = counter_events(topo)
    for invocation, scope, events, suffix in ((outer, '72', scf, '.scf.perf'),
                                            (inner, ','.join(map(str, range(72,144))), core, '.core.perf')):
        assert invocation[:2] == ['perf', 'stat'] and '-a' in invocation and '--no-scale' in invocation
        assert invocation[invocation.index('-C')+1] == scope
        assert invocation[invocation.index('-e')+1] == '{'+','.join(events)+'}'
        assert invocation[invocation.index('-o')+1] == '/capture/copy-b0-r10'+suffix
    assert workload == ['numactl', '--physcpubind='+','.join(map(str, range(72,144))),
                        '--membind=1', '/control', 'copy', '80000000', '10', '72']
    phases = list(phase_cases())
    assert len(phases) == len(set(phases)) == 36
    gated = phase_command(topo,Path('/control'),Path('/capture/phase'),'triad',80000000,'sweep',scf,(11,12,13,14))
    assert '--delay=-1' in gated and '--control=fd:11,14' in gated
    assert gated[gated.index('-C')+1] == '72'
    assert gated[-8:] == ['/control','triad','80000000','10','72','sweep','12','13']
    phase_records = [dict(block=b,mode=m,phase=p,read_bytes=10**9,write_bytes=2*10**9,
                          component_write_bytes=2*10**9) for b,m,p in phases]
    assert 'does not accept' in phase_report(phase_records)
    for invalid in (phase_records[:-1],phase_records+[phase_records[0]]):
        try:
            phase_report(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted incomplete or duplicate phases')
    assert len(list(capture_cases(False))) == 36
    cases = list(capture_cases(True))
    assert len(cases) == len(set(cases)) == 24
    diagnostic_scf = scf+[topo['pmu']+'/'+s+'/' for s in ('cmem_wb_data','cmem_wr_data')]
    for block in range(4):
        rotated = diagnostic_scf[block:]+diagnostic_scf[:block]
        idle = capture_command(topo, Path('/control'), Path('/capture/idle'), 'idle-before', 80000000, 0, rotated)
        assert idle[idle.index('-e')+1] == '{'+','.join(rotated)+'}'
        assert idle[idle.index('-C')+1] == '72'
        assert idle[-2:] == ['sleep','0.2'] and '/control' not in idle
    diagnostic_records = [dict(block=b,mode=m,repeat=r,read_bytes=10**9,
                               write_bytes=3*10**9,component_write_bytes=10**9) for b,m,r in cases]
    report = write_diagnostic_report(diagnostic_records)
    assert report.count('+2.000000') == 24 and 'does not accept' in report
    for bad_records in (diagnostic_records[:-1],diagnostic_records+[diagnostic_records[0]]):
        try:
            write_diagnostic_report(bad_records)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted incomplete or duplicate diagnostic')
    good = '100,,rd,1000000,100.00,,\n200,,wr,1000000,100.00,,\n'
    assert parse_perf(good, ['rd','wr']) == {'rd':100,'wr':200}
    bad = [good.replace('100.00','98.00'), good.replace('100.00','4.00'),
           good.replace('100,,','<not supported>,,'),
           good.replace('100,,','nan,,'), good.splitlines()[0], good+good,
           good.replace('100,,','-1,,'), good.replace('1000000','0')]
    for text in bad:
        try:
            parse_perf(text, ['rd','wr'])
        except ValueError:
            pass
        else:
            raise AssertionError('accepted invalid counter capture')
    records = [dict(mode=mode,block=block,repeat=repeat,read_bytes=1000+repeat*100*(8 if mode=='copy' else 16),
                    write_bytes=2000+repeat*100*8,fp_ops=100+repeat*100*(0 if mode=='copy' else 2))
               for mode in ('copy','triad') for block in range(6) for repeat in (10,40,70)]
    assert analyze(records,100)[1]
    corrupt = [dict(r,fp_ops=100) for r in records]
    assert not analyze(corrupt,100)[1]
    try:
        analyze(records[:-1],100)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted incomplete blocks')
    print('calibration CPU scopes, parser, slopes, missing counters and failed controls: PASS')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--binary',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--elements',type=int,default=80000000)
    diagnostics = ap.add_mutually_exclusive_group()
    diagnostics.add_argument('--diagnose-writes',action='store_true',help='collect write-counter components and idle controls without accepting calibration')
    diagnostics.add_argument('--diagnose-phases',action='store_true',help='count each control phase separately without accepting calibration')
    args = ap.parse_args()
    try:
        if args.selftest:
            selftest()
        elif args.binary and args.out and args.elements >= 80000000:
            if args.diagnose_phases:
                diagnose_phases(args.binary,args.out.resolve(),args.elements)
            else:
                collect(args.binary,args.out.resolve(),args.elements,args.diagnose_writes)
        else:
            ap.error('provide --binary and --out; arrays must exceed the Grace cache')
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('COUNTER CALIBRATION FAILED: '+str(error), file=sys.stderr)
        sys.exit(1)
