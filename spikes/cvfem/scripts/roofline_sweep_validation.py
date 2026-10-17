#!/usr/bin/env python3
"""Validate Grace counters around repeated copy/triad sweeps, excluding setup."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

from roofline_calibration import counter_events, control_seconds, parse_perf, prepare_capture


def cases():
    for block, order in enumerate(itertools.permutations((10,40,70))):
        for mode in (('copy','triad') if block % 2 == 0 else ('triad','copy')):
            yield block, mode, 'empty-before', 10
            for repeat in order:
                yield block, mode, 'sweep', repeat
            yield block, mode, 'empty-after', 10


def capture_command(topo, binary, stem, mode, elements, kind, repeat, descriptors):
    if len(descriptors) != 8 or len(set(descriptors)) != 8:
        raise ValueError('two independent control/acknowledgement pipe pairs required')
    scf, core = counter_events(topo)
    scf += [topo['pmu']+'/'+event+'/' for event in ('cmem_wb_data','cmem_wr_data')]
    cpus = ','.join(map(str,topo['cpus']))
    phase = 'sweep' if kind == 'sweep' else 'sweep-empty'
    command = ['numactl','--physcpubind='+cpus,'--membind='+str(topo['node']),
               str(binary),mode,str(elements),str(repeat),'72',phase,
               str(descriptors[1]),str(descriptors[2]),str(descriptors[5]),str(descriptors[6])]
    for scope, events, suffix, fds in ((cpus,core,'.core.perf',descriptors[4:]),
                                      (str(topo['scf_cpu']),scf,'.scf.perf',descriptors[:4])):
        command = ['perf','stat','--no-big-num','--no-scale','-x,','-a','-C',scope,
                   '--delay=-1','--control=fd:{},{}'.format(fds[0],fds[3]),
                   '-e','{'+','.join(events)+'}','-o',str(stem.with_suffix(suffix)),'--']+command
    return command


def run_capture(command, env, log, descriptors, timeout=60):
    with log.open('w') as output:
        process = subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT,
                                   pass_fds=tuple(descriptors),start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code,command)
        except BaseException:
            # A timed-out outer perf must not leave its inner perf or workload running.
            for sig in (signal.SIGTERM,signal.SIGKILL):
                try:
                    os.killpg(process.pid,sig)
                except ProcessLookupError:
                    pass
                if sig == signal.SIGTERM:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            process.wait()
            raise


def window_seconds(text, events, cpus):
    runtimes = [float(row[3])/1e9/cpus for row in csv.reader(text.splitlines())
                if len(row) >= 5 and row[2].strip() in events]
    if len(runtimes) != len(events) or not all(math.isfinite(x) and x > 0 for x in runtimes):
        raise ValueError('invalid counter window')
    if max(runtimes)-min(runtimes) > 1e-6*max(runtimes):
        raise ValueError('events in a group have inconsistent runtimes')
    return statistics.mean(runtimes)


def spread(values):
    return (max(values)-min(values))/statistics.mean(values) if min(values) > 0 else math.inf


def control_timing(output, phase):
    prefix = 'PERF_CONTROL phase='+phase+' controllers=2 '
    rows = [line for line in output.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError('missing or duplicate counter-control timing')
    fields = dict(word.split('=') for word in rows[0][len(prefix):].split())
    keys = ('enable_seconds','disable_seconds','inner_seconds','outer_seconds')
    if set(fields) != set(keys):
        raise ValueError('incomplete counter-control timing')
    values = {key:float(fields[key]) for key in keys}
    if not all(math.isfinite(v) and v >= 0 for v in values.values()) or values['outer_seconds'] <= 0:
        raise ValueError('invalid counter-control timing')
    total = sum(values[key] for key in keys[:3])
    if not math.isclose(total,values['outer_seconds'],rel_tol=1e-8,abs_tol=1e-9):
        raise ValueError('inconsistent counter-control timing')
    return values


def window_report(rows):
    lines = []
    for name,key in (('SCF','scf_seconds'),('FP','core_seconds')):
        ratios = [r[key]/r['seconds'] for r in rows]
        lines.append('  {}/loop range=[{:.6f}, {:.6f}]'.format(name,min(ratios),max(ratios)))
    for row in rows:
        scf, core = (row[key]/row['seconds'] for key in ('scf_seconds','core_seconds'))
        if 0.99 <= scf <= 1.10 and 0.99 <= core <= 1.10:
            continue
        lines.append('  FAILED {}-sweep-b{}-r{}: loop_ms={:.6f}, SCF_ms={:.6f} ({:.6f}x), FP_ms={:.6f} ({:.6f}x)'.format(
            row['mode'],row['block'],row['repeat'],row['seconds']*1e3,
            row['scf_seconds']*1e3,scf,row['core_seconds']*1e3,core))
        if 'control_timing' in row:
            values = row['control_timing']
            lines.append('    control: enable_ms={:.6f}, disable_ms={:.6f}, inner_ms={:.6f}, outer_ms={:.6f}'.format(
                *(values[key]*1e3 for key in ('enable_seconds','disable_seconds','inner_seconds','outer_seconds'))))
        else:
            lines.append('    control: switch timings unavailable in this older capture')
    return lines


def check_work(values, mode, elements):
    read, write, fp = (values[k] for k in ('read_bytes','write_bytes','fp_ops'))
    inputs = elements*(8 if mode == 'copy' else 16)
    useful = elements*(16 if mode == 'copy' else 24)
    expected_fp = elements*(0 if mode == 'copy' else 2)
    return dict(fp=abs(fp-expected_fp) <= 0.02*2*elements,
                # Include a possible read-for-ownership of the output array.
                read=0.85*inputs <= read <= 1.10*(inputs+8*elements),
                traffic=write > 0 and 0.85 <= (read+write)/useful <= 1.8)


def analyze(records, elements):
    actual = [(r['block'],r['mode'],r['kind'],r['repeat']) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != set(cases()):
        raise ValueError('incomplete or duplicate sweep validation captures')
    keys = ('read_bytes','write_bytes','fp_ops')
    for row in records:
        if not all(math.isfinite(row[k]) and row[k] >= 0 for k in keys+('component_write_bytes',)):
            raise ValueError('invalid validation count')
        if not all(math.isfinite(row[k]) and row[k] > 0 for k in ('seconds','scf_seconds','core_seconds')):
            raise ValueError('invalid validation duration')
    checks = [all(r['write_bytes'] == r['component_write_bytes'] for r in records)]
    lines = ['Sweep-window counter validation: six blocks, repeat counts 10, 40, 70.',
             'Both counter windows bracket the same timed sweep; switching is asynchronous.',
             'Empty controls open after warmup; counts are retained without subtraction.',
             'Ranges show sample variation, not confidence intervals.',
             'Practical screens do not establish fixed absolute counter accuracy.',
             'Write-component identity: '+str(checks[0]), '']
    for mode in ('copy','triad'):
        rows = [r for r in records if r['mode'] == mode and r['kind'] == 'sweep']
        empty = [r for r in records if r['mode'] == mode and r['kind'] != 'sweep']
        useful = elements*(16 if mode == 'copy' else 24)
        empty_ok = all(r['read_bytes']+r['write_bytes'] <= 0.01*10*useful and
                       r['fp_ops'] <= 0.01*10*2*elements for r in empty)
        windows_ok = all(0.99 <= r[k]/r['seconds'] <= 1.10
                         for r in rows for k in ('scf_seconds','core_seconds'))
        direct = [{k:r[k]/r['repeat'] for k in keys} for r in rows]
        slopes = []
        for block in range(6):
            by_repeat = {r['repeat']:r for r in rows if r['block'] == block}
            for lo, hi in ((10,40),(40,70)):
                slopes.append({k:(by_repeat[hi][k]-by_repeat[lo][k])/(hi-lo) for k in keys})
        checks += [empty_ok,windows_ok]
        lines += [mode+':',
                  '  empty <= 1% of ten-sweep useful bytes/triad FLOPs: '+str(empty_ok),
                  '  counter runtime / loop time in [0.99, 1.10]: '+str(windows_ok),
                  '  max empty bytes={:.0f}; max empty FP ops={:.0f}'.format(
                      max(r['read_bytes']+r['write_bytes'] for r in empty),max(r['fp_ops'] for r in empty))]
        lines += window_report(rows)
        for label, samples in (('direct',direct),('slopes',slopes)):
            work = [check_work(s,mode,elements) for s in samples]
            work_ok = {key:all(w[key] for w in work) for key in ('fp','read','traffic')}
            ranges = {key:spread([s[key] for s in samples]) for key in ('read_bytes','write_bytes')}
            totals = [s['read_bytes']+s['write_bytes'] for s in samples]
            stable = max(list(ranges.values())+[spread(totals)]) <= 0.10
            checks += list(work_ok.values())+[stable]
            lines += ['  {}: FP={}, read bounds={}, traffic bounds={}, stability={}'.format(
                          label,work_ok['fp'],work_ok['read'],work_ok['traffic'],stable),
                      '    read bytes/sweep range=[{:.3f}, {:.3f}]; write=[{:.3f}, {:.3f}]'.format(
                          min(s['read_bytes'] for s in samples),max(s['read_bytes'] for s in samples),
                          min(s['write_bytes'] for s in samples),max(s['write_bytes'] for s in samples)),
                      '    transported/useful range=[{:.6f}, {:.6f}]; FP ops/sweep=[{:.3f}, {:.3f}]'.format(
                          min(totals)/useful,max(totals)/useful,min(s['fp_ops'] for s in samples),max(s['fp_ops'] for s in samples))]
    lines += ['', 'A pass validates these controls and this measurement configuration only.',
              'No operator measurement or production performance claim is validated here.']
    return '\n'.join(lines)+'\n', all(checks)


def collect(binary, out, elements):
    topo, binary, env = prepare_capture(binary,out,elements,'sweep-validation',('cmem_wb_data','cmem_wr_data'))
    scf, core = counter_events(topo)
    scf += [topo['pmu']+'/'+event+'/' for event in ('cmem_wb_data','cmem_wr_data')]
    records = []
    for block, mode, kind, repeat in cases():
        stem = out/'{}-{}-b{}-r{}'.format(mode,kind,block,repeat)
        descriptors = []
        try:
            for unused in range(4):
                descriptors.extend(os.pipe())
            command = capture_command(topo,binary,stem,mode,elements,kind,repeat,descriptors)
            stem.with_suffix('.command.json').write_text(json.dumps(command)+'\n')
            run_capture(command,env,stem.with_suffix('.out'),descriptors)
        finally:
            for descriptor in descriptors:
                os.close(descriptor)
        output = stem.with_suffix('.out').read_text()
        seconds = control_seconds(output,mode,elements,repeat)
        phase = 'sweep' if kind == 'sweep' else 'sweep-empty'
        marker = 'PERF_PHASE phase='+phase+' controllers=2 transitions=4 PASS'
        if output.splitlines().count(marker) != 1:
            raise ValueError('both controllers must acknowledge enable and disable: '+str(stem))
        timing = control_timing(output,phase)
        scf_text, core_text = (stem.with_suffix(s).read_text() for s in ('.scf.perf','.core.perf'))
        memory = parse_perf(scf_text,scf)
        floating = parse_perf(core_text,core)
        records.append(dict(block=block,mode=mode,kind=kind,repeat=repeat,seconds=seconds,
            control_timing=timing,
            scf_seconds=window_seconds(scf_text,scf,1),core_seconds=window_seconds(core_text,core,72),
            read_bytes=32*memory[scf[0]],write_bytes=memory[scf[1]],fp_ops=sum(floating.values()),
            writeback_beats=memory[scf[2]],other_write_bytes=memory[scf[3]],
            component_write_bytes=32*memory[scf[2]]+memory[scf[3]]))
        (out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
        print('captured '+stem.name,flush=True)
    report, ok = analyze(records,elements)
    (out/'summary.txt').write_text(report)
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out/'files.sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(report,end='')
    if not ok:
        raise ValueError('sweep validation checks failed; retain logs before operator runs')
    print('SWEEP COUNTER VALIDATION PASSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--elements',type=int,default=80000000)
    args = parser.parse_args()
    try:
        if args.elements < 80000000:
            raise ValueError('arrays must exceed the Grace cache')
        collect(args.binary,args.out.resolve(),args.elements)
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        print('SWEEP COUNTER VALIDATION FAILED: '+str(error),file=sys.stderr)
        sys.exit(1)
