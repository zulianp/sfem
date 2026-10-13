#!/usr/bin/env python3
"""Measure checked, fixed-state CVFEM residuals with the validated Grace counter windows."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from roofline_calibration import counter_events, parse_perf, prepare_capture
from roofline_sweep_validation import control_timing, window_seconds, spread
from roofline_reuse_diagnostic import Session, analyze_measurements as analyze_controls


def groups():
    for block, order in enumerate(itertools.permutations((10,40,70))):
        for scheme in (('fo','darwish') if block % 2 == 0 else ('darwish','fo')):
            kinds = [('empty-before',10)] + [('sweep',r) for r in order] + [('empty-after',10)]
            kinds.insert(0 if block % 2 == 0 else len(kinds), ('normal',40))
            for layout in (('atomic','packed') if block % 2 == 0 else ('packed','atomic')):
                yield [(block,scheme,layout,kind,repeat) for kind,repeat in kinds]


def cases():
    return [case for group in groups() for case in group]


def session_name(block, scheme, layout):
    return '{}-{}-b{}-reuse'.format(scheme,layout,block)


def benchmark_command(binary, scheme, layout, window, repeat, warmup, n):
    command = [str(binary),'--n',str(n),'--pack-size','1024','--layout',layout,'--geom','affine',
               '--rhie-chow','--perturb','0.017','--warmup',str(warmup),'--repeat',str(repeat),
               '--roofline-window',window]
    if scheme == 'darwish':
        command += ['--conv-ho','3']
    return command


def capture_command(topo, binary, stem, scheme, layout, kind, repeat, fds, n=128):
    counted = kind != 'normal'
    if (counted and (len(fds) != 8 or len(set(fds)) != 8)) or (not counted and fds):
        raise ValueError('invalid operator control descriptors')
    window = 'normal' if not counted else ('sweep' if kind == 'sweep' else 'sweep-empty')
    cpus = ','.join(map(str,topo['cpus']))
    command = ['numactl','--physcpubind='+cpus,'--membind='+str(topo['node'])]
    command += benchmark_command(binary,scheme,layout,window,repeat,3,n)
    if counted:
        command += ['--perf-fds']+[str(fds[i]) for i in (1,2,5,6)]
        scf,core = counter_events(topo)
        scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data','cmem_wr_data')]
        for scope,events,suffix,pair in ((cpus,core,'.core.perf',fds[4:]),
                                       (str(topo['scf_cpu']),scf,'.scf.perf',fds[:4])):
            command = ['perf','stat','--no-big-num','--no-scale','-x,','-a','-C',scope,
                       '--delay=-1','--control=fd:{},{}'.format(pair[0],pair[3]),
                       '-e','{'+','.join(events)+'}','-o',str(stem.with_suffix(suffix)),'--']+command
    return command


class OperatorSession(Session):
    def __init__(self, topo, binary, env, out, name, scheme, layout, n=128, threads=72, launcher=True):
        self.scheme, self.layout, self.n = scheme, layout, n
        super().__init__(topo,binary,env,out,name,n**3,threads,launcher)

    def worker_command(self, binary, elements, threads):
        command = benchmark_command(binary,self.scheme,self.layout,'session',40,30,self.n)
        return command+['--perf-fds']+[str(self.fds[i]) for i in (1,2,5,6)]

    def capture(self, topo, binary, stem, kind, repeat, monitor_prefix=None, timeout=180):
        command = None
        phase = 'normal' if kind == 'normal' else ('sweep' if kind == 'sweep' else 'sweep-empty')
        if kind != 'normal':
            command = capture_command(topo,binary,stem,self.scheme,self.layout,kind,repeat,self.fds,self.n)
            command = command[:command.index('numactl')]+['cat']
            if monitor_prefix is not None:
                command = list(monitor_prefix)+[str(self.fds[i]) for i in (0,3,4,7)]
        return self.exchange(stem,'{} {}\n'.format(phase,repeat),command,timeout)


def parse_output(output, scheme, layout, kind, repeat, n=128, threads=72, warmup=3):
    rows = [line for line in output.splitlines() if line.startswith('ROOFLINE_OPERATOR ')]
    if len(rows) != 1 or not rows[0].endswith(' PASS'):
        raise ValueError('missing checked operator result')
    values = dict(word.split('=') for word in rows[0].split()[1:-1])
    phase = 'normal' if kind == 'normal' else ('sweep' if kind == 'sweep' else 'sweep-empty')
    expected = dict(window=phase,layout=layout,limiter=str(-1 if scheme == 'fo' else 3),
                    elements=str(n**3),nodes=str((n+1)**3),pack='1024',repeats=str(repeat),
                    warmup=str(warmup),threads=str(threads),precision='fp64',geometry='affine',rc='1',
                    state_gradients='cached',boundary='0',transient='0')
    if any(values.get(k) != v for k,v in expected.items()):
        raise ValueError('operator scope/team mismatch: '+str(values))
    seconds = float(values['seconds'])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('invalid operator duration')
    checks = [line for line in output.splitlines() if line.startswith('ROOFLINE_CHECK ')]
    if len(checks) != 8:
        raise ValueError('missing result checks')
    seen = set()
    for line in checks:
        if not line.endswith(' PASS'):
            raise ValueError('failed result check')
        fields = dict(word.split('=') for word in line.split()[1:-1])
        key = (fields['stage'],int(fields['field']))
        error = float(fields['relative'])
        if key in seen or not math.isfinite(error) or not 0 <= error <= 1e-10:
            raise ValueError('invalid result check')
        seen.add(key)
    if seen != {(stage,f) for stage in ('before','after') for f in range(4)}:
        raise ValueError('incomplete result checks')
    result = dict(seconds=seconds)
    if kind == 'normal':
        if 'PERF_PHASE' in output or 'PERF_CONTROL' in output:
            raise ValueError('normal run unexpectedly counted')
    else:
        if output.splitlines().count('PERF_PHASE phase='+phase+' controllers=2 transitions=4 PASS') != 1:
            raise ValueError('both counter controllers must complete')
        result['control_timing'] = control_timing(output,phase)
        # The control prints 12 significant digits; allow one nanosecond of text rounding.
        if kind == 'sweep' and result['control_timing']['inner_seconds'] + 1e-9 < seconds:
            raise ValueError('counter controls did not bracket timed loop')
    return result


def analyze(records):
    keys = ('block','scheme','layout','kind','repeat')
    actual = [tuple(r[k] for k in keys) for r in records]
    if actual != cases():
        raise ValueError('incomplete/duplicate operator campaign')
    sessions = {}
    for r in records:
        name = session_name(r['block'],r['scheme'],r['layout'])
        entries = sessions.setdefault(name,[])
        if (r['session'] != name or r['session_index'] != len(entries) or
                (entries and r['identity'] != entries[0]['identity'])):
            raise ValueError('operator process or buffers changed within a block')
        entries.append(r)
    for r in records:
        required = ('seconds',) if r['kind']=='normal' else ('seconds','read_bytes','write_bytes','fp_ops','scf_seconds','core_seconds')
        if not all(math.isfinite(r[k]) and r[k] >= 0 for k in required) or r['seconds'] <= 0:
            raise ValueError('invalid operator record')
    checks = []
    lines = ['Checked FP64 affine interior residuals: n=128, pack=1024, RC on, perturb=0.017.',
             'State gradients cached; no boundary/transient terms. Six blocks; sample ranges, not confidence intervals.',
             'One process/allocation per scheme/layout/block; 30 warmups and checks before every capture.',
             'FP counters count speculative operations of all precisions, not modelled useful FLOPs.',
             'Empty controls retained without subtraction. Rates use the timed loop bracketed by both counters.', '']
    for scheme in ('fo','darwish'):
        normal_means = {}
        for layout in ('atomic','packed'):
            group = [r for r in records if r['scheme']==scheme and r['layout']==layout]
            sweep = [r for r in group if r['kind']=='sweep']
            empty = [r for r in group if r['kind'].startswith('empty')]
            normal = [r for r in group if r['kind']=='normal']
            identity = all(r['write_bytes']==r['component_write_bytes'] for r in sweep+empty)
            windows = all(0.99 <= r[k]/r['seconds'] <= 1.10 for r in sweep for k in ('scf_seconds','core_seconds'))
            shortest = [r for r in sweep if r['repeat']==10]
            empty_ok = all(r['read_bytes']+r['write_bytes'] <= .01*min(s['read_bytes']+s['write_bytes'] for s in shortest)
                           and r['fp_ops'] <= .01*min(s['fp_ops'] for s in shortest) for r in empty)
            direct = [{k:r[k]/r['repeat'] for k in ('read_bytes','write_bytes','fp_ops')} for r in sweep]
            slopes = []
            for block in range(6):
                by_repeat = {r['repeat']:r for r in sweep if r['block']==block}
                for lo,hi in ((10,40),(40,70)):
                    slopes.append({k:(by_repeat[hi][k]-by_repeat[lo][k])/(hi-lo) for k in direct[0]})
            stable = all(spread([s[k] for s in samples]) <= .10 for samples in (direct,slopes) for k in direct[0])
            stable = stable and all(spread([s['read_bytes']+s['write_bytes'] for s in samples]) <= .10 for samples in (direct,slopes))
            paired_overhead = []
            for r in normal:
                full = next(s for s in sweep if s['block']==r['block'] and s['repeat']==40)
                paired_overhead.append(full['seconds']/r['seconds']-1)
            overhead_ok = abs(statistics.mean(paired_overhead)) <= .10
            checks += [identity,windows,empty_ok,stable,overhead_ok]
            normal_ms = [1000*r['seconds']/r['repeat'] for r in normal]
            normal_means[layout] = statistics.mean(normal_ms)
            lines += [scheme+' '+layout+': identity={}, windows={}, empty={}, count_stability={}, mean_timing_overhead={}'.format(identity,windows,empty_ok,stable,overhead_ok),
                      '  normal ms/sweep mean={:.6f}, range=[{:.6f}, {:.6f}]'.format(statistics.mean(normal_ms),min(normal_ms),max(normal_ms)),
                      '  paired instrumentation overhead mean={:+.2f}%, range=[{:+.2f}%, {:+.2f}%]'.format(100*statistics.mean(paired_overhead),100*min(paired_overhead),100*max(paired_overhead))]
            for label,values in (
                ('counted ms/sweep',[1000*r['seconds']/r['repeat'] for r in sweep]),
                ('read bytes/sweep',[s['read_bytes'] for s in direct]),
                ('write bytes/sweep',[s['write_bytes'] for s in direct]),
                ('speculative FP ops/sweep',[s['fp_ops'] for s in direct]),
                ('speculative FP ops/transported byte',[r['fp_ops']/(r['read_bytes']+r['write_bytes']) for r in sweep if r['read_bytes']+r['write_bytes']>0]),
                ('transported GB/s',[(r['read_bytes']+r['write_bytes'])/r['seconds']/1e9 for r in sweep]),
                ('speculative GFLOP/s',[r['fp_ops']/r['seconds']/1e9 for r in sweep]),
                ('SCF/loop',[r['scf_seconds']/r['seconds'] for r in sweep]),
                ('FP/loop',[r['core_seconds']/r['seconds'] for r in sweep])):
                if not values:
                    raise ValueError('empty rate data')
                lines.append('  {} mean={:.6f}, range=[{:.6f}, {:.6f}]'.format(label,statistics.mean(values),min(values),max(values)))
        lines.append('{} normal atomic/packed time ratio={:.6f}'.format(scheme,normal_means['atomic']/normal_means['packed']))
    lines += ['', 'All captures retained; failures prohibit accepting the campaign. These checks do not prove a bottleneck or solver speedup.']
    return '\n'.join(lines)+'\n', all(checks)


def calibration_reference(folder):
    records=json.loads((folder/'records.json').read_text())
    metadata=json.loads((folder/'collection.json').read_text())
    if metadata != dict(mode='reuse-validation',elements=80000000):
        raise ValueError('operator campaign requires the fixed reuse control validation')
    report,ok=analyze_controls(records,metadata['elements'],True)
    if not ok or report != (folder/'summary.txt').read_text():
        raise ValueError('same-allocation control validation failed')
    for name,digest in json.loads((folder/'files.sha256.json').read_text()).items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
            raise ValueError('control capture hash mismatch')
    lines=['Same-allocation STREAM-style references; transported bytes / timed sweep, not an official STREAM score.']
    for mode in ('copy','triad'):
        values=[(r['read_bytes']+r['write_bytes'])/r['seconds']/1e9 for r in records if r['mode']==mode and r['kind']=='sweep']
        lines.append('{} transported GB/s mean={:.6f}, range=[{:.6f}, {:.6f}]'.format(mode,statistics.mean(values),min(values),max(values)))
    return json.loads((folder/'topology.json').read_text()),'\n'.join(lines)+'\n'


def collect(binary, out, calibration):
    control_topo,control_report=calibration_reference(calibration)
    topo,binary,env=prepare_capture(binary,out,128**3,'residual-reuse',('cmem_wb_data','cmem_wr_data'))
    if topo != control_topo:
        raise ValueError('controls and operators must share the same allocation/topology')
    (out/'control-reference.txt').write_text(control_report)
    scf,core=counter_events(topo)
    scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data','cmem_wr_data')]
    records=[]
    try:
        for group in groups():
            block,scheme,layout,unused_kind,unused_repeat=group[0]
            session=None
            try:
                session=OperatorSession(topo,binary,env,out,session_name(block,scheme,layout),scheme,layout)
                for block,scheme,layout,kind,repeat in group:
                    stem=out/'{}-{}-{}-b{}-r{}'.format(scheme,layout,kind,block,repeat)
                    index=session.index
                    output=session.capture(topo,binary,stem,kind,repeat)
                    row=dict(block=block,scheme=scheme,layout=layout,kind=kind,repeat=repeat,
                             session=session.name,session_index=index,identity=session.identity)
                    row.update(parse_output(output,scheme,layout,kind,repeat,warmup=30))
                    if kind!='normal':
                        mt,ft=(stem.with_suffix(s).read_text() for s in ('.scf.perf','.core.perf'))
                        m,f=parse_perf(mt,scf),parse_perf(ft,core)
                        row.update(read_bytes=32*m[scf[0]],write_bytes=m[scf[1]],fp_ops=sum(f.values()),
                                   component_write_bytes=32*m[scf[2]]+m[scf[3]],
                                   scf_seconds=window_seconds(mt,scf,1),core_seconds=window_seconds(ft,core,72))
                    records.append(row)
                    (out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
                    print('captured '+stem.name,flush=True)
                session.close()
                session=None
            finally:
                if session is not None: session.close(abort=True)
        report,ok=analyze(records)
        (out/'summary.txt').write_text(control_report+'\n'+report)
        print(control_report+'\n'+report,end='')
        if not ok: raise ValueError('operator campaign checks failed; retain every capture')
        print('ROOFLINE OPERATOR CAMPAIGN PASSED')
    finally:
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='files.sha256.json'}
        (out/'files.sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    args=parser.parse_args()
    try:
        collect(args.binary,args.out.resolve(),args.calibration.resolve())
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        print('ROOFLINE OPERATOR CAMPAIGN FAILED: '+str(error),file=sys.stderr)
        sys.exit(1)
