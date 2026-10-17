#!/usr/bin/env python3
"""Compare three and thirty warmups without accepting a calibration or running operators."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from unittest.mock import patch

import roofline_sweep_validation as sweep
from roofline_calibration import counter_events, control_seconds, parse_perf, prepare_capture


def cases():
    for index,case in enumerate(sweep.cases()):
        for warmup in ((3,30) if index % 2 == 0 else (30,3)):
            yield case+(warmup,)


def parse_output(output, mode, elements, repeat, kind, warmup):
    seconds=control_seconds(output,mode,elements,repeat)
    marker='CALIBRATION_WARMUP sweeps='+str(warmup)
    rows=[line for line in output.splitlines() if line.startswith('CALIBRATION_WARMUP ')]
    if rows != [marker]:
        raise ValueError('missing or incorrect warmup count')
    phase='sweep' if kind=='sweep' else 'sweep-empty'
    if output.splitlines().count('PERF_PHASE phase='+phase+' controllers=2 transitions=4 PASS')!=1:
        raise ValueError('both counter controllers must acknowledge the window')
    return dict(seconds=seconds,control_timing=sweep.control_timing(output,phase))


def analyze(records,elements):
    actual=[tuple(r[k] for k in ('block','mode','kind','repeat','warmup')) for r in records]
    if len(actual)!=len(set(actual)) or set(actual)!=set(cases()):
        raise ValueError('incomplete or duplicate warmup diagnostic')
    lines=['Warmup diagnostic only: 3 versus 30 sweeps; no calibration acceptance.',
           'Independent processes/allocations for each capture; warmup order alternates within pairs.',
           'Six blocks, every permutation of 10/40/70 timed sweeps, copy and triad.',
           'No sample deletion, count correction, subtraction or changed acceptance thresholds.',
           'Ranges show sample variation, not confidence intervals.', '']
    for warmup in (3,30):
        rows=[r for r in records if r['warmup']==warmup]
        original,ok=sweep.analyze(rows,elements)
        lines += ['warmups={}: all unchanged screens pass={}'.format(warmup,ok),original.rstrip(),'']
        for mode in ('copy','triad'):
            values=[r for r in rows if r['mode']==mode and r['kind']=='sweep']
            for repeat in (10,40,70):
                selected=[r for r in values if r['repeat']==repeat]
                writes=[r['write_bytes']/repeat for r in selected]
                times=[1000*r['seconds']/repeat for r in selected]
                lines.append('{} warmup={} repeat={}: write MB/sweep mean={:.6f} range=[{:.6f},{:.6f}]; ms/sweep mean={:.6f} range=[{:.6f},{:.6f}]'.format(
                    mode,warmup,repeat,statistics.mean(writes)/1e6,min(writes)/1e6,max(writes)/1e6,
                    statistics.mean(times),min(times),max(times)))
            slopes=[]
            for block in range(6):
                by={r['repeat']:r for r in values if r['block']==block}
                for lo,hi in ((10,40),(40,70)):
                    entry=dict(block=block,lo=lo,hi=hi)
                    for k in ('read_bytes','write_bytes','writeback_beats','other_write_bytes'):
                        entry[k]=(by[hi][k]-by[lo][k])/(hi-lo)
                    slopes.append(entry)
            lines.append('{} warmup={}: write slope spread={:.6f}%'.format(mode,warmup,100*sweep.spread([r['write_bytes'] for r in slopes])))
            for tag,row in (('min',min(slopes,key=lambda r:r['write_bytes'])),('max',max(slopes,key=lambda r:r['write_bytes']))):
                lines.append('  {} block={} repeats={}->{} write MB/sweep={:.6f} (writeback={:.6f}, other={:.6f})'.format(tag,row['block'],row['lo'],row['hi'],row['write_bytes']/1e6,32*row['writeback_beats']/1e6,row['other_write_bytes']/1e6))
    lines += ['', 'Paired 30-minus-3 warmup differences in MB written per timed sweep:']
    for mode in ('copy','triad'):
        for repeat in (10,40,70):
            differences=[]
            for block in range(6):
                by={r['warmup']:r for r in records if r['block']==block and r['mode']==mode and r['kind']=='sweep' and r['repeat']==repeat}
                differences.append((by[30]['write_bytes']-by[3]['write_bytes'])/repeat/1e6)
            lines.append('{} repeat={}: mean={:+.6f}, range=[{:+.6f},{:+.6f}], positive={}/6'.format(mode,repeat,statistics.mean(differences),min(differences),max(differences),sum(x>0 for x in differences)))
    lines += ['', 'A reduced repeat-count trend with more warmup would support a settling effect, not identify a hardware mechanism.',
              'A persistent trend or spread leaves process/allocation effects and counter behavior unresolved.',
              'Both outcomes require review; this diagnostic cannot accept a roofline or start operator runs.']
    return '\n'.join(lines)+'\n'


def collect(binary,out,elements):
    topo,binary,env=prepare_capture(binary,out,elements,'warmup-diagnostic',('cmem_wb_data','cmem_wr_data'))
    scf,core=counter_events(topo)
    scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data','cmem_wr_data')]
    records=[]
    try:
        for block,mode,kind,repeat,warmup in cases():
            stem=out/'{}-{}-b{}-r{}-w{}'.format(mode,kind,block,repeat,warmup)
            fds=[]
            try:
                for unused in range(4):fds.extend(os.pipe())
                command=sweep.capture_command(topo,binary,stem,mode,elements,kind,repeat,fds)
                capture_env=dict(env,CVFEM_CALIBRATION_WARMUP=str(warmup))
                stem.with_suffix('.command.json').write_text(json.dumps(command)+'\n')
                stem.with_suffix('.environment.json').write_text(json.dumps(dict(CVFEM_CALIBRATION_WARMUP=str(warmup)))+'\n')
                sweep.run_capture(command,capture_env,stem.with_suffix('.out'),fds)
            finally:
                for fd in fds:os.close(fd)
            output=stem.with_suffix('.out').read_text()
            row=dict(block=block,mode=mode,kind=kind,repeat=repeat,warmup=warmup)
            row.update(parse_output(output,mode,elements,repeat,kind,warmup))
            mt,ft=(stem.with_suffix(s).read_text() for s in ('.scf.perf','.core.perf'))
            m,f=parse_perf(mt,scf),parse_perf(ft,core)
            row.update(scf_seconds=sweep.window_seconds(mt,scf,1),core_seconds=sweep.window_seconds(ft,core,72),
                       read_bytes=32*m[scf[0]],write_bytes=m[scf[1]],fp_ops=sum(f.values()),
                       writeback_beats=m[scf[2]],other_write_bytes=m[scf[3]],
                       component_write_bytes=32*m[scf[2]]+m[scf[3]])
            records.append(row)
            (out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
            print('captured '+stem.name,flush=True)
        report=analyze(records,elements)
        (out/'summary.txt').write_text(report)
        print(report,end='')
        print('WARMUP DIAGNOSTIC COLLECTED; calibration and operators remain unvalidated')
    finally:
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='files.sha256.json'}
        (out/'files.sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')


def selftest():
    from roofline_sweep_validation_test import fixture
    base=fixture(80000000)
    records=[]
    for warmup in (3,30):
        for row in base:
            record=dict(row,warmup=warmup,writeback_beats=0,other_write_bytes=row['write_bytes'])
            records.append(record)
    report=analyze(records,80000000)
    assert 'calibration acceptance' in report and 'write slope spread=0.000000%' in report
    for invalid in (records[:-1],records+[records[0]]):
        try:analyze(invalid,80000000)
        except ValueError:pass
        else:raise AssertionError('invalid diagnostic accepted')
    topo=dict(cpus=list(range(72)),node=0,scf_cpu=0,pmu='nvidia_scf_pmu_0')
    with tempfile.TemporaryDirectory() as temp:
        for fault in (None,'warmup','coverage','command','timeout'):
            out=Path(temp)/(fault or 'good');seen=[]
            def prepare(binary,path,elements,mode,extra):
                path.mkdir();return topo,Path('/binary'),{}
            def capture(command,env,log,fds):
                case=list(cases())[len(seen)];seen.append(tuple(fds))
                block,mode,kind,repeat,warmup=case
                for fd in fds:os.fstat(fd)
                assert env['CVFEM_CALIBRATION_WARMUP']==str(warmup)
                if fault=='command':raise subprocess.CalledProcessError(1,command)
                if fault=='timeout':raise subprocess.TimeoutExpired(command,60)
                row=next(r for r in records if tuple(r[k] for k in ('block','mode','kind','repeat','warmup'))==case)
                phase='sweep' if kind=='sweep' else 'sweep-empty'
                output='CALIBRATION mode={} elements=80000000 repeats={} threads=72 seconds={} useful_bytes={} flops={} PASS\n'.format(mode,repeat,row['seconds'],80000000*(16 if mode=='copy' else 24),80000000*(0 if mode=='copy' else 2))
                output+='PERF_PHASE phase={} controllers=2 transitions=4 PASS\n'.format(phase)
                output+='PERF_CONTROL phase={} controllers=2 enable_seconds=0 disable_seconds=0 inner_seconds={} outer_seconds={}\n'.format(phase,row['seconds'],row['seconds'])
                output+='CALIBRATION_WARMUP sweeps={}\n'.format(0 if fault=='warmup' else warmup)
                log.write_text(output)
                scf,core=counter_events(topo);scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data','cmem_wr_data')]
                for suffix,events,values,runtime in (('.scf.perf',scf,[row['read_bytes']/32,row['write_bytes'],0,row['write_bytes']],row['scf_seconds']),('.core.perf',core,[0,row['fp_ops']],row['core_seconds']*72)):
                    log.with_suffix(suffix).write_text(''.join('{},,{}, {},{},,\n'.format(value,event,runtime*1e9,'4' if fault=='coverage' else '100.00') for event,value in zip(events,values)))
            with patch.object(sys.modules[__name__],'prepare_capture',prepare),patch.object(sweep,'run_capture',capture),contextlib.redirect_stdout(io.StringIO()) as output:
                if fault:
                    try:collect(Path('/binary'),out,80000000)
                    except (ValueError,subprocess.SubprocessError):pass
                    else:raise AssertionError('failed capture accepted')
                else:
                    collect(Path('/binary'),out,80000000)
                    assert len(json.loads((out/'records.json').read_text()))==120
                    assert 'DIAGNOSTIC COLLECTED' in output.getvalue() and 'VALIDATION PASSED' not in output.getvalue()
                assert (out/'files.sha256.json').exists()
            for fds in seen:
                for fd in fds:
                    try:os.fstat(fd)
                    except OSError:pass
                    else:raise AssertionError('descriptor leaked')
    print('Warmup diagnostic: paired cases, complete collection, wrong warmup/coverage/failure refusal and descriptor cleanup PASS (no PMU measurement)')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--selftest',action='store_true')
    args=parser.parse_args()
    if args.selftest:selftest();sys.exit(0)
    if args.binary is None or args.out is None:parser.error('--binary and --out required')
    try:collect(args.binary,args.out.resolve(),80000000)
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        print('WARMUP DIAGNOSTIC FAILED: '+str(error),file=sys.stderr);sys.exit(1)
