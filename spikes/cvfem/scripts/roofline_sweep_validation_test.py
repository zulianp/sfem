#!/usr/bin/env python3
"""Reject broken accounting and collection paths without needing Grace counters."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
from unittest.mock import patch

import roofline_sweep_validation as validation


def fixture(elements=100):
    rows = []
    for block,mode,kind,repeat in validation.cases():
        empty = kind != 'sweep'
        rows.append(dict(block=block,mode=mode,kind=kind,repeat=repeat,
            seconds=repeat*0.003,scf_seconds=0.00001 if empty else repeat*0.003+0.0001,
            core_seconds=0.00001 if empty else repeat*0.003+0.0002,
            read_bytes=1 if empty else elements*(8 if mode == 'copy' else 16)*repeat,
            write_bytes=1 if empty else elements*8*repeat,
            component_write_bytes=1 if empty else elements*8*repeat,
            fp_ops=1 if empty else elements*(0 if mode == 'copy' else 2)*repeat))
    return rows


def selftest():
    rows = fixture()
    assert validation.analyze(rows,100)[1]
    timing_line = 'PERF_CONTROL phase=sweep controllers=2 enable_seconds=0.02 disable_seconds=0.003 inner_seconds=0.2 outer_seconds=0.223\n'
    assert validation.control_timing(timing_line,'sweep')['enable_seconds'] == 0.02
    for invalid in ('',timing_line+timing_line,timing_line.replace('0.223','0.1'),
                    timing_line.replace('0.02','nan'),timing_line.replace('controllers=2','controllers=1')):
        try:
            validation.control_timing(invalid,'sweep')
        except ValueError:
            pass
        else:
            raise AssertionError('accepted missing or invalid control timing')
    # Reproduce the user's timing-window failure without changing its acceptance.
    reported = dict(mode='triad',block=0,repeat=40,seconds=0.198390003,
                    scf_seconds=221931200/1e9,core_seconds=15898921088/72/1e9)
    report = '\n'.join(validation.window_report([reported]))
    assert 'FAILED triad-sweep-b0-r40' in report and '1.118661x' in report and '1.113052x' in report
    assert 'unavailable' in report
    for invalid in (rows[:-1],rows+[rows[0]]):
        try:
            validation.analyze(invalid,100)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted incomplete or duplicate campaign')
    for key,value,index in (('fp_ops',10000,1),('read_bytes',1,1),('write_bytes',100000,1),
                             ('component_write_bytes',0,1),('scf_seconds',1,1),
                             ('core_seconds',0.001,1),('fp_ops',10000,0),('read_bytes',10000,0)):
        bad = copy.deepcopy(rows)
        bad[index][key] = value
        assert not validation.analyze(bad,100)[1], (key,value,index)
    # A constant offset disappears from slopes but must fail direct accounting.
    bad = copy.deepcopy(rows)
    for row in bad:
        if row['kind'] == 'sweep':
            row['fp_ops'] += 10000
    assert not validation.analyze(bad,100)[1]
    # A repeat-dependent error can pass the broad bounds but fail slope stability.
    bad = copy.deepcopy(rows)
    row = next(r for r in bad if r['block'] == 0 and r['mode'] == 'copy' and r['repeat'] == 70)
    row['read_bytes'] *= 1.09
    assert not validation.analyze(bad,100)[1]
    topo = dict(cpus=list(range(72,144)),scf_cpu=72,pmu='nvidia_scf_pmu_1',node=1)
    command = validation.capture_command(topo,Path('/binary'),Path('/capture'),'triad',100,'sweep',40,tuple(range(11,19)))
    split = command.index('--')
    outer,inner = command[:split],command[split+1:]
    split = inner.index('--')
    inner,work = inner[:split],inner[split+1:]
    assert outer[outer.index('-C')+1] == '72' and '--control=fd:11,14' in outer
    assert inner[inner.index('-C')+1] == ','.join(map(str,range(72,144))) and '--control=fd:15,18' in inner
    assert '--delay=-1' in outer and '--delay=-1' in inner
    assert work[-10:] == ['/binary','triad','100','40','72','sweep','12','13','16','17']
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for fault in (None,'coverage','marker','timing','command','timeout'):
            out = root/(fault or 'good')
            seen = []
            def prepare(binary,path,elements,mode,extra):
                path.mkdir()
                return topo,Path('/binary'),{}
            def capture(command,env,log,fds):
                assert len(fds) == len(set(fds)) == 8
                for fd in fds:
                    os.fstat(fd)
                seen.append(tuple(fds))
                block,mode,kind,repeat = list(validation.cases())[len(seen)-1]
                row = next(r for r in rows if (r['block'],r['mode'],r['kind'],r['repeat']) == (block,mode,kind,repeat))
                if fault == 'command':
                    raise subprocess.CalledProcessError(1,command)
                if fault == 'timeout':
                    raise subprocess.TimeoutExpired(command,60)
                phase = 'sweep' if kind == 'sweep' else 'sweep-empty'
                marker = 'PERF_PHASE phase='+phase+' controllers=2 transitions=4 PASS\n'
                timing = 'PERF_CONTROL phase={} controllers=2 enable_seconds=0.001 disable_seconds=0.001 inner_seconds={} outer_seconds={}\n'.format(
                    phase,row['seconds'],row['seconds']+0.002)
                log.write_text('CALIBRATION mode={} elements=100 repeats={} threads=72 seconds={} useful_bytes={} flops={} PASS\n'.format(
                    mode,repeat,row['seconds'],100*(16 if mode == 'copy' else 24),100*(0 if mode == 'copy' else 2))+
                    ('' if fault == 'marker' else marker)+('' if fault == 'timing' else timing))
                scf,core = validation.counter_events(topo)
                scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data','cmem_wr_data')]
                for suffix,events,counts,runtime in (
                        ('.scf.perf',scf,[row['read_bytes']/32,row['write_bytes'],0,row['write_bytes']],row['scf_seconds']*1e9),
                        ('.core.perf',core,[row['fp_ops'],0],row['core_seconds']*1e9*72)):
                    coverage = '4.00' if fault == 'coverage' else '100.00'
                    log.with_suffix(suffix).write_text(''.join('{},,{}, {},{},,\n'.format(c,e,runtime,coverage) for e,c in zip(events,counts)))
            with patch.object(validation,'prepare_capture',prepare), patch.object(validation,'run_capture',capture), contextlib.redirect_stdout(io.StringIO()):
                try:
                    validation.collect(Path('/binary'),out,100)
                except (ValueError,subprocess.SubprocessError):
                    assert fault is not None
                    assert len(seen) == 1 and not (out/'summary.txt').exists()
                else:
                    assert fault is None
                    assert len(seen) == 60 and len(json.loads((out/'records.json').read_text())) == 60
                    assert validation.analyze(json.loads((out/'records.json').read_text()),100)[1]
            for fd in seen[-1]:
                try:
                    os.fstat(fd)
                except OSError:
                    pass
                else:
                    raise AssertionError('collector leaked pipe descriptor')
        # Exercise real process-group cleanup, not only an injected timeout.
        pid_file = root/'pids.txt'
        script = ('import os,signal,sys,time\n'
                  'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                  'child=os.fork()\n'
                  'if child == 0:\n'
                  ' time.sleep(30)\n'
                  'else:\n'
                  ' open(sys.argv[1],"w").write(str(os.getpid())+" "+str(child))\n'
                  ' time.sleep(30)\n')
        reader,writer = os.pipe()
        try:
            try:
                validation.run_capture([sys.executable,'-c',script,str(pid_file)],dict(os.environ),root/'timeout.log',[writer],timeout=0.5)
            except subprocess.TimeoutExpired:
                assert pid_file.exists()
            else:
                raise AssertionError('timeout was not propagated')
        finally:
            os.close(writer)
        try:
            # EOF requires both parent and child to release their inherited writer.
            assert select.select([reader],[],[],2)[0], 'nested workload survived timeout'
            assert os.read(reader,1) == b''
        finally:
            os.close(reader)
    print('Sweep validation: accounting, scopes, 60-capture collection, refusal paths and process cleanup PASS (no PMU measurement)')


if __name__ == '__main__':
    selftest()
