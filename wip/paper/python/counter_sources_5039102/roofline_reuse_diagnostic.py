#!/usr/bin/env python3
"""Compare process lifetimes, or validate the fixed recipe that reuses each allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import statistics
import subprocess
import sys
import time

import roofline_sweep_validation as sweep
import roofline_warmup_diagnostic as warmup
from roofline_calibration import counter_events, parse_perf, prepare_capture


def groups(validate_reuse=False):
    original = list(sweep.cases())
    for block in range(6):
        modes = ('copy','triad') if block % 2 == 0 else ('triad','copy')
        for mode in modes:
            lifetimes = ('reuse',) if validate_reuse else (('fresh','reuse') if block % 2 == 0 else ('reuse','fresh'))
            for lifetime in lifetimes:
                yield lifetime, [case for case in original if case[:2] == (block,mode)]


def cases(validate_reuse=False):
    return [case+(lifetime,) for lifetime, group in groups(validate_reuse) for case in group]


def session_name(case, lifetime):
    block, mode, kind, repeat = case
    name = '{}-b{}-{}'.format(mode,block,lifetime)
    return name if lifetime == 'reuse' else name+'-{}-r{}'.format(kind,repeat)


def stop(process):
    if process is None:
        return
    # Reap the leader first; its private group may still contain an inner perf.
    process.poll()
    def send(sig):
        try:
            os.killpg(process.pid,sig)
        except PermissionError:
            # Reap an exited leader before retrying a rejected group signal.
            if process.poll() is None:
                raise
            os.killpg(process.pid,sig)
    try:
        send(signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        send(signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class Session:
    def __init__(self, topo, binary, env, out, name, elements, threads=72, launcher=True):
        self.fds, self.process, self.pending = [], None, b''
        self.name, self.index, self.env = name, 0, env
        try:
            for unused in range(4):
                self.fds.extend(os.pipe())
            command = self.worker_command(binary,elements,threads)
            if launcher:
                command = ['numactl','--physcpubind='+','.join(map(str,topo['cpus'])),
                           '--membind='+str(topo['node'])]+command
            (out/(name+'.session.json')).write_text(json.dumps(dict(command=command,
                warmup=30, lifetime='reuse' if name.endswith('-reuse') else 'fresh'))+'\n')
            self.process = subprocess.Popen(command,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,pass_fds=tuple(self.fds[i] for i in (1,2,5,6)),
                bufsize=0,start_new_session=True)
            ready = self.read_until(b'SESSION_READY ',prefix=True)
            (out/(name+'.ready.out')).write_text(ready)
            marker = ready.splitlines()[-1]
            match = re.fullmatch(r'SESSION_READY pid=(\d+) a=(0x[0-9a-f]+) b=(0x[0-9a-f]+) c=(0x[0-9a-f]+)',marker)
            if not match or int(match.group(1)) != self.process.pid:
                raise ValueError('invalid session identity')
            self.identity = dict(pid=int(match.group(1)),buffers=list(match.groups()[1:]))
        except BaseException:
            self.close(abort=True)
            raise

    def worker_command(self, binary, elements, threads):
        return [str(binary),'session',str(elements),str(threads)]+[str(self.fds[i]) for i in (1,2,5,6)]

    def read_until(self, marker, prefix=False, monitor=None, timeout=60):
        deadline, lines = time.monotonic()+timeout, []
        self.partial_output = self.pending
        while True:
            while b'\n' in self.pending:
                line,self.pending = self.pending.split(b'\n',1)
                lines.append(line+b'\n')
                if (line.startswith(marker) if prefix else line == marker):
                    return b''.join(lines).decode('utf-8')
            if monitor is not None and monitor.poll() is not None:
                raise ValueError('perf monitor stopped before the session finished')
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired('calibration session',timeout)
            if select.select([self.process.stdout],[],[],min(0.1,max(0,deadline-time.monotonic())))[0]:
                data = os.read(self.process.stdout.fileno(),65536)
                if not data:
                    raise ValueError('session ended early: '+b''.join(lines).decode('utf-8',errors='replace'))
                self.pending += data
                self.partial_output += data
                if sum(map(len,lines))+len(self.pending) > 1048576:
                    raise ValueError('session output exceeds limit')

    def capture(self, topo, binary, stem, mode, elements, kind, repeat, monitor_prefix=None, timeout=60):
        # These counters are socket/CPU-wide, so the worker need not be perf's child.
        command = sweep.capture_command(topo,binary,stem,mode,elements,kind,repeat,self.fds)
        command = command[:command.index('numactl')]+['cat']
        if monitor_prefix is not None:
            command = list(monitor_prefix)+[str(self.fds[i]) for i in (0,3,4,7)]
        request = '{} {} {}\n'.format(mode,repeat,'sweep' if kind == 'sweep' else 'sweep-empty')
        return self.exchange(stem,request,command,timeout)

    def exchange(self, stem, request, command, timeout=60):
        stem.with_suffix('.command.json').write_text(json.dumps(command)+'\n')
        stem.with_suffix('.request.json').write_text(json.dumps(dict(session=self.name,
            index=self.index,request=request,identity=self.identity))+'\n')
        monitor = None
        try:
            with stem.with_suffix('.monitor.log').open('w') as log:
                if command is not None:
                    monitor = subprocess.Popen(command,env=self.env,stdin=subprocess.PIPE,stdout=log,
                        stderr=subprocess.STDOUT,pass_fds=tuple(self.fds),start_new_session=True)
                self.process.stdin.write(request.encode())
                output = self.read_until(('SESSION_DONE index='+str(self.index)).encode(),
                                         monitor=monitor,timeout=timeout)
                stem.with_suffix('.out').write_text(output)
                expected = 'SESSION_BUFFERS pid={} a={} b={} c={}'.format(
                    self.identity['pid'],*self.identity['buffers'])
                if [line for line in output.splitlines() if line.startswith('SESSION_BUFFERS ')] != [expected]:
                    raise ValueError('session process or arrays changed')
                if monitor is not None:
                    monitor.stdin.close()
                    code = monitor.wait(timeout=timeout)
                    if code:
                        raise subprocess.CalledProcessError(code,command)
                self.index += 1
                return output
        except BaseException:
            if not stem.with_suffix('.out').exists():
                stem.with_suffix('.out').write_text(self.partial_output.decode('utf-8',errors='replace'))
            stop(monitor)
            self.close(abort=True)
            raise
        finally:
            if monitor is not None and monitor.stdin is not None:
                monitor.stdin.close()

    def close(self, abort=False):
        try:
            if self.process is not None:
                if not abort and self.process.poll() is None:
                    self.process.stdin.close()
                    if self.process.wait(timeout=10):
                        raise ValueError('session failed at shutdown')
                elif not abort and self.process.returncode:
                    raise ValueError('session stopped unexpectedly')
        finally:
            stop(self.process)
            if self.process is not None:
                self.process.stdin.close()
                self.process.stdout.close()
            for fd in self.fds:
                os.close(fd)
            self.fds = []


def analyze_measurements(records, elements, validate_reuse=False):
    keys = ('block','mode','kind','repeat','lifetime')
    actual = [tuple(row[k] for k in keys) for row in records]
    if actual != cases(validate_reuse):
        raise ValueError('incomplete, duplicate or reordered reuse diagnostic')
    sessions = {}
    for row in records:
        case = tuple(row[k] for k in keys[:4])
        name = session_name(case,row['lifetime'])
        if row['session'] != name:
            raise ValueError('incorrect session assignment')
        entries = sessions.setdefault(name,[])
        if row['session_index'] != len(entries) or (entries and row['identity'] != entries[0]['identity']):
            raise ValueError('process or buffers changed within a reuse session')
        entries.append(row)
    lines = ['Process/array reuse diagnostic only; no calibration acceptance.',
        'Both arms initialize the same arrays and execute 30 warmups before every capture.',
        'Fresh: one worker per capture. Reuse: one worker/allocation per five-capture block.',
        'Both arms use fresh perf monitors for every window; CPU/event scopes are unchanged.',
        'Six repeat-order permutations; lifetime order alternates by block.',
        'No discarded samples, count adjustment or changed acceptance limits.',
        'Ranges show sample variation, not confidence intervals.','']
    if validate_reuse:
        lines = ['Control validation with reused processes and arrays.',
            'One worker/allocation per five-capture block; initialization and 30 warmups before each capture.',
            'Six repeat-order permutations; 12 workers, 36 sweep captures and 24 empty controls.',
            'Every window opens fresh perf monitors with the same CPU/event scopes and acknowledged controls.',
            'No sample deletion, count adjustment or changed acceptance limits.',
            'Ranges show sample variation, not confidence intervals.','']
    accepted = True
    for lifetime in (('reuse',) if validate_reuse else ('fresh','reuse')):
        rows = [r for r in records if r['lifetime'] == lifetime]
        report,ok = sweep.analyze(rows,elements)
        accepted = accepted and ok
        lines += ['lifetime={}: all unchanged screens pass={}'.format(lifetime,ok),report.rstrip(),'']
        for mode in ('copy','triad'):
            selected = [r for r in rows if r['mode'] == mode and r['kind'] == 'sweep']
            for repeat in (10,40,70):
                values = [r for r in selected if r['repeat'] == repeat]
                writes = [r['write_bytes']/repeat/1e6 for r in values]
                times = [1000*r['seconds']/repeat for r in values]
                lines.append('{} {} r{}: write MB/sweep mean={:.6f} range=[{:.6f},{:.6f}]; ms/sweep range=[{:.6f},{:.6f}]'.format(
                    lifetime,mode,repeat,statistics.mean(writes),min(writes),max(writes),min(times),max(times)))
            slopes = []
            for block in range(6):
                by = {r['repeat']:r for r in selected if r['block'] == block}
                slopes += [(by[hi]['write_bytes']-by[lo]['write_bytes'])/(hi-lo) for lo,hi in ((10,40),(40,70))]
            lines.append('{} {}: write slope spread={:.6f}%'.format(lifetime,mode,100*sweep.spread(slopes)))
    if validate_reuse:
        lines += ['', 'A pass validates these copy/triad controls and this reuse configuration in this allocation only.',
                  'It does not accept the earlier fresh-process recipe, a CVFEM operator or a paper performance claim.']
    else:
        lines += ['', 'A difference supports a process/allocation-history effect, not a specific hardware cause.',
                  'Initialization and warmup are retained in both arms; physical pages may still migrate.',
                  'Review both arms before a separate validation. No operator measurement is accepted here.']
    return '\n'.join(lines)+'\n',accepted


def analyze(records, elements):
    return analyze_measurements(records,elements)[0]


def collect(binary, out, elements=80000000, validate_reuse=False):
    mode = 'reuse-validation' if validate_reuse else 'reuse-diagnostic'
    topo,binary,env = prepare_capture(binary,out,elements,mode,('cmem_wb_data','cmem_wr_data'))
    env['CVFEM_CALIBRATION_WARMUP'] = '30'
    environment = json.loads((out/'environment.json').read_text())
    environment['CVFEM_CALIBRATION_WARMUP'] = '30'
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    scf,core = counter_events(topo)
    scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data','cmem_wr_data')]
    records,session = [],None
    try:
        for lifetime,group in groups(validate_reuse):
            try:
                for case in group:
                    block,mode,kind,repeat = case
                    if session is None:
                        session = Session(topo,binary,env,out,session_name(case,lifetime),elements)
                    stem = out/'{}-{}-b{}-r{}-{}'.format(mode,kind,block,repeat,lifetime)
                    index = session.index
                    output = session.capture(topo,binary,stem,mode,elements,kind,repeat)
                    row = dict(block=block,mode=mode,kind=kind,repeat=repeat,lifetime=lifetime,
                               session=session.name,session_index=index,identity=session.identity)
                    row.update(warmup.parse_output(output,mode,elements,repeat,kind,30))
                    mt,ft = (stem.with_suffix(s).read_text() for s in ('.scf.perf','.core.perf'))
                    m,f = parse_perf(mt,scf),parse_perf(ft,core)
                    row.update(scf_seconds=sweep.window_seconds(mt,scf,1),core_seconds=sweep.window_seconds(ft,core,72),
                        read_bytes=32*m[scf[0]],write_bytes=m[scf[1]],fp_ops=sum(f.values()),
                        writeback_beats=m[scf[2]],other_write_bytes=m[scf[3]],component_write_bytes=32*m[scf[2]]+m[scf[3]])
                    records.append(row)
                    (out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
                    print('captured '+stem.name,flush=True)
                    if lifetime == 'fresh':
                        session.close()
                        session = None
                if session is not None:
                    session.close()
                    session = None
            finally:
                if session is not None:
                    session.close(abort=True)
                    session = None
        report,accepted = analyze_measurements(records,elements,validate_reuse)
        (out/'summary.txt').write_text(report)
        print(report,end='')
        if validate_reuse:
            if not accepted:
                raise ValueError('reuse validation checks failed; retain logs before operator runs')
            print('REUSE CONTROL VALIDATION PASSED; operator measurements remain pending')
        else:
            print('REUSE DIAGNOSTIC COLLECTED; calibration and operators remain unvalidated')
    finally:
        hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
                  if p.is_file() and p.name != 'files.sha256.json'}
        (out/'files.sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--validate-reuse',action='store_true',help='validate 60 new captures with the fixed reuse recipe')
    args = parser.parse_args()
    try:
        collect(args.binary,args.out.resolve(),validate_reuse=args.validate_reuse)
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        label = 'REUSE CONTROL VALIDATION' if args.validate_reuse else 'REUSE DIAGNOSTIC'
        print(label+' FAILED: '+str(error),file=sys.stderr)
        sys.exit(1)
