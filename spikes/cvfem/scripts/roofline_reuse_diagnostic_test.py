#!/usr/bin/env python3
"""Check process reuse, fresh counters, failure cleanup and complete collection locally."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import roofline_reuse_diagnostic as reuse
from roofline_calibration import counter_events
from roofline_sweep_validation_test import fixture


def collection_test(validate_reuse=False):
    base = fixture(80000000)
    topo = dict(cpus=list(range(72)),node=0,scf_cpu=0,pmu='nvidia_scf_pmu_0')
    faults = (None,'identity','coverage','capture','close')+ (('stability',) if validate_reuse else ())
    for fault in faults:
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'captures'
            sessions = []
            def prepare(binary,path,elements,mode,events):
                assert mode == ('reuse-validation' if validate_reuse else 'reuse-diagnostic')
                path.mkdir();(path/'environment.json').write_text('{}')
                return topo,Path('/binary'),{}
            class FakeSession:
                def __init__(self,topo,binary,env,path,name,elements):
                    assert env['CVFEM_CALIBRATION_WARMUP']=='30'
                    self.name,self.index,self.closed = name,0,False
                    self.identity = dict(pid=100+len(sessions),buffers=['0x1','0x2','0x3'])
                    sessions.append(self)
                def capture(self,topo,binary,stem,mode,elements,kind,repeat):
                    if fault=='capture':raise subprocess.TimeoutExpired('mock',1)
                    block=int(self.name.split('-b')[1].split('-')[0])
                    row=dict(next(r for r in base if (r['block'],r['mode'],r['kind'],r['repeat'])==(block,mode,kind,repeat)))
                    if fault=='stability' and (block,mode,kind,repeat)==(0,'copy','sweep',40):
                        row['write_bytes']*=1.3
                    scf,core=counter_events(topo);scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data','cmem_wr_data')]
                    for suffix,events,values,runtime in (('.scf.perf',scf,[row['read_bytes']/32,row['write_bytes'],0,row['write_bytes']],row['scf_seconds']),('.core.perf',core,[0,row['fp_ops']],row['core_seconds']*72)):
                        stem.with_suffix(suffix).write_text(''.join('{},,{}, {},{},,\n'.format(v,e,runtime*1e9,'4' if fault=='coverage' else '100.00') for e,v in zip(events,values)))
                    phase='sweep' if kind=='sweep' else 'sweep-empty'
                    output='CALIBRATION mode={} elements=80000000 repeats={} threads=72 seconds={} useful_bytes={} flops={} PASS\n'.format(mode,repeat,row['seconds'],80000000*(16 if mode=='copy' else 24),80000000*(0 if mode=='copy' else 2))
                    output+='PERF_PHASE phase={} controllers=2 transitions=4 PASS\n'.format(phase)
                    output+='PERF_CONTROL phase={} controllers=2 enable_seconds=0 disable_seconds=0 inner_seconds={} outer_seconds={}\n'.format(phase,row['seconds'],row['seconds'])
                    output+='CALIBRATION_WARMUP sweeps=30\nSESSION_DONE index={}\n'.format(self.index)
                    self.index+=1
                    if fault=='identity' and self.index==2:self.identity=dict(self.identity,pid=99999)
                    return output
                def close(self,abort=False):
                    self.closed=True
                    if fault=='close' and not abort:raise ValueError('mock shutdown failure')
            with patch.object(reuse,'prepare_capture',prepare),patch.object(reuse,'Session',FakeSession),contextlib.redirect_stdout(io.StringIO()) as log:
                if fault:
                    try:reuse.collect(Path('/binary'),out,validate_reuse=validate_reuse)
                    except (ValueError,subprocess.SubprocessError):pass
                    else:raise AssertionError('failed case accepted: '+fault)
                    assert 'VALIDATION PASSED' not in log.getvalue()
                    if fault=='stability':
                        assert 'stability=False' in (out/'summary.txt').read_text()
                else:
                    reuse.collect(Path('/binary'),out,validate_reuse=validate_reuse)
                    rows=json.loads((out/'records.json').read_text())
                    assert len(rows)==(60 if validate_reuse else 120) and len(sessions)==(12 if validate_reuse else 72)
                    assert sorted(s.index for s in sessions)==([5]*12 if validate_reuse else [1]*60+[5]*12)
                    if validate_reuse:
                        assert 'REUSE CONTROL VALIDATION PASSED' in log.getvalue() and 'DIAGNOSTIC COLLECTED' not in log.getvalue()
                    else:
                        assert 'REUSE DIAGNOSTIC COLLECTED' in log.getvalue() and 'VALIDATION PASSED' not in log.getvalue()
                    for invalid in (rows[:-1],rows+[rows[0]],list(reversed(rows))):
                        try:reuse.analyze_measurements(invalid,80000000,validate_reuse)
                        except ValueError:pass
                        else:raise AssertionError('incomplete/reordered diagnostic accepted')
                assert (out/'files.sha256.json').exists() and all(s.closed for s in sessions)
    print(('Reuse validation: 60 captures, 12 sessions, unchanged stability gate and failure refusal' if validate_reuse else
           'Reuse collection: 120 captures, 72 sessions, identity/completeness/coverage/timeout/shutdown refusal')+' PASS (synthetic counters)')


MOCK = '''import os,select,sys,time
fault=sys.argv[1]
fds=list(map(int,sys.argv[2:]))
peers=[(fds[0],fds[1]),(fds[2],fds[3])]
if fault=='exit':sys.exit(1)
if fault=='silent':time.sleep(60)
messages=[[],[]]
pending={p[0]:b'' for p in peers}
while sum(map(len,messages))<4:
 ready=select.select(list(pending),[],[],6)[0]
 if not ready:sys.exit(2)
 for fd in ready:
  pending[fd]+=os.read(fd,128)
  while b'\\n' in pending[fd]:
   line,pending[fd]=pending[fd].split(b'\\n',1)
   i=next(i for i,p in enumerate(peers) if p[0]==fd)
   messages[i].append(line)
   reply=b'bad\\n\\0' if fault=='bad' and i==1 else b'ack\\n\\0'
   for piece in (reply[:2],reply[2:]):os.write(peers[i][1],piece)
assert messages==[[b'enable',b'disable'],[b'enable',b'disable']],messages
sys.stdin.buffer.read()
'''


def protocol_test(binary):
    topo=dict(cpus=[0,1],node=0,scf_cpu=0,pmu='mock')
    env=dict(os.environ,OMP_NUM_THREADS='2',OMP_DYNAMIC='FALSE',CVFEM_CALIBRATION_WARMUP='30')
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp);mock=out/'mock.py';mock.write_text(MOCK)
        for fault in (None,'bad','exit','silent'):
            session=reuse.Session(topo,binary,env,out,'test-'+(fault or 'good'),4096,threads=2,launcher=False)
            descriptors=list(session.fds)
            try:
                if fault:
                    try:session.capture(topo,binary,out/('bad-'+fault),'copy',4096,'sweep',10,
                            monitor_prefix=[sys.executable,str(mock),fault],timeout=0.5 if fault=='silent' else 10)
                    except (ValueError,subprocess.SubprocessError):pass
                    else:raise AssertionError('failed controller accepted: '+fault)
                    assert session.process.poll() is not None
                else:
                    for index,(mode,kind,repeat) in enumerate((('copy','empty-before',10),('copy','sweep',10),
                            ('copy','sweep',40),('copy','sweep',70),('triad','sweep',40),('triad','empty-after',10))):
                        output=session.capture(topo,binary,out/('good-'+str(index)),mode,4096,kind,repeat,
                                               monitor_prefix=[sys.executable,str(mock),'good'])
                        assert 'SESSION_DONE index='+str(index) in output
                        assert 'CALIBRATION mode='+mode in output and 'CALIBRATION_WARMUP sweeps=30' in output
                        assert 'controllers=2 transitions=4 PASS' in output
                    session.close()
                    assert session.process.returncode==0
            finally:session.close(abort=True)
            for fd in descriptors:
                try:os.fstat(fd)
                except OSError:pass
                else:raise AssertionError('leaked session descriptor')
        for warmups in ('3','0','nan'):
            try:reuse.Session(topo,binary,dict(env,CVFEM_CALIBRATION_WARMUP=warmups),out,'bad-warmup',4096,threads=2,launcher=False)
            except ValueError:pass
            else:raise AssertionError('wrong warmup accepted')
        for request in ('copy 0 sweep\n','triad -1 sweep\n','copy 40 setup\n','copy 10 sweep extra\n'):
            session=reuse.Session(topo,binary,env,out,'invalid-request',4096,threads=2,launcher=False)
            try:
                session.process.stdin.write(request.encode());session.process.stdin.close()
                assert session.process.wait(timeout=5)!=0
            finally:session.close(abort=True)
    print('Reuse protocol: six windows share one worker; fresh dual controllers, bad reply/early exit/timeout/invalid request and descriptor cleanup PASS (no PMU measurement)')


NESTED = '''import os,signal,subprocess,sys,threading,time
args=sys.argv[1:];split=args.index('--');command=args[split+1:]
own=next(a for a in args[:split] if a.startswith('--control=fd:'))
read,write=map(int,own.split(':')[1].split(','))
path=args[args.index('-o')+1];events=args[args.index('-e')+1][1:-1].split(',')
heartbeat=os.environ.get('MOCK_ORPHAN_HEARTBEAT')
if heartbeat and events[0]=='r80c0':
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 while True:
  with open(heartbeat,'w') as output:output.write(str(time.monotonic()))
  time.sleep(0.02)
inherited=[]
for a in command:
 if a.startswith('--control=fd:'):inherited.extend(map(int,a.split(':')[1].split(',')))
messages=[]
def control():
 pending=b''
 while len(messages)<2:
  pending+=os.read(read,128)
  while b'\\n' in pending:
   message,pending=pending.split(b'\\n',1);messages.append(message);os.write(write,b'ack\\n\\0')
thread=threading.Thread(target=control,daemon=True);thread.start()
child=subprocess.Popen(command,pass_fds=tuple(inherited))
if heartbeat:
 deadline=time.monotonic()+5
 while not os.path.exists(heartbeat) and time.monotonic()<deadline:time.sleep(0.01)
 assert os.path.exists(heartbeat)
 sys.exit(1)
code=child.wait();thread.join(timeout=6)
assert code==0 and messages==[b'enable',b'disable']
with open(path,'w') as output:
 for event in events:output.write('0,,{},1000000,100.00,,\\n'.format(event))
'''


def nested_test(binary):
    topo=dict(cpus=[0,1],node=0,scf_cpu=0,pmu='mock')
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp);perf=out/'perf';perf.write_text('#!'+sys.executable+'\n'+NESTED);perf.chmod(0o755)
        env=dict(os.environ,PATH=str(out)+os.pathsep+os.environ['PATH'],OMP_NUM_THREADS='2',
                 OMP_DYNAMIC='FALSE',CVFEM_CALIBRATION_WARMUP='30')
        session=reuse.Session(topo,binary,env,out,'nested-reuse',4096,threads=2,launcher=False)
        try:
            for index in range(3):
                stem=out/('nested-'+str(index))
                output=session.capture(topo,binary,stem,'triad',4096,'sweep',10)
                assert 'SESSION_DONE index='+str(index) in output
                assert stem.with_suffix('.scf.perf').exists() and stem.with_suffix('.core.perf').exists()
            session.close()
        finally:session.close(abort=True)
        heartbeat=out/'heartbeat'
        session=reuse.Session(topo,binary,dict(env,MOCK_ORPHAN_HEARTBEAT=str(heartbeat)),out,
                              'nested-failure',4096,threads=2,launcher=False)
        try:
            try:session.capture(topo,binary,out/'nested-failure','copy',4096,'sweep',10)
            except (ValueError,subprocess.SubprocessError):pass
            else:raise AssertionError('exited outer monitor accepted')
            assert heartbeat.exists()
            time.sleep(0.1);last=heartbeat.read_text();time.sleep(0.15)
            assert heartbeat.read_text()==last,'inner monitor survived its failed parent'
        finally:session.close(abort=True)
    print('Nested perf command: three windows restart both monitors; failed parent cleanup stops an inner monitor that ignores TERM PASS (mock perf)')


if __name__=='__main__':
    collection_test()
    collection_test(validate_reuse=True)
    if len(sys.argv)==2:
        binary=Path(sys.argv[1]).resolve()
        protocol_test(binary)
        nested_test(binary)
