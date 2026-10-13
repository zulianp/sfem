#!/usr/bin/env python3
"""Check operator accounting and optional local handshake/result gates without a PMU."""
import copy
import contextlib
import io
import hashlib
from unittest.mock import patch
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import roofline_operator as op
from roofline_reuse_diagnostic_test import MOCK, NESTED
from roofline_sweep_validation_test import fixture
import roofline_reuse_diagnostic as reuse


def refused(call):
    try:
        call()
    except (ValueError,KeyError,AssertionError):
        return
    raise AssertionError('invalid evidence accepted')


def synthetic_records(repeat_counts=op.REPEAT_COUNTS):
    result=[]
    indices={}
    for block,scheme,layout,kind,repeat in op.cases(repeat_counts):
        row=dict(block=block,scheme=scheme,layout=layout,kind=kind,repeat=repeat,seconds=.01*repeat)
        name=op.session_name(block,scheme,layout)
        index=indices.get(name,0);indices[name]=index+1
        row.update(session=name,session_index=index,identity=dict(pid=len(indices),buffers=['0x1','0x2','0x3']))
        if kind!='normal':
            factor=repeat if kind=='sweep' else .001
            row.update(read_bytes=2e8*factor,write_bytes=1e8*factor,fp_ops=4e8*factor,
                       component_write_bytes=1e8*factor,scf_seconds=.01*repeat,core_seconds=.01*repeat)
        result.append(row)
    return result


def accounting_tests():
    records=synthetic_records()
    assert len(records)==144 and op.analyze(records)[1]
    refused(lambda:op.analyze(records[:-1]))
    refused(lambda:op.analyze(records+[records[0]]))
    refused(lambda:op.analyze(list(reversed(records))))
    bad=copy.deepcopy(records);bad[1]['identity']['buffers'][0]='0xf'
    refused(lambda:op.analyze(bad))
    for key,value in (('read_bytes',float('nan')),('seconds',0)):
        bad=copy.deepcopy(records)
        row=next(r for r in bad if r['kind']=='sweep'); row[key]=value
        refused(lambda:op.analyze(bad))
    for kind,key,value in (('sweep','scf_seconds',99),('sweep','component_write_bytes',0),
                            ('sweep','fp_ops',1),('empty-before','fp_ops',1e20),('normal','seconds',.01)):
        bad=copy.deepcopy(records)
        next(r for r in bad if r['kind']==kind)[key]=value
        assert not op.analyze(bad)[1],(kind,key)
    topo=dict(cpus=list(range(72)),node=0,scf_cpu=0,pmu='nvidia_scf_pmu_0')
    command=op.capture_command(topo,Path('/bench'),Path('/capture'),'fo','packed','sweep',40,list(range(3,11)))
    scopes=[command[i+1] for i,x in enumerate(command) if x=='-C']
    assert scopes==['0',','.join(map(str,range(72)))]
    assert command.count('--delay=-1')==2 and '--perf-fds' in command
    normal=op.capture_command(topo,Path('/bench'),Path('/capture'),'fo','atomic','normal',40,[])
    assert 'perf' not in normal and '--perf-fds' not in normal
    refused(lambda:op.capture_command(topo,Path('/bench'),Path('/capture'),'fo','packed','sweep',40,[3]*8))
    longer=synthetic_records(op.LONG_REPEAT_COUNTS)
    assert op.analyze(longer,op.LONG_REPEAT_COUNTS)[1]
    refused(lambda:op.analyze(longer))
    refused(lambda:op.analyze(records,op.LONG_REPEAT_COUNTS))
    for counts in ((10,40,90),(30,30,90),(90,60,30)):
        refused(lambda:op.cases(counts))
    for kind,key,value in (('sweep','scf_seconds',99),('sweep','component_write_bytes',0),
                            ('sweep','fp_ops',1),('empty-before','fp_ops',1e20),('normal','seconds',.01)):
        bad=copy.deepcopy(longer)
        next(r for r in bad if r['kind']==kind)[key]=value
        assert not op.analyze(bad,op.LONG_REPEAT_COUNTS)[1],(kind,key)
    print('Operator accounting: completeness, scopes and failed acceptance checks PASS (synthetic)')


def collection_tests(repeat_counts=op.REPEAT_COUNTS):
    topo=dict(cpus=list(range(72)),node=0,scf_cpu=0,pmu='nvidia_scf_pmu_0')
    fixture=synthetic_records(repeat_counts)
    with tempfile.TemporaryDirectory() as directory:
        for fault in (None,'coverage','marker','command','timeout','identity','shutdown','stability'):
            out=Path(directory)/(fault or 'good')
            seen=[];sessions=[]
            def prepare(binary,path,elements,mode,extra):
                path.mkdir()
                assert mode=='residual-reuse'
                return topo,Path('/bench'),{}
            class FakeSession:
                def __init__(self,topo,binary,env,path,name,scheme,layout):
                    assert all(s.closed for s in sessions)
                    self.name,self.index,self.closed=name,0,False
                    self.identity=dict(pid=1+len(sessions),buffers=['0x1','0x2','0x3'])
                    sessions.append(self)
                def capture(self,topo,binary,stem,kind,repeat):
                    row=dict(fixture[len(seen)]);seen.append(row)
                    assert (kind,repeat,self.name,self.index)==(row['kind'],row['repeat'],row['session'],row['session_index'])
                    if fault=='command':raise subprocess.CalledProcessError(1,'mock')
                    if fault=='timeout':raise subprocess.TimeoutExpired('mock',1)
                    if fault=='identity' and self.index==1:self.identity=dict(self.identity,pid=9999)
                    if fault=='stability' and kind=='sweep' and repeat==repeat_counts[1] and row['block']==0:row['write_bytes']*=1.3
                    phase='normal' if kind=='normal' else ('sweep' if kind=='sweep' else 'sweep-empty')
                    output=''
                    for stage in ('before','after'):
                        for f in range(4):output+='ROOFLINE_CHECK stage={} field={} relative=0 PASS\n'.format(stage,f)
                    if kind!='normal':
                        output+='PERF_PHASE phase={} controllers=2 transitions=4 PASS\n'.format(phase)
                        output+='PERF_CONTROL phase={} controllers=2 enable_seconds=0 disable_seconds=0 inner_seconds={} outer_seconds={}\n'.format(phase,row['seconds'],row['seconds'])
                    output+='ROOFLINE_OPERATOR window={} layout={} limiter={} elements={} nodes={} pack=1024 repeats={} warmup=30 threads=72 seconds={} precision=fp64 geometry=affine rc=1 state_gradients=cached boundary=0 transient=0 PASS\n'.format(phase,row['layout'],-1 if row['scheme']=='fo' else 3,128**3,129**3,repeat,row['seconds'])
                    if fault=='marker':output=output.replace('ROOFLINE_CHECK','MISSING_CHECK')
                    stem.with_suffix('.out').write_text(output)
                    if kind!='normal':
                        scf,core=op.counter_events(topo)
                        scf += [topo['pmu']+'/'+e+'/' for e in ('cmem_wb_data','cmem_wr_data')]
                        for suffix,events,values,runtime in (('.scf.perf',scf,[row['read_bytes']/32,row['write_bytes'],0,row['write_bytes']],row['scf_seconds']),('.core.perf',core,[0,row['fp_ops']],row['core_seconds']*72)):
                            stem.with_suffix(suffix).write_text(''.join('{:.0f},,{}, {:.0f},{},,\n'.format(v,e,runtime*1e9,'4.0' if fault=='coverage' else '100.00') for e,v in zip(events,values)))
                    self.index+=1
                    return output
                def close(self,abort=False):
                    self.closed=True
                    if fault=='shutdown' and not abort:raise ValueError('worker failed at shutdown')
            with patch.object(op,'prepare_capture',prepare),patch.object(op,'OperatorSession',FakeSession),patch.object(op,'calibration_reference',return_value=(topo,'control fixture\n')),contextlib.redirect_stdout(io.StringIO()) as log:
                if fault:
                    try:op.collect(Path('/bench'),out,Path('/control'),repeat_counts)
                    except (ValueError,subprocess.SubprocessError):pass
                    else:raise AssertionError('failed collection accepted')
                    assert 'CAMPAIGN PASSED' not in log.getvalue()
                else:
                    op.collect(Path('/bench'),out,Path('/control'),repeat_counts)
                    assert len(json.loads((out/'records.json').read_text()))==144
                    assert len(sessions)==24 and all(s.index==6 for s in sessions)
                    metadata=json.loads((out/'collection.json').read_text())
                    assert metadata['repeat_counts']==list(repeat_counts)
                    assert metadata['normal_repeat']==repeat_counts[1]
                assert (out/'files.sha256.json').exists() and all(s.closed for s in sessions)
    print('Operator collection: 144 captures in 24 retained workers; coverage/result/lifetime/timeout/shutdown/stability refusal PASS (synthetic)')


def reference_tests():
    base=fixture(80000000)
    records=[]
    indices={}
    for block,mode,kind,repeat,lifetime in reuse.cases(True):
        row=dict(next(r for r in base if (r['block'],r['mode'],r['kind'],r['repeat'])==(block,mode,kind,repeat)))
        name=reuse.session_name((block,mode,kind,repeat),lifetime)
        index=indices.get(name,0);indices[name]=index+1
        row.update(lifetime=lifetime,session=name,session_index=index,identity=dict(pid=len(indices),buffers=['0x1','0x2','0x3']))
        records.append(row)
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp)
        (folder/'records.json').write_text(json.dumps(records))
        (folder/'summary.txt').write_text(reuse.analyze_measurements(records,80000000,True)[0])
        (folder/'collection.json').write_text(json.dumps(dict(mode='reuse-validation',elements=80000000)))
        (folder/'topology.json').write_text('{}')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
        (folder/'files.sha256.json').write_text(json.dumps(hashes))
        assert 'copy transported GB/s' in op.calibration_reference(folder)[1]
        for mode in ('reuse-diagnostic','sweep-validation'):
            (folder/'collection.json').write_text(json.dumps(dict(mode=mode,elements=80000000)))
            refused(lambda:op.calibration_reference(folder))
        (folder/'collection.json').write_text(json.dumps(dict(mode='reuse-validation',elements=80000000)))
        (folder/'topology.json').write_text('{"changed":true}')
        refused(lambda:op.calibration_reference(folder))
        with patch.object(op,'calibration_reference',return_value=({'job':'old'},'')),patch.object(op,'prepare_capture',return_value=({'job':'new'},Path('/bench'),{})),patch.object(op,'OperatorSession') as worker:
            refused(lambda:op.collect(Path('/bench'),folder,folder))
            worker.assert_not_called()
    print('Operator reference: accepts fixed reuse controls, rejects diagnostics/fresh recipe/hash changes/different allocation PASS')


def run_binary(binary, layout, scheme, kind, threads, fault=None):
    env=dict(os.environ,OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',
             OMPI_MCA_btl='self',OMPI_MCA_pml='ob1')
    phase='normal' if kind=='normal' else ('sweep' if kind=='sweep' else 'sweep-empty')
    command=[str(binary),'--n','4','--pack-size','1024','--layout',layout,'--rhie-chow',
             '--perturb','0.017','--warmup','3','--repeat','10','--roofline-window',phase]
    if scheme=='darwish': command+=['--conv-ho','3']
    descriptors=[]; child=[]; peers=[]; messages=[[],[]]; workers=[]; process=None
    try:
        if kind!='normal':
            for i in range(2):
                cr,cw=os.pipe(); ar,aw=os.pipe()
                descriptors += [cr,cw,ar,aw]; child += [cw,ar]; peers.append((cr,aw))
            command+=['--perf-fds']+list(map(str,child))
        process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 universal_newlines=True,pass_fds=tuple(child))
        for fd in child: os.close(fd); descriptors.remove(fd)
        def acknowledge(i,cr,aw):
            try:
                with os.fdopen(os.dup(cr),'rb') as reader:
                    for unused in range(2):
                        line=reader.readline()
                        if not line: break
                        messages[i].append(line)
                        reply=b'bad\n\0' if fault==i else b'ack\n\0'
                        os.write(aw,reply[:2]); os.write(aw,reply[2:])
            except BrokenPipeError:
                pass
        for i,(cr,aw) in enumerate(peers):
            worker=threading.Thread(target=acknowledge,args=(i,cr,aw));worker.daemon=True;worker.start();workers.append(worker)
        output=process.communicate(timeout=30)[0]
        if fault is not None:
            assert process.returncode!=0 and 'ROOFLINE_OPERATOR window=' not in output,output
        else:
            assert process.returncode==0,output
            result=op.parse_output(output,scheme,layout,kind,10,n=4,threads=threads)
            assert result['seconds']>0
            if kind!='normal': assert messages==[[b'enable\n',b'disable\n']]*2,messages
            refused(lambda:op.parse_output(output.replace('ROOFLINE_CHECK stage=before field=0','OMITTED'),scheme,layout,kind,10,n=4,threads=threads))
            refused(lambda:op.parse_output(output.replace('state_gradients=cached','state_gradients=per_apply'),scheme,layout,kind,10,n=4,threads=threads))
    finally:
        if process is not None and process.poll() is None: process.kill();process.communicate()
        for fd in descriptors: os.close(fd)
        for worker in workers: worker.join(timeout=2)


def binary_tests(binary):
    for threads in (1,2):
        for layout in ('atomic','packed'):
            for scheme in ('fo','darwish'):
                for kind in ('normal','sweep','empty-before'):
                    run_binary(binary,layout,scheme,kind,threads)
    for controller in (0,1): run_binary(binary,'packed','darwish','sweep',2,fault=controller)
    env=dict(os.environ,OMP_NUM_THREADS='1',OMPI_MCA_btl='self',OMPI_MCA_pml='ob1')
    for extra in ([],['--assemble'],['--boundary'],['--grad-precision','single'],['--conv-ho','2'],['--pgrad-per-apply'],['--repeat','0'],['--warp','0.1'],['--rho','2'],['--no-sfc']):
        command=[str(binary),'--n','4','--perturb','0.017','--roofline-window','normal']
        if extra: command+=['--rhie-chow']+extra
        result=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=30)
        assert result.returncode!=0 and 'ROOFLINE_OPERATOR window=' not in result.stdout,result.stdout
    print('Operator binary: 24 checked runs, two controller failures and ten scope refusals PASS (no PMU measurement)')


def session_tests(binary):
    topo=dict(cpus=[0,1],node=0,scf_cpu=0,pmu='mock')
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp);mock=out/'mock.py';mock.write_text(MOCK)
        for threads in (1,2):
            env=dict(os.environ,OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',OMPI_MCA_btl='self',OMPI_MCA_pml='ob1')
            low,middle,high=op.REPEAT_COUNTS if threads==1 else op.LONG_REPEAT_COUNTS
            for layout in ('atomic','packed'):
                for scheme in ('fo','darwish'):
                    name='{}-{}-{}-reuse'.format(scheme,layout,threads)
                    session=op.OperatorSession(topo,binary,env,out,name,scheme,layout,n=4,threads=threads,launcher=False)
                    descriptors=list(session.fds)
                    try:
                        for index,(kind,repeat) in enumerate((('normal',middle),('empty-before',low),('sweep',high),('sweep',middle),('sweep',low),('empty-after',low))):
                            stem=out/(name+'-'+str(index))
                            output=session.capture(topo,binary,stem,kind,repeat,monitor_prefix=[sys.executable,str(mock),'good'])
                            assert op.parse_output(output,scheme,layout,kind,repeat,n=4,threads=threads,warmup=30)['seconds']>0
                            assert json.loads(stem.with_suffix('.command.json').read_text()) is None if kind=='normal' else 'controllers=2 transitions=4 PASS' in output
                            assert session.index==index+1
                        session.close()
                        assert session.process.returncode==0
                    finally:session.close(abort=True)
                    for fd in descriptors:
                        try:os.fstat(fd)
                        except OSError:pass
                        else:raise AssertionError('session descriptor leaked')
        for fault in ('bad','exit','silent'):
            session=op.OperatorSession(topo,binary,env,out,'failure-reuse','darwish','packed',n=4,threads=2,launcher=False)
            try:
                try:session.capture(topo,binary,out/('failed-'+fault),'sweep',10,
                        monitor_prefix=[sys.executable,str(mock),fault],timeout=.5 if fault=='silent' else 10)
                except (ValueError,subprocess.SubprocessError):pass
                else:raise AssertionError('failed controller accepted')
                assert session.process.poll() is not None
                assert (out/('failed-'+fault)).with_suffix('.out').exists()
            finally:session.close(abort=True)
        for request in ('sweep 0\n','sweep -1\n','sweep 1001\n','setup 10\n','normal 10 extra\n',''):
            session=op.OperatorSession(topo,binary,env,out,'bad-request-reuse','fo','atomic',n=4,threads=2,launcher=False)
            try:
                session.process.stdin.write(request.encode());session.process.stdin.close()
                assert session.process.wait(timeout=10)!=0
            finally:session.close(abort=True)
        perf=out/'perf';perf.write_text('#!'+sys.executable+'\n'+NESTED);perf.chmod(0o755)
        env=dict(env,PATH=str(out)+os.pathsep+os.environ['PATH'])
        session=op.OperatorSession(topo,binary,env,out,'nested-reuse','darwish','packed',n=4,threads=2,launcher=False)
        try:
            for index,kind in enumerate(('sweep','normal','empty-before','sweep')):
                stem=out/('nested-'+str(index))
                output=session.capture(topo,binary,stem,kind,10)
                op.parse_output(output,'darwish','packed',kind,10,n=4,threads=2,warmup=30)
                assert stem.with_suffix('.scf.perf').exists()==(kind!='normal')
                assert stem.with_suffix('.core.perf').exists()==(kind!='normal')
            session.close()
        finally:session.close(abort=True)
    print('Operator sessions: 48 checked windows on 1/2 threads; nested monitors, normal gaps, invalid requests and failed controllers PASS (no PMU)')


if __name__=='__main__':
    accounting_tests()
    collection_tests()
    collection_tests(op.LONG_REPEAT_COUNTS)
    reference_tests()
    if len(sys.argv)>1:
        binary=Path(sys.argv[1]).resolve()
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as work:
            try:
                os.chdir(work)
                binary_tests(binary)
                session_tests(binary)
            finally:
                os.chdir(str(previous))
