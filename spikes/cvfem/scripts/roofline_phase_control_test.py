#!/usr/bin/env python3
"""Check the control binary's perf handshake without hardware counters."""
import os
import math
from pathlib import Path
import select
import subprocess
import sys
import time


def run(binary, mode, threads, phase=None, fault=None, split_ack=False, controllers=1, fault_controller=0,
        delayed_action=None):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE')
    command = [str(binary), mode, '4096', '100', str(threads)]
    if phase is None:
        result = subprocess.check_output(command, env=env, universal_newlines=True)
        assert ' PASS' in result and 'PERF_PHASE' not in result
        assert 'CALIBRATION_WARMUP sweeps='+env.get('CVFEM_CALIBRATION_WARMUP','3') in result
        return
    descriptors = []
    process = None
    try:
        peers, child_fds = [], []
        command += [phase]
        for unused in range(controllers):
            fds = list(os.pipe())+list(os.pipe())
            descriptors.extend(fds)
            control_read, control_write, ack_read, ack_write = fds
            peers.append((control_read,ack_write))
            child_fds.extend((control_write,ack_read))
            command += [str(control_write),str(ack_read)]
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True, pass_fds=tuple(child_fds))
        for fd in child_fds:
            os.close(fd)
            descriptors.remove(fd)
        messages, pending = [[] for peer in peers], {peer[0]:b'' for peer in peers}
        readable = list(pending)
        while readable and sum(map(len,messages)) < 2*controllers:
            ready = select.select(readable,[],[],7)[0]
            if not ready:
                raise AssertionError('control request did not arrive')
            for control_read in ready:
                index = next(i for i,peer in enumerate(peers) if peer[0] == control_read)
                ack_write = peers[index][1]
                data = os.read(control_read,128)
                if not data:
                    readable.remove(control_read)
                    continue
                pending[control_read] += data
                while b'\n' in pending[control_read]:
                    message,pending[control_read] = pending[control_read].split(b'\n',1)
                    messages[index].append(message)
                    active_fault = fault if index == fault_controller else None
                    if active_fault == 'silent':
                        continue
                    if index == 0 and message.decode() == delayed_action:
                        time.sleep(0.025)
                    # Match perf's sizeof("ack\n") write, including the NUL.
                    reply = {'bad': b'bad\n\0', 'terminator': b'ack\nx',
                             'short': b'ack\n'}.get(active_fault, b'ack\n\0')
                    if split_ack:
                        for part in (reply[:2],reply[2:4],reply[4:]):
                            os.write(ack_write,part)
                            time.sleep(0.01)
                    else:
                        try:
                            os.write(ack_write,reply)
                        except BrokenPipeError:
                            # The other controller's bad reply may already have stopped the child.
                            if not fault:
                                raise
                    if active_fault == 'short':
                        os.close(ack_write)
                        descriptors.remove(ack_write)
        output = process.communicate(timeout=7)[0]
        if fault:
            assert process.returncode != 0 and 'PERF_PHASE' not in output, output
            if fault in ('bad','terminator'):
                assert 'invalid perf acknowledgement' in output, output
            elif fault in ('silent','short'):
                assert 'acknowledgement' in output, output
        else:
            assert process.returncode == 0, output
            assert 'CALIBRATION_WARMUP sweeps='+env.get('CVFEM_CALIBRATION_WARMUP','3') in output
            assert messages == [[b'enable',b'disable'] for peer in peers], messages
            marker = 'PERF_PHASE phase='+phase+(' transitions=2 PASS' if controllers == 1 else ' controllers=2 transitions=4 PASS')
            assert output.splitlines().count(marker) == 1
            assert 'CALIBRATION mode='+mode in output
            prefix = 'PERF_CONTROL phase='+phase+' controllers='+str(controllers)+' '
            timing_rows = [line for line in output.splitlines() if line.startswith(prefix)]
            assert len(timing_rows) == 1, output
            timing = {key:float(value) for key,value in
                      (word.split('=') for word in timing_rows[0][len(prefix):].split())}
            assert all(math.isfinite(v) and v >= 0 for v in timing.values()), timing
            assert math.isclose(timing['outer_seconds'],sum(timing[k] for k in
                ('enable_seconds','inner_seconds','disable_seconds')),rel_tol=1e-8,abs_tol=1e-9), timing
            if phase == 'sweep':
                line = next(line for line in output.splitlines() if line.startswith('CALIBRATION '))
                seconds = float(dict(word.split('=') for word in line.split()[1:-1])['seconds'])
                assert timing['inner_seconds'] >= seconds, (timing,seconds)
            if delayed_action:
                assert timing[delayed_action+'_seconds'] >= 0.020, timing
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        for fd in descriptors:
            os.close(fd)


if __name__ == '__main__':
    binary = Path(sys.argv[1]).resolve()
    os.environ['CVFEM_CALIBRATION_WARMUP']='3'
    for threads in (1,2):
        for mode in ('copy','triad'):
            run(binary,mode,threads)
            for phase in ('empty','setup','warmup','sweep','check','teardown'):
                run(binary,mode,threads,phase,split_ack=(threads == 2))
    run(binary,'copy',1,'sweep','bad')
    run(binary,'copy',1,'sweep','terminator')
    run(binary,'copy',1,'sweep','short')
    run(binary,'copy',1,'sweep','silent')
    run(binary,'copy',1,'unknown','unknown')
    print('Phase protocol: 24 gated controls (12 with split replies), 4 normal controls and 5 failure cases PASS (no PMU measurement)')
    for threads in (1,2):
        for mode in ('copy','triad'):
            for phase in ('sweep-empty','sweep'):
                run(binary,mode,threads,phase,split_ack=(threads == 2),controllers=2)
    for controller in (0,1):
        for fault in ('bad','terminator','short','silent'):
            run(binary,'copy',1,'sweep',fault,controllers=2,fault_controller=controller)
    print('Dual-controller protocol: 8 gated controls and 8 failure cases PASS (no PMU measurement)')
    for action in ('enable','disable'):
        run(binary,'triad',2,'sweep',controllers=2,delayed_action=action)
    print('Control timing: delayed enable/disable replies identified PASS (no PMU measurement)')

    os.environ['CVFEM_CALIBRATION_WARMUP']='30'
    for mode in ('copy','triad'):
        for phase in (None,'sweep','sweep-empty'):
            run(binary,mode,2,phase,controllers=2)
    for invalid in ('0','-1','nan','1001'):
        env=dict(os.environ,OMP_NUM_THREADS='1',CVFEM_CALIBRATION_WARMUP=invalid)
        result=subprocess.run([str(binary),'copy','4096','10','1'],env=env,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=10)
        assert result.returncode!=0 and 'CALIBRATION mode=' not in result.stdout,result.stdout
    print('Warmup control: six 30-sweep cases and four invalid-count refusals PASS (no PMU measurement)')
