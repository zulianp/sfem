"""Reconstruct the accepted Grace campaign from its captured analysis and raw files."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def samples(rows):
    fields = ('block', 'repeat', 'seconds', 'fp_ops', 'read_bytes', 'write_bytes')
    return [{key: row[key] for key in fields} for row in rows if row['kind'] == 'sweep']


def verify(job):
    study = job/'study'
    sources = {}
    bundle = dict(line.split(None, 1)[::-1] for line in (job/'SOURCE-SHA256.txt').read_text().splitlines())
    for folder, manifest in ((study, 'measurement-source.sha256'), (study/'calibration', 'source.sha256')):
        for line in (folder/manifest).read_text().splitlines():
            sha, original = line.split(None, 1)
            path = folder/Path(original.strip()).name
            require(digest(path) == sha, 'changed captured source: '+str(path))
            relative = 'sfem/'+original.strip().split('/sfem/', 1)[1]
            require(bundle.get(relative) == sha, 'source differs from checked bundle: '+path.name)
            if path.name in sources:
                require(sources[path.name] == sha, 'inconsistent captured source')
            sources[path.name] = sha
    sys.path[:0] = [str(study), str(study/'calibration')]
    import roofline_operator as op
    import roofline_actions as act
    import roofline_gemm as gemm
    import roofline_reuse_diagnostic as reuse
    import roofline_memory_policy as policy
    import roofline_warmup_diagnostic as warmup
    from roofline_calibration import counter_events, parse_perf
    from roofline_sweep_validation import window_seconds
    os.environ[policy.VARIABLE] = 'disabled'
    for module in (op, act, gemm, reuse, policy, warmup):
        path = Path(module.__file__)
        require(path.name in sources and digest(path) == sources[path.name], 'analysis was not captured')
    topo = read(study/'actions/topology.json')
    require(topo['job'] == '5047137' and topo['host'] == 'nid006550', 'wrong allocation')
    require(topo['cpus'] == list(range(72)) and topo['node'] == topo['scf_cpu'] == 0,
            'wrong CPU/NUMA scopes')
    scf, core = counter_events(topo)
    scf += [topo['pmu']+'/'+event+'/' for event in ('cmem_wb_data', 'cmem_wr_data')]
    control_topo, reference = op.calibration_reference(study/'calibration/captures')
    require(control_topo == topo, 'different control topology')
    result = dict(job=topo['job'], host=topo['host'], policy='disabled', stages={}, sources=sources)
    metrics = {}
    counted_samples = {}
    checks = 0
    for name, module, count in (('calibration/captures', reuse, 60), ('residuals', op, 144), ('actions', act, 252), ('gemm', gemm, 36)):
        folder = study/name
        hashes = read(folder/'files.sha256.json')
        for file, sha in hashes.items():
            require(Path(file).name == file and digest(folder/file) == sha, 'capture hash mismatch: '+file)
        require({p.name for p in folder.iterdir() if p.is_file()} == set(hashes) | {'files.sha256.json'}, 'unbound capture files')
        require(read(folder/'topology.json') == topo, 'different topology: '+name)
        rows = read(folder/'records.json')
        require(len(rows) == count, 'wrong capture count: '+name)
        policy.verify_reference(folder, rows)
        if module in (op, act):
            require(read(folder/'collection.json') == dict(mode='action-reuse' if module == act else 'residual-reuse',
                    repeat_counts=[30,60,90], warmup=30, normal_repeat=60), 'wrong operator recipe')
        for row in rows:
            if module == reuse:
                stem = folder/'{}-{}-b{}-r{}-reuse'.format(row['mode'], row['kind'], row['block'], row['repeat'])
            elif module == gemm:
                stem = folder/'gemm-{}-b{}-r{}'.format(row['kind'], row['block'], row['repeat'])
            else:
                stem = folder/((''+row['operation']+'-' if module == act else '')+
                               '{}-{}-{}-b{}-r{}'.format(row['scheme'],row['layout'],row['kind'],row['block'],row['repeat']))
            output = stem.with_suffix('.out').read_text()
            if module == reuse:
                parsed = warmup.parse_output(output, row['mode'], 80000000, row['repeat'], row['kind'], 30)
            elif module == gemm:
                parsed = gemm.parse_output(output, row['kind'], row['repeat'])
                checks += output.count('GEMM_CHECK ')
                library = Path(read(folder/'blas-library.json')['path'])
                ready = (folder/(row['session']+'.ready.out')).read_text()
                identity = gemm.library_identity(ready, 72, production=True)
                # The uenv symlink is only resolvable on Grace; check its saved mapping.
                require('file='+str(library) in ready.split(), 'explicit BLAS absent from worker mappings')
                require(identity == row['library'] == read(study/'gemm-library.json'), 'DGEMM library changed')
            else:
                parsed = (act.parse_output(output,row['operation'],row['scheme'],row['layout'],row['kind'],row['repeat']) if module == act
                          else op.parse_output(output,row['scheme'],row['layout'],row['kind'],row['repeat'],warmup=30))
                checks += output.count('ROOFLINE_CHECK ')
                if module == act:
                    require(act.direction_identity((folder/(row['session']+'.ready.out')).read_text(),128) == row['direction'], 'direction mismatch')
            require(all(row[key] == value for key,value in parsed.items()), 'output differs from record: '+str(stem))
            buffers = 'SESSION_BUFFERS pid={} a={} b={} c={}'.format(row['identity']['pid'],*row['identity']['buffers'])
            require([line for line in output.splitlines() if line.startswith('SESSION_BUFFERS ')] == [buffers], 'worker buffers changed')
            require(output.splitlines()[-1] == 'SESSION_DONE index='+str(row['session_index']), 'wrong session index')
            request = read(stem.with_suffix('.request.json'))
            require((request['session'],request['index'],request['identity']) == (row['session'],row['session_index'],row['identity']), 'request differs')
            command = read(stem.with_suffix('.command.json'))
            if row['kind'] == 'normal':
                require(command is None, 'normal run has counters')
            else:
                require([command[i+1] for i,part in enumerate(command) if part == '-C'] == ['0',','.join(map(str,range(72)))], 'wrong monitor CPU scopes')
                require(command.count('--no-scale') == command.count('--delay=-1') == 2, 'unexpected counter controls')
                mt, ft = (stem.with_suffix(ext).read_text() for ext in ('.scf.perf','.core.perf'))
                m, f = parse_perf(mt,scf), parse_perf(ft,core)
                parsed = dict(read_bytes=32*m[scf[0]],write_bytes=m[scf[1]],component_write_bytes=32*m[scf[2]]+m[scf[3]],
                              fp_ops=sum(f.values()),scf_seconds=window_seconds(mt,scf,1),core_seconds=window_seconds(ft,core,72))
                require(all(row[key] == value for key,value in parsed.items()), 'raw counts differ: '+str(stem))
        if module == reuse:
            report, ok = reuse.analyze_measurements(rows,80000000,True)
        else:
            require((folder/'control-reference.txt').read_text() == reference, 'different control report')
            report, ok = module.analyze(rows) if module == gemm else module.analyze(rows,op.LONG_REPEAT_COUNTS)
            report = reference+'\n'+report
        require(ok and report == (folder/'summary.txt').read_text(), 'failed or changed report: '+name)
        result['stages'][name] = dict(captures=len(rows), hashes=len(hashes), sessions=len({row['session'] for row in rows}), accepted=True)
        if module in (op, act):
            arms = act.ARMS if module == act else [(scheme,layout) for scheme in ('fo','darwish') for layout in ('atomic','packed')]
            for arm in arms:
                keys = ('operation','scheme','layout') if module == act else ('scheme','layout')
                selected = [row for row in rows if tuple(row[k] for k in keys) == arm]
                normal = [1e3*row['seconds']/row['repeat'] for row in selected if row['kind']=='normal']
                counted = [row for row in selected if row['kind']=='sweep']
                counted_samples[('-'.join(arm) if module == act else 'residual-'+'-'.join(arm))] = samples(counted)
                metrics[('-'.join(arm) if module == act else 'residual-'+'-'.join(arm))] = dict(
                    normal_ms=statistics.mean(normal), normal_range=[min(normal),max(normal)],
                    mb=statistics.mean((row['read_bytes']+row['write_bytes'])/row['repeat']/1e6 for row in counted),
                    gbps=statistics.mean((row['read_bytes']+row['write_bytes'])/row['seconds']/1e9 for row in counted),
                    gop=statistics.mean(row['fp_ops']/row['repeat']/1e9 for row in counted),
                    gops=statistics.mean(row['fp_ops']/row['seconds']/1e9 for row in counted))
        elif module == gemm:
            counted_samples['dgemm'] = samples(rows)
            metrics['dgemm'] = dict(useful_gflops=statistics.mean(row['repeat']*row['useful_flops']/row['seconds']/1e9 for row in rows if row['kind']=='normal'),
                                   speculative_gops=statistics.mean(row['fp_ops']/row['seconds']/1e9 for row in rows if row['kind']=='sweep'))
            result['blas'] = rows[0]['library']
        else:
            for mode in ('copy','triad'):
                counted_samples[mode] = samples([row for row in rows if row['mode'] == mode])
                metrics[mode] = dict(gbps=statistics.mean((row['read_bytes']+row['write_bytes'])/row['seconds']/1e9
                    for row in rows if row['kind']=='sweep' and row['mode']==mode))
    require((study/'residuals/binary.sha256').read_text() == (study/'actions/binary.sha256').read_text(), 'operator binary mismatch')
    for path, sha in read(study/'gemm/sources.sha256.json').items():
        require(sources[Path(path).name] == sha, 'DGEMM source differs')
    source_checks = (job/'source-check.log').read_text().splitlines()
    source_manifest = (job/'SOURCE-SHA256.txt').read_text().splitlines()
    require(len(source_checks) == len(source_manifest) == 4299 and all(line.endswith(': OK') for line in source_checks), 'bundle source checks failed')
    require([line[:-4] for line in source_checks] == [line.split(None,1)[1] for line in source_manifest], 'source check list differs')
    require(read(study/'gemm/collection.json') == dict(mode='gemm-reuse',elements=8192), 'wrong DGEMM scope')
    inputs = read(job/'gemm-build/build-inputs.sha256.json')
    library = read(study/'gemm/blas-library.json')
    require(inputs[library['path']] == library['sha256'], 'DGEMM library hash changed after build')
    require(read(job/'gemm-build/runtime-library.json') == result['blas'], 'DGEMM preflight identity differs')
    require('MATCHED RESIDUAL, ACTION AND COMPUTE STUDY PASSED:' in (study/'run.log').read_text(), 'campaign incomplete')
    result.update(numerical_checks=checks, bundle_checks=len(source_checks), metrics=metrics,
                  counted_samples=counted_samples)
    return result


if __name__ == '__main__':
    print(json.dumps(verify(Path(sys.argv[1]).resolve()), indent=2))
