#!/usr/bin/env python3
"""Run and check paired residual/action timings on the hashed public meshes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


def mesh_identity(path):
    path = Path(path)
    data = json.loads((path/'mesh.json').read_text())
    expected = {'meta.yaml'} | {'i{}.int32'.format(i) for i in range(8)} | {c+'.float32' for c in 'xyz'}
    require(set(data['files']) == expected, 'incomplete mesh manifest')
    require(data['generator'] == 'gmsh 4.15.0, Delaunay then all-hex subdivision', 'unknown generator')
    require(data['coordinate_dtype'] == 'float32' and data['index_dtype'] == 'int32', 'unsupported mesh types')
    require(data['nodes'] > 0 and data['elements'] > 0, 'empty mesh')
    require(math.isfinite(data['bernstein_det_lower']) and data['bernstein_det_lower'] > 0, 'uncertified geometry')
    require(len(data['interior_incident_cells']) > 1, 'no irregular interior connectivity')
    for name, digest in data['files'].items():
        p = path/name
        require(hashlib.sha256(p.read_bytes()).hexdigest() == digest, 'mesh hash mismatch: '+name)
        if name != 'meta.yaml':
            count = data['elements'] if name.startswith('i') else data['nodes']
            require(p.stat().st_size == 4*count, 'mesh size mismatch: '+name)
    return data


def rows(text, kind):
    result = []
    for line in text.splitlines():
        if line.startswith(kind+','):
            parts = line.split(',')[1:]
            result.append(dict(p.split('=', 1) for p in parts if '=' in p))
    return result


def number(row, key):
    value = float(row[key])
    require(math.isfinite(value), 'nonfinite '+key)
    return value


def check(text, mesh, threads, pack, blocks=12, repeat=12, verify=False, scheme="first_order"):
    marker = 'UNSTRUCTURED NUMERICAL CHECKS PASSED' if verify else 'UNSTRUCTURED OPERATOR STUDY PASSED'
    require(text.splitlines()[-1] == marker, 'missing final success marker')
    scope = rows(text, 'scope')
    require(len(scope) == 1, 'missing or duplicate scope')
    for key, value in (('threads', threads), ('pack', pack), ('nodes', mesh['nodes']), ('elements', mesh['elements'])):
        require(int(scope[0][key]) == value, 'wrong '+key)
    require(scheme in ('first_order','darwish'), 'unsupported convection scheme')
    require('scope,'+scheme+'_RC,isoparametric,steady_interior,' in text, 'wrong operator scope')
    geometry = rows(text, 'geometry')
    require(len(geometry) == 1 and number(geometry[0], 'min_sampled_det') > 0, 'invalid geometry check')
    linear=rows(text,'linear_gradient')
    require(len(linear)==2 and {r['layout'] for r in linear}=={'atomic','packed'} and
            all(0 <= number(r,'error') < 1e-11 for r in linear), 'failed linear-gradient check')
    steps = (1e-7,1e-8) if scheme == 'darwish' else (1e-6,1e-7)
    fd = rows(text, 'fd')
    require(len(fd) == 12, 'incomplete derivative check')
    seen = set()
    for line in text.splitlines():
        if not line.startswith('fd,'):
            continue
        layout = line.split(',')[1]
        row = rows(line, 'fd')[0]
        key = (layout, int(row['direction']), number(row, 'eps'))
        require(key not in seen, 'duplicate derivative check')
        seen.add(key)
        if scheme == 'darwish':
            require(int(row['crossings']) == 0, 'finite difference crosses a limiter switch')
        require(0 <= number(row, 'error') < 1e-6 and number(row, 'no_qgrad') > 1e-5, 'failed derivative or negative control')
    require(seen == {(layout,k,e) for layout in ('atomic','packed') for k in range(3) for e in steps}, 'wrong derivative coverage')
    coarse = rows(text, 'fd_coarse')
    if scheme == 'darwish':
        require(len(coarse) == 6 and
                {(line.split(',')[1],int(rows(line,'fd_coarse')[0]['direction']))
                 for line in text.splitlines() if line.startswith('fd_coarse,')} ==
                {(layout,k) for layout in ('atomic','packed') for k in range(3)},
                'missing coarse-step diagnostic')
        require(all(number(r,'eps') == 1e-6 and number(r,'error') >= 0 and
                    number(r,'no_qgrad') > 1e-5 and int(r['crossings']) >= 0 for r in coarse),
                'invalid coarse-step diagnostic')
    else:
        require(not coarse, 'unexpected coarse-step diagnostic')
    controls = rows(text, 'darwish_control')
    if scheme == 'darwish':
        require(len(controls) == 12 and
                {(r['layout'],int(r['direction']),number(r,'eps')) for r in controls} == seen,
                'incomplete Darwish derivative controls')
        require(all(number(r,'no_vgrad') > 1e-5 for r in controls), 'missing velocity-gradient sensitivity')
    else:
        require(not controls, 'unexpected Darwish controls')
    agreement = rows(text, 'agreement')
    require(len(agreement) == 4 and {(int(r['op']),int(r['direction'])) for r in agreement} == {(0,0),(1,0),(1,1),(1,2)}, 'incomplete layout check')
    require(all(0 <= number(r,'error') < 1e-11 for r in agreement), 'layout mismatch')
    storage, setup, payload = rows(text,'storage'), rows(text,'setup'), rows(text,'payload')
    require(len(storage) == len(setup) == 1 and len(payload) == 2, 'incomplete setup or storage accounting')
    require({r['layout'] for r in payload} == {'atomic','packed'}, 'wrong storage layouts')
    for row in storage+setup+payload:
        for key in row:
            if key != 'layout':
                require(number(row,key) >= 0, 'negative accounting value')
    samples = rows(text,'sample')
    require(len(samples) == (0 if verify else 4*blocks), 'incomplete timing blocks')
    seen = set()
    for r in samples:
        key = (r['op'],int(r['block']),r['layout'])
        require(key not in seen, 'duplicate timing sample')
        seen.add(key)
        require(int(r['repeat']) == repeat and number(r,'seconds') > 0 and 0 <= number(r,'error') < 1e-11, 'invalid timing or changed output')
        require(int(r['order']) == (int(r['block'])%2 if r['layout']=='atomic' else 1-int(r['block'])%2), 'wrong paired order')
    if not verify:
        require(seen == {(op,b,layout) for op in ('residual','jacobian') for b in range(blocks) for layout in ('atomic','packed')}, 'wrong timing coverage')
    return dict(storage=storage[0],setup=setup[0],payload=payload,samples=samples)


def summary(records):
    lines = ['Paired normal wall times; each arm includes its state-gradient reconstruction.',
             'Jacobian state gradients are cached; direction gradients are rebuilt per apply.',
             'Ranges show all sample variation, not confidence intervals. No samples are removed.',
             'Storage is an explicit array ledger, not peak process memory or a solver footprint.',
             '', '| Scheme | Mesh | Threads | Pack | Operator | Atomic ms range | Packed ms range | Mean time ratio | Paired ratio range |',
             '|---|---|---:|---:|---|---|---|---:|---|']
    for rec in records:
        if rec['verify']:
            continue
        for op in ('residual','jacobian'):
            s = [r for r in rec['result']['samples'] if r['op']==op]
            a = [float(r['seconds'])*1000 for r in s if r['layout']=='atomic']
            p = [float(r['seconds'])*1000 for r in s if r['layout']=='packed']
            ratios = [x/y for x,y in zip(a,p)]
            lines.append('| {} | {} | {} | {} | {} | {:.6f} [{:.6f}, {:.6f}] | {:.6f} [{:.6f}, {:.6f}] | {:.6f} | [{:.6f}, {:.6f}] |'.format(
                rec.get('scheme','first_order'),rec['mesh'],rec['threads'],rec['pack'],op,statistics.mean(a),min(a),max(a),
                statistics.mean(p),min(p),max(p),statistics.mean(a)/statistics.mean(p),min(ratios),max(ratios)))
    lines += ['', 'All setup times, array capacities, mesh identities, commands and samples are retained in records.json.',
              'This tests a procedural irregular mesh and interior operators; it does not establish PDE accuracy or solver speedup.']
    return '\n'.join(lines)+'\n'


def collect(binary, meshes, out, local=False, scheme="both"):
    binary=Path(binary).resolve();meshes=Path(meshes).resolve();out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    if local:
        cases=[('small',t,p,True) for t in (1,2) for p in (64,256)] + [('small',2,256,False)]
        blocks,repeat,warmup=4,3,3
    else:
        require(os.environ.get('SLURM_CPUS_PER_TASK') == '72', 'request 72 cores')
        cases=[('small',t,p,True) for t in (1,72) for p in (64,256)]
        # Alternate pack order across grid/thread combinations as well as arm order within each run.
        for index,(m,t) in enumerate((m,t) for m in ('medium','large') for t in (1,72)):
            cases += [(m,t,p,False) for p in ((256,1024) if index%2==0 else (1024,256))]
        blocks,repeat,warmup=12,12,10
    identity={m:mesh_identity(meshes/('mesh-'+m)) for m in {c[0] for c in cases}}
    schemes = ('first_order','darwish') if scheme == 'both' else (scheme,)
    for m,t,p,verify,scheme in (case+(scheme,) for case in cases for scheme in schemes):
        name='{}-{}-t{}-p{}-{}'.format(scheme,m,t,p,'check' if verify else 'timing')
        work=out/name;work.mkdir()
        command=[str(binary),'--scheme',scheme,'--mesh',str(meshes/('mesh-'+m)),'--pack-size',str(p)]
        command += ['--verify-only'] if verify else ['--blocks',str(blocks),'--repeat',str(repeat),'--warmup',str(warmup)]
        if not local:
            # Only the benchmark worker changes policy; the wrapper reads it back before exec.
            command=[sys.executable,str(Path(__file__).with_name('roofline_thp_exec.py')),'disabled','--']+command
        env=os.environ.copy()
        for key in list(env):
            if key.startswith(('SFEM_','CVFEM_')):
                del env[key]
        env.update(OMP_NUM_THREADS=str(t),OMP_DYNAMIC='FALSE',OMP_PLACES='cores',OMP_PROC_BIND='close')
        rec=dict(scheme=scheme,mesh=m,threads=t,pack=p,verify=verify,command=command,identity=identity[m],
                 binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),memory_policy='default' if local else 'THP disabled')
        (work/'command.json').write_text(json.dumps(rec,indent=2)+'\n')
        with (work/'run.log').open('w') as log:
            code=subprocess.call(command,cwd=str(work),env=env,stdout=log,stderr=subprocess.STDOUT)
        require(code == 0, 'failed '+name+'; retain '+str(work/'run.log'))
        captured=(work/'run.log').read_text()
        if not local:
            require('THP_LAUNCH policy=disabled disabled=1 ' in captured, 'missing worker memory-policy readback')
        rec['result']=check(captured,identity[m],t,p,blocks,repeat,verify,scheme)
        records.append(rec)
        (out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
        print('PASSED: '+name,flush=True)
    report=summary(records)
    (out/'summary.md').write_text(report)
    print(report,end='')
    print('UNSTRUCTURED CAMPAIGN PASSED')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',required=True);parser.add_argument('--meshes',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--local-check',action='store_true')
    parser.add_argument('--scheme',choices=('both','first_order','darwish'),default='both')
    args=parser.parse_args()
    try:
        collect(args.binary,args.meshes,args.out,args.local_check,args.scheme)
    except (ValueError,KeyError,OSError) as error:
        print('UNSTRUCTURED CAMPAIGN FAILED: '+str(error),file=sys.stderr)
        sys.exit(1)
