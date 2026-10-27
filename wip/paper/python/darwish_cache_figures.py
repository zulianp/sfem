"""Generate the fixed-state Darwish timing table from the retained Grace evidence."""
import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import zipfile

ARCHIVE_SHA256 = '226de8ebd3dba370bdc132add5dfb2af447df10844157897a223fed0c9edf1d4'
ARCHIVE_NAME = 'pasc27-darwish-5108749.zip'
ARMS = ('original', 'control', 'cached', 'lagged', 'spmv')
LABELS = ('Consistent Darwish, direct', 'Direct, cache allocated', 'Consistent Darwish, cached', 'Packed lagged action', 'Assembled lagged SpMV')
ROOT = Path(__file__).resolve().parents[1]

def validate(data):
    rows,workers=data['captures'],data['workers']
    expected={(b,a,r) for b in range(6) for a in ARMS for r in (30,60,90)}
    if data['local_check'] or len(rows)!=90 or {(r['block'],r['arm'],r['repeats']) for r in rows}!=expected:
        raise ValueError('incomplete or local timing study')
    if len(workers)!=30 or {(w['block'],w['arm']) for w in workers}!={(b,a) for b in range(6) for a in ARMS}:
        raise ValueError('incomplete workers')
    directions=set()
    for w in workers:
        if w['exit']!=0: raise ValueError('failed worker')
        cmd=w['command'];directions.add(tuple(w['direction']))
        for option,value in (('--n','128'),('--pack-size','1024'),('--layout','packed'),('--geom','affine'),('--perturb','0.017'),('--warmup','30'),('--retained-windows','3')):
            if cmd[cmd.index(option)+1]!=value:raise ValueError('changed measurement scope')
        if '--rhie-chow' not in cmd or '--membind=0' not in cmd or '--physcpubind='+','.join(map(str,range(72))) not in cmd:
            raise ValueError('changed RC or CPU binding')
        part=[r for r in rows if (r['block'],r['arm'])==(w['block'],w['arm'])]
        if len({(r['pid'],r['direction'],r['output'],r['fingerprint']) for r in part})!=1:
            raise ValueError('changed worker, arrays or output')
        if {r['index'] for r in part}!={0,1,2}:raise ValueError('incomplete session sequence')
        for r in part:
            if [str(r['pid']),r['direction'],r['output']]!=w['identity']:raise ValueError('worker identity mismatch')
            for key in ('seconds','ms','relative'):
                if not math.isfinite(r[key]) or r[key]<0:raise ValueError('invalid measured value')
            if r['seconds']<=0 or r['relative']>1e-10 or abs(r['ms']-1000*r['seconds']/r['repeats'])>1e-10:
                raise ValueError('invalid timing or numerical check')
        if w['arm'] in ('cached','control') and (w['bytes']<=0 or not math.isfinite(w['setup']) or w['setup']<=0):raise ValueError('invalid cache cost')
    if len(directions)!=1:raise ValueError('different physical directions')
    for b in range(6):
        for arm,tolerance in (('cached',5e-12),('control',5e-12),('lagged_spmv',1e-10)):
            values=data['checks'][str(b)][arm]
            if len(values)!=4 or any(not math.isfinite(x) or x<0 or x>tolerance for x in values):raise ValueError('failed full-vector check')
    return rows,workers

def read_archive(path):
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=ARCHIVE_SHA256:raise ValueError('Darwish evidence archive hash mismatch')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        def read(name):return z.read('darwish_5108749/'+name).decode()
        data=json.loads(read('captures/records.json'));rows,workers=validate(data)
        for w in workers:
            text=read('captures/{}-b{}.log'.format(w['arm'],w['block']))
            if 'THP_LAUNCH policy=disabled disabled=1 pid=' not in text or 'RETAINED_DONE windows=3' not in text:raise ValueError('missing policy or completion')
            found=re.findall(r'^RETAINED_CAPTURE index=(\d+) repeats=(\d+) seconds=(\S+) relative=(\S+) fingerprint=(\S+) pid=(\d+) direction=(\S+) output=(\S+)$',text,re.M)
            if len(found)!=3:raise ValueError('missing capture logs')
            for index,repeats,seconds,relative,fingerprint,pid,direction,output in found:
                row=next(r for r in rows if r['block']==w['block'] and r['arm']==w['arm'] and r['index']==int(index))
                if (row['repeats'],row['seconds'],row['relative'],row['fingerprint'],row['pid'],row['direction'],row['output'])!=(int(repeats),float(seconds),float(relative),fingerprint,int(pid),direction,output):raise ValueError('records disagree with logs')
        run=read('run.log')
        if 'nid006549\n' not in run or '100% tests passed, 0 tests failed out of 7' not in run or 'MATCHED DARWISH CACHE TIMINGS COLLECTED' not in run:raise ValueError('missing validation/provenance')
        return data

def lagged_matrix_bytes(n):
    # On the tensor grid, each axis contributes 3*(n+1)-2 neighbour pairs.
    nodes = (n + 1)**3
    blocks = (3*n + 1)**3
    return blocks * (16*8 + 4) + (nodes + 1)*4


def generate(data,out):
    rows,workers=validate(data)
    out.mkdir(parents=True,exist_ok=True)
    means={a:statistics.mean(r['ms'] for r in rows if r['arm']==a and r['repeats']==60) for a in ARMS}
    prefix='% Generated by python/darwish_cache_figures.py; do not edit.\n'
    table=[prefix,r'\begin{tabular}{lrrrr}',r'\toprule',r'Application & Mean (ms) & Range (ms) & Matrix (GiB) & Added cache (GiB) \\',r'\midrule']
    for arm,label in zip(ARMS,LABELS):
        values=[r['ms'] for r in rows if r['arm']==arm and r['repeats']==60]
        matrix = format(lagged_matrix_bytes(128)/2**30, '.2f') + r'$^{\dagger}$' if arm == 'spmv' else '---'
        sizes = {w['bytes'] for w in workers if w['arm'] == arm} if arm in ('control', 'cached') else set()
        if sizes and len(sizes) != 1: raise ValueError('cache size changed')
        cache = format(next(iter(sizes))/2**30, '.2f') if sizes else '---'
        table.append('{} & {:.2f} & [{:.2f}, {:.2f}] & {} & {} \\\\'.format(label,means[arm],min(values),max(values),matrix,cache))
    table += [r'\bottomrule',r'\end{tabular}']
    (out/'darwish_cache.tex').write_text('\n'.join(table)+'\n')
    cached=[w for w in workers if w['arm']=='cached'];sizes={w['bytes'] for w in cached}
    if len(sizes)!=1:raise ValueError('cache size changed')
    setup=statistics.mean(w['setup'] for w in cached)
    definitions={'dcSpeedup':format(means['original']/means['cached'],'.2f'),
        'dcTimeReduction':format(100*(1-means['cached']/means['original']),'.1f'),
        'dcSpmvReduction':format(100*(1-means['cached']/means['spmv']),'.1f'),
        'dcLaggedGiB':format(lagged_matrix_bytes(128)/2**30,'.2f'),
        'dcGiB':format(sizes.pop()/2**30,'.2f'),'dcSetup':format(setup,'.2f'),
        'dcPayoff':str(math.ceil(1000*setup/(means['original']-means['cached']))),
        'dcOriginalMs':format(means['original'],'.2f'),'dcCachedMs':format(means['cached'],'.2f'),
        'dcSpmvMs':format(means['spmv'],'.2f')}
    (out/'darwish_cache_macros.tex').write_text(prefix+''.join('\\newcommand{{\\{}}}{{{}}}\n'.format(k,v) for k,v in definitions.items()))

def selftest(data):
    # Build small element adjacencies independently of the tensor-grid formula.
    for n in (1, 2, 3):
        pairs = set()
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    nodes = [(i+di, j+dj, k+dk) for di in (0,1) for dj in (0,1) for dk in (0,1)]
                    pairs.update((a,b) for a in nodes for b in nodes)
        assert lagged_matrix_bytes(n) == len(pairs)*132 + ((n+1)**3+1)*4
    validate(data)
    mutations=[lambda d:d['captures'].pop(),lambda d:d['captures'].append(copy.deepcopy(d['captures'][0])),
        lambda d:d['captures'][0].update(seconds=float('nan')),lambda d:d['captures'][0].update(relative=1),
        lambda d:d['workers'][0].update(exit=1),lambda d:d.update(local_check=True),
        lambda d:d['checks']['0'].update(cached=[1,1,1,1])]
    for mutate in mutations:
        changed=copy.deepcopy(data);mutate(changed)
        try:validate(changed)
        except ValueError:pass
        else:raise AssertionError('invalid evidence accepted')
    print('Darwish evidence: replay and seven refusal checks PASS')


MATRIX_SHA256 = '497ec8ae48b1e13030836e84adf4be81089334ac36b0800103c5fe836078dda1'
MATRIX_ARMS = ('direct', 'cached', 'consistent-csr', 'consistent-bsr')
MATRIX_ORDER = (('direct','cached','consistent-bsr','consistent-csr'),
    ('consistent-csr','consistent-bsr','cached','direct'),
    ('cached','consistent-bsr','consistent-csr','direct'),
    ('direct','consistent-csr','consistent-bsr','cached'),
    ('consistent-bsr','consistent-csr','direct','cached'),
    ('cached','direct','consistent-csr','consistent-bsr'),
    ('consistent-csr','direct','cached','consistent-bsr'),
    ('consistent-bsr','cached','direct','consistent-csr'))

def validate_matrix(records):
    expected=[('check-n4-t1-0',4,1),('check-n8-t1-1',8,1),('check-n8-t72-2',8,72),
              ('check-n9-t72-3',9,72),('check-n16-t72-4',16,72)]+[('timing-worker'+str(i),128,72) for i in range(3)]
    if len(records)!=len(expected):raise ValueError('incomplete matrix study')
    pids=set()
    for record,(name,n,threads) in zip(records,expected):
        r=record['result'];timing=name.startswith('timing-')
        if (record['name'],record['exit'],record['threads'],r['n'],r['threads'])!=(name,0,threads,n,threads):raise ValueError('changed matrix run scope')
        if r['pid'] in pids:raise ValueError('reused independent worker')
        pids.add(r['pid'])
        for key in ('max_scaled','omission_scaled'):
            if not math.isfinite(r[key]) or r[key]<0:raise ValueError('invalid matrix error')
        if r['max_scaled']>5e-12 or r['omission_scaled']<=5e-12 or r['checks']!=(4112 if n==4 else 208):raise ValueError('failed matrix check')
        if any(not math.isfinite(v) or v<=0 for v in r['costs'].values()):raise ValueError('invalid matrix cost')
        cmd=record['command']
        for option,value in (('--n',str(n)),('--layout','packed'),('--geom','affine'),('--conv-ho','3'),('--perturb','0.017'),('--warmup','30' if timing else '1'),('--repeat','60' if timing else '2')):
            if cmd[cmd.index(option)+1]!=value:raise ValueError('changed matrix command')
        if any(v not in cmd for v in ('--rhie-chow','--darwish-cache-experiment','--membind=0','--physcpubind='+','.join(map(str,range(72))))):raise ValueError('changed matrix configuration')
        rows=r['captures'];coverage={(c['block'],c['position']) for c in rows}
        if len(rows)!=(32 if timing else 0) or coverage!=({(b,p) for b in range(8) for p in range(4)} if timing else set()):raise ValueError('incomplete matrix timing')
        for c in rows:
            if c['arm']!=MATRIX_ORDER[c['block']][c['position']] or c['repeats']!=60:raise ValueError('changed matrix timing order')
            if any(not math.isfinite(c[k]) or c[k]<=0 for k in ('seconds','ms')) or abs(c['ms']-1000*c['seconds']/60)>1e-10:raise ValueError('invalid matrix timing')
    return records[5:]

def read_matrix_archive(path):
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=MATRIX_SHA256:raise ValueError('consistent matrix archive hash mismatch')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        records=json.loads(z.read('captures/records.json'));validate_matrix(records)
        for record in records:
            text=z.read('captures/'+record['name']+'.log').decode();r=record['result']
            if 'THP_LAUNCH policy=disabled disabled=1 pid='+str(r['pid'])+' PASS' not in text:raise ValueError('missing matrix memory policy')
            timing=record['name'].startswith('timing-')
            if text.count('CONSISTENT_MATRIX_CHECKS_PASSED timing='+str(int(timing)))!=1:raise ValueError('missing matrix completion')
            storage = re.findall(r'^MATRIX_STORAGE full_blocks=\d+ retained_blocks=(\d+) scalar_nonzeros=(\d+) bytes=(\d+) assembly_seconds=\S+$', text, re.M)
            csr = re.findall(r'^MATRIX_CSR bytes=(\d+) conversion_seconds=\S+$', text, re.M)
            if len(storage) != 1 or len(csr) != 1: raise ValueError('missing matrix storage ledger')
            blocks, nonzeros, bsr_bytes = map(int, storage[0])
            nodes = (r['n']+1)**3
            if bsr_bytes != blocks*132 + (nodes+1)*4 or bsr_bytes != r['costs']['bsr_bytes']:
                raise ValueError('changed BSR storage widths or ledger')
            if int(csr[0]) != nonzeros*12 + (nodes*4+1)*8 or int(csr[0]) != r['costs']['csr_bytes']:
                raise ValueError('changed CSR storage widths or ledger')
            check=re.findall(r'^MATRIX_VALIDATION checks=(\d+) max_scaled=(\S+) basis_columns=(\d+)$',text,re.M)
            if len(check)!=1 or (int(check[0][0]),float(check[0][1]))!=(r['checks'],r['max_scaled']):raise ValueError('matrix check log differs')
            found=re.findall(r'^MATRIX_TIMING block=(\d+) position=(\d+) arm=(\S+) repeats=(\d+) seconds=(\S+)$',text,re.M)
            values=[(int(b),int(p),a,int(count),float(seconds)) for b,p,a,count,seconds in found]
            if values!=[(c['block'],c['position'],c['arm'],c['repeats'],c['seconds']) for c in r['captures']]:raise ValueError('matrix timing log differs')
        run=z.read('run.log').decode()
        if 'nid006544\n' not in run or '100% tests passed, 0 tests failed out of 3' not in run:raise ValueError('missing matrix provenance')
    return records

def generate_matrix(records,out):
    workers=validate_matrix(records);out.mkdir(parents=True,exist_ok=True)
    values={a:[c['ms'] for w in workers for c in w['result']['captures'] if c['arm']==a] for a in MATRIX_ARMS}
    means={a:statistics.mean(v) for a,v in values.items()}
    prefix='% Generated by python/darwish_cache_figures.py --consistent-matrix; do not edit.\n'
    table=[prefix,r'\begin{tabular}{lrrrr}',r'\toprule',r'Application & Mean (ms) & Range (ms) & Matrix (GiB) & Added cache (GiB) \\',r'\midrule']
    for arm,label in zip(MATRIX_ARMS,('Consistent Darwish, direct','Consistent Darwish, cached','Consistent CSR SpMV','Consistent BSR SpMV')):
        v=values[arm]
        key = {'consistent-csr':'csr_bytes', 'consistent-bsr':'bsr_bytes'}.get(arm)
        sizes = {w['result']['costs'][key] for w in workers} if key else set()
        if sizes and len(sizes) != 1: raise ValueError('matrix storage changed')
        matrix = format(next(iter(sizes))/2**30, '.2f') if sizes else '---'
        cache_sizes = {w['result']['costs']['cache_bytes'] for w in workers}
        if len(cache_sizes) != 1: raise ValueError('cache size changed')
        cache = format(next(iter(cache_sizes))/2**30, '.2f') if arm == 'cached' else '---'
        table.append('{} & {:.2f} & [{:.2f}, {:.2f}] & {} & {} \\\\'.format(label,means[arm],min(v),max(v),matrix,cache))
    table += [r'\bottomrule',r'\end{tabular}']
    (out/'consistent_matrix.tex').write_text('\n'.join(table)+'\n')
    definitions={'cmCsrSpeedup':format(means['consistent-csr']/means['cached'],'.2f'),
        'cmBsrSpeedup':format(means['consistent-bsr']/means['cached'],'.2f'),
        'cmTimeReduction':format(100*(1-means['cached']/means['direct']),'.1f')}
    for arm,macro in zip(MATRIX_ARMS,('cmDirectMs','cmCachedMs','cmCsrMs','cmBsrMs')):definitions[macro]=format(means[arm],'.2f')
    for key,macro in (('cache_bytes','cmCacheGiB'),('csr_bytes','cmCsrGiB'),('bsr_bytes','cmBsrGiB')):
        sizes={w['result']['costs'][key] for w in workers}
        if len(sizes)!=1:raise ValueError('matrix storage changed')
        definitions[macro]=format(sizes.pop()/2**30,'.2f')
    for key,macro in (('cache_setup_seconds','cmCacheSetup'),('assembly_seconds','cmAssembly'),('csr_conversion_seconds','cmConversion')):
        definitions[macro]=format(statistics.mean(w['result']['costs'][key] for w in workers),'.2f')
    (out/'consistent_matrix_macros.tex').write_text(prefix+''.join('\\newcommand{{\\{}}}{{{}}}\n'.format(k,v) for k,v in definitions.items()))

def matrix_selftest(records):
    validate_matrix(records)
    mutations=[lambda d:d.pop(),lambda d:d[5].update(exit=1),
        lambda d:d[5]['result']['captures'].pop(),
        lambda d:d[5]['result'].update(max_scaled=float('nan')),
        lambda d:d[5]['result'].update(omission_scaled=0),
        lambda d:d[5]['result']['captures'][0].update(seconds=-1),
        lambda d:d[5]['result']['captures'][0].update(arm='cached'),
        lambda d:d[5]['result'].update(threads=1)]
    for mutate in mutations:
        changed=copy.deepcopy(records);mutate(changed)
        try:validate_matrix(changed)
        except ValueError:pass
        else:raise AssertionError('invalid matrix evidence accepted')
    print('Consistent matrix evidence: log replay and eight refusal checks PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path)
    p.add_argument('--consistent-matrix',action='store_true')
    p.add_argument('--selftest',action='store_true');p.add_argument('--out',type=Path,default=ROOT/'tables')
    a=p.parse_args()
    if a.consistent_matrix:
        path=a.archive or Path(os.environ.get('SFEM_CONSISTENT_MATRIX_ARCHIVE','pasc27-consistent-matrix-5109074.zip'))
        data=read_matrix_archive(path)
        if a.selftest:matrix_selftest(data)
        else:generate_matrix(data,a.out)
    else:
        data=read_archive(a.archive or Path(os.environ.get('SFEM_DARWISH_ARCHIVE',ARCHIVE_NAME)))
        if a.selftest:selftest(data)
        else:generate(data,a.out)
