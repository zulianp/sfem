#!/usr/bin/env python3
"""Check refusal paths using one real, complete local campaign capture."""
import argparse
import json
from pathlib import Path
import tempfile
import shutil
from unstructured_study import check, mesh_identity


def main(log, mesh):
    text=Path(log).read_text();identity=mesh_identity(mesh)
    def accept(value):
        return check(value,identity,2,256,4,3,False)
    accept(text)
    lines=text.splitlines()
    sample=next(line for line in lines if line.startswith('sample,'))
    fd=next(line for line in lines if line.startswith('fd,'))
    bad=[text.replace('UNSTRUCTURED OPERATOR STUDY PASSED',''),
         text.replace(sample+'\n',''),text.replace(sample,sample+'\n'+sample),
         text.replace(fd+'\n',''),text.replace(fd,fd+'\n'+fd),
         text.replace('threads=2,','threads=1,'),text.replace('repeat=3,','repeat=4,'),
         text.replace('op=residual','op=unknown')]
    for key,value in (('seconds','nan'),('seconds','-1'),('error','1')):
        parts=sample.split(',')
        changed=','.join(key+'='+value if p.startswith(key+'=') else p for p in parts)
        bad.append(text.replace(sample,changed))
    changed=','.join('no_qgrad=0' if p.startswith('no_qgrad=') else p for p in fd.split(','))
    bad.append(text.replace(fd,changed))
    for index,value in enumerate(bad):
        try:
            accept(value)
        except ValueError:
            continue
        raise AssertionError('accepted invalid capture {}'.format(index))
    with tempfile.TemporaryDirectory() as tmp:
        dst=Path(tmp)/'mesh';shutil.copytree(str(mesh),str(dst))
        p=dst/'x.float32';data=bytearray(p.read_bytes());data[0]^=1;p.write_bytes(data)
        try:
            mesh_identity(dst)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted changed mesh bytes')
    print('Unstructured accounting: real capture and 13 incomplete/invalid/hash refusal controls PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--log',required=True);parser.add_argument('--mesh',required=True)
    args=parser.parse_args();main(args.log,Path(args.mesh))
