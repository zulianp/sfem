#!/usr/bin/env python3
"""Generate a public graded box-with-hole mesh using Gmsh's all-hex subdivision."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

CORNERS = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                    [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
FACES = np.array([[0,1,2,3],[4,5,6,7],[0,1,5,4],
                  [1,2,6,5],[2,3,7,6],[3,0,4,7]])


def certify(points, cells):
    # det(J) is at most quadratic in each reference coordinate. Positive
    # tensor-product Bernstein coefficients bound it throughout each cell.
    lower = float('inf'); upper = 0.0
    for start in range(0, len(cells), 4096):
        xyz = points[cells[start:start+4096]].astype(np.float64)
        values = np.empty((len(xyz), 3, 3, 3))
        for ix, x in enumerate((0., .5, 1.)):
            for iy, y in enumerate((0., .5, 1.)):
                for iz, z in enumerate((0., .5, 1.)):
                    q = np.array([x,y,z]); factors = np.where(CORNERS, q, 1-q)
                    dn = np.empty((8,3))
                    for d in range(3):
                        axes = [k for k in range(3) if k != d]
                        dn[:,d] = (2*CORNERS[:,d]-1)*factors[:,axes[0]]*factors[:,axes[1]]
                    values[:,ix,iy,iz] = np.linalg.det(np.einsum('eac,ad->ecd',xyz,dn))
        for axis in (1,2,3):
            middle = [slice(None)]*4; middle[axis]=1
            lo=list(middle);lo[axis]=0
            hi=list(middle);hi[axis]=2
            values[tuple(middle)] = 2*values[tuple(middle)]-.5*(values[tuple(lo)]+values[tuple(hi)])
        local_lo=values.min(axis=(1,2,3));local_hi=values.max(axis=(1,2,3))
        if not (np.isfinite(values).all() and np.all(local_lo > 1e-10*local_hi)):
            raise ValueError('a cell lacks a positive Bernstein Jacobian certificate')
        lower=min(lower,float(local_lo.min()));upper=max(upper,float(local_hi.max()))
    return lower, upper


def generate(out, h):
    import gmsh
    if gmsh.__version__ != '4.15.0': raise ValueError('use gmsh==4.15.0')
    if not np.isfinite(h) or not 0 < h <= .4: raise ValueError('invalid mesh size')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.NumThreads',1)
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.option.setNumber('Mesh.RandomSeed',314159)
        box=gmsh.model.occ.addBox(0,0,0,1,1,1)
        hole=gmsh.model.occ.addCylinder(.5,.5,0,0,0,1,.18)
        gmsh.model.occ.cut([(3,box)],[(3,hole)])
        gmsh.model.occ.synchronize()
        f=gmsh.model.mesh.field.add('MathEval')
        gmsh.model.mesh.field.setString(f,'F','{}*(0.55+0.45*x)'.format(h))
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        for name in ('MeshSizeExtendFromBoundary','MeshSizeFromPoints','MeshSizeFromCurvature'):
            gmsh.option.setNumber('Mesh.'+name,0)
        gmsh.option.setNumber('Mesh.Algorithm3D',1)
        gmsh.option.setNumber('Mesh.SubdivisionAlgorithm',2)
        gmsh.model.mesh.generate(3)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        types, _, conn = gmsh.model.mesh.getElements(3)
        if list(types) != [5]: raise ValueError('mesh contains non-HEX8 cells')
        order=np.argsort(tags);tags=tags[order]
        points=np.asarray(xyz).reshape(-1,3)[order].astype('<f4')
        cells=np.searchsorted(tags,np.asarray(conn[0])).reshape(-1,8).astype('<i4')
        used=np.unique(cells)
        remap=np.full(len(points),-1,dtype=np.int32);remap[used]=np.arange(len(used))
        points=points[used];cells=remap[cells]
    finally:
        gmsh.finalize()
    if len(points) >= 2**31 or len(cells)==0: raise ValueError('unsupported mesh size')
    if np.any(np.diff(np.sort(cells,axis=1),axis=1)==0): raise ValueError('repeated cell nodes')
    lo,hi=certify(points,cells)
    faces=np.sort(cells[:,FACES].reshape(-1,4),axis=1)
    unique,counts=np.unique(faces,axis=0,return_counts=True)
    if not np.all((counts==1)|(counts==2)): raise ValueError('nonmanifold mesh')
    boundary=np.unique(unique[counts==1]); interior=np.ones(len(points),dtype=bool);interior[boundary]=False
    degree=np.bincount(cells.ravel(),minlength=len(points))
    values,numbers=np.unique(degree[interior],return_counts=True)
    if len(values)<2 or np.all(values==8): raise ValueError('fixture lacks irregular interior connectivity')
    for d,name in enumerate('xyz'): points[:,d].tofile(out/(name+'.float32'))
    for a in range(8): cells[:,a].astype('<i4').tofile(out/('i{}.int32'.format(a)))
    meta='spatial_dimension: 3\nelem_num_nodes: 8\nelement_type: HEX8\ngeom_map: IsoParametric\nn_elements: {}\nn_nodes: {}\nelements:\n'.format(len(cells),len(points))
    meta+=''.join('- i{0}: i{0}.int32\n'.format(a) for a in range(8))
    meta+='points:\n'+''.join('- {0}: {0}.float32\n'.format(c) for c in 'xyz')+'rpath: true\n'
    (out/'meta.yaml').write_text(meta)
    report=dict(generator='gmsh 4.15.0, Delaunay then all-hex subdivision',seed=314159,
        geometry='unit box minus through-cylinder radius 0.18',h=h,nodes=len(points),elements=len(cells),
        coordinate_dtype='float32',index_dtype='int32',bernstein_det_lower=lo,bernstein_det_upper=hi,
        interior_incident_cells={str(int(v)):int(n) for v,n in zip(values,numbers)},
        scope='procedural public mesh; not an industrial mesh or PDE accuracy result')
    report['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}
    (out/'mesh.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out');parser.add_argument('--h',type=float,required=True)
    args=parser.parse_args();generate(args.out,args.h)
