#!/usr/bin/env python3

from sfem_codegen import *
import matplotlib.pyplot as plt

# import pdb

debug=False
use_jac=not debug

# Tet4 nodes
p0 = sp.Matrix(3, 1, [x0, y0, z0])
p1 = sp.Matrix(3, 1, [x1, y1, z1])
p2 = sp.Matrix(3, 1, [x2, y2, z2])
p3 = sp.Matrix(3, 1, [x3, y3, z3])

if use_jac:
    J = matrix_coeff('J', 3, 3)
    p0 = sp.Matrix(3, 1, [0, 0, 0])
    p1 = sp.Matrix(3, 1, [J[0, 0], J[1, 0], J[2, 0]])
    p2 = sp.Matrix(3, 1, [J[0, 1], J[1, 1], J[2, 1]])
    p3 = sp.Matrix(3, 1, [J[0, 2], J[1, 2], J[2, 2]])

# Debug override
if debug:
    p0 = sp.Matrix(3, 1, [0, 0, 0])
    p1 = sp.Matrix(3, 1, [1, 0, 0])
    p2 = sp.Matrix(3, 1, [0, 1, 0])
    p3 = sp.Matrix(3, 1, [0, 0, 1])

ref0 = sp.Matrix(3, 1, [0, 0, 0])
ref1 = sp.Matrix(3, 1, [1, 0, 0])
ref2 = sp.Matrix(3, 1, [0, 1, 0])
ref3 = sp.Matrix(3, 1, [0, 0, 1])

ptet = [p0, p1, p2, p3]
rtet = [ref0, ref1, ref2, ref3]

vx = coeffs('vx', 4)
vy = coeffs('vy', 4)
vz = coeffs('vz', 4)

# Debug override
if debug:
    vx = sp.ones(4, 1)
    vy = sp.zeros(4, 1)
    vz = sp.zeros(4, 1)

def fun(x, y, z):
    return [1 - x - y - z, x, y, z]

# Centroid
centroid  = sp.Matrix(3, 1, [sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 4)])

# Sides opposite to corner
tri = [
    [1, 2, 3],
    [2, 0, 3],
    [1, 3, 0],
    [2, 1, 0]
];

def poly_surf_area_normal(poly):
    ndA = sp.zeros(3, 1);
    area = 0;
    n = len(poly);
    p0 = poly[0];

    for i in range(1, n-1):
        ip1 = i + 1;

        p1 = poly[i];
        p2 = poly[ip1];

        u = p1 - p0
        v = p2 - p0;

        ndAi = cross(u, v) / 2;
        
        if debug:
            if i > 1:
                assert(dot3(ndAi, ndA) > 0)

        ndA += ndAi;
        area += sp.sqrt(ndAi[0]*ndAi[0] + ndAi[1]*ndAi[1] + ndAi[2]*ndAi[2]);

    if debug:
        # print("------------")
        # print(area*area)
        # print(dot3(ndA, ndA))
        assert( abs(area*area -  dot3(ndA, ndA)) < 1e-8 )

    return ndA, area

def cv_faces(v0, v1, v2, v3):
    m01 = (v0 + v1)/2
    m02 = (v0 + v2)/2
    m03 = (v0 + v3)/2

    bf012 = (v0 + v1 + v2)/3;
    bf013 = (v0 + v1 + v3)/3;
    bf023 = (v0 + v2 + v3)/3;

    # Order here matters!!!
    f0 = [m01,  bf013, centroid, bf012];
    f1 = [m02,  bf012, centroid, bf023];
    f2 = [m03,  bf023, centroid, bf013];
    return f0, f1, f2

def cv_ips(v0, v1, v2, v3):
   f0, f1, f2 = cv_faces(v0, v1, v2, v3)

   ip0 = (f0[0] + f0[1] + f0[2] + f0[3]) / 4
   ip1 = (f1[0] + f1[1] + f1[2] + f1[3]) / 4
   ip2 = (f2[0] + f2[1] + f2[2] + f2[3]) / 4

   return ip0, ip1, ip2

def cv_interp(ip, v):
    f = fun(ip[0], ip[1], ip[2])
    ret = 0
    for j in range(0, 4):
        ret += f[j] * v[j]
    return ret

def cv_interp4(ips, v):
    ret = []
    for ip in ips:
        ret.append(cv_interp(ip, v))
    return ret

def advective_fluxes(vc, dn):
    q = []
    for i in range(0, 3):
        qi = 0
        for d in range(0, 3):
            qi += vc[d][i] * dn[i][d]

        # Quite important reduction of register usage here
        qi = sp.simplify(qi)
        q.append(qi)

    return q

def pw_max(a, b):
    return sp.Piecewise((b, a < b), (a, True))

def assign_matrix(name, mat):
    rows, cols = mat.shape
    expr = []
    for i in range(0, rows):
        for j in range(0, cols):
            var = sp.symbols(f'{name}[{i*cols + j}]')
            expr.append(ast.Assignment(var, mat[i, j]))
    return expr

def ref_subs(expr):
    if use_jac:
        for i in range(0, 3):
            for j in range(0, 3):
                expr = expr.subs(J[i, j], 1 if i==j else 0)
    else:
        expr = expr.subs(x0, 0)
        expr = expr.subs(y0, 0)
        expr = expr.subs(z0, 0)

        expr = expr.subs(x1, 1)
        expr = expr.subs(y1, 0)
        expr = expr.subs(z1, 0)

        expr = expr.subs(x2, 0)
        expr = expr.subs(y2, 1)
        expr = expr.subs(z2, 0)

        expr = expr.subs(x3, 0)
        expr = expr.subs(y3, 0)
        expr = expr.subs(z3, 1)
    return expr

def advection_op(q, i0, i1, i2, i3):
    A = sp.zeros(4, 4)
    A[i0, i0] += -pw_max(-q[0], 0) - pw_max(-q[1], 0) - pw_max(-q[2], 0)
    A[i0, i1] +=  pw_max(q[0], 0)
    A[i0, i2] +=  pw_max(q[1], 0)
    A[i0, i3] +=  pw_max(q[2], 0)
    return A

A = sp.zeros(4, 4)

for i in range(0, 4):
    o = tri[i]

    v0 = ptet[i]
    v1 = ptet[o[0]]
    v2 = ptet[o[1]]
    v3 = ptet[o[2]]

    f0, f1, f2 = cv_faces(v0, v1, v2, v3)

    r0 = rtet[i]
    r1 = rtet[o[0]]
    r2 = rtet[o[1]]
    r3 = rtet[o[2]]

    ip0, ip1, ip2 = cv_ips(r0, r1, r2, r3)

    vcx = cv_interp4([ip0, ip1, ip2], vx)
    vcy = cv_interp4([ip0, ip1, ip2], vy)
    vcz = cv_interp4([ip0, ip1, ip2], vz)

    # Scaled normals
    dn0, area0 = poly_surf_area_normal(f0);
    dn1, area1 = poly_surf_area_normal(f1);
    dn2, area2 = poly_surf_area_normal(f2);

    q = advective_fluxes([vcx, vcy, vcz], [dn0, dn1, dn2])
    B = advection_op(q, i, o[0], o[1], o[2])
    A += B

    # print(f'idx = {i}, {o[0]}, {o[1]}, {o[2]}')
    # print(vcx, vcy, vcz)
    # print(dn0[:], dn1[:], dn2[:])
    # print(dn0[0], dn1[0], dn2[0])
    # print(q)

    #
    # print(f'area = {area0}, {area1}, {area2}')
    # print(q)
    # print(B)

    if debug:
        assert(dot3(dn0, dn1) > 0)
        assert(dot3(dn0, dn2) > 0)
        
        # Normals should be pointing towards node
        dc0 = dot3(v0 - centroid, dn0)
        dc1 = dot3(v0 - centroid, dn1)
        dc2 = dot3(v0 - centroid, dn2)
        assert(dc0 > 0)
        assert(dc1 > 0)
        assert(dc2 > 0)

    # pdb.set_trace()

if not debug:    

    print('----------------------------')
    print('Hessian')
    print('----------------------------')

    expr = assign_matrix('element_matrix', A)
    c_code(expr)

    print('----------------------------')
    print('Apply')
    print('----------------------------')

    x = coeffs('x', 4)
    y = A * x
    expr = assign_matrix('element_vector', y)
    c_code(expr)

# Check on ref element
# if False:
if debug:
    print('------------------')
    sum_A = 0
    for i in range(0, 4):
        line = ""
        for j in range(0, 4):
            su = ref_subs(A[i, j])
            for v in vx:
                su = su.subs(v, 1)

            for v in vy:
                su = su.subs(v, 0)

            for v in vz:
                su = su.subs(v, 0)

            sum_A += su

            line += f"{round(su, 5)} "

        print(line)
        print('\n')

    print(f'sum_A={sum_A}')
