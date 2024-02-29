function cvfem_advection_diffusion()
% Advection diffusion equation
% d phi/ dt + div(v phi) - div( k grad phi ) - Q = 0
% v is velocity
% phi is the conserved quantity
% k is the diffusivity
% Q is the source term

% 1) Solve the steady state advection-diffusion equation
% $int_{\Gamma} <v, n> phi -  int_{\Gamma} k <grad phi, n> = 0$
% where $\Gamma$ is the boundary of the domain

% Cylinder example


if 0
mesh = '/Users/patrickzulian/Desktop/code/sfem/tests/cvfem3/mesh';
lsystem = '/Users/patrickzulian/Desktop/code/sfem/tests/cvfem3/system';

rowptr = read_array([lsystem  '/rowptr.raw'], 'int32') + 1;
colidx = read_array([lsystem  '/colidx.raw'], 'int32') + 1;
% nneighs = rowptr(2:end) - rowptr(1:end-1);
% max_nneighs = max(nneighs);

i0 = read_array([mesh '/i0.raw'], 'int32') + 1;
i1 = read_array([mesh '/i1.raw'], 'int32') + 1;
i2 = read_array([mesh '/i2.raw'], 'int32') + 1;
i3 = read_array([mesh '/i3.raw'], 'int32') + 1;

x = read_array([mesh '/x.raw'], 'float');
y = read_array([mesh '/y.raw'], 'float');
z = read_array([mesh '/z.raw'], 'float');

dirichlet = unique(sort(read_array([mesh '/sidesets_aos/sinlet.raw'], 'int32') + 1));
neumann   = read_array([mesh '/sidesets_aos/soutlet.raw'], 'int32') + 1;
neumann   = reshape(neumann, 3, length(neumann)/ 3);

else 
% Basic test

rowptr = [1, 6, 11, 16, 21, 26];
colidx = [
    1, 2, 3, 4, 5, ...
    1, 2, 3, 4, 5, ...
    1, 2, 3, 4, 5, ... 
    1, 2, 3, 4, 5, ...
    1, 2, 3, 4, 5];

i0 = [1,5]';
i1 = [2,4]';
i2 = [3,3]';
i3 = [4,2]';

x = [0.0, 1.0, 0.0, 0.,  1.0]';
y = [0.0, 0.0, 1.0, 0.,  1.0]';
z = [0.0, 0.0, 0.0, 1., 1.0]';

dirichlet = [1];
neumann = [5, 3, 4];
end


phi = zeros(length(x), 1);
phi(dirichlet) = 1;

elems = [i0'; i1'; i2'; i3'];
points = [x'; y'; z'];
evol = element_volumes(elems, points);
cv_vol = CV_volume(length(x), elems, evol);

disp('evol')
disp(sum(evol))
disp('cv_vol')
disp(sum(cv_vol))
disp(sum(cv_vol)./sum(evol))


[dn1, dn2, dn3, dn4, dn5, dn6] = CV_normals(elems, points);

% disp(size(dn1));

vx0 = ones(size(x));
zzz = zeros(size(x));
vx1 = zeros(size(x));
vx2 = zeros(size(x));
vx3 = zeros(size(x));

% Compute velocties at the centroids
vxc = p_CV_face_centroids_interp(vx0, vx1, vx2, vx3);

% FIXME now velocities are zero
vyc = p_CV_face_centroids_interp(zzz, zzz, zzz, zzz);
vzc = p_CV_face_centroids_interp(zzz, zzz, zzz, zzz);

q = advective_fluxes(vxc, vyc, vzc, dn1, dn2, dn3, dn4, dn5, dn6);
% size(q)

qphi = upwind_scheme(elems, phi, q);
% size(qphi)

problem = {};
problem.q = q;
problem.qphi = qphi;
problem.vxc = vxc;
problem.vyc = vyc;
problem.vzc = vzc;
problem.dn1 = dn1;
problem.dn2 = dn2;
problem.dn3 = dn3;
problem.dn4 = dn4;
problem.dn5 = dn5;
problem.dn6 = dn6;
problem.elems = elems;
problem.points = points;
problem.evol = evol;
problem.cv_vol = cv_vol;
problem.neumann = neumann;
problem.dirichlet = dirichlet;
problem.rowptr = rowptr;
problem.colidx = colidx;

problem.nelements = size(elems, 2);
problem.nnodes = length(x);

mat = assemble(problem);

%% Plot stuff
close all;
if(size(neumann, 1) == 1)
    ptri = triangulation(neumann, x, y, z);
else
    ptri = triangulation(neumann', x, y, z);
end
trisurf(ptri, 'FaceAlpha', 0.1);
hold on;
plot3(x(dirichlet), y(dirichlet), z(dirichlet), '*');
return

%% Algebraic system
function values = assemble(problem)
rowptr = problem.rowptr;
colidx = problem.colidx;

lap_e = e_assemble_laplacian(problem);
adv_e = e_assemble_advection(problem);

% disp(reshape(lap_e(1, :, :), 4, 4))
% disp(reshape(adv_e(1, :, :), 4, 4))
% 
% disp(reshape(lap_e(2, :, :), 4, 4))
% disp(reshape(adv_e(2, :, :), 4, 4))

mat_e = lap_e + adv_e;

values = elemental_to_crs(problem.elems, mat_e, rowptr, colidx);

D = crs_to_dense(rowptr, colidx, values);
disp(D)
return

function values = elemental_to_crs(elems, mat_e, rowptr, colidx)
values = zeros(size(colidx));

assert(length(values) == rowptr(end)-1);

for e=1:size(elems, 2)
    idx = elems(:, e);

    for ii=1:4
        i = idx(ii);
        row = colidx(rowptr(i):(rowptr(i+1)-1));
       
        for jj=1:4
            j=idx(jj);
    
            found = 0;
            for kk=1:length(row)
                if j == row(kk)
                    found = 1;
                    break
                end
            end

            k = rowptr(i) + kk - 1;

            assert(found);
            assert(kk <= length(row));
            assert(k <= length(values));
            assert(j == colidx(k));
            values(k) = values(k) + mat_e(e, ii, jj);
        end
    end
end
   
return

function Ae = e_assemble_laplacian(p)
Ae = zeros(p.nelements, 4, 4);
elems = p.elems;
points = p.points;

dn1 = p.dn1;
dn2 = p.dn2;
dn3 = p.dn3;
dn4 = p.dn4;
dn5 = p.dn5;
dn6 = p.dn6;
evol = p.evol;

for e=1:p.nelements
    enodes = elems(:, e);
    x = points(1, enodes);
    y = points(2, enodes);
    z = points(3, enodes);
    p = [x; y; z];

    a = p(:, 1);
    b = p(:, 2);
    c = p(:, 3);
    d = p(:, 4);

    agrad = 1./6 * cross(d - b, c - b) / evol(e);
    bgrad = 1./6 * cross(c - a, d - a) / evol(e);
    cgrad = 1./6 * cross(d - a, b - a) / evol(e);
    dgrad = 1./6 * cross(b - a, c - a) / evol(e);

    grads = [agrad, bgrad, cgrad, dgrad]';
    
    assert(size(grads,1) == 4);
    assert(size(grads,2) == 3);

    gtest1 = -dn1(:, e) - dn2(:, e) - dn3(:, e);
    
    Ae(e, 1, 1) = dot(grads(1, :), gtest1);
    Ae(e, 1, 2) = dot(grads(2, :), gtest1);
    Ae(e, 1, 3) = dot(grads(3, :), gtest1);
    Ae(e, 1, 4) = dot(grads(4, :), gtest1);

    gtest2 =  dn1(:, e) - dn4(:, e) - dn5(:, e);
    
    Ae(e, 2, 1) = dot(grads(1, :), gtest2);
    Ae(e, 2, 2) = dot(grads(2, :), gtest2);
    Ae(e, 2, 3) = dot(grads(3, :), gtest2);
    Ae(e, 2, 4) = dot(grads(4, :), gtest2);
    
    gtest3 =  dn2(:, e) + dn4(:, e) - dn6(:, e);
    
    Ae(e, 3, 1) = dot(grads(1, :), gtest3);
    Ae(e, 3, 2) = dot(grads(2, :), gtest3);
    Ae(e, 3, 3) = dot(grads(3, :), gtest3);
    Ae(e, 3, 4) = dot(grads(4, :), gtest3);
    
    gtest4 = dn3(:, e) + dn5(:, e) + dn6(:, e);
    
    Ae(e, 4, 1) = dot(grads(1, :), gtest4);
    Ae(e, 4, 2) = dot(grads(2, :), gtest4);
    Ae(e, 4, 3) = dot(grads(3, :), gtest4);
    Ae(e, 4, 4) = dot(grads(4, :), gtest4);    
end


return

function Ae = e_assemble_advection(p)
Ae = zeros(p.nelements, 4, 4);
elems = p.elems;
points = p.points;
q = p.q;

% Node a
Ae(:, 1, 1) = -max( q(1, :), 0) - max(q(2, :), 0) - max(q(3, :), 0);
Ae(:, 1, 2) =  max(-q(1, :), 0);
Ae(:, 1, 3) =  max(-q(2, :), 0);
Ae(:, 1, 4) =  max(-q(3, :), 0);

% Node b
Ae(:, 2, 2) = -max( q(1, :), 0) - max(q(4, :), 0) - max(q(5, :), 0);
Ae(:, 2, 1) =  max(-q(1, :), 0);
Ae(:, 2, 3) =  max(-q(4, :), 0);
Ae(:, 2, 4) =  max(-q(5, :), 0);

% Node c
Ae(:, 3, 3) = -max( q(2, :), 0) - max(q(4, :), 0) - max(q(6, :), 0);
Ae(:, 3, 1) =  max(-q(2, :), 0);
Ae(:, 3, 2) =  max(-q(4, :), 0);
Ae(:, 3, 4) =  max(-q(6, :), 0);

% Node d
Ae(:, 4, 4) = -max( q(3, :), 0) - max(q(5, :), 0) - max(q(6, :), 0);
Ae(:, 4, 1) =  max(-q(3, :), 0);
Ae(:, 4, 2) =  max(-q(5, :), 0);
Ae(:, 4, 3) =  max(-q(6, :), 0);
return


%% Elemental quantities
function qphi = upwind_scheme(elems, phi, q)
phi1 = phi(elems(1, :))';
phi2 = phi(elems(2, :))';
phi3 = phi(elems(3, :))';
phi4 = phi(elems(4, :))';

qphi1 = max(q(1, :), 0) .* phi1 - max(-q(1, :), 0) .* phi2;
qphi2 = max(q(2, :), 0) .* phi1 - max(-q(2, :), 0) .* phi3;
qphi3 = max(q(3, :), 0) .* phi1 - max(-q(3, :), 0) .* phi4;
qphi4 = max(q(4, :), 0) .* phi2 - max(-q(4, :), 0) .* phi3;
qphi5 = max(q(5, :), 0) .* phi2 - max(-q(5, :), 0) .* phi4;
qphi6 = max(q(6, :), 0) .* phi3 - max(-q(6, :), 0) .* phi4;

qphi=[qphi1; qphi2; qphi3; qphi4; qphi5; qphi6];
return

function q = advective_fluxes(vxc, vyc, vzc, dn1, dn2, dn3, dn4, dn5, dn6)
q1 = vxc(1, :) .* dn1(1, :) + vyc(1, :) .* dn1(2, :) + vzc(1, :) .* dn1(2, :);
q2 = vxc(2, :) .* dn2(1, :) + vyc(2, :) .* dn2(2, :) + vzc(2, :) .* dn2(2, :);
q3 = vxc(3, :) .* dn3(1, :) + vyc(3, :) .* dn3(2, :) + vzc(3, :) .* dn3(2, :);
q4 = vxc(4, :) .* dn4(1, :) + vyc(4, :) .* dn4(2, :) + vzc(4, :) .* dn4(2, :);
q5 = vxc(5, :) .* dn5(1, :) + vyc(5, :) .* dn5(2, :) + vzc(5, :) .* dn5(2, :);
q6 = vxc(6, :) .* dn6(1, :) + vyc(6, :) .* dn6(2, :) + vzc(6, :) * dn6(2, :);
q = [q1; q2; q3; q4; q5; q6];
return

function vol = element_volumes(elems, points)
x0 = points(1, elems(1, :));
x1 = points(1, elems(2, :));
x2 = points(1, elems(3, :));
x3 = points(1, elems(4, :));

y0 = points(2, elems(1, :));
y1 = points(2, elems(2, :));
y2 = points(2, elems(3, :));
y3 = points(2, elems(4, :));

z0 = points(3, elems(1, :));
z1 = points(3, elems(2, :));
z2 = points(3, elems(3, :));
z3 = points(3, elems(4, :));

u0 = x1 - x0;
u1 = x2 - x0;
u2 = x3 - x0;

v0 = y1 - y0;
v1 = y2 - y0;
v2 = y3 - y0;

w0 = z1 - z0;
w1 = z2 - z0;
w2 = z3 - z0;

[n0, n1, n2] = batched_cross(u0, u1, u2, v0, v1, v2);
vol = 1./6 * (w0 .* n0 + w1 .* n1 + w2 .* n2)';

return

function [dn1, dn2, dn3, dn4, dn5, dn6] = CV_normals(elems, points)
xa = points(1, elems(1, :));
xb = points(1, elems(2, :));
xc = points(1, elems(3, :));
xd = points(1, elems(4, :));

ya = points(2, elems(1, :));
yb = points(2, elems(2, :));
yc = points(2, elems(3, :));
yd = points(2, elems(4, :));

za = points(3, elems(1, :));
zb = points(3, elems(2, :));
zc = points(3, elems(3, :));
zd = points(3, elems(4, :));

cross_ab = p_cross(xa, ya, za, xb, yb, zb);
cross_ac = p_cross(xa, ya, za, xc, yc, zc);
cross_ad = p_cross(xa, ya, za, xd, yd, zd);

cross_bc = p_cross(xb, yb, zb, xc, yc, zc);
cross_bd = p_cross(xb, yb, zb, xd, yd, zd);

cross_cd = p_cross(xc, yc, zc, xd, yd, zd);

dn1 =  1./24 * cross_ac - 1./24 * cross_ad + 1./24 *cross_bc - 1./24 * cross_bd + 1./12 * cross_cd;
dn2 = -1./24 * cross_ab + 1./24 * cross_ad + 1./24 *cross_bc - 1./12 * cross_bd + 1./24 * cross_cd;
dn3 =  1./24 * cross_ab - 1./24 * cross_ac + 1./12 *cross_bc - 1./24 * cross_bd + 1./24 * cross_cd;
dn4 = -1./24 * cross_ab - 1./24 * cross_ac + 1./12 *cross_ad - 1./24 * cross_bd - 1./24 * cross_cd;
dn5 =  1./24 * cross_ab - 1./12 * cross_ac + 1./24 *cross_ad + 1./24 * cross_bc - 1./24 * cross_cd;
dn6 =  1./12 * cross_ab - 1./24 * cross_ac - 1./24 *cross_ad + 1./24 * cross_bc + 1./24 * cross_bd;

return

function cv_vol = CV_volume(ncvs, elems, evol)
cv_vol = zeros(ncvs, 1);

nelements = size(elems, 2);

for d=1:4
    for i=1:nelements
        idx = elems(d, i);
        cv_vol(idx) = cv_vol(idx) + evol(i);
    end
end

cv_vol = cv_vol ./ 4;
return

function [n0, n1, n2] = batched_cross(a0, a1, a2, b0, b1, b2)
n0 = a1 .* b2 - a2 .* b1;
n1 = a2 .* b0 - a0 .* b2;
n2 = a0 .* b1 - a1 .* b0;
return

function [n]= p_cross(a0, a1, a2, b0, b1, b2)
[n0, n1, n2] = batched_cross(a0, a1, a2, b0, b1, b2);
n = [n0; n1; n2];
return

function [vc1, vc2, vc3, vc4, vc5, vc6] = CV_face_centroids_interp(v0, v1, v2, v3)
w0 = 13./36;
w1 =  5./36;

vc1 = w0 * v0 + w0 * v1 + w1 * v2 + w1 * v3;
vc2 = w0 * v0 + w1 * v1 + w0 * v2 + w1 * v3;
vc3 = w0 * v0 + w1 * v1 + w1 * v2 + w0 * v3;
vc4 = w1 * v0 + w0 * v1 + w0 * v2 + w1 * v3;
vc5 = w1 * v0 + w0 * v1 + w1 * v2 + w0 * v3;
vc6 = w1 * v0 + w1 * v1 + w0 * v2 + w0 * v3;
return

function [vc] = p_CV_face_centroids_interp(v0, v1, v2, v3)
[vc1, vc2, vc3, vc4, vc5, vc6] = CV_face_centroids_interp(v0, v1, v2, v3);
vc =  [vc1; vc2; vc3; vc4; vc5; vc6];
return
