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

if 1
% mesh = '/Users/patrickzulian/Desktop/code/sfem/tests/cvfem3/mesh';
% lsystem = '/Users/patrickzulian/Desktop/code/sfem/tests/cvfem3/system';
mesh = './annulus/mesh';
lsystem = './annulus/crs';

% ../python/sfem/mesh/raw_to_db.py /Users/patrickzulian/Desktop/code/sfem/tests/cvfem3/mesh out.vtk --point_data=solution.raw

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

% dirichlet = unique(sort(read_array([mesh '/sidesets_aos/sinlet.raw'], 'int32') + 1));
dirichlet = read_array([mesh '/sidesets_aos/sinlet.raw'], 'int32') + 1;
dirichlet   = reshape(dirichlet, 3, length(dirichlet)/ 3);

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

x = [0.0, 1.0, 0.0, 0., 1.0]';
y = [0.0, 0.0, 1.0, 0., 1.0]';
z = [0.0, 0.0, 0.0, 1., 1.0]';

dirichlet = [1];
neumann = [5, 3, 4];
end

% 
% phi = zeros(length(x), 1);
% phi(dirichlet) = 1;

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

vx = ones(size(x));
vy = zeros(size(x));
vz = zeros(size(x));

% Compute velocties at the centroids
vxc = p_nodal_to_CV_centroids(elems, vx);
vyc = p_nodal_to_CV_centroids(elems, vy);
vzc = p_nodal_to_CV_centroids(elems, vz);

q = advective_fluxes(vxc, vyc, vzc, dn1, dn2, dn3, dn4, dn5, dn6);

% qphi = upwind_scheme(elems, phi, q);


problem = {};
problem.q = q;
% problem.qphi = qphi;
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

problem.BC_penalization = 1e16;
problem.diffusivity = 10;

problem.dirichlet_nodes = unique(sort(dirichlet(:)));
problem.dirichlet = dirichlet;
problem.dirichlet_fun = @(x, y, z) 1.;

problem.rowptr = rowptr;
problem.colidx = colidx;

problem.nelements = size(elems, 2);
problem.nnodes = length(x);

[A, b] = assemble(problem);
sol = A\b;

write_array('solution.raw', sol, 'double');

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
xlabel('x');
ylabel('y');
zlabel('z');
return

%% Algebraic system
function [A, b] = assemble(problem)
rowptr = problem.rowptr;
colidx = problem.colidx;

lap_e = e_assemble_laplacian(problem);
adv_e = e_assemble_advection(problem);
mat_e = lap_e + adv_e;

values = elemental_to_crs(problem.elems, mat_e, rowptr, colidx);

A = sparse(crs_to_dense(rowptr, colidx, values));
[ad, b] = assemble_dirichlet(problem);
vs = assemble_source_term(problem);

A = A + diag(ad);
b = b + vs;

% Without BCs this should sum to zero
disp(sum(A(:)))
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

function vs = assemble_source_term(p)
     vs = zeros(p.nnodes, 1);
return

function [ad, vd] = assemble_dirichlet(p)
    vd = zeros(p.nnodes, 1);
    ad = zeros(p.nnodes, 1);

%   Not a big fan of this
    h = p.BC_penalization;

    d = p.dirichlet;
    points = p.points;
    
    x = points(1, :);
    y = points(2, :);
    z = points(3, :);

    ux = x(d(2, :)) - x(d(1, :));
    uy = y(d(2, :)) - y(d(1, :));
    uz = z(d(2, :)) - z(d(1, :));

    vx = x(d(3, :)) - x(d(1, :));
    vy = y(d(3, :)) - y(d(1, :));
    vz = z(d(3, :)) - z(d(1, :));

    areas = p_cross(ux, uy, uz, vx, vy, vz);
    areas = sqrt(sum(areas .* areas, 1))/2;

    for i=1:3
        dd = d(1, :);
        for kk=1:length(dd)
            k = dd(kk);
            a = h * (areas(kk) / 3);
            x = points(1, k);
            y = points(2, k);
            z = points(3, k);
            vd(k) = vd(k) + a * p.dirichlet_fun(x, y, z);
            ad(k) = ad(k) + a;
        end
    end
return

function Ae = e_assemble_laplacian(p)
Ae = zeros(p.nelements, 4, 4);
elems = p.elems;
points = p.points;

diffusivity = p.diffusivity;

dn1 = p.dn1;
dn2 = p.dn2;
dn3 = p.dn3;
dn4 = p.dn4;
dn5 = p.dn5;
dn6 = p.dn6;
evol = p.evol;

a = points(:, elems(1, :));
b = points(:, elems(2, :));
c = points(:, elems(3, :));
d = points(:, elems(4, :));

dmb = d - b;
cmb = c - b;
cma = c - a;
dma = d - a;
bma = b - a;

agrad = 1./6 .* p_cross(dmb(1, :), dmb(2, :), dmb(3, :), cmb(1, :), cmb(2, :), cmb(3, :)) ./ evol';
bgrad = 1./6 .* p_cross(cma(1, :), cma(2, :), cma(3, :), dma(1, :), dma(2, :), dma(3, :)) ./ evol';
cgrad = 1./6 .* p_cross(dma(1, :), dma(2, :), dma(3, :), bma(1, :), bma(2, :), bma(3, :)) ./ evol';
dgrad = 1./6 .* p_cross(bma(1, :), bma(2, :), bma(3, :), cma(1, :), cma(2, :), cma(3, :)) ./ evol';

gtest1 = -dn1 - dn2 - dn3;
gtest2 =  dn1 - dn4 - dn5;
gtest3 =  dn2 + dn4 - dn6;
gtest4 =  dn3 + dn5 + dn6;

Ae(:, 1, 1) = sum(agrad .* gtest1, 1);
Ae(:, 1, 2) = sum(bgrad .* gtest1, 1);
Ae(:, 1, 3) = sum(cgrad .* gtest1, 1);
Ae(:, 1, 4) = sum(dgrad .* gtest1, 1);

Ae(:, 2, 1) = sum(agrad .* gtest2, 1);
Ae(:, 2, 2) = sum(bgrad .* gtest2, 1);
Ae(:, 2, 3) = sum(cgrad .* gtest2, 1);
Ae(:, 2, 4) = sum(dgrad .* gtest2, 1);

Ae(:, 3, 1) = sum(agrad .* gtest3, 1);
Ae(:, 3, 2) = sum(bgrad .* gtest3, 1);
Ae(:, 3, 3) = sum(cgrad .* gtest3, 1);
Ae(:, 3, 4) = sum(dgrad .* gtest3, 1);

Ae(:, 4, 1) = sum(agrad .* gtest4, 1);
Ae(:, 4, 2) = sum(bgrad .* gtest4, 1);
Ae(:, 4, 3) = sum(cgrad .* gtest4, 1);
Ae(:, 4, 4) = sum(dgrad .* gtest4, 1);

Ae = -Ae * diffusivity;
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
Ae(:, 2, 2) = -max(-q(1, :), 0) - max(q(4, :), 0) - max(q(5, :), 0);
Ae(:, 2, 1) =  max( q(1, :), 0);
Ae(:, 2, 3) =  max(-q(4, :), 0);
Ae(:, 2, 4) =  max(-q(5, :), 0);

% Node c
Ae(:, 3, 3) = -max(-q(2, :), 0) - max(-q(4, :), 0) - max(q(6, :), 0);
Ae(:, 3, 1) =  max( q(2, :), 0);
Ae(:, 3, 2) =  max( q(4, :), 0);
Ae(:, 3, 4) =  max(-q(6, :), 0);

% Node d
Ae(:, 4, 4) = -max(-q(3, :), 0) - max(-q(5, :), 0) - max(-q(6, :), 0);
Ae(:, 4, 1) =  max(q(3, :), 0);
Ae(:, 4, 2) =  max(q(5, :), 0);
Ae(:, 4, 3) =  max(q(6, :), 0);
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
q6 = vxc(6, :) .* dn6(1, :) + vyc(6, :) .* dn6(2, :) + vzc(6, :) .* dn6(2, :);
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

function [vc] = p_nodal_to_CV_centroids(elems, v)
v0 = v(elems(1, :));
v1 = v(elems(2, :));
v2 = v(elems(3, :));
v3 = v(elems(4, :));
vc = p_CV_face_centroids_interp(v0, v1, v2, v3);
return
