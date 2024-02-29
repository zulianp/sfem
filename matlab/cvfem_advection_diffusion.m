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

mesh = '/Users/patrickzulian/Desktop/code/sfem/tests/cvfem3/mesh';
lapl = '/Users/patrickzulian/Desktop/code/sfem/tests/cvfem3/system';

crs_rowtr = read_array([lapl '/rowptr.raw'], 'int32') + 1;
crs_colidx = read_array([lapl '/colidx.raw'], 'int32') + 1;
crs_values = zeros(size(crs_colidx));

nneighs = crs_rowtr(2:end) - crs_rowtr(1:end-1);
max_nneighs = max(nneighs);


i0 = read_array([mesh '/i0.raw'], 'int32') + 1;
i1 = read_array([mesh '/i1.raw'], 'int32') + 1;
i2 = read_array([mesh '/i2.raw'], 'int32') + 1;
i3 = read_array([mesh '/i3.raw'], 'int32') + 1;

x = read_array([mesh '/x.raw'], 'float');
y = read_array([mesh '/y.raw'], 'float');
z = read_array([mesh '/z.raw'], 'float');

% Basic test
% i0 = [1, 1]';
% i1 = [2, 2]';
% i2 = [3, 3]';
% i3 = [4, 4]';
% 
% x = [0.0, 1.0, 0.0, 0.]';
% y = [0.0, 0.0, 1.0, 0.]';
% z = [0.0, 1.0, 0.0, 10.]';

dirichlet = unique(sort(read_array([mesh '/sidesets_aos/sinlet.raw'], 'int32') + 1));
neumann   = read_array([mesh '/sidesets_aos/soutlet.raw'], 'int32') + 1;
neumann   = reshape(neumann, 3, length(neumann)/ 3);

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

disp(size(dn1));

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
size(qphi)


%% Plot stuff
close all;
ptri = triangulation(neumann', x, y, z);
trisurf(ptri, 'FaceAlpha', 0.1);
hold on;
plot3(x(dirichlet), y(dirichlet), z(dirichlet), '*');


return

%% Algebraic system
[ap, anb] = assemble(nnodes, max_nneighs, )
anb = zeros(nnodes, max_nneighs);

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

[n0, n1, n2] = cross(u0, u1, u2, v0, v1, v2);
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
dn6 =  1./12 * cross_ab - 1./24 * cross_ac - 1./24 *cross_ad + 1./24 * cross_bc - 1./24 * cross_bd;
    
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

function [n0, n1, n2] = cross(a0, a1, a2, b0, b1, b2)
n0 = a1 .* b2 - a2 .* b1;
n1 = a2 .* b0 - a0 .* b2;
n2 = a0 .* b1 - a1 .* b0;
return

function [n]= p_cross(a0, a1, a2, b0, b1, b2)
[n0, n1, n2] = cross(a0, a1, a2, b0, b1, b2);
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
