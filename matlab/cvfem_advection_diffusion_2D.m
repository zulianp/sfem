function cvfem_advection_diffusion_2D()
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

mesh = './square/mesh';
lsystem = './square/crs';

% ../python/sfem/mesh/raw_to_db.py /Users/patrickzulian/Desktop/code/sfem/tests/cvfem3/mesh out.vtk --point_data=solution.raw

rowptr = read_array([lsystem  '/rowptr.raw'], 'int32') + 1;
colidx = read_array([lsystem  '/colidx.raw'], 'int32') + 1;


i0 = read_array([mesh '/i0.raw'], 'int32') + 1;
i1 = read_array([mesh '/i1.raw'], 'int32') + 1;
i2 = read_array([mesh '/i2.raw'], 'int32') + 1;

x = read_array([mesh '/x.raw'], 'float');
y = read_array([mesh '/y.raw'], 'float');


% dirichlet = unique(sort(read_array([mesh '/sidesets_aos/sinlet.raw'], 'int32') + 1));
dirichlet = read_array([mesh '/sidesets_aos/sleft.raw'], 'int32') + 1;
dirichlet   = reshape(dirichlet, 2, length(dirichlet)/ 2);

neumann   = read_array([mesh '/sidesets_aos/sright.raw'], 'int32') + 1;
neumann   = reshape(neumann, 2, length(neumann)/ 2);

else

rowptr = [1, 5, 9, 13, 17];
colidx = [
    1, 2, 3, 4, ...
    1, 2, 3, 4, ...
    1, 2, 3, 4, ... 
    1, 2, 3, 4 ];

i0 = [1,2]';
i1 = [2,4]';
i2 = [3,3]';

x = [0.0, 1.0, 0.0, 1.0]';
y = [0.0, 0.0, 1.0, 1.0]';

dirichlet = [2; 1];
neumann = [2, 4];

end

elems = [i0'; i1'; i2'];
points = [x'; y'];
evol = element_volumes(elems, points);
cv_vol = CV_volume(length(x), elems, evol);

disp('evol')
disp(sum(evol))
disp('cv_vol')
disp(sum(cv_vol))
disp('cv_vol/evol')
disp(sum(cv_vol)./sum(evol))

[dn1, dn2, dn3] = CV_normals(elems, points);

% disp('dn')
% disp(dn1);
% disp(dn2);
% disp(dn3);

vx = ones(size(x));
vy = zeros(size(x));

% Compute velocties at the centroids
vxc = p_nodal_to_CV_centroids(elems, vx);
vyc = p_nodal_to_CV_centroids(elems, vy);

q = advective_fluxes(vxc, vyc, dn1, dn2, dn3);
% qphi = upwind_scheme(elems, phi, q);

problem = {};
problem.q = q;
% problem.qphi = qphi;
problem.vxc = vxc;
problem.vyc = vyc;

problem.dn1 = dn1;
problem.dn2 = dn2;
problem.dn3 = dn3;

problem.elems = elems;
problem.points = points;
problem.evol = evol;
problem.cv_vol = cv_vol;
problem.neumann = neumann;

problem.BC_penalization = 1e16;
problem.diffusivity = 1;

problem.dirichlet_nodes = unique(sort(dirichlet(:)));
problem.dirichlet = dirichlet;
problem.dirichlet_fun = @(x, y) 1.;

problem.rowptr = rowptr;
problem.colidx = colidx;

problem.nelements = size(elems, 2);
problem.nnodes = length(x);

[A, b] = assemble(problem);
sol = A\b;
% sol = ones(size(b));

write_array('solution.raw', sol, 'double');

%% Plot stuff
% close all;
% if(size(neumann, 1) == 1)
%     ptri = triangulation(neumann, x, y);
% else
ptri = triangulation(elems', x, y);
% end
trimesh(elems', x, y, sol, 'FaceAlpha', 0.1);
% hold on;
% plot3(x(dirichlet), y(dirichlet), z(dirichlet), '*');
% xlabel('x');
% ylabel('y');
% zlabel('z');
return

%% Algebraic system
function [A, b] = assemble(problem)
rowptr = problem.rowptr;
colidx = problem.colidx;

lap_e = e_assemble_laplacian(problem);
adv_e = e_assemble_advection(problem);
% mat_e = lap_e + adv_e;
mat_e = lap_e;

% disp('adv')
% disp(reshape(lap_e(1, :, :), 3, 3));
% disp(reshape(adv_e(1, :, :), 3, 3));
% disp(reshape(adv_e(2, :, :), 3, 3));

% values = elemental_to_crs(problem.elems, adv_e, rowptr, colidx);
values = elemental_to_crs(problem.elems, mat_e, rowptr, colidx);

A = sparse(crs_to_dense(rowptr, colidx, values));
[ad, b] = assemble_dirichlet(problem);
vs = assemble_source_term(problem);

format short;
% disp(full(A))
disp(sum(A(:)))

A = A + diag(ad);
b = b + vs;

% Without BCs this should sum to zero

return

function values = elemental_to_crs(elems, mat_e, rowptr, colidx)
values = zeros(size(colidx));

assert(length(values) == rowptr(end)-1);

for e=1:size(elems, 2)
    idx = elems(:, e);

    for ii=1:3
        i = idx(ii);
        row = colidx(rowptr(i):(rowptr(i+1)-1));
       
        for jj=1:3
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
   
    ux = x(d(2, :)) - x(d(1, :));
    uy = y(d(2, :)) - y(d(1, :));
   
    
    areas = sqrt(ux.*ux + uy.*uy);
    
    for i=1:2
        dd = d(1, :);
        for kk=1:length(dd)
            k = dd(kk);
            a = h * (areas(kk) / 3);
            x = points(1, k);
            y = points(2, k);
            
            vd(k) = vd(k) + a * p.dirichlet_fun(x, y);
            ad(k) = ad(k) + a;
        end
    end
return

function Jinv = inv2(u, v)
det = (u(1, :) .* v(2, :) - u(2, :) .* v(1, :));

Jinv = [
    v(2, :);
    -v(1, :);
    -u(2, :);
    u(1, :)
] ./det;

return

function Ae = e_assemble_laplacian(p)
Ae = zeros(p.nelements, 3, 3);
elems = p.elems;
points = p.points;

diffusivity = p.diffusivity;

dn1 = p.dn1;
dn2 = p.dn2;
dn3 = p.dn3;

evol = p.evol;

p1 = points(:, elems(1, :));
p2 = points(:, elems(2, :));
p3 = points(:, elems(3, :));

u = p2 - p1; 
v = p3 - p1;

Jinv = inv2(u, v);

% Inverse transpose
J11 = Jinv(1, :);
J12 = Jinv(3, :);
J21 = Jinv(2, :);
J22 = Jinv(4, :);

agradx = -J11 - J12;
agrady = -J21 - J22;

bgradx = J11;
bgrady = J21;

cgradx = J21;
cgrady = J22;

agrad = 1./2 .* [agradx; agrady] ./ evol;
bgrad = 1./2 .* [bgradx; bgrady] ./ evol;
cgrad = 1./2 .* [cgradx; cgrady] ./ evol;

gtest1 =  dn1 - dn3;
gtest2 = -dn1 + dn2;
gtest3 = -dn2 + dn3;

Ae(:, 1, 1) = sum(agrad .* gtest1, 1);
Ae(:, 1, 2) = sum(bgrad .* gtest1, 1);
Ae(:, 1, 3) = sum(cgrad .* gtest1, 1);

Ae(:, 2, 1) = sum(agrad .* gtest2, 1);
Ae(:, 2, 2) = sum(bgrad .* gtest2, 1);
Ae(:, 2, 3) = sum(cgrad .* gtest2, 1);

Ae(:, 3, 1) = sum(agrad .* gtest3, 1);
Ae(:, 3, 2) = sum(bgrad .* gtest3, 1);
Ae(:, 3, 3) = sum(cgrad .* gtest3, 1);

Ae = -Ae * diffusivity;
return

function Ae = e_assemble_advection(p)
Ae = zeros(p.nelements, 3, 3);
elems = p.elems;
points = p.points;
q = p.q;

% disp('q')
% disp(q)

% Node a
Ae(:, 1, 1) = -max( q(1, :), 0) - max(-q(3, :), 0);
Ae(:, 1, 2) = -max(-q(1, :), 0);
Ae(:, 1, 3) = max(q(3, :), 0);

% Node b
Ae(:, 2, 2) = -max(-q(1, :), 0) - max(q(2, :), 0);
Ae(:, 2, 1) = max(q(1, :), 0);
Ae(:, 2, 3) = max(-q(2, :), 0);


% Node c
Ae(:, 3, 3) = -max(-q(2, :), 0) - max(q(3, :), 0);
Ae(:, 3, 1) = max(-q(3, :), 0);
Ae(:, 3, 2) = max(q(2, :), 0);

return


%% Elemental quantities
function qphi = upwind_scheme(elems, phi, q)
phi1 = phi(elems(1, :))';
phi2 = phi(elems(2, :))';
phi3 = phi(elems(3, :))';

qphi1 = max(q(1, :), 0) .* phi1 - max(-q(1, :), 0) .* phi2;
qphi2 = max(q(2, :), 0) .* phi1 - max(-q(2, :), 0) .* phi3;
qphi3 = max(q(3, :), 0) .* phi1 - max(-q(3, :), 0) .* phi4;

qphi = [qphi1; qphi2; qphi3];
return

function q = advective_fluxes(vxc, vyc, dn1, dn2, dn3)
q1 = vxc(1, :) .* dn1(1, :) + vyc(1, :) .* dn1(2, :);
q2 = vxc(2, :) .* dn2(1, :) + vyc(2, :) .* dn2(2, :);
q3 = vxc(3, :) .* dn3(1, :) + vyc(3, :) .* dn3(2, :);
q = [q1; q2; q3];
return

function vol = element_volumes(elems, points)
x0 = points(1, elems(1, :));
x1 = points(1, elems(2, :));
x2 = points(1, elems(3, :));

y0 = points(2, elems(1, :));
y1 = points(2, elems(2, :));
y2 = points(2, elems(3, :));

u0 = x1 - x0;
u1 = x2 - x0;

v0 = y1 - y0;
v1 = y2 - y0;

vol = (u0 .* v1 - u1 .* v0) ./ 2;
return

function ret = perp(v)
    ret = [
        -v(2, :);
        v(1, :)
        ];
return

function [dn1, dn2, dn3] = CV_normals(elems, points)
a = points(:, elems(1, :));
b = points(:, elems(2, :));
c = points(:, elems(3, :));

bary = (a + b + c) ./ 3;
e1 = (a + b) / 2;
e2 = (b + c) / 2;
e3 = (c + a) / 2;

% s1 = bary - e1;
% s2 = bary - e2;
% s3 = e3 - bary;

s1 = e1 - bary;
s2 = e2 - bary;
s3 = e3 - bary;


dn1 = perp(s1);
dn2 = perp(s2);
dn3 = perp(s3);
return

function cv_vol = CV_volume(ncvs, elems, evol)
cv_vol = zeros(ncvs, 1);

nelements = size(elems, 2);

for d=1:3
    for i=1:nelements
        idx = elems(d, i);
        cv_vol(idx) = cv_vol(idx) + evol(i);
    end
end

cv_vol = cv_vol ./ 3;
return

function [vc1, vc2, vc3] = CV_face_centroids_interp(v0, v1, v2)
w0 = 5./12;
w1 = 1./6;
vc1 = (w0 * v0) + (w0 * v1) + (w1 * v2);
vc2 = (w1 * v0) + (w0 * v1) + (w0 * v2);
vc3 = (w0 * v0) + (w1 * v1) + (w0 * v2);
return

function [vc] = p_CV_face_centroids_interp(v0, v1, v2)
[vc1, vc2, vc3] = CV_face_centroids_interp(v0, v1, v2);
vc =  [vc1; vc2; vc3];
return

function [vc] = p_nodal_to_CV_centroids(elems, v)
v0 = v(elems(1, :));
v1 = v(elems(2, :));
v2 = v(elems(3, :));
vc = p_CV_face_centroids_interp(v0, v1, v2);
return
