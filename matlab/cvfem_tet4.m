t = [
    0, 0, 0;
    1, 0, 0;
    0, 1, 0;
    0, 0, 1
]';

% Faces opposite to corner
tri = [ 
    2, 3, 4, ...
    3, 1, 4, ...
    2, 4, 1, ...
    3, 2, 1
]';

tri = reshape(tri, 4, 3);


% Edge midpoints
emp = [];

for i=1:4
    for j=(i+1):4
        mp = (t(:, i) + t(:, j))/2;
        emp = [emp, mp];
    end
end

% Face centroids
temp = 1:3;
cf = [];
for i=1:4
    cfi = sum(t(:, tri(temp)),  2)/3;
    cf = [cf, cfi];
    temp = temp + 3;
end

% Volume centroid
c = sum(t, 2)/4;

close all;

figure(1);
hold on;

ptri = triangulation(tri, t(1, :)', t(2, :)', t(3, :)');
trisurf(ptri, 'FaceAlpha', 0.1);
plot3(t(1, :), t(2, :), t(3, :), 'b*');
plot3(c(1, :), c(2, :), c(3, :), 'r*');
plot3(cf(1, :), cf(2, :), cf(3, :), 'g*');
plot3(emp(1, :), emp(2, :), emp(3, :), 'm*');
xlabel('x')
ylabel('y')
zlabel('z')

eps = 0.1;
axis([-eps, 1 + eps, -eps, 1 + eps, -eps, 1 + eps])



figure(2);
for i=1:4
% for i=2
    subplot(2, 2, i);
    hold on;
    trisurf(ptri, 'FaceAlpha', 0.1);
    xlabel('x')
    ylabel('y')
    zlabel('z')

    b = (i-1) * 3 + 1;
    e = b + 2;
    o = tri(b:e);  
    
    p0 = t(:, i);
    p1 = t(:, o(1));
    p2 = t(:, o(2));
    p3 = t(:, o(3));
    
    m01 = (p0 + p1)/2;
    m02 = (p0 + p2)/2;
    m03 = (p0 + p3)/2;

    bf012 = (p0 + p1 + p2)/3;
    bf013 = (p0 + p1 + p3)/3;
    bf023 = (p0 + p2 + p3)/3;

    f1 = [m01,  bf013, c, bf012];
    f2 = [m03,  bf023, c, bf013];
    f3 = [m02,  bf012, c, bf023];

    [a1, area1] = poly_surf_area_normal(f1);
    [a2, area2] = poly_surf_area_normal(f2);
    [a3, area3] = poly_surf_area_normal(f3);

    pf1 = [f1, f1(:, 1)];
    pf2 = [f2, f2(:, 1)];
    pf3 = [f3, f3(:, 1)];

    plot3(pf1(1, :), pf1(2, :), pf1(3, :), 'r', 'LineWidth', 2);
    plot3(pf2(1, :), pf2(2, :), pf2(3, :), 'g');
    plot3(pf3(1, :), pf3(2, :), pf3(3, :), 'm');
    
    
    bf1 = sum(f1, 2)/4;
    bf2 = sum(f2, 2)/4;
    bf3 = sum(f3, 2)/4;

    quiver3(bf1(1), bf1(2), bf1(3), a1(1), a1(2), a1(3), 'LineWidth', 2);
    quiver3(bf2(1), bf2(2), bf2(3), a2(1), a2(2), a2(3), 'g');
    quiver3(bf3(1), bf3(2), bf3(3), a3(1), a3(2), a3(3), 'm');

    legend('', ['f1 (' num2str(area1) ')'],  ['f2 (' num2str(area2) ')'],  ['f3 (' num2str(area3) ')'], 'n1', 'n2', 'n3')
end

axis([-eps, 1 + eps, -eps, 1 + eps, -eps, 1 + eps])

function [a, area] = poly_surf_area_normal(poly)
    a = 0;
    area = 0;
    n = size(poly, 2);
    for i=1:n-1
        ip1 = i+1;

        if ip1 == n
            ip2 = 1;
        else
            ip2 = ip1 + 1;
        end

        p0 = poly(:, i);
        p1 = poly(:, ip1);
        p2 = poly(:,ip2);

        J = [p1 - p0, p2 - p0];
        J2 = J'*J;

        dA = cross(J(:, 1), J(:, 2)) / 2;
        a = a+dA;
        area = area + sqrt(det(J2))/2;
    end

%     dA / norm(dA, 2)
%     a / norm(a, 2)
% 
%     area
%     norm(a)    
end

