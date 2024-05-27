clear variables;

r = readmatrix("rule56.csv");

s = 0 ; 
nodes = zeros(3, size(r,1));
e = [0 0 0 ; 1 0 0 ; 0 1 0 ; 0 0 1]';

for n = 1:size(r,1)

    s = 0;
    for i = 1:4 
        s = s +  e(:,i) * r(n, i) ;
    end
    nodes(:, n) = s;
end

scatter3(nodes(1,:), nodes(2,:), nodes(3,:))

