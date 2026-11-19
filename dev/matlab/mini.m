% https://link.springer.com/article/10.1007/s00466-019-01760-w
close all;

d = 2;

if d == 2

    phi0 = @(x, y) (1 - x - y);
    phi1 = @(x, y) x;
    phi2 = @(x, y) y;

    bubble = @(x, y) (x + y <= 1) .* 27 .* (1.0 - x - y) .* x .* y;

    range = linspace (0, 1, 20);
    [X, Y] = meshgrid (range, range);


    f=cell(4, 1);

    f{1} =  @(x, y) 27 .* (1.0 - x - y) .* x .* y;
    f{2} = phi0;
    f{3} = phi1;
    f{4} = phi2;


    syms x y
    dfdx=cell(4,1);
    dfdy=cell(4,1);

    for k=1:4
        dfdx{k} = diff(f{k}(x, y), x);
        dfdy{k} = diff(f{k}(x, y), y);
    end

    L = zeros(4, 4);
    for i=1:4
        for j=1:4
            expr = dfdx{i} * dfdx{j} + dfdy{i} * dfdy{j};

            inty = int(expr, y, [0, 1 - x]);
            intx = int(inty, x, [0, 1]);
            L(i, j) = intx;
        end
    end

    Dx = zeros(4, 3);
    for i=1:4
        for j=1:3
            expr = (dfdx{i}) * f{j+1};
            inty = int(expr, y, [0, 1 - x]);
            intx = int(inty, x, [0, 1]);
            Dx(i, j) = intx;
        end
    end

    Dy = zeros(4, 3);
    for i=1:4
        for j=1:3
            expr = (dfdy{i}) * f{j+1};

            inty = int(expr, y, [0, 1 - x]);
            intx = int(inty, x, [0, 1]);
            Dy(i, j) = intx;
        end
    end
    
    disp('L')
    disp(L)
    disp('Dx')
    disp(Dx)
    disp('Dy')
    disp(Dy)

    surf(X, Y, bubble(X, Y));
else
    phi0 = @(x, y, z) (1 - x - y - z);
    phi1 = @(x, y, z) x;
    phi2 = @(x, y, z) y;
    phi3 = @(x, y, z) z;

    bubble = @(x, y, z) (x + y + z <= 1) .* 256 .* (1.0 - x - y - z) .* x .* y .* z

    range = linspace (0, 1, 10);
    [X, Y, Z] = meshgrid (range, range, range);

    v = bubble(X, Y, Z);

    f = {bubble, phi0, phi1, phi2, phi3};
    % f = {phi0, phi1, phi2, phi3};

    X = X(:)
    Y = Y(:)
    Z = Z(:)

    idx = (X + Y + Z) < 1

    XX = X(idx)
    YY = Y(idx)
    ZZ = Z(idx)

    fx = zeros(size(XX));
    for i=1:length(f)
        fx += f{i}(XX, YY, ZZ);
    end

    fx = fx(:)

    disp(fx)

    % disp(max(v(:)))
end
