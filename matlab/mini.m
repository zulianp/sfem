% https://link.springer.com/article/10.1007/s00466-019-01760-w
close all;

d = 2

if d == 2

  phi0 = @(x, y) (1 - x - y);
  phi1 = @(x, y) x;
  phi2 = @(x, y) y;

  bubble = @(x, y) (x + y <= 1) .* 27 .* (1.0 - x - y) .* x .* y

  range = linspace (0, 1, 50);
  [X, Y] = meshgrid (range, range);


  surf (X, Y, bubble(X, Y));
else
  phi0 = @(x, y, z) (1 - x - y - z);
  phi1 = @(x, y, z) x;
  phi2 = @(x, y, z) y;


  bubble = @(x, y, z) (x + y + z <= 1) .* 256 .* (1.0 - x - y - z) .* x .* y .* z

  range = linspace (0, 1, 500);
  [X, Y, Z] = meshgrid (range, range, range);

  v = bubble(X, Y, Z);

  disp(max(v(:)))
end
