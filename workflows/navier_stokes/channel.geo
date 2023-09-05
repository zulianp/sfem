//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {3, 0, 0, 1.0};
//+
Point(3) = {3, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Point(5) = {0.70, 0.30, 0, 1.0};
//+
Point(6) = {0.70, 0.70, 0, 1.0};
//+
Point(7) = {0.3, 0.5, 0, 1.0};
//+
Point(8) = {0.7, 0.5, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {3, 4};
//+
Line(3) = {2, 3};
//+
Line(4) = {4, 1};
//+
Circle(5) = {5, 8, 6};
//+
Circle(6) = {6, 8, 5};
//+
Recursive Delete {
  Point{7}; Point{8}; 
}
//+
Recursive Delete {
  Point{8}; 
}
//+
Curve Loop(1) = {2, 4, 1, 3};
//+
Curve Loop(2) = {5, 6};
//+
Plane Surface(1) = {1, 2};
