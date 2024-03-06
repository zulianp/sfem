// Center
Point(1) = { 0, 0, 0, 1.0};

// Inner circle
Point(2) = { 1, 0, 0, 1.0};
Point(3) = { 0, 1, 0, 1.0};

//  Outer circle
Point(4) = { 2, 0, 0, 1.0};
Point(5) = { 0, 2, 0, 1.0};

Circle(1) = {2, 1, 3};
Circle(2) = {4, 1, 5};
Line(3) = {5, 3};
Line(4) = {2, 4};

Curve Loop(1) = {3, -1, 4, 2};
Plane Surface(1) = {1};

Extrude {0, 0, 1} {
  Curve{2}; Curve{3}; Curve{1}; Curve{4}; 
}

Curve Loop(2) = {9, -13, 17, 5};
Surface(21) = {2};


Curve Loop(3) = {13, -11, -1, 14};
Surface(22) = {3};

Curve Loop(4) = {2, 7, -5, -6};
Surface(23) = {4};


//+
Surface Loop(1) = {21, 12, 1, 20, 16, 8};
//+
Volume(1) = {1};
//+
Physical Volume("domain", 24) = {1};
