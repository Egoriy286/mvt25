l = 1;
r = 0.2;
ls = 0.02;
Point(1) = {0, 0, 0, ls};
Point(2) = {0, l, 0, ls};
Point(3) = {l, l, 0, ls};
Point(4) = {l, 0, 0, ls};

cx = l/2;
cy = l/2;
Point(5) = {cx,cy, 0, ls};
Point(6) = {cx-r,cy, 0, ls};
Point(7) = {cx,cy+r, 0, ls};
Point(8) = {cx+r,cy, 0, ls};
Point(9) = {cx,cy-r, 0, ls};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};
Plane Surface(2) = {2};


Physical Surface(1) = {1};
Physical Surface(2) = {2};