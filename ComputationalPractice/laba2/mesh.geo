l1 = 1;
l2 = 0.5;
ls = 0.015;
h = 1;
Point(1) = {0,0,0,ls};
Point(2) = {0, h,0,ls};
Point(3) = {l1,h,0,ls};
Point(4) = {l1, 0,0,ls};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Point(5) = {l1+l2, h, 0,ls};
Point(6) = {l1+l2, 0, 0,ls};


//+
Line(5) = {3, 5};
//+
Line(6) = {5, 6};
//+
Line(7) = {6, 4};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {3, -7, -6, -5};
//+
Plane Surface(2) = {2};
