// Pract 2: [0,2] x [0,1] with two wells of radius 0.02.
// Mesh parameters are edited here. Physical tags survive dolfin-convert.
L = 2;
H = 1;
r = 0.02;
x1 = 0.5;
x2 = 1.5;
yc = 0.5;
If (!Exists(h))
  h = 0.08;
EndIf
hw = 0.008;
nArc = 16; // 64 boundary segments per well on the coarse mesh.

Point(1) = {0, 0, 0, h};
Point(2) = {L, 0, 0, h};
Point(3) = {L, H, 0, h};
Point(4) = {0, H, 0, h};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};

Point(5) = {x1, yc, 0, hw};
Point(6) = {x1+r, yc, 0, hw};
Point(7) = {x1, yc+r, 0, hw};
Point(8) = {x1-r, yc, 0, hw};
Point(9) = {x1, yc-r, 0, hw};
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};
Curve Loop(2) = {5, 6, 7, 8};

Point(10) = {x2, yc, 0, hw};
Point(11) = {x2+r, yc, 0, hw};
Point(12) = {x2, yc+r, 0, hw};
Point(13) = {x2-r, yc, 0, hw};
Point(14) = {x2, yc-r, 0, hw};
Circle(9) = {11, 10, 12};
Circle(10) = {12, 10, 13};
Circle(11) = {13, 10, 14};
Circle(12) = {14, 10, 11};
Curve Loop(3) = {9, 10, 11, 12};
Plane Surface(1) = {1, 2, 3};

Transfinite Curve {5:12} = nArc + 1;
Physical Curve("outer", 1) = {1, 2, 3, 4};
Physical Curve("well_left", 2) = {5, 6, 7, 8};
Physical Curve("well_right", 3) = {9, 10, 11, 12};
Physical Surface("reservoir", 10) = {1};
Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
Mesh.MshFileVersion = 2.2;
Mesh.SaveAll = 0;
