// Квадрат 1x1. Грубая сетка n x n, внутри каждой ячейки fine разбиений.
If (!Exists(n))
  n = 10;
EndIf
If (!Exists(fine))
  fine = 4;
EndIf

For j In {0:n}
  For i In {0:n}
    Point(1 + j*(n+1) + i) = {i/n, j/n, 0, 1/(n*fine)};
  EndFor
EndFor

horizontal[] = {};
For j In {0:n}
  For i In {0:n-1}
    id = newl;
    Line(id) = {1+j*(n+1)+i, 2+j*(n+1)+i};
    horizontal[] += {id};
  EndFor
EndFor

vertical[] = {};
For j In {0:n-1}
  For i In {0:n}
    id = newl;
    Line(id) = {1+j*(n+1)+i, 1+(j+1)*(n+1)+i};
    vertical[] += {id};
  EndFor
EndFor

surfaces[] = {};
For j In {0:n-1}
  For i In {0:n-1}
    loop = newll;
    Curve Loop(loop) = {horizontal[j*n+i], vertical[j*(n+1)+i+1],
                      -horizontal[(j+1)*n+i], -vertical[j*(n+1)+i]};
    s = news;
    Plane Surface(s) = {loop};
    surfaces[] += {s};
  EndFor
EndFor

Transfinite Curve{:} = fine + 1;
Transfinite Surface{:};
Recombine Surface{:};
Mesh.MshFileVersion = 2.2;
Mesh.SaveAll = 1;

// Вытягиваем плоскую сетку в куб.
For k In {0:n-1}
  out[] = Extrude {0, 0, 1/n} {
    Surface{surfaces[]}; Layers{fine}; Recombine;
  };
  // Для четырёхугольного основания: верх, объём, четыре боковые грани.
  For s In {0:n*n-1}
    surfaces[s] = out[6*s];
  EndFor
EndFor
