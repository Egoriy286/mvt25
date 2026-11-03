// Куб 1x1x1 из n x n x n блоков с тетраэдральной сеткой шага примерно h.
SetFactory("OpenCASCADE");
n = 10;
fine = 4;
h = 1/(n*fine);

For k In {0:n-1}
  For j In {0:n-1}
    For i In {0:n-1}
      Box(newv) = {i/n, j/n, k/n, 1/n, 1/n, 1/n};
    EndFor
  EndFor
EndFor

// Соседние блоки используют общие грани.
Coherence;
Mesh.MeshSizeMin = h;
Mesh.MeshSizeMax = h;
Mesh.MshFileVersion = 2.2;
Mesh.SaveAll = 1;
