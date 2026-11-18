// Квадрат 1x1 из 10x10 грубых ячеек.
SetFactory("OpenCASCADE");
n = 10;
fine = 8;
h = 1/(n*fine);

For j In {0:n-1}
  For i In {0:n-1}
    Rectangle(news) = {i/n, j/n, 0, 1/n, 1/n};
  EndFor
EndFor

// Объединяем совпадающие линии: у соседних квадратов общие рёбра.
Coherence;
Mesh.MeshSizeMin = h;
Mesh.MeshSizeMax = h;
Mesh.MshFileVersion = 2.2;
Mesh.SaveAll = 1;
