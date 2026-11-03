# Legacy DOLFIN + Gmsh, with native ARM64 support. Pin the upstream manifest.
FROM ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30@sha256:0bf22f477ef594ac5df53bc0b62bee426ad9b1bb184604dd6442718c6f898851

# Notebook execution and VS Code debugging live only inside this image.
RUN python3 -m pip install --no-cache-dir \
    ipykernel==6.29.5 \
    debugpy==1.8.14 \
    nbformat==5.10.4 \
    nbclient==0.10.2

RUN python3 -m ipykernel install --sys-prefix --name fenics-legacy \
    --display-name "Python (FEniCS Legacy)"

WORKDIR /workspace/fenics-legacy
