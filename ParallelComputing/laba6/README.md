# Laba 6

## Requirements

Install a C++ compiler and an MPI implementation (for example, Open MPI). The MPI compiler wrapper `mpic++` and launcher `mpirun` must be available on your `PATH`.

## Compile

From this directory, compile the project's C++ source file. Replace `source.cpp` with the actual source filename; include any additional project source files in the command if needed.

```bash
mpicc laba6.c -o laba6
```

## Run

Launch the compiled program with the desired number of processes. For example, to use four processes:

```bash
mpirun -np 4 ./laba6
```
