
import subprocess
import sys
import numpy as np

solvers = ["default", "bicgstab", "cg", "gmres", "minres", "tfqmr"]
preconditioners = ["none", "default", "sor", "icc"]
processors = [1, 2, 4, 6, 8] 
data = []
N = len(processors)*len(preconditioners)*len(solvers)
i = 0
for proc in processors:
    for solver in solvers:
        for precond in preconditioners:
            i += 1
            print(f"{solver} - {precond} \t: {100*i/N:.3f}%")
            
            # Build the mpirun command list
            cmd = [
                "mpirun",
                "-np",
                f"{proc}",
                sys.executable,  # Uses the current Python interpreter
                "solve.py",
                "--arg1",
                f"{solver}",
                "--arg2",
                f"{precond}",
            ]

            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.stdout:
                raw = {
                    "Precond": precond,
                    "Solver": solver,
                    "proc": proc,
                    "time": result.stdout
                    }
                data.append(raw)
            if result.stderr:
                raw = {
                    "Precond": precond,
                    "Solver": solver,
                    "proc": proc,
                    "time": None
                    }
                data.append(raw)

np.savez("data.npz", data=np.array(data, dtype=object))