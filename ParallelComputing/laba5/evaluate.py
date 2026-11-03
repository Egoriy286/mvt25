import numpy as np
import pandas as pd



with np.load("data.npz", allow_pickle=True) as archive:
    data = archive["data"].tolist()
    

df = pd.DataFrame(data)

columns = ["Solver", "Precond", "proc", "time"]
print(df[columns].to_string(index=False))

df.to_csv("data.csv", index=False)