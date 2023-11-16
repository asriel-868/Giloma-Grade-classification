import pandas as pd
import os

df = pd.read_csv("training_data.csv")
df["labels"] = pd.read_csv("training_data_targets.csv", header=None)

# df.replace(["--", "not reported"], pd.NA, inplace=True)
# df.dropna(axis=0, how="any", inplace=True)
# df.reset_index(drop=True, inplace=True)


if os.path.exists("analysis.txt"):
    os.remove("analysis.txt")
with open("analysis.txt", "w") as f:
    count = 0
    for _, row in df.iterrows():
        if row["labels"] == "GBM" and row["Primary_Diagnosis"] != "Glioblastoma":
            f.write("---------------\n")
            f.write(str(row))
            f.write("\n")



