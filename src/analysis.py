from utils import *

data = pd.read_csv("training_data.csv")
data["labels"] = pd.read_csv("training_data_targets.csv", header=None)

data.replace(["--", "not reported"], pd.NA, inplace=True)
data.dropna(axis=0, how="any", inplace=True)
data.reset_index(drop=True, inplace=True)

data["Age_at_diagnosis"] = data["Age_at_diagnosis"].apply(utils.convert_to_year)


data = pd.get_dummies(data)

# getting the correlation matrix
correlation_matrix = data.corr()
correlation_with_target = correlation_matrix["labels_GBM"].drop(["labels_GBM", "labels_LGG"])

# Writing out the correlation of all columns to the 'labels_GBM' column
if os.path.exists("analysis.txt"):
    os.remove("analysis.txt")
with open("analysis.txt", "w") as f:
    f.write("----------------- Correlation Matrix with Target class GBM -----------------\n\n")
    f.write(correlation_with_target.to_string())
    f.write("\n\n")
