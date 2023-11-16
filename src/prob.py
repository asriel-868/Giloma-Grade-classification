from imports import *


def main():
    data = pd.read_csv("training_data.csv")
    data["labels"] = pd.read_csv("training_data_targets.csv", names=["Target"])

    data.replace(["--", "not reported"], pd.NA, inplace=True)
    data.dropna(axis=0, how="any", inplace=True)
    data.reset_index(drop=True, inplace=True)

    labels = data["labels"]
    data.drop(["labels"], axis=1, inplace=True)
    data["Age_at_diagnosis"] = data["Age_at_diagnosis"].apply(utils.convert_to_year)
    print(data.shape)


main()
