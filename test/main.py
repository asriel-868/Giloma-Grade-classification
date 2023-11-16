from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

def main():
    data = pd.read_csv("training_data.csv")
    labels = pd.read_csv("training_data_targets.csv", names=["Labels"])

    # Appending labels to dataframe
    data["Labels"] = labels

    if os.path.exists("freq.txt"):
        os.remove("freq.txt")

    # Getting some preliminary analysis
    for i in data.columns:
        with open("freq.txt", "a") as f:
            f.write("--------------------------\n")
            f.write(str(data[i].value_counts()))
            f.write("\n\n\n")

    # Dropping missing values from the dataset 
    data.replace(["--", "not reported"], pd.NA, inplace=True)
    data.dropna(axis=0, how="any", inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Converting age to numerical value
    data["Age_at_diagnosis"] = data["Age_at_diagnosis"].apply(convert_to_year)
    
    # Dropping the labels column from the dataframe and storing it seperately
    labels = data["Labels"]
    data.drop("Labels", axis=1, inplace=True)
    
    # Getting the numerical and categorical features 
    numerical_features = data.select_dtypes(include=["number"]).columns
    categorical_features = data.select_dtypes(include=["object"]).columns
    
    data_encoded = pd.get_dummies(data, columns=categorical_features, dtype=int)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(data_encoded, encoded_labels, test_size=0.92, stratify=encoded_labels, random_state=42)
    
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    
    svm = SVC(kernel="rbf", C=1.0)
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    print(f"Accuracy : {accuracy_score(y_test, pred)}")
    print(f"Report : \n {classification_report(y_test, pred)}")
        

def convert_to_year(inp):
    pattern = r"(\d+) years(?: (\d+) days)?"
    match = re.search(pattern, inp)

    if match:
        x = int(match.group(1))
        if match.group(2):
            y = int(match.group(2))
        else:
            y = 0

        return float(x + (y / 365))
    else:
        print(inp)
        print("Age is in wrong format")

main()