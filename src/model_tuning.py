""" I have commented out all the code which I used for tuning various models """

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import re

""" Function to convert age to float. The age is expected to be in the format : X years Y days """


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

data = pd.read_csv("training_data.csv")
data["labels"] = pd.read_csv("training_data_targets.csv", header=None)
data.drop(["Primary_Diagnosis", ], axis=1, inplace=True)

missing_values = ["--", "not reported"]

data.replace(missing_values, pd.NA, inplace=True)
data.dropna(axis=0, how="any", inplace=True)
data.reset_index(drop=True, inplace=True)

labels = data["labels"]
data.drop(["labels", ], axis=1, inplace=True)

data["Age_at_diagnosis"] = data["Age_at_diagnosis"].apply(convert_to_year)

categorical_features = data.select_dtypes(include=["object"]).columns
numeric_features = ["Age_at_diagnosis"]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", StandardScaler(), numeric_features)
    ])

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("knn", KNeighborsClassifier())
    ]
)
x_train, x_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.2, random_state=42)

param_grid_lasso = {"feature_selector__alpha": [0.001, 0.01, 0.1]}

# grid_search_lasso = GridSearchCV(pipeline_lasso, param_grid=param_grid_lasso, cv=5, scoring="neg_mean_squared_error")
# grid_search_lasso.fit(x_train, y_train)
# best_alpha = grid_search_lasso.best_params_["feature_selector__alpha"]

# print(f"Best Alpha: {best_alpha}")

# param_grid_svm = {
#     'svm__C': [0.1, 1, 10, 100],  # Regularization parameter
#     'svm__kernel': ['linear', 'rbf'],  # Kernel type
#     'svm__gamma': ["auto", "scale"]  # Kernel coefficient for 'rbf' kernel
# }

# param_grid_lr = {
#     'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
#     'lr__penalty': ['l1', 'l2'],  # Regularization penalty ('l1' or 'l2')
#     'lr__solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg']  # Algorithm to use in the optimization problem
# }

# param_grid_rfc = {
#     'rf__n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
#     'rf__max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
#     'rf__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
#     'rf__min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
# }

# base_estimators = [
#     DecisionTreeClassifier(max_depth=1),
#     DecisionTreeClassifier(max_depth=2),
#     SVC(kernel='linear', C=1),
#     KNeighborsClassifier(n_neighbors=3)
# ]

# param_grid_adaboost = {
#     'adaboost__n_estimators': [50, 100, 200],  # Number of weak learners (base estimators)
#     'adaboost__learning_rate': [0.001, 0.01, 0.1, 1],  # Weighting of weak learners
#     'adaboost__base_estimator': base_estimators
# }

# param_grid_knn = {
#     'knn__n_neighbors': [3, 5, 7, 10],  # Number of neighbors
#     'knn__weights': ['uniform', 'distance'],  # Weight function used in prediction
#     'knn__metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
# }

# grid_search = GridSearchCV(pipeline, param_grid=param_grid_knn, cv=5, scoring='accuracy')
# grid_search.fit(x_train, y_train)
# print("Best Parameters: ", grid_search.best_params_)

