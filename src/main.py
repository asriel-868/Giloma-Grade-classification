import pandas as pd

from imports import *


def main():
    data = pd.read_csv("training_data.csv")
    data["labels"] = pd.read_csv("training_data_targets.csv", header=None)
    data.drop(["Primary_Diagnosis", ], axis=1, inplace=True)

    missing_values = ["--", "not reported"]

    data.replace(missing_values, pd.NA, inplace=True)
    data.dropna(axis=0, how="any", inplace=True)
    data.reset_index(drop=True, inplace=True)

    labels = data["labels"]
    data.drop(["labels", ], axis=1, inplace=True)
    data["Age_at_diagnosis"] = data["Age_at_diagnosis"].apply(utils.convert_to_year)

    data.replace("MUTATED", 1, inplace=True)
    data.replace("NOT_MUTATED", 0, inplace=True)
    data.replace("MALE", 1, inplace=True)
    data.replace("FEMALE", 0, inplace=True)

    one_hot_features = ["Gender", "Race"]
    numeric_features = ["Age_at_diagnosis"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), one_hot_features),
            ("numerical", StandardScaler(), numeric_features)
        ])

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    classifiers = {
        "LR": LogisticRegression(penalty="l2", dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                                 class_weight=None),
        "RF": RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=42),
        "SVM": SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42, probability=True),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm="SAMME.R"),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="minkowski")
    }

    pipeline_lasso = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selector", SelectFromModel(LinearSVC(dual=False, penalty='l1', random_state=42)))
        ]
    )

    pipeline_without_lasso = Pipeline(
        steps=[
            ("preprocessor", preprocessor)
        ]
    )

    individual_models_with_lasso = []
    individual_models_without_lasso = []

    for clf in classifiers:
        result_dict_lasso = utils.classification_results(pipeline_lasso, classifiers[clf], x_train, y_train, x_test,
                                                         y_test)
        result_dict_without_lasso = utils.classification_results(pipeline_without_lasso, classifiers[clf], x_train,
                                                                 y_train, x_test,
                                                                 y_test)
        individual_models_with_lasso.append(pd.DataFrame([result_dict_lasso]))
        individual_models_without_lasso.append(pd.DataFrame([result_dict_without_lasso]))

    results_individual_models_with_lasso = pd.concat(individual_models_with_lasso, ignore_index=True)
    results_individual_models_without_lasso = pd.concat(individual_models_without_lasso, ignore_index=True)

    classifier_combinations_3 = list(combinations(classifiers, 3))
    classifier_combinations_4 = list(combinations(classifiers, 4))
    classifier_combinations_5 = list(combinations(classifiers, 5))
    combinations_with_lasso = []

    for combo in classifier_combinations_3:
        voting_clf = VotingClassifier(estimators=[(combo[0], classifiers[combo[0]]), (combo[1], classifiers[combo[1]]),
                                                  (combo[2], classifiers[combo[2]])],
                                      voting='soft')
        result_dict_comb_lasso = utils.classification_results(pipeline_lasso, voting_clf, x_train, y_train, x_test,
                                                              y_test)
        combinations_with_lasso.append(pd.DataFrame([result_dict_comb_lasso]))

    for combo in classifier_combinations_4:
        voting_clf = VotingClassifier(estimators=[(combo[0], classifiers[combo[0]]), (combo[1], classifiers[combo[1]]),
                                                  (combo[2], classifiers[combo[2]]), (combo[3], classifiers[combo[3]])],
                                      voting='soft')
        result_dict_comb_lasso = utils.classification_results(pipeline_lasso, voting_clf, x_train, y_train, x_test,
                                                             y_test)
        combinations_with_lasso.append(pd.DataFrame([result_dict_comb_lasso]))

    for combo in classifier_combinations_5:
        voting_clf = VotingClassifier(estimators=[(combo[0], classifiers[combo[0]]), (combo[1], classifiers[combo[1]]),
                                                  (combo[2], classifiers[combo[2]]), (combo[3], classifiers[combo[3]]),
                                                  (combo[4], classifiers[combo[4]])],
                                      voting='soft')
        result_dict_comb_lasso = utils.classification_results(pipeline_lasso, voting_clf, x_train, y_train, x_test,
                                                              y_test)
        combinations_with_lasso.append(pd.DataFrame([result_dict_comb_lasso]))

    results_combinations_with_lasso = pd.concat(combinations_with_lasso, ignore_index=True)
    if os.path.exists("results.txt"):
        os.remove("results.txt")
    with open("results.txt", "w") as f:
        f.write("----------------- Classification Result with Individual Models and LASSO -----------------\n\n")
        f.write(results_individual_models_with_lasso.to_string(index=False))
        f.write("\n\n")
        f.write("----------------- Classification Results with only Individual Models -----------------\n\n")
        f.write(results_individual_models_without_lasso.to_string(index=False))
        f.write("\n\n")
        f.write("----------------- Classification Result with Ensemble Methods and Lasso -----------------\n\n")
        f.write(results_combinations_with_lasso.to_string(index=False))
        f.write("\n\n")


main()
