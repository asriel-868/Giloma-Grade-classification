import warnings
from utils import *

# Ignore all the warnings
warnings.filterwarnings("ignore")


def main():
    data = pd.read_csv("training_data.csv")
    data["labels"] = pd.read_csv("training_data_targets.csv", header=None)
    data.drop(["Primary_Diagnosis", ], axis=1, inplace=True)

    missing_values = ["--", "not reported"]

    # Replacing all the missing values with NA and dropping them
    data.replace(missing_values, pd.NA, inplace=True)
    data.dropna(axis=0, how="any", inplace=True)
    data.reset_index(drop=True, inplace=True)

    labels = data["labels"]
    data.drop(["labels", ], axis=1, inplace=True)

    # Converting Age at diagnosis to float
    data["Age_at_diagnosis"] = data["Age_at_diagnosis"].apply(utils.convert_to_year)

    categorical_features = data.select_dtypes(include=["object"]).columns
    numeric_features = ["Age_at_diagnosis"]

    # Preprocessor which OneHot encodes all the categorical columns and normalizes all the numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numerical", StandardScaler(), numeric_features)
        ])

    # Splitting data to test and train set
    x_train, x_test, y_train, y_test = train_test_split(data, labels,  test_size=0.1, random_state=42)

    # Encoding the labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # List of classifiers
    classifiers = {
        "LR": LogisticRegression(dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                                 class_weight=None),
        "RF": RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=42),
        "SVM": SVC(random_state=42, kernel="rbf", gamma="scale", C=1.0, probability=True),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm="SAMME.R", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="minkowski")
    }.0
    # classifiers = {
    #     "LR": LogisticRegression(C=0.1, solver="liblinear", penalty="l2"),
    #     "RF": RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=10,
    #                                  min_samples_leaf=1, random_state=42),
    #     "SVM": SVC(random_state=42, kernel="rbf", gamma="scale", C=0.1, probability=True),
    #     "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.1,  random_state=42),
    #     "KNN": KNeighborsClassifier(n_neighbors=7, weights="uniform", metric="euclidean")
    # }

    # Pipeline which applies lasso feature_selection

    pipeline_lasso = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selector", SelectFromModel(Lasso(alpha=0.001, random_state=42)))
        ]
    )

    # Pipeline without lasso feature selection
    pipeline_without_lasso = Pipeline(
        steps=[
            ("preprocessor", preprocessor)
        ]
    )

    individual_models_with_lasso = []
    individual_models_without_lasso = []

    # Results with individual models
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

    # Getting all possible combination of models
    classifier_combinations_3 = list(combinations(classifiers, 3))
    classifier_combinations_4 = list(combinations(classifiers, 4))
    classifier_combinations_5 = list(combinations(classifiers, 5))
    combinations_with_lasso = []
    combinations_without_lasso = []

    # Results with all possible combination of models. Uses soft voting (at least 3 models)
    for combo in classifier_combinations_3:
        voting_clf = VotingClassifier(estimators=[(combo[0], classifiers[combo[0]]), (combo[1], classifiers[combo[1]]),
                                                  (combo[2], classifiers[combo[2]])],
                                      voting='soft')
        result_dict_comb_lasso = utils.classification_results(pipeline_lasso, voting_clf, x_train, y_train, x_test,
                                                              y_test)
        result_dict_comb_without_lasso = utils.classification_results(pipeline_without_lasso, voting_clf, x_train,
                                                                      y_train, x_test,
                                                                      y_test)
        combinations_with_lasso.append(pd.DataFrame([result_dict_comb_lasso]))
        combinations_without_lasso.append(pd.DataFrame([result_dict_comb_without_lasso]))

    for combo in classifier_combinations_4:
        voting_clf = VotingClassifier(estimators=[(combo[0], classifiers[combo[0]]), (combo[1], classifiers[combo[1]]),
                                                  (combo[2], classifiers[combo[2]]), (combo[3], classifiers[combo[3]])],
                                      voting='soft')
        result_dict_comb_lasso = utils.classification_results(pipeline_lasso, voting_clf, x_train, y_train, x_test,
                                                              y_test)
        result_dict_comb_without_lasso = utils.classification_results(pipeline_without_lasso, voting_clf, x_train,
                                                                      y_train, x_test,
                                                                      y_test)
        combinations_with_lasso.append(pd.DataFrame([result_dict_comb_lasso]))
        combinations_without_lasso.append(pd.DataFrame([result_dict_comb_without_lasso]))

    for combo in classifier_combinations_5:
        voting_clf = VotingClassifier(estimators=[(combo[0], classifiers[combo[0]]), (combo[1], classifiers[combo[1]]),
                                                  (combo[2], classifiers[combo[2]]), (combo[3], classifiers[combo[3]]),
                                                  (combo[4], classifiers[combo[4]])],
                                      voting='soft')
        result_dict_comb_lasso = utils.classification_results(pipeline_lasso, voting_clf, x_train, y_train, x_test,
                                                              y_test)
        result_dict_comb_without_lasso = utils.classification_results(pipeline_without_lasso, voting_clf, x_train,
                                                                      y_train, x_test,
                                                                      y_test)
        combinations_with_lasso.append(pd.DataFrame([result_dict_comb_lasso]))
        combinations_without_lasso.append(pd.DataFrame([result_dict_comb_without_lasso]))

    results_combinations_with_lasso = pd.concat(combinations_with_lasso, ignore_index=True)
    results_combinations_without_lasso = pd.concat(combinations_without_lasso, ignore_index=True)

    # Writing out the results to a file
    if os.path.exists("results.txt"):
        os.remove("results.txt")
    with open("results.txt", "w") as f:
        f.write("----------------- Classification Result with Individual Models and LASSO -----------------\n\n")
        f.write(results_individual_models_with_lasso.to_string(index=False))
        f.write("\n\n")
        f.write("----------------- Classification Results with only Individual Models -----------------\n\n")
        f.write(results_individual_models_without_lasso.to_string(index=False))
        f.write("\n\n")
        f.write("----------------- Classification Result with Voting Methods and Lasso -----------------\n\n")
        f.write(results_combinations_with_lasso.to_string(index=False))
        f.write("\n\n")
        f.write("----------------- Classification Result with only Voting Methods -----------------\n\n")
        f.write(results_combinations_without_lasso.to_string(index=False))
        f.write("\n\n")


main()
