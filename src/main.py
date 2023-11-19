"""" 
Author : Rishikesh S
roll no : 21222

The commented code contains the experimental models I tried on this dataset. If you run this code right now, it predicts the labels for 
the test set using the best model .

Best Model : A soft Voting combination of LR + SVM + AdaBoosts

"""


from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from itertools import combinations
from sklearn.svm import SVC
import pandas as pd
import warnings
import os
import re


warnings.filterwarnings("ignore")


def main():
    data = pd.read_csv("training_data.csv")
    data["labels"] = pd.read_csv("training_data_targets.csv", header=None)
        
    missing_values = ["--", "not reported"]
    
    # Replacing all the missing values with NA and dropping them
    data.replace(missing_values, pd.NA, inplace=True)
    data.dropna(axis=0, how="any", inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    labels = data["labels"]
    data.drop(["labels", ], axis=1, inplace=True)
    
    # Converting Age at diagnosis to float
    data["Age_at_diagnosis"] = data["Age_at_diagnosis"].apply(convert_to_year)
    
    categorical_features = data.select_dtypes(include=["object"]).columns
    numeric_features = ["Age_at_diagnosis"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numerical", StandardScaler(), numeric_features)
        ])
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    classifiers = {
        "LR": LogisticRegression(dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                                 class_weight=None),
        "RF": RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=42),
        "SVM": SVC(random_state=42, kernel="rbf", gamma="scale", C=1.0, probability=True),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm="SAMME.R", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="minkowski")
    }
    
    # pipeline_lasso = Pipeline(
    #     steps=[
    #         ("preprocessor", preprocessor),
    #         ("feature_selector", SelectFromModel(Lasso(alpha=0.001, random_state=42)))
    #     ]
    # )
    
    # # Pipeline without lasso feature selection
    # pipeline_without_lasso = Pipeline(
    #     steps=[
    #         ("preprocessor", preprocessor)
    #     ]
    # )
    
    # individual_models_with_lasso = []
    # individual_models_without_lasso = []
    
    # # Results with individual models
    # for clf in classifiers:
    #     result_dict_lasso = `zclassification_results(pipeline_lasso, classifiers[clf], data, labels)
    #     result_dict_without_lasso = classification_results(pipeline_without_lasso, classifiers[clf], data, labels)
    #     individual_models_with_lasso.append(pd.DataFrame([result_dict_lasso]))
    #     individual_models_without_lasso.append(pd.DataFrame([result_dict_without_lasso]))

    # results_individual_models_with_lasso = pd.concat(individual_models_with_lasso, ignore_index=True)
    # results_individual_models_without_lasso = pd.concat(individual_models_without_lasso, ignore_index=True)
    
    # # Getting all possible combination of models
    # classifier_combinations_3 = list(combinations(classifiers, 3))
    # classifier_combinations_4 = list(combinations(classifiers, 4))
    # classifier_combinations_5 = list(combinations(classifiers, 5))
    # combinations_with_lasso = []
    # combinations_without_lasso = []
    
    # # Results with all possible combination of models. Uses soft voting (at least 3 models)
    # for combo in classifier_combinations_3:
    #     voting_clf = VotingClassifier(estimators=[(combo[0], classifiers[combo[0]]), (combo[1], classifiers[combo[1]]),
    #                                               (combo[2], classifiers[combo[2]])], voting='soft')
        
    #     result_dict_comb_lasso = classification_results(pipeline_lasso, voting_clf, data, labels)
    #     result_dict_comb_without_lasso = utils.classification_results(pipeline_without_lasso, voting_clf, data, labels)
        
    #     combinations_with_lasso.append(pd.DataFrame([result_dict_comb_lasso]))
    #     combinations_without_lasso.append(pd.DataFrame([result_dict_comb_without_lasso]))
    
    # for combo in classifier_combinations_4:
    #     voting_clf = VotingClassifier(estimators=[(combo[0], classifiers[combo[0]]), (combo[1], classifiers[combo[1]]),
    #                                             (combo[2], classifiers[combo[2]]), (combo[3], classifiers[combo[3]])], voting='soft')
            
    #     result_dict_comb_lasso = classification_results(pipeline_lasso, voting_clf, data, labels)
    #     result_dict_comb_without_lasso = utils.classification_results(pipeline_without_lasso, voting_clf, data, labels)
            
    #     combinations_with_lasso.append(pd.DataFrame([result_dict_comb_lasso]))
    #     combinations_without_lasso.append(pd.DataFrame([result_dict_comb_without_lasso]))
        
    # for combo in classifier_combinations_5:
    #     voting_clf = VotingClassifier(estimators=[(combo[0], classifiers[combo[0]]), (combo[1], classifiers[combo[1]]),
    #                                             (combo[2], classifiers[combo[2]]), (combo[3], classifiers[combo[3]]),
    #                                             (combo[4], classifiers[combo[4]])], voting='soft')
            
    #     result_dict_comb_lasso = classification_results(pipeline_lasso, voting_clf, data, labels)
    #     result_dict_comb_without_lasso = utils.classification_results(pipeline_without_lasso, voting_clf, data, labels)
            
    #     combinations_with_lasso.append(pd.DataFrame([result_dict_comb_lasso]))
    #     combinations_without_lasso.append(pd.DataFrame([result_dict_comb_without_lasso]))
        
    # results_combinations_with_lasso = pd.concat(combinations_with_lasso, ignore_index=True)
    # results_combinations_without_lasso = pd.concat(combinations_without_lasso, ignore_index=True)
    
    # if os.path.exists("results.txt"):
    #     os.remove("results.txt")
    # with open("results.txt", "w") as f:
    #     f.write("----------------- Classification Result with Individual Models and LASSO -----------------\n\n")
    #     f.write(results_individual_models_with_lasso.to_string(index=False))
    #     f.write("\n\n")
    #     f.write("----------------- Classification Results with only Individual Models -----------------\n\n")
    #     f.write(results_individual_models_without_lasso.to_string(index=False))
    #     f.write("\n\n")
    #     f.write("----------------- Classification Result with Voting Methods and Lasso -----------------\n\n")
    #     f.write(results_combinations_with_lasso.to_string(index=False))
    #     f.write("\n\n")
    #     f.write("----------------- Classification Result with only Voting Methods -----------------\n\n")
    #     f.write(results_combinations_without_lasso.to_string(index=False))
    #     f.write("\n\n")


    test_data = pd.read_csv("test_data.csv")
    test_data["Age_at_diagnosis"] = test_data["Age_at_diagnosis"].apply(convert_to_year)
    
    final_clf = VotingClassifier(estimators=[("LR", classifiers["LR"]), ("SVM", classifiers["SVM"]), ("AdaBoost", classifiers["AdaBoost"])])
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", final_clf)
        ]
    )
    
    final_pipeline.fit(data, labels)
    test_labels = final_pipeline.predict(test_data)

    if os.path.exists("final_labels.txt"):
        os.remove("final_labels.txt")
    with open("final_labels.txt", "w") as f:
        for i in test_labels:
            if i == 0:
                f.write("GBM\n")
            elif i == 1:
                f.write("LGG\n")


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
    

""" Returns all the classification metrics in a dictionary format """


def classification_results(pipeline, clf, data, labels):
    pipeline.steps.append(("clf", clf))
    
    stratified_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    cv_accuracy = cross_val_score(pipeline, data, labels, cv=stratified_kf, scoring='accuracy')
    cv_precision = cross_val_score(pipeline, data, labels, cv=stratified_kf, scoring='precision_macro')
    cv_recall = cross_val_score(pipeline, data, labels, cv=stratified_kf, scoring='recall_macro')
    cv_f1 = cross_val_score(pipeline, data, labels, cv=stratified_kf, scoring='f1_macro')

    if type(clf).__name__ != "VotingClassifier":
        result_dict = {
            'Classifier': type(clf).__name__,
            'Accuracy': cv_accuracy.mean(),
            'Precision': cv_precision.mean(),
            'Recall': cv_recall.mean(),
            'F1 Score': cv_f1.mean()
        }
    else:
        result_dict = {
            'Classifier': "+".join(name for name, _ in clf.estimators),
            'Accuracy': cv_accuracy.mean(),
            'Precision': cv_precision.mean(),
            'Recall': cv_recall.mean(),
            'F1 Score': cv_f1.mean()
        }
    pipeline.steps.pop()

    return result_dict


main()