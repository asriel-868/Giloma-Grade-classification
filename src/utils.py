from imports import *

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


""" Returns the classification metrics """


def classification_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")


    return accuracy, precision, recall, f1


""" Returns the predictions of the pipeline """


def pipeline_predictions(pipeline, clf, x_train, y_train, x_test):
    pipeline.steps.append(("clf", clf))
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    pipeline.steps.pop()
    return predictions


""" Returns all the classification metrics in a dictionary format """


def classification_results(pipeline, clf, x_train, y_train, x_test, y_test):
    y_pred = pipeline_predictions(pipeline, clf, x_train, y_train, x_test)
    metrics = classification_metrics(y_test, y_pred)

    if type(clf).__name__ != "VotingClassifier":
        result_dict = {
            'Classifier': type(clf).__name__,
            'Accuracy': metrics[0],
            'Precision': metrics[1],
            'Recall': metrics[2],
            'F1 Score': metrics[3]
        }
    else:
        result_dict = {
            'Classifier': "+".join(name for name, _ in clf.estimators),
            'Accuracy': metrics[0],
            'Precision': metrics[1],
            'Recall': metrics[2],
            'F1 Score': metrics[3]
        }

    return result_dict
