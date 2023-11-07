import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier  # for big datas
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, \
    roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


model_memory = joblib.Memory('./model_cache')
model_memory_sgd_clf = joblib.Memory('./tmp/model_memory_sgd_clf')
model_memory_predict_sgd_clf = joblib.Memory('./tmp/model_memory_predict_sgd_clf')
model_memory_never_5_clf = joblib.Memory('./tmp/model_memory_never_5_clf')
model_memory_sgd_clf_decision_function = joblib.Memory('./tmp/model_memory_sgd_clf_decision_function')
model_memory_OneVsRestClassifier_model_memory = joblib.Memory('./tmp/model_memory_OneVsRestClassifier_model_memory')
model_memory_model_scaling_inputs = joblib.Memory('./tmp/model_memory_model_scaling_inputs')

svm_clf_model = joblib.Memory('./tmp/svm_clf_model')


@model_memory_model_scaling_inputs.cache
def train_sgd_scaling():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    val_score = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
    return X_train_scaled, val_score, scaler


# Define a function to train and cache the model
@model_memory.cache
def train_sgd_classifier(X, y, some_digit):
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(X, y)
    sgd_clf.predict([some_digit])
    return sgd_clf


@model_memory_OneVsRestClassifier_model_memory.cache
def train_OneVsRestClassifier_model():
    over_clf = OneVsRestClassifier(SVC())
    fit = over_clf.fit(X_train, y_train)
    predict = over_clf.predict([some_digit])

    return over_clf, fit, predict


@svm_clf_model.cache
def train_sgd_classifier(X_train, y_train, some_digit):
    print("ok")
    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)
    svm_clf.predict([some_digit])
    return svm_clf


@model_memory_sgd_clf.cache
def train_model_memory_sgd_clf():
    return cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


@model_memory_sgd_clf_decision_function.cache
def train_model_memory_sgd_clf_decision_function():
    return cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")


@model_memory_never_5_clf.cache
def train_model_memory_never_5_clf():
    return cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")


@model_memory_predict_sgd_clf.cache
def train_model_memory_predict_sgd_clf():
    return cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal


if __name__ == '__main__':
    memory = joblib.Memory('./tmp')

    fetch_openml_cached = memory.cache(fetch_openml)

    mnist = fetch_openml_cached('mnist_784')
    X, y = mnist["data"], mnist["target"]

    # First part of example
    # print("shape of x", X.shape)
    # print("shape of y ", y.shape)
    some_digit = X.iloc[0]
    some_digit = some_digit.to_numpy()
    # some_digit_in_y = y.iloc[0]
    # print(some_digit_in_y)  # gives corresponding number for x
    # some_digit_image = some_digit.reshape(28, 28)
    # print(some_digit_image)
    # plt.imshow(some_digit_image, cmap="binary")
    # plt.axis("off")
    # plt.show()

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    y_train_5 = (y_train == '5')  # '5' instead of 5
    y_test_5 = (y_test == '5')  # '5' instead of 5

    #    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    #    sgd_clf.fit(X_train, y_train_5)

    sgd_clf = train_sgd_classifier(X_train, y_train_5, some_digit)
    # cached_predictions = sgd_clf.predict([some_digit])
    # print(cached_predictions)
    print("ok")
    arr_sgd_clf = train_model_memory_sgd_clf()
    never_5_clf = Never5Classifier()

    arr_never_5_clf = train_model_memory_never_5_clf()

    print(arr_sgd_clf)
    print(arr_never_5_clf)
    y_train_pred = train_model_memory_predict_sgd_clf()
    print(confusion_matrix(y_train_5, y_train_pred))

    print(precision_score(y_train_5, y_train_pred))
    print(recall_score(y_train_5, y_train_pred))
    print(f1_score(y_train_5, y_train_pred))

    y_scores = sgd_clf.decision_function([some_digit])
    print(y_scores)
    threshold = 0
    y_some_digit_pred = (y_scores > threshold)
    print(y_some_digit_pred)

    y_scores = train_model_memory_sgd_clf_decision_function()
    # precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

    # plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    # threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
    # print(threshold_90_precision)
    #
    # y_train_pred_90 = (y_scores >= threshold_90_precision)
    # print(precision_score(y_train_5, y_train_pred_90))
    # print(recall_score(y_train_5, y_train_pred_90))

    #    plt.show()
    #     fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    #     print(plot_roc_curve(fpr, tpr))

    #     print(roc_auc_score(y_train_5, y_scores))
    # plt.show()
    print("ok")
    svm_clf = train_sgd_classifier(X_train, y_train, some_digit)

    some_digit_score = svm_clf.decision_function([some_digit])
    print(some_digit_score)

    np.argmax(some_digit_score)
    print(svm_clf.classes_)

    # over_clf, over_fit, over_predict = train_OneVsRestClassifier_model()

    # print(over_predict)
    # len(over_clf.estimators_)

    # X_train_scaled, val_score, scaler = train_sgd_scaling()
    # print(val_score)

