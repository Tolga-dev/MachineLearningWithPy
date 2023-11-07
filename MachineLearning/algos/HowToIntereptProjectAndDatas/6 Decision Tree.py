# Decision Trees
# it is like svm it is versatile
# fundamental components of random forests
# CART training algorithms

# CART - classification and regression training
# to train decision trees, algo works by first splitting the training set
# into two subsets using a single feature k and a threshold t(k)
# J(K,T(K)) = m(left)* G(left)/m  + m(right)* G(right)/m

# gini impurity
# it is generally lower than its parent's due to cart training algorithm's cost function
#

# Regularization hyperparameters
# decision trees make very few assumptions about thr training data
# it is often called a nonparametric model, (it has parameters) just parameters,
# is not determined prior to training,
# to avoid over-fitting the training data, we can use regularization.

# decision tree complexity
# n x m log(m)

# Exercises
# log2(m)^ 2 depth of a decision tree


import os

import joblib
import numpy as np
from scipy.stats import mode
from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source
from sklearn.datasets import make_moons


class DecisionTree:
    def __init__(self):
        iris = load_iris()
        X = iris.data[:, 2:]
        y = iris.target

        tree_clf = DecisionTreeClassifier(max_depth=2)
        tree_clf.fit(X, y)

        # to get graph illustration of graph
        # IMAGES_PATH = "/home/xamblot/PycharmProjects/pythonProject/datasets/iris_tree"
        #
        # export_graphviz(
        #     tree_clf,
        #     out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
        #     feature_names=iris.feature_names[2:],
        #     class_names=iris.target_names,
        #     rounded=True,
        #     filled=True
        # )
        #
        # Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))
        #

        # get this dot file
        # dot -Tpng iris_tree.dot -o iris_tree.png
        # and make it png and see Visualizing
        self.prediction(tree_clf)

    def prediction(self, tree_clf):
        print(tree_clf.predict_proba([[5, 1.5]]))  # 5 , 1.5 cm
        print(tree_clf.predict([[5, 1.5]]))


train_model_cache_moon = joblib.Memory('./tmp/TrainingModelsCache/moon')


@train_model_cache_moon.cache
def cache(X_train, y_train, X_test, y_test):
    # use grid search with cross validation
    params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}

    grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)

    grid_search_cv.fit(X_train, y_train)
    print(grid_search_cv.best_estimator_)

    y_gred = grid_search_cv.predict(X_test)
    accuracy_score(y_test, y_gred)
    return params, grid_search_cv, y_gred


class Example1:
    def __init__(self):
        # generate a moons data set
        self.example1()

    def example1(self):
        X, y = make_moons(n_samples=10000, noise=.4, random_state=42)
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        params, grid_search_cv, y_pred = cache(X_train, y_train, X_test, y_test)
        print(grid_search_cv.best_estimator_)
        print(accuracy_score(y_test, y_pred))
        n_tree = 1000
        n_instance = 100
        mini_sets = []

        rs = ShuffleSplit(n_splits=n_tree, test_size=len(X_train) - n_instance, random_state=42)
        for mini_train_index, mini_test_index in rs.split(X_train):
            X_mini_train = X_train[mini_train_index]
            y_mini_train = y_train[mini_train_index]
            mini_sets.append((X_mini_train, y_mini_train))

        forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_tree)]

        accuracy_scores = []

        for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
            tree.fit(X_mini_train, y_mini_train)

            y_pred = tree.predict(X_test)
            accuracy_scores.append(accuracy_score(y_test, y_pred))

        print(np.mean(accuracy_scores))

        Y_pred = np.empty([n_tree, len(X_test)], dtype=np.uint8)

        for tree_index, tree in enumerate(forest):
            Y_pred[tree_index] = tree.predict(X_test)

        y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

        print(accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))

def program1():
    DecisionTree()


def program2():
    Example1()


if __name__ == '__main__':
    program2()
