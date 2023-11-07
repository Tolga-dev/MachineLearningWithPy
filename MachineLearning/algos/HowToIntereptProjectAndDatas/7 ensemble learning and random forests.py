# ensemble learning and random forests
# a group of predictors is called an ensemble, thereby this technique is called ensemble learning
# an ensemble learning algorithm is called an ensemble method
# predict the class that gets the most votes it is random forest
# most powerful machine learning algorithms available today
import joblib
# Ensemble methods, bagging, boosting and stacking


# Voting classifiers
# if you trained a few classifiers and each one achieving about 80 percent accuracy, you may use
# logistic, svm, random forest, k nearest classifier and perhaps more
# to get most votes and this majority-vote classifier is called a hard voting classifier
# each classifier a weak learner, ensemble can still be a strong learner

# Bagging and Pasting
# when sapling is performed with replacement, this is called bagging
# without replacement it is called pasting
# it allows training instance to be sampled several times across multiple predictors
# and aggregate them with function is called statistical mode
# they scale very well

# out of bag evaluation
# soem instances may be samples several times for any given predictors


# Random forest
# typically with max_samples set to the size of the training set.
# instead of using bagging classifier and passing it a decisiontreeclassifier, you can use random forest classifier
# with a few exceptions, a random forest classifier has all the hyperparameters of a decision, bagging classifier
# it is introduces extra randomness,

# Boosting
# refers to any ensemble method that can  combine several weak learners into a strong learner
#  adaptive boosting and gradient boosting

# ada boost
# when training an adaboost classifier, the algorithm first trains a base classifier like decision tree and, uses it
# to make predictions on the training set. then algo increases the relative weight of misclassified training instances
# then it trains a second classifier, using the updated weights and again makes predictions on the training set,
# updates the instance weights and so on

# Gradient boosting
# it works by sequentially adding predictors to an ensemble, each on correcting its predecessor.
# using decision trees as the base predictors, it is called gradient tree boosting

# stacking
# stacked generalization
# instead of using trivial functions such as hard voting,
# its final predictor called a blender or a meta learner
# to train blender, a common approach is to use a hold-out set
# first training set is split into two subsets, first one is used to train predictors
# on the second later, it is used to make predictions

# exercises
# a hard voting classifier just counts the votes of each classifier in the ensemble and picks the class that
# gets the most votes
# a soft voting classifier computes the average estimated class probability for each class and picks the class
# this gives high-confidence votes more weight and often performs better.
#

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, fetch_openml
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import joblib as joblib


class VotingClassifiers:
    def __init__(self):
        log_clf = LogisticRegression()
        rnd_clf = RandomForestClassifier()
        svm_clf = SVC()
        X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        voting_clf = VotingClassifier(
            estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
            voting='hard')
        voting_clf.fit(X_train, y_train)
        for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


class BaggingAndPasting:
    def __init__(self):
        # self.baggingAndPastingInScikit()
        # self.out_of_evaluation()
        self.RandomForest()
        pass

    def RandomForest(self):
        X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
        rnd_clf.fit(X_train, y_train)
        y_pred_rf = rnd_clf.predict(X_test)
        print(accuracy_score(y_test, y_pred_rf))

        bag_clf = BaggingClassifier(
            DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
            n_estimators=500, random_state=42)
        bag_clf.fit(X_train, y_train)
        y_pred = bag_clf.predict(X_test)
        print(np.sum(y_pred == y_pred_rf) / len(y_pred))  # very similar predictions

    def out_of_evaluation(self):
        X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                                    bootstrap=True, n_jobs=-1, oob_score=True)
        bag_clf.fit(X_train, y_train)
        print(bag_clf.oob_score_)
        y_pred = bag_clf.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(bag_clf.oob_decision_function_)

    def baggingAndPastingInScikit(self):
        X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        # bagging classifier performs soft voting
        bag_clf = BaggingClassifier(
            DecisionTreeClassifier(), n_estimators=500,
            max_samples=100, bootstrap=True, n_jobs=-1)
        bag_clf.fit(X_train, y_train)
        y_pred = bag_clf.predict(X_test)

        print(accuracy_score(y_test, y_pred))

        tree_clf = DecisionTreeClassifier(random_state=42)
        tree_clf.fit(X_train, y_train)
        y_pred_tree = tree_clf.predict(X_test)
        print(accuracy_score(y_test, y_pred_tree))


class GradientBoost:
    def __init__(self):
        np.random.seed(42)
        X = np.random.rand(100, 1) - 0.5
        y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

        tree_Reg = DecisionTreeRegressor(max_depth=2, random_state=42)
        tree_Reg.fit(X, y)

        # residual errors made by the first predictor
        y2 = y - tree_Reg.predict(X)
        tree_Reg_2 = DecisionTreeRegressor(max_depth=2, random_state=42)
        tree_Reg_2.fit(X, y2)

        # for residual error made by the second predictor
        y3 = y2 - tree_Reg_2.predict(X)
        tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
        tree_reg3.fit(X, y3)
        X_new = np.array([[0.8]])
        y_pred = sum(tree.predict(X_new) for tree in (tree_Reg, tree_Reg_2, tree_reg3))
        print(y_pred)


train_model_for_example = joblib.Memory('./tmp/TrainingModelsCache/train_model_for_example')
train_model_for_voting = joblib.Memory('./tmp/TrainingModelsCache/train_model_for_voting')


@train_model_for_example.cache
def Train_model_for_example(random_forest_clf, extra_trees_clf, svm_clf, mlp_clf, X_train, y_train):
    estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
    for estimator in estimators:
        print("Training the", estimator)
        estimator.fit(X_train, y_train)
    return estimators


@train_model_for_voting.cache
def train_model_for_voting_function(named_estimators):
    return VotingClassifier(named_estimators)


class Example: # https://github.com/ageron/handson-ml2/blob/master/07_ensemble_learning_and_random_forests.ipynb
    def __init__(self):
        # loading data
        memory = joblib.Memory('./tmp')

        fetch_openml_cached = memory.cache(fetch_openml)

        mnist = fetch_openml_cached('mnist_784')

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            mnist.data, mnist.target, test_size=10000, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=10000, random_state=42)
        # train various classifiers, RFC and extra-trees and an svm
        random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
        svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)
        mlp_clf = MLPClassifier(random_state=42)
        estimators = Train_model_for_example(random_forest_clf, extra_trees_clf, svm_clf, mlp_clf, X_train, y_train)

        print([estimator.score(X_val, y_val) for estimator in estimators])

        # ensemble that outperforms them all on the validation set using a soft or hard voting classifier
        named_estimators = [
            ("random_forest_clf", random_forest_clf),
            ("extra_trees_clf", extra_trees_clf),
            ("svm_clf", svm_clf),
            ("mlp_clf", mlp_clf),
        ]

        voting_clf = train_model_for_voting_function(named_estimators)
        voting_clf.fit(X_train, y_train)
        print(voting_clf.score(X_val, y_val))
        print([estimator.score(X_val, y_val) for estimator in voting_clf.estimators_])

        voting_clf.set_params(svm_clf=None)
        print(voting_clf.estimators)
        print(voting_clf.estimators_)
        del voting_clf.estimators_[2]

        voting_clf.voting = "soft"
        print(voting_clf.score(X_val, y_val))

        voting_clf.voting = "hard"
        print(voting_clf.score(X_test, y_test))

        print([estimator.score(X_test, y_test) for estimator in voting_clf.estimators_])



def program1():
    # VotingClassifiers()
    # BaggingAndPasting()
    # GradientBoost()
    pass


def program2():
    Example()


if __name__ == '__main__':
    program2()
