# SUPPORT VECTOR MACHINE
# is a powerful and versatile model. most popular model.
# it is well suited for small and medium-sized data sets
from multiprocessing import Pipe

# Linear SVM Classification
# it is two classes can clearly be separated easily with a straight line
# the fairest line between classes = large margin classification

# Soft Margin classification
# to use a more flexible model, as large as possible,

# Hard Margin classification
# if we impose that all instance must be on the one side, this is called hard margin classification

# Non-Linear SVM classification
# although linear efficient and work in many cases when linearly separable.
# to handle nonlinear datasets is to add more features, polynomial features.

# Gaussian RBF Kernel
# on large training sets, it may slow

# Computational Complexity
# Class   Time complexity     Out-of-core support     Scaling required    Kernel trick
# LinearSVC       O(m × n) No Yes No
# SGDClassifier   O(m × n) Yes Yes No
# SVC             O(m² × n) to O(m³ × n) No Yes Yes

# SVM Regression
# versatile
# support linear and nonlinear classifications and regressions


# Under the hood
# how svms make predictions and their training algorithms work
# Decision FUnctions and predictions
# linear svm classifier model predicts

# Exercises

# to fit widest possible "width" between the classes, in other words the gold is to have the largest possible
# margin between the decision




import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC


class NonLinearSVMClassification:
    def __init__(self, X, y):
        self.y = y
        self.X = X
        print("ok")
        # self.plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
        # self.pipelineScm(X, y)
        self.rbf_kernel_svm(X, y)

    # Gaussian RBF Kernel
    def rbf_kernel_svm(self, X, y):
        gamma1, gamma2 = 0.1, 5
        C1, C2 = 0.001, 1000
        hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

        svm_clfs = []
        for gamma, C in hyperparams:
            rbf_kernel_svm_clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
            ])
            rbf_kernel_svm_clf.fit(X, y)
            svm_clfs.append(rbf_kernel_svm_clf)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 7), sharex=True, sharey=True)

        for i, svm_clf in enumerate(svm_clfs):
            plt.sca(axes[i // 2, i % 2])
            self.plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
            self.plot_dataset(X, y, [-1.5, 2.45, -1, 1.5])
            gamma, C = hyperparams[i]
            plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)
            if i in (0, 1):
                plt.xlabel("")
            if i in (1, 3):
                plt.ylabel("")

        plt.show()

    # second polynomial kernel
    def pipelineScm(self, X, y):
        polynomial_svm_clf = Pipeline([
            ("poly_features", PolynomialFeatures(degree=3)),
            ("scaler", StandardScaler()),
            ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
        ])

        polynomial_svm_clf.fit(X, y)

        self.plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
        self.plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
        plt.show()

    def plot_predictions(self, clf, axes):
        x0s = np.linspace(axes[0], axes[1], 100)
        x1s = np.linspace(axes[2], axes[3], 100)
        x0, x1 = np.meshgrid(x0s, x1s)
        X = np.c_[x0.ravel(), x1.ravel()]
        y_pred = clf.predict(X).reshape(x0.shape)
        y_decision = clf.decision_function(X).reshape(x0.shape)
        plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
        plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

    # first
    def plot_dataset(self, X, y, axes):
        plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
        plt.axis(axes)
        plt.grid(True, which='both')
        plt.xlabel(r"$x_1$", fontsize=20)
        plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


def formula1():
    X, y = make_moons(n_samples=100, noise=0.15)

    NonLinearSVMClassification(X, y)


if __name__ == '__main__':
    formula1()
