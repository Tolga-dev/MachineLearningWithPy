# Dimensionality reduction
# many ml problems involve thousands features for each training instance
# it is referred as curse of dimensionality
# PCA , KERNEL PCA, LLE
# Main approaches for dimensionality reduction
# projection and Manifold learning
# projection
# not good idea in twist and turn, such as swiss roll, it is example of manifold
import time

import joblib
from matplotlib import pyplot as plt
from sklearn.datasets import make_swiss_roll, fetch_openml
# manifold learning
# it is work by modeling the manifold on which the training instances lie; it is called manifold learning
# it relies on the manifold assumption# manifold hypothesis

# PCA
# principal component analysis
# most popular
# it identifies the hyperplane lies closest to the data, and then it projects the data onto it.

# preserving the variance

# principal components
# i th axis is called the ith principal component pc of the data
# to find principal components of training set, standard matrix factorization called singular value decomposition
# pca assumes that the dataset is centered around the origin.

# obtain a reduced set X(d-proj) of dimensionality d, compute the matrix * of the training set matrix X by matrix
# Wd defined as the matrix containing the first d columns of v
# X-d-proj = XWd

# PCA class uses svd decomposition to implement pca, it automatically takes care data

# Variance Ratio
# explained_variance_ratio_ it indicates the proportion of the dataset's variance that lies along each principal
# component.
#

# Choosing right number of dimensions
# pca = PCA()
# pca.fit(X_train)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1

# PCA for compression
# the msd between the original data and reconstructed data is called the reconstruction error
# X recovered = Xd . Wd,t

# Randomized PCA
# uses stochastic algorithm call randomized pca and finds an first D principal compoinents

# LLE
# locally linear embedding
# nonlinear
# manifold learning
# it works by first measuring how each training instance linearly relates to its closest neighbors and then looking
# for a low dimensional representation of the training set where these local relationships are best preserved
# at unrolling twisted manifolds

# random projections
# uses random linear projection

# MDS

# isomap

# t-sne

# lda


# Exercises

# main motivation to speed up, visualize and save space
# but some information is lost,  can be computationally intensive, adds some complexity, hard to interpret


from sklearn.decomposition import PCA, KernelPCA
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class DimensionalityReduction:
    def __init__(self):
        # self.usingScikitLearn()
        self.KernelPca()
        pass

    def KernelPca(self):
        X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

        lin_pca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True)
        rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
        sig_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

        y = t > 6.9

        plt.figure(figsize=(11, 4))
        for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"),
                                    (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
            X_reduced = pca.fit_transform(X)
            if subplot == 132:
                X_reduced_rbf = X_reduced

            plt.subplot(subplot)
            plt.title(title, fontsize=14)
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
            plt.xlabel("$z_1$", fontsize=18)
            if subplot == 131:
                plt.ylabel("$z_2$", fontsize=18, rotation=0)
            plt.grid(True)
        plt.show()

    def usingScikitLearn(self):
        np.random.seed(4)
        m = 60
        w1, w2 = .1, .3
        noise = .1
        angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
        X = np.empty((m, 3))
        X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
        X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
        X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

        # principal components
        X_centered = X - X.mean(axis=0)
        U, s, Vt = np.linalg.svd(X_centered)
        c1 = Vt.T[:, 0]
        c2 = Vt.T[:, 1]

        m, n = X.shape

        S = np.zeros(X_centered.shape)
        S[:n, :n] = np.diag(s)

        np.allclose(X_centered, U.dot(S).dot(Vt))

        # projecting down to d dimenstions
        W2 = Vt.T[:, :2]
        X2D = X_centered.dot(W2)
        X2D_using_svd = X2D

        pca = PCA(n_components=2)
        X2D = pca.fit_transform(X)
        print(pca.components_)  # hold Wd
        print(
            pca.explained_variance_ratio_)  # it is tell you 84% dataset's variance lies along the first pc, others second


model_memory_sgd_clf = joblib.Memory('./tmp/model_memory_sgd_clf2_')
model_memory_sgd_clf_new = joblib.Memory('./tmp/model_memory_sgd_clf_new')


@model_memory_sgd_clf.cache
def train_model_memory_sgd_clf(X_train, y_train, X_test):
    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    t0 = time.time()
    rnd_clf.fit(X_train, y_train)
    t1 = time.time()
    print("Training took {:.2f}s".format(t1 - t0))

    y_pred = rnd_clf.predict(X_test)
    return rnd_clf, y_pred


@model_memory_sgd_clf_new.cache
def train_model_memory_sgd_clf_new(X_train_reduced, y_train):
    rnd_clf2 = RandomForestClassifier(n_estimators=100, random_state=42)

    t0 = time.time()
    rnd_clf2.fit(X_train_reduced, y_train)
    t1 = time.time()

    print("Training took {:.2f}s".format(t1 - t0))

    return rnd_clf2


class Example:
    def __init__(self):
        memory = joblib.Memory('./tmp')

        fetch_openml_cached = memory.cache(fetch_openml)

        mnist = fetch_openml_cached('mnist_784')
        X_train = mnist['data'][:60000]
        y_train = mnist['target'][:60000]

        X_test = mnist['data'][60000:]
        y_test = mnist['target'][60000:]

        rnd_clf, y_pred = train_model_memory_sgd_clf(X_train, y_train, X_test)
        print(accuracy_score(y_test, y_pred))

        # pca to reduce dataset's dimensionality

        pca = PCA(n_components=0.95)
        X_train_reduced = pca.fit_transform(X_train)

        rnd_clf2 = train_model_memory_sgd_clf_new(X_train_reduced, y_train)

        X_test_reduced = pca.transform(X_test)
        y_pred = rnd_clf2.predict(X_test_reduced)
        print(accuracy_score(y_test, y_pred))



def program1():
    Example()


if __name__ == '__main__':
    program1()
