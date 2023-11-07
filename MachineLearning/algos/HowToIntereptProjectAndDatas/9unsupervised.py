# Unsupervised learning techniques
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
# most of the applications of machine learning are based on supervised learning
# the most common unsupervised learning task is dimensionality reduction
# clustering, anomaly detection, density estimation

# clustering
# grouping species according to their similarity
# without the labels, we cant use classification algorithms

# applications
# customer segmentation
# data analysis
# a dimensionality reduction technique
# anomaly detection
# semi-supervised learning
# customer segmentation
# data analysis
# a dimensionality reduction technique

# two popular clustering algorithms, K-Means and DBSCAN
# nonlinear dimensionality reduction, semi-supervised learning, and anomaly detection

# K means
# unlabeled dataset, cluster easy dataset, pulse code modulation
#

# Algo
# centroid initialization methods
# first set init hyperparameters to a numpy array
# second, to run the algorithm multiple times with different random inits, keep
# the best sols, to find best, it uses performance metric,
# greater is better ,
#

# limits of k means
# fast and scalable but not perfect

# Using Clustering for Image-Segmentation

# image segmentation is the task of partitioning an image into multiple segments

# semantic segmentation, all pixs that are part of the same object type get assigned to the same segment
# in a self-driving car's vision system, all pixs that are part of a pedestrian's image might be assigned to the
# pedestrian segment.

# instance segmentation, all pixels, all pixels that are part of the same individual object are assigned to the same
# segment.

# colour segmentation
# assign pixels to the same segment if they have a similar color

# using clustering for preprocessing
# it can be an efficient way to dimensionality reduction

# using clustering for semi supervised learning
# unlabeled and labeled instance in same data
# propagated the labels to all other instances in the same cluster, label propagation

# DBSCAN
# ANOTHER POPULAR CLUSTERING ALGORITHM THAT ILLUSTRATES A VERY DIFFERENT APPROACH BASED ON LOCAL DENSITY ESTIMATION
# it is allow to identify clusters of arbitrary shapes

# Hierarchical DBSCAN HDBSCAN
# https://github.com/scikit-learn-contrib/hdbscan/


# Gaussian Mixtures
# it is a probabilistic model that assumes that the instances were generated from a mixture of several
# gaussian distributions whose parameters are unknown
#
# anomaly detection using gaussian mixtures
# out-liner detection is the task of detecting instances that deviate strongly from the norm
# fraud, defective products, removing outliers from a dataset
# any instance located in a low density region can be considered an anomaly
# it requires the number of clusters

# selecting tue number of clusters
# theoretical information criterion, bayesian information criterion bic, Aka-ike information criterion aic
# bic = log(m)p - 2logL
# aic = 2p - 2logL
# m = number of instances
# p number of parameters
# l maximized value of likelihood function

# bayesian gaussian mixture models
# rather than manually searching for the optimal number of clusters, u can use it

# Exercises
# clustering is the unsupervised task of grouping similar instances together

# to select the right number of clusters : elbow rule, plotting silhouette score

# label propagation, labeling a dataset is costly and time-consuming
# therefore it is common to have plenty of unlabeled instances

# to scale to large datasets, kmeans and birch scale well.

# whenever you have plenty of unlabeled instances, active learning can be used

# https://github.com/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb


from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import load_iris, make_blobs, load_digits, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def drawCluster(self, X, y):  # draw labeled and unlabeled
    plt.figure(figsize=(9, 3.5))

    plt.subplot(121)
    plt.plot(X[y == 0, 2], X[y == 0, 3], "yo", label="Iris setosa")
    plt.plot(X[y == 1, 2], X[y == 1, 3], "bs", label="Iris versicolor")
    plt.plot(X[y == 2, 2], X[y == 2, 3], "g^", label="Iris virginica")

    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)

    plt.legend(fontsize=12)

    plt.subplot(122)
    plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")
    plt.xlabel("Petal length", fontsize=14)
    plt.tick_params(labelleft=False)

    plt.show()


class KMean:
    def __init__(self):
        data = load_iris()
        X = data.data
        y = data.target
        print(data.target_names)

        blob_centers = np.array(
            [[0.2, 2.3],
             [-1.5, 2.3],
             [-2.8, 1.8],
             [-2.8, 2.8],
             [-2.8, 1.3]])
        blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        X, y = make_blobs(n_samples=2000, centers=blob_centers,
                          cluster_std=blob_std, random_state=7)

        # self.plot_clusters(X)
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=42)
        y_pred = kmeans.fit_predict(X)
        print(y_pred)
        print(kmeans.cluster_centers_)
        X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
        print(kmeans.predict(X_new))

    def plot_clusters(self, X, y=None):
        plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
        plt.xlabel("$x_1$", fontsize=14)
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
        plt.figure(figsize=(8, 4))
        plt.show()


class KMeanImageSegmentation:
    def __init__(self):
        # self.imageSegmentation()
        # self.clusteringForPreprocesssing()
        self.dbScan()

    def imageSegmentation(self):
        image = imread(os.path.join("images", "unsupervised_learning", "ladybug.png"))
        X = image.reshape(-1, 3)

        segmented_imgs = []
        n_colors = (10, 8, 6, 4, 2)
        for n_clusters in n_colors:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            segmented_img = kmeans.cluster_centers_[kmeans.labels_]
            segmented_imgs.append(segmented_img.reshape(image.shape))
        plt.figure(figsize=(10, 5))
        plt.subplots_adjust(wspace=0.05, hspace=0.1)

        plt.subplot(231)
        plt.imshow(image)
        plt.title("Original image")
        plt.axis('off')

        for idx, n_clusters in enumerate(n_colors):
            plt.subplot(232 + idx)
            plt.imshow(segmented_imgs[idx])
            plt.title("{} colors".format(n_clusters))
            plt.axis('off')
        plt.show()

    def clusteringForPreprocesssing(self):
        X_digits, y_digits = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)
        # logistic regression
        # log_reg = LogisticRegression()
        # log_reg.fit(X_train, y_train)
        # print(log_reg.score(X_test, y_test))

        pipeline = Pipeline([
            ("kmeans", KMeans(n_clusters=50)),
            ("log_reg", LogisticRegression()),
        ])
        pipeline.fit(X_train, y_train)
        print(pipeline.score(X_test, y_test))

        param_grid = dict(kmeans__n_clusters=range(2, 100))
        # grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
        # grid_clf.fit(X_train, y_train)

        # print(grid_clf.best_params_) # 99
        # print(grid_clf.score(X_test, y_test)) # .98

    def Semi_supervised_learning(self):
        X_digits, y_digits = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)
        n_labeled = 50
        log_res = LogisticRegression()
        log_res.fit(X_train[:n_labeled], y_train[:n_labeled])
        print(log_res.score(X_test, y_test))
        k = 50
        kmeans = KMeans(n_clusters=k)
        X_digits_dist = kmeans.fit_transform(X_train)
        representative_digit_idx = np.argmin(X_digits_dist, axis=0)
        X_representative_digits = X_train[representative_digit_idx]

        plt.figure(figsize=(8, 2))
        for index, X_representative_digit in enumerate(X_representative_digits):
            plt.subplot(k // 10, 10, index + 1)
            plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
            plt.axis('off')

        # plt.show()

        print(y_train[representative_digit_idx])

        dummy_array = np.array(y_train[representative_digit_idx])

        y_representative_digits = np.array(dummy_array)

        log_reg = LogisticRegression()
        log_reg.fit(X_representative_digits, y_representative_digits)
        print(log_reg.score(X_test, y_test))

    def dbScan(self):
        X, y = make_moons(n_samples=1000, noise=0.05)
        dbscan = DBSCAN(eps=0.05, min_samples=5)
        dbscan.fit(X)
        print(dbscan.labels_)
        print(dbscan.core_sample_indices_)
        print(len(dbscan.core_sample_indices_))

        knn = KNeighborsClassifier(n_neighbors=50)
        knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
        X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
        print(knn.predict(X_new))
        print(knn.predict_proba(X_new))


class GaussianMixtures:
    def __init__(self):
        self.Gaussian_Mixture()

        pass

    def Gaussian_Mixture(self):
        X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
        X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
        X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
        X2 = X2 + [6, -8]
        X = np.r_[X1, X2]
        y = np.r_[y1, y2]

        gm = GaussianMixture(n_components=3, n_init=10)
        gm.fit(X)

        print(gm.weights_)
        print(gm.means_)
        print(gm.covariances_)


class Example:
    def __init__(self):
        pass


def program1():
    Example()


if __name__ == '__main__':
    program1()
