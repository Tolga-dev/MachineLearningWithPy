import urllib

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import pandas as pd
import sklearn.linear_model

# from hands on ml with scikit learn
# https://github.com/ageron/handson-ml2/blob/master/01_the_machine_learning_landscape.ipynb


class LinearRegressionLandscape:
    def __init__(self):
        datapath = os.path.join("datasets", "lifesat", "")
        mpl.rc('axes', labelsize=14)
        mpl.rc('xtick', labelsize=12)
        mpl.rc('ytick', labelsize=12)

        os.makedirs(datapath, exist_ok=True)

        oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
        gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv", thousands=',', delimiter='\t',
                                     encoding='latin1', na_values="n/a")

        # Prepare the data
        country_stats = self.prepare_country_stats(oecd_bli, gdp_per_capita)
        X = np.c_[country_stats["GDP per capita"]]
        y = np.c_[country_stats["Life satisfaction"]]

        # Visualize the data
        country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
        plt.show()

        # Select a linear model
        model = sklearn.linear_model.LinearRegression()

        # Train the model
        model.fit(X, y)

        # Make a prediction for Cyprus
        X_new = [[22587]]  # Cyprus' GDP per capita
        print(model.predict(X_new))  # outputs [[ 5.96242338]]

    def prepare_country_stats(self,oecd_bli, gdp_per_capita):
        oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
        oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
        gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
        gdp_per_capita.set_index("Country", inplace=True)
        full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                      left_index=True, right_index=True)
        full_country_stats.sort_values(by="GDP per capita", inplace=True)
        remove_indices = [0, 1, 6, 8, 33, 34, 35]
        keep_indices = list(set(range(36)) - set(remove_indices))
        return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# https://brilliant.org/wiki/linear-regression/
class LinearReg1:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def estimate_coef(self):
        n = np.size(self.x)

        m_x = np.mean(self.x)
        m_y = np.mean(self.y)

        summed_xy = np.sum(self.x * self.y) - n * m_x * m_y
        summed_xx = np.sum(self.x * self.x) - n * m_x * m_x

        b_1 = summed_xy / summed_xx
        b_0 = m_y - b_1 * m_x

        return b_0, b_1


def Runner():
    raw_data = []
    with open('MachineLearning/data/test.csv', 'r') as md:
        next(md)
        for line in md.readlines():
            data_row = line.strip().split(',')
            raw_data.append(data_row)

    processed_data_x = []
    processed_data_y = []

    for row in raw_data:
        data_row = list(map(float, row))
        processed_data_x.append(data_row[0])
        processed_data_y.append(data_row[1])
    processed_data_y = np.array(processed_data_y)
    processed_data_x = np.array(processed_data_x)

    plt.scatter(processed_data_x, processed_data_y)
    lin = LinearReg1(processed_data_x, processed_data_y)
    result = lin.estimate_coef()
    y = result[0] + result[1] * processed_data_x
    plt.plot(processed_data_x, y, color="g")
    plt.show()


#

class LinearRegression2:

    def __init__(self, lr=0.001, n_iters=2000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


def runner2():
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
    plt.show()

    reg = LinearRegression2(lr=0.01)
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    def mse(y_test, predictions):
        return np.mean((y_test - predictions) ** 2)

    mse = mse(y_test, predictions)
    print(mse)

    y_pred_line = reg.predict(X)
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
    plt.show()
