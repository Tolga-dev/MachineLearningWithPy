import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import sklearn.datasets as dt
from numpy import array
from sklearn.model_selection import train_test_split


# small gradient descent example

class First:
    def gradient_descent(self, gradient, start, learn_rate, n_iter=10, tolerance=1e-06):
        vector = start
        vals = []
        for _ in range(n_iter):
            diff = -learn_rate * gradient(vector)
            if np.all(np.abs(diff) <= tolerance):
                break
            vector += diff
            vals.append(vector)
            print(vector)

        return vals

    def Runner(self):
        # gradient_descent(gradient=lambda v: 2 * v, start=10, learn_rate=0.8)
        # gradient_descent(gradient=lambda v: 2 * v, start=10, learn_rate=0.08)
        # datas = gradient_descent(gradient=lambda v: v**2 - 3, start=2.5, learn_rate=0.2)

        datas = self.gradient_descent(gradient=lambda v: 1 - 1 / v, start=2.5, learn_rate=0.5)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(datas)), datas, marker='o', linestyle='-', color='b')
        plt.title('Objective Function Value')
        plt.xlabel('Iteration')
        plt.ylabel('Value')

        plt.tight_layout()
        plt.show()


# Stochastic Gradient Descent

class First2:
    def ssr_gradient(self, x, y, b):
        res = b[0] + b[1] * x - y
        return res.mean(), (res * x).mean()

    def gradient_descent4(self, gradient, x, y, start, learn_rate, n_iter=10, tolerance=1e-06):
        vector = start
        for _ in range(n_iter):
            diff = -learn_rate * np.array(gradient(x, y, vector))
            if np.all(np.abs(diff) <= tolerance):
                break
            vector += diff
            print(vector)

        return vector

    def Runner4(self):
        x = np.array([5, 15, 25, 35, 45, 55])
        y = np.array([5, 20, 14, 32, 22, 38])

        self.gradient_descent4(self.ssr_gradient, x, y, start=[0.5, 0.51], learn_rate=0.0008, n_iter=100_000)


# Stochastic Gradient Descent2 with tensor flow

class First3:
    def Runner1(self):
        sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        var = tf.Variable(2.5)
        cost = lambda: 2 + var ** 2

        # Perform optimization
        for _ in range(100):
            sgd.minimize(cost, var_list=[var])

        # Extract results
        var.numpy()

        cost().numpy()


# Use it with data
# 𝑦=𝛽+θnXn


class First4:
    def __init__(self):
        self.df = pd.read_csv('/MachineLearning/data/Advertising.csv')
        self.X = array([])
        self.Y = array([])
        self.b, self.theta = array([]), array([])
        self.y_hat = array([])
        self.num_iterations = 200

        self.NormalizedData()
        self.runner()

        self.gd_iterations_df, self.b, self.theta = self.runner()
        self.makeGraph()
        print(self.gd_iterations_df)

    def runner(self):
        print(self.X.shape[1])
        self.b, self.theta = self.Initialize(self.X.shape[1])

        iter_num = 0
        result_idx = 0
        gd_iterations_df = pd.DataFrame(columns=['iteration', 'cost'])

        for each_iter in range(self.num_iterations):
            self.y_hat = self.predict_y()
            this_cost = self.Cost()

            self.b, self.theta = self.update_theta(0.01)

            if iter_num % 10 == 0:
                gd_iterations_df.loc[result_idx] = [iter_num, this_cost]

            result_idx = result_idx + 1
            iter_num += 1
            # print("final estimate b and theta", self.b, self.theta)

        return gd_iterations_df, self.b, self.theta

    def makeGraph(self):
        plt.plot(self.gd_iterations_df['iteration'], self.gd_iterations_df['cost'])
        plt.xlabel("number of iteration")
        plt.ylabel("cost or mse")
        plt.show()

    def Cost(self):
        Y_resd = self.Y - self.y_hat
        return np.sum(np.dot(Y_resd.T, Y_resd)) / len(self.Y - Y_resd)

    def update_theta(self, LearningRate):
        db = (np.sum(self.y_hat - self.Y) * 2) / len(self.Y)
        dw = (np.dot((self.y_hat - self.Y), self.X) * 2) / len(self.Y)
        b_1 = self.b - LearningRate * db
        theta_1 = self.theta - LearningRate * dw
        return b_1, theta_1

    def NormalizedData(self):
        self.X = self.df[['TV', 'radio', 'newspaper']]
        self.Y = self.df['sales']
        self.Y = np.array((self.Y - self.Y.mean()) / self.Y.std())
        self.X = self.X.apply(lambda rec: (rec - rec.mean()) / rec.std(), axis=0)

    def Initialize(self, dim):
        return random.random(), np.random.rand(dim)

    def predict_y(self):
        return self.b + np.dot(self.X, self.theta)
