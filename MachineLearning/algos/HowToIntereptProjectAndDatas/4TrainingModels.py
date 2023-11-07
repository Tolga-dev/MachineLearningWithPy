# LINEAR REGRESSION
# training models
# Linear Regression
# out = o0 + o1 * base
# vectorized form = y = h0(x) = 0.x
# 0 = parameter vector, containing bias term 0 and 01 - 0n
# x feature vector = containing x0 to xn with x0 always equal to 1
# 0*x is the dot product of the vectors 0 and x
# h0 is the hypothesis functions using model parameters 0

# vectors are generally represented as column vectors which are 2d arrays with a single column
# to check a model fits to data we use root-mean-square error
# mse(x, h0) = 1/m (mEi=1(0 ^ t * x ^ (i) - y(i))^ 2

# to find the value of 0 that minimizes the cost function, there is a closed form
# 0 = (X^T*X)^-1 * X^T*Y
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(self.y)

# inv() -> gives inverse of matrix and dot() method for matrix multiplications

# GRADIENT DESCENT

# Batch Gradient Descent
# generic optimization algorithm - tweak parameters iteratively in order to minimize a cost functions
# you need to calculate how much the cost function will change if you change  0 justa little bit. 'partial derivative'
# (derivative with regard to 0i)mse(x, h0) = 2/m (mEi=1(0 ^ t * x ^ (i) - y(i)) *x ^ (i)
# 0(next step) = 0 - n(learning rate)MSE(0)

# Stochastic Gradient Descent
# Main problem with batch gradient descent is  very slow in large sets
# it is much less regular than batch gradient descent

# Mini Batch gradient descent
# main advantage of its over Stochastic is that u can get a performance boost from hardware when using gpu
# IT CAN END UP A BIT CLOSER TO minimum than stochastic

# Polynomial Regression
# if ur data is not straight line you can use

# learning curves
# if we perform high degree polynomial regression, we will get better fitting

# Regularized Linear Models
# for a linear model regularization is typically achieved by constraining the weights of the model

# Ridge Regression
# tikhonov regularization, = a*(Ei,n 0i^2) is added to the cost functions
# a = how much you want to regularize the model, if a = 0, means just linear regression, if a is large,
# all weights end up very lose to zero, flat line
# MSE(0) + (a*1/2)(Ei,n 0i^2)
# 0 = (X^T*X + aA)^-1 * X^T*Y

# lasso regression
# the least absolute shrinkage and selection operator regression = usually called lasso regression
# MSE(0) + (a)(Ei,n 0i) uses vector
# eliminate the weights of the least important features

# Elastic Net
# middle ground between ridge and lasso
# u can control the mix ratio r
# MSE(0) + (a*(1-r)/2)(Ei,n 0i^2) + (a*r)(Ei,n 0i)

# logistic regression it is called as logit regression, it computes a weighted sum of the input features, but instead
# of outputting the result directly like the linear regression, regression model does,
# it outputs the logistic of this result.
# p = h0(x)  = a((x^t)0)
# a between 0 and 1
# a(t) = 1/(1 + exp(-t))
# J(0) = - (1/m)Em,i=1[y^(i)log(p^i) + (1 - y^i)log(1 - p^i)]
# there is no closed for equation to compute value of 0
# but, it is convex, thereby gradient descent is guarantee to find the global minimum

# soft-max regression
# logistic regression can be generalized to support multiple classes directly,
# without having to train and combine multiple binary classifiers
# this is softmax and multinomial logistic regression
# it computes a score s(x) for each class k, estimates probability of each class bu applying
# the soft-max function
# a(s(x))k = (exp(sk(x))) / (Ek,j=1 (exp(s(x))))

# algo large m, out-of-core-support Larger-n hyperparams scaling-required scikit-learn
# Normal Equation Fast No Slow 0 No N/A
# SVD  Fast     No                  Slow     0          No               LinearRegression
# BatchGD Slow  No                  Fast     2          Yes              SGDRegressor
# Stochastic GD Fast       Yes                 Fast     ≥2 Yes SGDRegressor
# Mini-batch GD Fast                Yes Fast ≥2 Yes SGDRegressor

# Exercises

# if you have millions of features you can use sgd or mini batch gd
# perhaps batch gradient descent if the training set fits in memory

# if we have very different scales, cost function will have the shape of an elongated bowl
# gda will take a long time to converge
# we may regularize the data and use normal equ or svd

# gd cannot get stuck in a local min in logistic regression mode, cost function is convex



import joblib as joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import random
import threading

train_model_cache_voting_classifier = joblib.Memory('./tmp/TrainingModelsCache/train_model_cache_voting_classifier')


@train_model_cache_voting_classifier.cache
def General_Training_Linear():
    pass


class LinearRegressionUserDefined:
    def __init__(self, X, y, X_b):
        # self.useFormula()
        self.X_b = X_b
        self.ling_reg = None
        self.X = X
        self.y = y
        self.y_predict = None
        self.X_new_b = None
        self.X_new = None
        self.useFormula()

    def usingSkLearn(self, X, y):
        self.ling_reg = LinearRegression()
        self.ling_reg.fit(X, y)
        print(self.ling_reg.intercept_, self.ling_reg.coef_)
        print(self.ling_reg.predict(self.X_new))

    def useFormula(self):
        theta_best = np.linalg.inv(self.X_b.T.dot(self.X_b)).dot(self.X_b.T).dot(self.y)

        # print(X_b)
        # print(theta_best)

        self.X_new = np.array([[0], [2]])
        self.X_new_b = np.c_[np.ones((2, 1)), self.X_new]  # add x0 = 1 to each instance
        self.y_predict = self.X_new_b.dot(theta_best)

        # print(self.y_predict)

        self.ex1Graph()
        # self.usingSkLearn(self.X, self.y)

    def ex1Graph(self):
        self.plotMe(self.X, self.y)

    def plotMe(self, X, y):
        plt.plot(X, y, "b.")
        plt.plot(self.X_new, self.y_predict, "r-")

        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$y$", rotation=0, fontsize=18)
        plt.axis([0, max(X), 0, max(y)])
        plt.show()


class GradientDescentUserDefined:  # batch
    def __init__(self, X, y, X_b):
        self.eta = None
        self.m = None
        self.n_iteration = None
        self.gradients = None
        self.theta = None
        self.X_b = X_b
        self.X = X
        self.y = y
        self.usingFormula()

    def usingFormula(self):
        self.eta = 0.1
        self.n_iteration = 2000
        self.m = 100
        self.theta = np.random.randn(2, 1)  # random initialzation

        for iteration in range(self.n_iteration):
            self.gradients = 2 / self.m * self.X_b.T.dot(self.X_b.dot(self.theta) - self.y)
            self.theta = self.theta - self.eta * self.gradients
        print(self.theta)


class StochasticGradientDescentUserDefined:
    def __init__(self, X, y, X_b):
        self.t1 = None
        self.t0 = None
        self.n_epochs = None
        self.eta = None
        self.m = None
        self.n_iteration = None
        self.gradients = None
        self.theta = None
        self.X_b = X_b
        self.X = X
        self.y = y
        self.usingFormula()
        self.useSklearn()

    def usingFormula(self):
        self.n_epochs = 50
        self.t0 = 5
        self.t1 = 50
        self.theta = np.random.randn(2, 1)
        self.m = 100

        for epoch in range(self.n_epochs):
            for i in range(self.m):
                random_index = np.random.randint(self.m)
                xi = self.X_b[random_index:random_index + 1]
                yi = self.y[random_index:random_index + 1]
                gradients = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                eta = self.learning_schedule(epoch * self.m + i)
                self.theta = self.theta - eta * gradients
        print(self.theta)

    def useSklearn(self):
        self.sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
        self.sgd_reg.fit(self.X, self.y.ravel())
        print(self.sgd_reg.intercept_, self.sgd_reg.coef_)

    def learning_schedule(self, t):
        return self.t0 / (t + self.t1)


class PolynomialRegression:
    def __init__(self, X, y, m):
        self.lin_reg = None
        self.X_poly = None
        self.poly_features = None
        self.X = X
        self.y = y
        self.m = m
        self.useSkLearn()

    def useSkLearn(self):
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.X_poly = self.poly_features.fit_transform(self.X)
        print(self.X[0])
        print(self.X_poly[0])
        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.X_poly, self.y)
        print(self.lin_reg.intercept_, self.lin_reg.coef_)


class LearningCurves:
    def __init__(self, X, y):
        self.lin_reg = None
        self.X_poly = None
        self.poly_features = None
        self.X = X
        self.y = y
        polynomial_regression = Pipeline([
            ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
            ("lin_reg", LinearRegression()),
        ])

        # self.plot_learning_curves(LinearRegression(), self.X, self.y)
        self.plot_learning_curves(polynomial_regression, self.X, self.y)

    def plot_learning_curves(self, model, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        train_errors, val_errors = [], []

        for m in range(1, len(X_train)):
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
            val_errors.append(mean_squared_error(y_val, y_val_predict))
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
        plt.show()


class RidgeRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        t1 = threading.Thread(target=self.skLearnRidge(self.X, self.y))
        t2 = threading.Thread(target=self.skLearnSgdRegressor(self.X, self.y))
        t1.start()
        t2.start()

    def skLearnRidge(self, X, y):
        ridge_reg = Ridge(alpha=1, solver="cholesky")  # andre louis cholesky
        ridge_reg.fit(X, y)
        print(ridge_reg.predict([[1.5]]))

    def skLearnSgdRegressor(self, X, y):
        sgd_reg = SGDRegressor(penalty="l2")  # andre louis cholesky
        sgd_reg.fit(X, y.ravel())
        print(sgd_reg.predict([[1.5]]))


class LassoRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        print("ok")
        t1 = threading.Thread(target=self.skLearnLasso(self.X, self.y))
        t2 = threading.Thread(target=self.skLearnSgdRegressor(self.X, self.y))
        t1.start()
        t2.start()

    def skLearnLasso(self, X, y):
        Lasso_reg = Lasso(alpha=0.1)  # andre louis cholesky
        Lasso_reg.fit(X, y)
        print(Lasso_reg.predict([[1.5]]))

    def skLearnSgdRegressor(self, X, y):
        sgd_reg = SGDRegressor(penalty="l2")  # andre louis cholesky
        sgd_reg.fit(X, y.ravel())
        print(sgd_reg.predict([[1.5]]))


class ElasticNetRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        print("ok")
        t1 = threading.Thread(target=self.skLearnElasticNet(self.X, self.y))
        t2 = threading.Thread(target=self.skLearnSgdRegressor(self.X, self.y))
        t1.start()
        t2.start()

    def skLearnElasticNet(self, X, y):
        ElasticNet_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)  # andre louis cholesky
        ElasticNet_reg.fit(X, y)
        print(ElasticNet_reg.predict([[1.5]]))

    def skLearnSgdRegressor(self, X, y):
        sgd_reg = SGDRegressor(penalty="l2")  # andre louis cholesky
        sgd_reg.fit(X, y.ravel())
        print(sgd_reg.predict([[1.5]]))


class DecisionBoundariesLogisticRegression:
    def __init__(self):
        self.iris = datasets.load_iris()
        self.X = self.iris["data"][:, 3:]
        self.y = (self.iris["target"] == 2)
        print("data ok")
        self.useSKlearn(self.X, self.y)

    def useSKlearn(self, X, y):
        log_reg = LogisticRegression()
        log_reg.fit(X, y)
        X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
        y_proba = log_reg.predict_proba(X_new)
        plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica")
        plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris virginica")
        print(log_reg.predict([[1.7], [1.5]]))
        plt.show()


def formula1():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    StochasticGradientDescentUserDefined(X, y, X_b)
    GradientDescentUserDefined(X, y, X_b)
    LinearRegressionUserDefined(X, y, X_b)


def formula2():
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
    # PolynomialRegression(X, y, m)
    # LearningCurves(X, y)
    # RidgeRegression(X, y)
    # LassoRegression(X, y)
    DecisionBoundariesLogisticRegression()



formula2()
