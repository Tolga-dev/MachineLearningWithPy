# introduction to artificial neural networks with keras
import joblib
# ANNs
# inspired from human brain,
# w(i,j) next step = w(i,j) + n(yi - y|j)xi
# wij = wight between ith input neuron and jth output neuron
#  y|j = jth
# yj = target
# n = learning rate


# implementing Mlp with KERAS
# KERAS is a high level deep learning api that allows you to easily build, train, evaluate,
# and execute all sorts of neural networks
# TENSOR-flow, cntk, theano

# summarize
# neural nets came from
# what an mlp is and how we can use it for classification and regression
# how to use tf.keras's sequential api to build mlp and how to use the functional api
# how to sasve and restore a model and callbacks and checkpoints
# tensor-board


# examples
# a classical perceptron will converge only if the dataset is linearly separable
# ,and it won't be able to estimate class probabilities,

#


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import reciprocal
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
import os
import joblib as joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_model_cache_fashion_mnist = joblib.Memory('./tmp/TrainingModelsCacheData/fashionData')


@train_model_cache_fashion_mnist.cache
def train_model_cache_fashion_mnist_keeper():
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    return (X_train_full, y_train_full), (X_test, y_test)


class DeepLearning:
    def __init__(self):
        self.root_logdir = os.path.join(os.curdir, "TensorBoardLogs")
        self.UsingTensorBoardForVisualization()
        pass

    def perceptron(self):
        iris = load_iris()
        X = iris.data[:, (2, 3)]  # petal length, petal width
        y = (iris.target == 0).astype(np.int64)

        per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
        per_clf.fit(X, y)

        y_pred = per_clf.predict([[2, 0.5]])

        print(y_pred)

    def ClassificationMLP(self):
        # print(tf.__version__)
        # print(keras.__internal__)

        fashion_mnist = keras.datasets.fashion_mnist
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        print(X_train_full.info())
        print(y_train_full.info())
        print(X_test.info())
        print(y_test.info())
        print(X_train_full.shape)
        print(y_train_full.shape)

        X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
        y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        print(class_names[y_train[0]])

        # model = keras.models.Sequential()
        # model.add(keras.layers.Flatten(input_shape=[28, 28]))
        # model.add(keras.layers.Dense(300, activation="relu"))
        # model.add(keras.layers.Dense(100, activation="relu"))
        # model.add(keras.layers.Dense(10, activation="softmax"))
        # https://keras.io/activations/

        # alternatively we could use
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="relu"),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])
        print(model.summary())

        # get layer names
        print(model.layers)
        hidden1 = model.layers[1]
        print(hidden1.name)

        # also can get weights and bias
        weights, biases = hidden1.get_weights()
        print(weights)
        print(weights.shape)
        print(biases)
        print(biases.shape)

        model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

        history = model.fit(X_train, y_train, epochs=30,
                            validation_data=(X_valid, y_valid))
        print(history)

        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
        plt.show()

        model.evaluate(X_test, y_test)

        X_new = X_test[:3]
        y_proba = model.predict(X_new)
        y_proba.round(2)

        y_pred = model.predict_classes(X_new)
        print(y_pred)
        print(np.array(class_names)[y_pred])
        y_new = y_test[:3]
        print(y_new)

    def RegressionMLP(self):
        housing = fetch_california_housing()  # using fetch california housing

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full)
        scaler = StandardScaler()

        # after loading data, we will split it into a training set and,
        # a validation and test set

        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

        # to make simple predictions
        # it is same as classification but main difference is it has
        # a single neuron
        # the loss function is just basically a mean square error
        # the dataset can be quite noisy, we use a single hidden layer with fewer neurons
        # it is quite common with more complex topologies, or with multiple inputs
        # or outputs
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
            keras.layers.Dense(1)
        ])
        model.compile(loss="mean_squared_error", optimizer="sgd")
        history = model.fit(X_train, y_train, epochs=20,
                            validation_data=(X_valid, y_valid))
        mse_test = model.evaluate(X_test, y_test)
        X_new = X_test[:3]  # pretend these are new instances
        y_pred = model.predict(X_new)
        print(y_pred)

        # one example for non-sequential neural network is a wide deep neurol network

        # first we need to create an input object, it is just a specification
        input_ = keras.layers.Input(shape=X_train.shape[1:])
        # it is passing a object the input, that's why it is 'functional api'
        # no data is processed yet, it is just a hidden layer
        hidden1 = keras.layers.Dense(30, activation="relu")(input_)
        # it is same with hidden1
        hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
        concat = keras.layers.Concatenate()([input_, hidden2])
        # a single neuron and no activation function
        output = keras.layers.Dense(1)(concat)
        model = keras.Model(inputs=[input_], outputs=[output])
        print(model.summary())

        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        history = model.fit(X_train, y_train, epochs=20,
                            validation_data=(X_valid, y_valid))
        mse_test = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_new)

        # if we wanna send a subset of the features through the wide path and a different subset,
        # through the deep path, in that case, use multiple inputs
        # input a
        input_A = keras.layers.Input(shape=[5], name="wide_input")
        # input b
        input_B = keras.layers.Input(shape=[6], name="deep_input")
        # connected to b
        hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
        # connected to hidden
        hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
        # connected to both inputs
        concat = keras.layers.concatenate([input_A, hidden2])
        # output of neuron
        output = keras.layers.Dense(1, name="output")(concat)

        # compile model, same as before
        model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
        model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
        X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
        X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
        X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]
        history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                            validation_data=((X_valid_A, X_valid_B), y_valid))
        mse_test = model.evaluate((X_test_A, X_test_B), y_test)
        y_pred = model.predict((X_new_A, X_new_B))

        # to create multiple outputs we may basically connect to aux_output
        output = keras.layers.Dense(1, name="main_output")(concat)
        aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
        model = keras.models.Model(inputs=[input_A, input_B],
                                   outputs=[output, aux_output])
        # each output need its loss function
        # therefore, when we compile the model, we should pass a list of losses
        model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
                            validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))
        # after fitting it to save ur model  use save
        model.save('tmp/models/dummy_model.h')
        # to load ur model u can use load
        model = keras.models.load_model("tmp/models/dummy_model.h")
        print(history)
        # use checkpoint for computer crashing, with callbacks
        checkpoint_cb = keras.callbacks.ModelCheckpoint("tmp/models/checkpoints/checkpointA.h")
        # dummy history
        history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])
        # fit method accepts callbacks arguments that lets you specific a list of objects

        # in here if we set save_best_only, don't need to worry about over-fitting the training set
        #
        checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
        history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb])
        model = keras.models.load_model("my_keras_model.h5")  # roll back to best model

        # we can use early stopping callback
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
                            callbacks=[checkpoint_cb, early_stopping_cb])

        total_loss, main_loss, aux_loss = model.evaluate(
            [X_test, X_test_B], [y_test, y_test])

        # use predict method will return a prediction for each output
        y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])
        # we can build any sort of architecture

    def UsingTensorBoardForVisualization(self):
        # it is a good interactive visualization tool
        # binary log files called event files
        # each binary data record is called summary
        run_log_dir = self.get_run_log_dir()
        print(run_log_dir)

        # settings
        keras.backend.clear_session()
        np.random.seed(42)  # check for cache again
        tf.random.set_seed(42)
        checkpoint_cb = keras.callbacks.ModelCheckpoint("tmp/training_check_points.h5", save_best_only=True)

        housing = fetch_california_housing()

        X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)

        model = keras.models.Sequential(
            [
                keras.layers.Dense(30, activation="relu", input_shape=[8]),
                keras.layers.Dense(30, activation="relu"),
                keras.layers.Dense(1)
            ]
        )
        model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=0.05))
        tensorboard_cb = keras.callbacks.TensorBoard(run_log_dir)
        history = model.fit(X_train, y_train, epochs=15,
                            validation_data=(X_valid, y_valid),
                            callbacks=[checkpoint_cb, tensorboard_cb])
        print(history)
        print(help(keras.callbacks.TensorBoard.__init__))

        pass

    def get_run_log_dir(self):
        import time
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(self.root_logdir, run_id)


# using the subclassing api to build dynamic models
# both sequential and functional are declarative
class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)  # handles standard args (e.g., name)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output


class FineTuningNeuralNetworkHyperparameters:
    def __init__(self):
        # how do we know what combination of hyperparameters is the best for ur task?
        #  k fold cross validation, we can use
        # grid-search or randomized search cv
        housing = fetch_california_housing()  # using fetch california housing

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full)
        X_new = X_test[:3]

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        keras_reg = keras.wrappers.scikit_learn.KerasRegressor(self.build_model)

        keras_reg.fit(X_train, y_train, epochs=100,
                      validation_data=(X_valid, y_valid),
                      callbacks=[keras.callbacks.EarlyStopping(patience=10)])

        mse_test = keras_reg.score(X_test, y_test)
        print(mse_test)
        y_pred = keras_reg.predict(X_new)
        print(y_pred)

        param_distribs = {
            "n_hidden": [0, 1, 2, 3],
            "n_neurons": np.arange(1, 100),
            "learning_rate": reciprocal(3e-4, 3e-2),
        }
        rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
        rnd_search_cv.fit(X_train, y_train, epochs=100,
                          validation_data=(X_valid, y_valid),
                          callbacks=[keras.callbacks.EarlyStopping(patience=10)])
        print(rnd_search_cv.best_params_)
        print(rnd_search_cv.best_score_)
        model = rnd_search_cv.best_estimator_.model
        ## save it !

        pass

    def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
        model = keras.models.Sequential()

        model.add(keras.layers.InputLayer(input_shape=input_shape))
        for layer in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation="relu"))
        model.add(keras.layers.Dense(1))
        optimizer = keras.optimizers.SGD(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)

        return model


class Exercises:
    def __init__(self):

        pass


def program1():
    # DeepLearning()
    # model = WideAndDeepModel()
    FineTuningNeuralNetworkHyperparameters()


if __name__ == '__main__':
    program1()
