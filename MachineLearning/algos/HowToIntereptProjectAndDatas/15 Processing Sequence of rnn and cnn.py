# processing sequences using rnn and cnn
from pathlib import Path

import keras.losses
# unstable gradients
# a very limited short-term memory, which can be extended using lstm and gru cells
# we will implement wavenet

# recurrent neurons and layers
# it can also point backward, but it is very much like feedforward neural networks
# composed of one neuron-receiving input and producing an output and sending output back to itself
# bidirectional artificial neurol network,

# lstm
# long short-term memory, and set accuracy records,

# libs
# caffe c++
# torch C
# tensorflow

# architectures
# fully recurrent
# elman and jordan
# hopfield
# bidirectional associative memory

# apps
# machine translation
# robot control
# speech and handwriting recognition
# speech synthesis
# brain computer
# grammar learning


# memory cells

# input and output sequences.
# sequence to sequence network is useful for predicting time series such as stock prices.
# sequence to vector network called an encoder, opposite of it called decoder
# it could be used for translating a sentence from one lang to another

# training rnn
# simply use regular backpropagation, it is called backpropagation through time

# forecasting a time series
# suppose we re working on daily temp in our city, using multiple metrics, data will be a
# sequence of one or more values per time step
# this is called time series

# forecasting time series
# task of predicting future values based on historical data based on historical data
# to predicting price trends for cryptocurrencies such as bitcoin and ethereum,
# the most commonly used is autoregressive moving average (arma)
# ARIMA and SARIMA


# baseline metrics
# its good idea to have a few baseline metrics, or else we may end up thinking our model works great
# to predict the last value in each series, this is called naive forecasting

# deep rnn
# it is simple in implementation, basically using a rnn

# forecasting several time steps ahead.
# we can predict 10 steps away instead of 1 step.
# to predict steps ahead, change the targets to be the value 10 steps ahead instead of 1 step ahead.
# the first option is to use model and make it predict the next value, then add that value to input

# handling long sequences
# to train a rnn on long sequences, we must run it over many time steps
# unstable gradients problem

# layer normalization
# another form of normalization often works better with rnn
# very similarly to batch normalization
# it normalizes features dimension instead of batch

# Tackling the Short-Term Memory Problem
# due to transformations, some data can be lost at each time step.
# After a while, it contains no trace of the first inputs, it is show-stopper
# u read a long sentence to translate but you forgot where did u start
# to solve this problem we have lstm a.k.a long short-term memory

# LSTM
# it is easy to use just use LSTM instead of simpleRNN

# peephole connections
# lstm variant with extra connections called peephole

# Gru cells
# Gated Recurrent Unit
# Encoder decoder network


import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


class TrainingExample:
    def __init__(self):
        print("ok")
        np.random.seed(42)
        n_steps = 50
        # it creates as many time series as requested
        series = self.generate_time_series(10000, n_steps + 1)
        X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
        X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
        X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

        # show data
        # self.plot_series(n_steps, X_valid[0, :, 0], y_valid[0, 0])
        # plt.show()

        # computing some baselines
        # naive predictions
        # y_pred = X_valid[:, -1]
        # print(np.mean(keras.losses.mean_squared_error(y_valid, y_pred)))
        # self.plot_series(n_steps, X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
        # plt.show()

        # linear prediction, first option
        # model = keras.models.Sequential([
        #     keras.layers.Flatten(input_shape=[50,1]),
        #     keras.layers.Dense(1)
        # ])
        # model.compile(loss="mse", optimizer="adam")
        # history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))
        # print(model.evaluate(X_valid, y_valid))
        # self.plot_learning_curves(history.history["loss"], history.history["val_loss"])
        # plt.show()
        # y_pred = model.predict(X_valid)
        # self.plot_series(n_steps, X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
        # plt.show()

        # implementing a simple rnn
        # model = keras.models.Sequential(
        #     [
        #         keras.layers.SimpleRNN(1, input_shape=[None, 1])
        #     ])

        # deep rnn, second option,
        # just stack recurrent layers, three simplernn, lstm and gru layer can add
        # last layer is not ideal,
        # simple rnn layer uses the tanh activation functions.
        # model = keras.models.Sequential([
        #     keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        #     keras.layers.SimpleRNN(20, return_sequences=True),
        #     keras.layers.SimpleRNN(1)
        # ])
        #
        # model.compile(loss="mse", optimizer="adam")
        # history = model.fit(X_train, y_train, epochs=5,
        #                     validation_data=(X_valid, y_valid))
        # print(model.evaluate(X_valid, y_valid))
        # self.plot_learning_curves(history.history["loss"], history.history["val_loss"])
        # # plt.show()
        # y_pred = model.predict(X_valid)
        # # self.plot_series(n_steps, X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
        # # plt.show()
        #
        # # prediction of 10 steps
        # series = self.generate_time_series(1, n_steps + 10)
        # X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
        # X = X_new
        # for step_ahead in range(10):
        #     y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
        #     X = np.concatenate([X, y_pred_one], axis=1)
        #
        # Y_pred = X[:, n_steps:]
        # # self.plot_multiple_forecasts(X_new, Y_new, Y_pred)
        # # plt.show()
        # # using model then predicting next 10 val
        # n_steps = 50
        # series = self.generate_time_series(10000, n_steps + 10)
        # X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
        # X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
        # X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]
        #
        # X = X_valid
        # for step_ahead in range(10):
        #     y_pred_one = model.predict(X)[:, np.newaxis, :]
        #     X = np.concatenate([X, y_pred_one], axis=1)
        #
        # Y_pred = X[:, n_steps:, 0]
        #
        # print(Y_pred.shape)
        # print(np.mean(keras.metrics.mean_squared_error(Y_valid, Y_pred)))
        # # comparing this performance with some baselines, naive and simple linear
        #
        # model = keras.models.Sequential([
        #     keras.layers.Flatten(input_shape=[50, 1]),
        #     keras.layers.Dense(10)
        # ])
        #
        # model.compile(loss="mse", optimizer="adam")
        # history = model.fit(X_train, Y_train, epochs=20,
        #                     validation_data=(X_valid, Y_valid))

        # create rnn that predicts all 10 next vals at once

        # model = keras.models.Sequential([
        #     keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        #     keras.layers.SimpleRNN(20),
        #     keras.layers.Dense(10)
        # ])

        # model.compile(loss="mse", optimizer="adam")
        # history = model.fit(X_train, Y_train, epochs=20,
        #                     validation_data=(X_valid, Y_valid))
        # series = self.generate_time_series(1, 50 + 10)
        # X_new, Y_new = series[:, :50, :], series[:, -10:, :]
        # Y_pred = model.predict(X_new)[..., np.newaxis]
        # plot_multiple_forecasts(X_new, Y_new, Y_pred)
        # plt.show()

    def plot_learning_curves(self, loss, val_loss):
        plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
        plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
        plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        plt.axis([1, 20, 0, 0.05])
        plt.legend(fontsize=14)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)

    def plot_multiple_forecasts(self, X, Y, Y_pred):
        n_steps = X.shape[1]
        ahead = Y.shape[1]
        self.plot_series(n_steps, X[0, :, 0])
        plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "bo-", label="Actual")
        plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "rx-", label="Forecast", markersize=10)
        plt.axis([0, n_steps + ahead, -1, 1])
        plt.legend(fontsize=14)

    def generate_time_series(self, batch_size, n_steps):
        freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
        time = np.linspace(0, 1, n_steps)
        series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1
        series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + wave 2
        series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # + noise
        return series[..., np.newaxis].astype(np.float32)

    def plot_series(self, n_steps, series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$", legend=True):
        plt.plot(series, ".-")
        if y is not None:
            plt.plot(n_steps, y, "bo", label="Target")
        if y_pred is not None:
            plt.plot(n_steps, y_pred, "rx", markersize=10, label="Prediction")
        plt.grid(True)
        if x_label:
            plt.xlabel(x_label, fontsize=16)
        if y_label:
            plt.ylabel(y_label, fontsize=16, rotation=0)
        plt.hlines(0, 0, 100, linewidth=1)
        plt.axis([0, n_steps + 1, -1, 1])
        if legend and (y or y_pred):
            plt.legend(fontsize=14, loc="upper left")




def program1():
    TrainingExample()


if __name__ == '__main__':
    program1()
