# fan(avg) = (fan(in) + fan(out)) / 2
# glorot and bengio
# Xavier or Glorot initializations
# normal distribution with mean 0 and variance a^2 = 1 / fan(avg)
# a uniform distribution between -r and +r with r, r = sqrt(3 / fan(avg))
# for the uniform distribution, r = sqrt(3 * a^2)
import math
from functools import partial

# Initialization  Activation functions    σ² (Normal)
# Glorot          None, tanh, logistic, softmax   1 / fan avg
# He              ReLU and variants               2 / fan in
# LeCun           SELU                            1 / fan in

# by default keras uses glorot initialization with a uniform distribution
# you can change it to he initialization by setting, kernel_initializer="he_uniform" or "he_normal"

# Relu activation function is not perfect, it suffers from a problem known as the dying ReLUs .
# if u use them they can give just 0 as output
# when this happens, it just keeps outputting zeros and Gradient Descent does not affect it anymore because
# the gradient of the ReLU function is zero when its input is negative

# u may use leaky ReLU(a,z) = max(az,z)
# a defines how much the function leaks, z < 0 and typically set 0.01 and it is ensures ReLU never dies
# but can be in long come, hue leak seemed to result in better performance than small leak

# also RReLU stands for randomized leaky ReLU
# a is random. it is fairly well and seemed to act as a regularize

# and finally parametric leaky ReLU, a is learned by a training, PReLU was reported to strongly outperform ReLU
# on large image datasets but smaller there is a over-fitting training set problem

# exponential linear unit, ELU, out-performed all the ReLU variants in the author's experiments.

# elu looks a lot like the ReLU function with a few major differences.
# negative values when z<0, helps alleviate the vanishing gradients problem.
# nonzero gradient for z < 0, avoids the dead neurons.
# if a equal to 1, function is smooth everywhere, included z=0.
# it is faster.

# Scaled ELU

# it is a scaled variant of the ELU activation function.

# which activation function should we use for the hidden layers?
# in general SELU > ELU > LEAKY RELU > RELU > TANH > LOGISTIC
# if we care about runtime latency, then we can use leaky relu,


# batch normalization
# applied to a same image, classification model, batch achieves it with 14 times fewer taining steps
# acts like regularize
# but it add some complxity to the model
#

# faster optimizations
# momentum optimization, nesterov accelerated gradient, adagrad, rmsprop, adam, nadam optimization

# momentum
# optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

# nesterov accelerated gradient
# optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

# AdaGrad

# RMSProp, better than ada
# optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)

# Exercise
# is it ok to initalize all the wights to same val as long as that val is selected randly using he init
# no, not good solution

# bias terms to 0 can be initialized


import numpy as np
from keras.src.activations import selu
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import keras.layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def exponential_decacy_fn(epoch):
    return 0.01 * 0.1 ** (epoch / 20)


def exponential_decay_fn(epoch, lr):
    return lr * 0.1 ** (1 / 20)


def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001


def piecewise_constant(boundaries, values):
    boundaries = np.array([0] + boundaries)
    values = np.array(values)

    def piecewise_constant_fn(epoch):
        return values[np.argmax(boundaries > epoch) - 1]

    return piecewise_constant_fn


class TrainingDeepNN:
    def __init__(self):
        # print([name for name in dir(keras.initializers) if not name.startswith("_")])

        # to change kernel initializer
        # print(keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal"))
        # we can use variance scaling initializer for fan,avg, to fan,in,

        # he_avg_init = keras.initializers.VarianceScaling(scale=2, mode="fan_avg", distribution="uniform")
        # print(keras.layers.Dense(10, activation="relu", kernel_initializer=he_avg_init))

        #
        # np.random.seed(42)
        # Z = np.random.normal(size=(500, 100))
        # for layer in range(1000):
        #     W = np.random.normal(size=(100, 100), scale=np.sqrt(1 / 100))
        #     Z = selu(np.dot(Z, W))
        #     means = np.mean(Z, axis=0).mean()
        #     stds = np.std(Z, axis=0).mean()
        #
        #     if layer % 100 == 0:
        #         print("{} , {:.2f},   {:.2f}".format(layer, means, stds))

        # selu in keras
        # keras.layers.Dense(10, activation="selu", kernel_initializer="lecun_normal")

        np.random.seed(42)
        tf.random.set_seed(42)

        fashion_mnist = keras.datasets.fashion_mnist
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
        y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

        # model = keras.models.Sequential()
        # model.add(keras.layers.Flatten(input_shape=[28, 28]))
        # model.add(keras.layers.Dense(300, activation="relu",  # changing activator to relu and see what is happening
        #                              kernel_initializer="he_normal"))
        # for layer in range(99):
        #     model.add(keras.layers.Dense(100, activation="relu",
        #                               # Not great at all, we suffered from the vanishing/exploding gradients problem.
        #                                  kernel_initializer="he_normal"))
        # model.add(keras.layers.Dense(10, activation="softmax"))
        # model.compile(loss="sparse_categorical_crossentropy",
        #               optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        #               metrics=["accuracy"])

        pixel_means = X_train.mean(axis=0, keepdims=True)
        pixel_stds = X_train.std(axis=0, keepdims=True)
        X_train_scaled = (X_train - pixel_means) / pixel_stds
        X_valid_scaled = (X_valid - pixel_means) / pixel_stds
        X_test_scaled = (X_test - pixel_means) / pixel_stds
        # history = model.fit(X_train_scaled, y_train, epochs=5,
        #                     validation_data=(X_valid_scaled, y_valid))
        #
        # print(history)

        # batch normalization

        # model = keras.models.Sequential([
        #     keras.layers.Flatten(input_shape=[28, 28]),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.Dense(10, activation="softmax")
        # ])
        # print(model.summary())
        # # check its params
        # print([(var.name, var.trainable) for var in model.layers[1].variables])
        # # a  bn layer in keras, also creates two ops
        # print(model.layers[1].updates)

        # bn layers before activations functions
        # model = keras.models.Sequential([
        #     keras.layers.Flatten(input_shape=[28, 28]),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.Activation("elu"),
        #     keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.Activation("elu"),
        #     keras.layers.Dense(10, activation="softmax")
        # ])

        # to mitigate the exploding gradient-problem is to clip the gradient during backprop
        # gradient clipping
        # it is used often in recurrent neural networks as BN is tricky to us in rnn

        # Reusing pretrained layers
        # (X_train_A, y_train_A), (X_train_B, y_train_B) = self.split_dataset(X_train, y_train)
        # (X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = self.split_dataset(X_valid, y_valid)
        # (X_test_A, y_test_A), (X_test_B, y_test_B) = self.split_dataset(X_test, y_test)
        # X_train_B = X_train_B[:200]
        # y_train_B = y_train_B[:200]
        # # print(X_train_A.shape)
        # # print(X_train_B.shape)
        # # print(y_train_A[:30])
        #
        # model_A = keras.models.Sequential()
        # model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
        # for n_hidden in (300, 100, 50, 50, 50):
        #     model_A.add(keras.layers.Dense(n_hidden, activation="selu"))
        # model_A.add(keras.layers.Dense(8, activation="softmax"))
        # model_A.compile(loss="sparse_categorical_crossentropy",
        #                 optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        #                 metrics=["accuracy"])
        # history = model_A.fit(X_train_A, y_train_A, epochs=5,
        #                       validation_data=(X_valid_A, y_valid_A))
        # print(model_A.history)
        # print(model_A.layers)
        # print(model_A.summary())
        # model_A.save("tmp/models/my_model_A.h5")
        # print("model a is finished")
        #
        # model_B = keras.models.Sequential()
        # model_B.add(keras.layers.Flatten(input_shape=[28, 28]))
        # for n_hidden in (300, 100, 50, 50, 50):
        #     model_B.add(keras.layers.Dense(n_hidden, activation="selu"))
        # model_B.add(keras.layers.Dense(1, activation="sigmoid"))
        # model_B.compile(loss="binary_crossentropy",
        #                 optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        #                 metrics=["accuracy"])
        # history = model_B.fit(X_train_B, y_train_B, epochs=5,
        #                       validation_data=(X_valid_B, y_valid_B))
        # print(model_B.history)
        # print(model_B.layers)
        # print(model_B.summary())
        # print(history)
        # print("model b is finished")
        #
        # model_A_clone = keras.models.clone_model(model_A)
        # model_A_clone.set_weights(model_A.get_weights())
        # model_B_on_A = keras.models.Sequential(model_A_clone.layers[:-1])
        # model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))
        # print("model a is cloned to be")
        #
        # for layer in model_B_on_A.layers[:-1]:
        #     layer.trainable = False
        #
        # model_B_on_A.compile(loss="binary_crossentropy",
        #                      optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        #                      metrics=["accuracy"])
        # history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
        #                            validation_data=(X_valid_B, y_valid_B))
        # print(history)
        # print(model_B_on_A.history)
        # print(model_B_on_A.layers)
        # print(model_B_on_A.summary())
        #
        # for layer in model_B_on_A.layers[:-1]:
        #     layer.trainable = True
        #
        # model_B_on_A.compile(loss="binary_crossentropy",
        #                      optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        #                      metrics=["accuracy"])
        # history = model_B_on_A.fit(X_train_B, y_train_B, epochs=5,
        #                            validation_data=(X_valid_B, y_valid_B))
        #
        # print(history)
        # print(model_B.evaluate(X_test_B, y_test_B))
        # print(model_B_on_A.evaluate(X_test_B, y_test_B))

        # power scheduling
        # self.powerScheduling(X_train_scaled, y_train, X_valid_scaled, y_valid, X_train)

        # exponential scheduling
        # self.exponential(self, X_train_scaled, y_train, X_valid_scaled)

        # piece constant scheduling
        # self.PieceConstantSche(X_train_scaled, y_train, X_valid_scaled)

        # performance scheduling
        # self.PerformanceSche(X_train_scaled, y_train, X_valid_scaled)

        # Avoiding Over-fitting Through Regularization
        # l1 and l2 regularization
        # l2 regularization to constrain a nn connection weights and l1 regularization if

        # Avoiding Over-fitting Through Regularization
        # l1 and l2 regularization
        # l2 regularization to constrain a nn connection weights and l1 regularization if
        # we want a sparse model
        # layer = keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal",
        #                            kernel_regularizer=keras.regularizers.l2(.01))

        # model = keras.models.Sequential([
        #     keras.layers.Flatten(input_shape=[28, 28]),
        #     keras.layers.Dense(300, activation="elu",
        #                        kernel_initializer="he_normal",
        #                        kernel_regularizer=keras.regularizers.l2(0.01)),
        #     keras.layers.Dense(100, activation="elu",
        #                        kernel_initializer="he_normal",
        #                        kernel_regularizer=keras.regularizers.l2(0.01)),
        #     keras.layers.Dense(10, activation="softmax",
        #                        kernel_regularizer=keras.regularizers.l2(0.01))
        # ])
        # model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        # n_epochs = 2
        # history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
        #                     validation_data=(X_valid_scaled, y_valid))
        # print(model.summary())

        # l2 function returns a regularizing, it is added to final loss.
        # we can use l1 and l1_l2 to use
        # to not repeat args we can use partial in python

        # RegularizedDense = partial(keras.layers.Dense,
        #                            activation="elu",
        #                            kernel_initializer="he_normal",
        #                            kernel_regularizer=keras.regularizers.l2(0.01))
        #
        # model = keras.models.Sequential([
        #     keras.layers.Flatten(input_shape=[28, 28]),
        #     RegularizedDense(300),
        #     RegularizedDense(100),
        #     RegularizedDense(10, activation="softmax")
        # ])
        # model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        # n_epochs = 2
        # history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
        #                     validation_data=(X_valid_scaled, y_valid))
        # print(model.history)

        # Drop out
        # most popular regularization techniques for deep nn
        # increasing acc by 1 and 2
        # at every training step; every neuron, has a probability p of being temp.
        # It may be active in the next step
        # p is dropout rate.

        # model = keras.models.Sequential([
        #     keras.layers.Flatten(input_shape=[28, 28]),
        #     keras.layers.Dropout(rate=0.2),
        #     keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
        #     keras.layers.Dropout(rate=0.2),
        #     keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
        #     keras.layers.Dropout(rate=0.2),
        #     keras.layers.Dense(10, activation="softmax")
        # ])
        # model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        # n_epochs = 2
        # history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
        #                     validation_data=(X_valid_scaled, y_valid))
        # print(model.summary())

        # Monte Carlo Drop-out
        # tf.random.set_seed(42)
        # np.random.seed(42)
        #
        # y_probas = np.stack([model(X_test_scaled, training=True) for sample in range(100)])
        # y_proba = y_probas.mean(axis=0)
        # y_std = y_probas.std(axis=0)
        # print(np.round(model.predict(X_test_scaled[:1]), 2))
        # print(np.round(y_probas[:, :1], 2))
        # print(np.round(y_proba[:1], 2))
        #
        # y_std = y_probas.std(axis=0)
        # print(np.round(y_std[:1], 2))
        #
        # y_pred = np.argmax(y_proba, axis=1)
        # accuracy = np.sum(y_pred == y_test) / len(y_test)
        # print(accuracy)

        # max-norm regularization
        # popular for neural networks,
        # it does not add a regularization loss term to the overall loss function, instead,
        # in each steps, it rescaling w if needed, recuding r increases the amount of
        # regularization and helps reduce overfitting. Also help alleviate the unstable gradient
        # problem.
        # layer = keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal",
        #                            kernel_constraint=keras.constraints.max_norm(1.))
        # MaxNormDense = partial(keras.layers.Dense,
        #                        activation="selu", kernel_initializer="lecun_normal",
        #                        kernel_constraint=keras.constraints.max_norm(1.))
        #
        # model = keras.models.Sequential([
        #     keras.layers.Flatten(input_shape=[28, 28]),
        #     MaxNormDense(300),
        #     MaxNormDense(100),
        #     keras.layers.Dense(10, activation="softmax")
        # ])
        # model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        # n_epochs = 2
        # history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
        #                     validation_data=(X_valid_scaled, y_valid))

        # default dnn configuration
        # hyperparameters Default val
        # kernel init     he init
        # activation func elu
        # normalization   non if shallow, batch norm if deep
        # regularization  early stopping
        # optimizer       momentum optimization
        # learning rate   Icycle

        # dnn configuration for a self-normalizing net
        # hyperparameters Default val
        # kernel init     LeCun int
        # Activation      SELU
        # Normalization   None
        # Regularization  Alpha drop if needed
        # Optimizer       Momentum optimization
        # Learning rate   Icycle

        # if we need a sparse model, we can use l1 regularization, even sparser model
        # we can use tensorflow model optimization toolkit
        # this will break self-normalization, use in this case,u have to use default
        # configurations.

        pass

    def exponential_decay(self, lr0, s):
        def exponential_decay_fn(epoch):
            return lr0 * 0.1 ** (epoch / s)

        return exponential_decay_fn

    def split_dataset(self, X, y):
        y_5_or_6 = (y == 5) | (y == 6)  # sandals or shirts
        y_A = y[~y_5_or_6]
        y_A[y_A > 6] -= 2  # class indices 7, 8, 9 should be moved to 5, 6, 7
        y_B = (y[y_5_or_6] == 6).astype(np.float32)  # binary classification task: is it a shirt (class 6)?
        return ((X[~y_5_or_6], y_A),
                (X[y_5_or_6], y_B))

    def powerScheduling(self, X_train_scaled, y_train, X_valid_scaled, y_valid, X_train):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,  # Initial learning rate
            decay_steps=10000,  # Decay every 10,000 steps
            decay_rate=0.9  # Decay rate
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(10, activation="softmax")
        ])

        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        n_epochs = 5
        history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                            validation_data=(X_valid_scaled, y_valid))
        print(history)

        learning_rate = 0.01
        decay = 1e-4
        batch_size = 32
        n_steps_per_epoch = math.ceil(len(X_train) / batch_size)
        epochs = np.arange(n_epochs)
        lrs = learning_rate / (1 + decay * epochs * n_steps_per_epoch)

        plt.plot(epochs, lrs, "o-")
        plt.axis([0, n_epochs - 1, 0, 0.01])
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Power Scheduling", fontsize=14)
        plt.grid(True)
        plt.show()

    def PerformanceSche(self, X_train_scaled, y_train, X_valid_scaled, y_valid):
        tf.random.set_seed(42)
        np.random.seed(42)
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(10, activation="softmax")
        ])
        optimizer = keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        n_epochs = 9
        history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                            validation_data=(X_valid_scaled, y_valid),
                            callbacks=[lr_scheduler])

        plt.plot(history.epoch, history.history["lr"], "bo-")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate", color='b')
        plt.tick_params('y', colors='b')
        plt.gca().set_xlim(0, n_epochs - 1)
        plt.grid(True)

        ax2 = plt.gca().twinx()
        ax2.plot(history.epoch, history.history["val_loss"], "r^-")
        ax2.set_ylabel('Validation Loss', color='r')
        ax2.tick_params('y', colors='r')

        plt.title("Reduce LR on Plateau", fontsize=14)
        plt.show()

    def PieceConstantSche(self, X_train_scaled, y_train, X_valid_scaled, y_valid):
        piecewise_constant_fn = piecewise_constant([5, 15], [0.01, 0.005, 0.001])
        lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)

        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        n_epochs = 8
        history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                            validation_data=(X_valid_scaled, y_valid),
                            callbacks=[lr_scheduler])

        plt.plot(history.epoch, [piecewise_constant_fn(epoch) for epoch in history.epoch], "o-")
        plt.axis([0, n_epochs - 1, 0, 0.011])
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Piecewise Constant Scheduling", fontsize=14)
        plt.grid(True)
        plt.show()

    def exponential(self, X_train_scaled, y_train, X_valid_scaled, y_valid):
        exponential_decay_fn = self.exponential_decay(lr0=0.01, s=20)
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        n_epochs = 5
        lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

        history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                            validation_data=(X_valid_scaled, y_valid),
                            callbacks=[lr_scheduler])
        print(history)
        plt.plot(history.epoch, history.history["lr"], "o-")
        plt.axis([0, n_epochs - 1, 0, 0.011])
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Exponential Scheduling", fontsize=14)
        plt.grid(True)
        plt.show()


class TrainingExample:
    def __init__(self):
        # trying regularizing the model with alpha dropout
        print("ok")

    def BatchNOrmalizationWithDropout(self):
        keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)
        model_path = "tmp/models/my_cifar10_alpha_dropout_model.h5"
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))

        for _ in range(20):
            model.add(keras.layers.Dense(100,
                                         kernel_initializer="lecun_normal",
                                         activation="selu"))

        model.add(keras.layers.AlphaDropout(rate=0.1))
        model.add(keras.layers.Dense(10, activation="softmax"))

        optimizer = keras.optimizers.Nadam(learning_rate=5e-4)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
        model_checkpoint_cb = keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
        run_index = 1  # increment every time you train the model
        run_logdir = os.path.join(os.curdir, "my_cifar10_logs", "run_alpha_dropout_{:03d}".format(run_index))
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
        callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

        (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
        X_train = X_train_full[5000:]
        y_train = y_train_full[5000:]
        X_valid = X_train_full[:5000]
        y_valid = y_train_full[:5000]

        X_means = X_train.mean(axis=0)
        X_stds = X_train.std(axis=0)
        X_train_scaled = (X_train - X_means) / X_stds
        X_valid_scaled = (X_valid - X_means) / X_stds
        X_test_scaled = (X_test - X_means) / X_stds

        X_means = X_train.mean(axis=0)
        X_stds = X_train.std(axis=0)
        X_train_scaled = (X_train - X_means) / X_stds
        X_valid_scaled = (X_valid - X_means) / X_stds
        X_test_scaled = (X_test - X_means) / X_stds

        model.fit(X_train_scaled, y_train, epochs=5,
                  validation_data=(X_valid_scaled, y_valid),
                  callbacks=callbacks)

        model = keras.models.load_model(model_path)
        print(model.evaluate(X_valid_scaled, y_valid))

    def BatchNormalizationWithSelu(self):
        keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)

        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
        for _ in range(20):
            model.add(keras.layers.Dense(100,
                                         kernel_initializer="lecun_normal",
                                         activation="selu"))
        model.add(keras.layers.Dense(10, activation="softmax"))

        optimizer = keras.optimizers.Nadam(learning_rate=7e-4)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
        model_checkpoint_cb = keras.callbacks.ModelCheckpoint("tmp/models/my_cifar10_model.h5", save_best_only=True)
        run_index = 1  # increment every time you train the model
        run_logdir = os.path.join(os.curdir, "my_cifar10_logs", "run_selu_{:03d}".format(run_index))
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
        callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

        (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
        X_train = X_train_full[5000:]
        y_train = y_train_full[5000:]
        X_valid = X_train_full[:5000]
        y_valid = y_train_full[:5000]

        X_means = X_train.mean(axis=0)
        X_stds = X_train.std(axis=0)
        X_train_scaled = (X_train - X_means) / X_stds
        X_valid_scaled = (X_valid - X_means) / X_stds
        X_test_scaled = (X_test - X_means) / X_stds

        model.fit(X_train_scaled, y_train, epochs=5,
                  validation_data=(X_valid_scaled, y_valid),
                  callbacks=callbacks)

        model = keras.models.load_model("tmp/models/my_cifar10_model.h5")
        model.evaluate(X_valid_scaled, y_valid)
        print("yes")

    def BatchNormalization(self):
        keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)

        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
        model.add(keras.layers.BatchNormalization())
        for _ in range(20):
            model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation("elu"))
        model.add(keras.layers.Dense(10, activation="softmax"))

        optimizer = keras.optimizers.Nadam(learning_rate=5e-4)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
        model_checkpoint_cb = keras.callbacks.ModelCheckpoint("tmp/models/my_cifar10_model.h5", save_best_only=True)
        run_index = 1  # increment every time you train the model
        run_logdir = os.path.join(os.curdir, "my_cifar10_logs", "run_bn_{:03d}".format(run_index))
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
        callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

        (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
        X_train = X_train_full[5000:]
        y_train = y_train_full[5000:]
        X_valid = X_train_full[:5000]
        y_valid = y_train_full[:5000]

        model.fit(X_train, y_train, epochs=10,
                  validation_data=(X_valid, y_valid),
                  callbacks=callbacks)

        model = keras.models.load_model("tmp/models/my_cifar10_model.h5")
        print(model.evaluate(X_valid, y_valid))

    def UsingNadamOptimization(self):
        keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)

        # dnn with 20 hidden layers of 100 neurons
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
        for _ in range(20):
            model.add(keras.layers.Dense(100,
                                         activation="elu",
                                         kernel_initializer="he_normal"))
        # using nadam optimization and early stopping, use softmax with 10 neurons.

        model.add(keras.layers.Dense(10, activation="softmax"))
        # using optimizer
        optimizer = keras.optimizers.Nadam(learning_rate=5e-5)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        # load data and use early stopping, and we need a validation set

        (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
        X_train = X_train_full[5000:]
        y_train = y_train_full[5000:]
        X_valid = X_train_full[:5000]
        y_valid = y_train_full[:5000]

        # create callbacks and train the model
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
        model_check_point_cb = keras.callbacks.ModelCheckpoint(
            "tmp/models/my_cifar10_model.h5",
            save_best_only=True
        )
        run_index = 1  # increment every time you train the model
        run_logdir = os.path.join(os.curdir, "my_cifar10_model_logs", "run_{:03d}".format(run_index))
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
        callbacks = [early_stopping_cb, model_check_point_cb, tensorboard_cb]

        model.fit(X_train, y_train, epochs=10,
                  validation_data=(X_valid, y_valid),
                  callbacks=callbacks)


def program1():
    # TrainingDeepNN()
    TrainingExample()


if __name__ == '__main__':
    program1()
