# it is a powerful lib for numerical computations
# for large machine learning
# it is used in image classification, natural language processing, recommender
# system, and time series forecasting
import numpy as np
# using tensorflow like numpy
# tensorflow is working around tensors which flow from operation to operation
# when creating custom cost functions, metrics, layers, and more

# custom loss function,
# suppose we want to train a regression model, but set is a bit noisy, after removing some noise
# dataset is still noisy, we can use huber loss function instead of good old mse
# it is not currently part of the official keras api, it is available in tf.keras.
# let's pretend it is not there.



import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tqdm.notebook import trange
from collections import OrderedDict

def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    return huber_fn

def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

class TrainingExample:
    def __init__(self):
        housing = fetch_california_housing()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        input_shape = X_train.shape[1:]

        path = "tmp/models/my_custom_model.ckpt"

        exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))

        print(exponential_layer([-1., 0., 1.]))

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        X_new_scaled = X_test_scaled




        pass

    class ReconstructingRegressor(keras.Model):
        def __init__(self, output_dim, **kwargs):
            super().__init__(**kwargs)
            self.hidden = [keras.layers.Dense(30, activation="selu",
                                              kernel_initializer="lecun_normal")
                           for _ in range(5)]
            self.out = keras.layers.Dense(output_dim)
            self.reconstruction_mean = keras.metrics.Mean(name="reconstruction_error")

        def build(self, batch_input_shape):
            n_inputs = batch_input_shape[-1]
            self.reconstruct = keras.layers.Dense(n_inputs)
            # super().build(batch_input_shape)

        def call(self, inputs, training=None):
            Z = inputs
            for layer in self.hidden:
                Z = layer(Z)
            reconstruction = self.reconstruct(Z)
            recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
            self.add_loss(0.05 * recon_loss)
            if training:
                result = self.reconstruction_mean(recon_loss)
                self.add_metric(result)
            return self.out(Z)
    class ResidualBlock(keras.layers.Layer):
        def __init__(self, n_layers, n_neurons, **kwargs):
            super().__init__(**kwargs)
            self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                              kernel_initializer="he_normal")
                           for _ in range(n_layers)]

        def call(self, inputs):
            Z = inputs
            for layer in self.hidden:
                Z = layer(Z)
            return inputs + Z

    class ResidualRegressor(keras.models.Model):
        def __init__(self, output_dim, **kwargs):
            super().__init__(**kwargs)
            self.hidden1 = keras.layers.Dense(30, activation="elu",
                                              kernel_initializer="he_normal")
            self.block1 = self.ResidualBlock(2, 30)
            self.block2 = self.ResidualBlock(2, 30)
            self.out = keras.layers.Dense(output_dim)

        def call(self, inputs):
            Z = self.hidden1(inputs)
            for _ in range(1 + 3):
                Z = self.block1(Z)
            Z = self.block2(Z)
            return self.out(Z)

    class AddGaussianNoise(keras.layers.Layer):
        def __init__(self, stddev, **kwargs):
            super().__init__(**kwargs)
            self.stddev = stddev

        def call(self, X, training=None):
            if training:
                noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
                return X + noise
            else:
                return X

        def compute_output_shape(self, batch_input_shape):
            return batch_input_shape

    def split_data(self, data):
        columns_count = data.shape[-1]
        half = columns_count // 2
        return data[:, :half], data[:, half:]

    class MyMultiLayer(keras.layers.Layer):
        def call(self, X):
            X1, X2 = X
            print("X1.shape: ", X1.shape, " X2.shape: ", X2.shape)  # Debugging of custom layer
            return X1 + X2, X1 * X2

        def compute_output_shape(self, batch_input_shape):
            batch_input_shape1, batch_input_shape2 = batch_input_shape
            return [batch_input_shape1, batch_input_shape2]

    class MyDense(keras.layers.Layer):
        def __init__(self, units, activation=None, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.activation = keras.activations.get(activation)

        def build(self, batch_input_shape):
            self.kernel = self.add_weight(
                name="kernel", shape=[batch_input_shape[-1], self.units],
                initializer="glorot_normal")
            self.bias = self.add_weight(
                name="bias", shape=[self.units], initializer="zeros")
            super().build(batch_input_shape)  # must be at the end

        def call(self, X):
            return self.activation(X @ self.kernel + self.bias)

        def compute_output_shape(self, batch_input_shape):
            return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "units": self.units,
                    "activation": keras.activations.serialize(self.activation)}

    def CustomModel(self):
        housing = fetch_california_housing()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        input_shape = X_train.shape[1:]

        path = "tmp/models/my_custom_model.ckpt"

        exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))

        print(exponential_layer([-1., 0., 1.]))

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        X_new_scaled = X_test_scaled

        model = self.ResidualRegressor(1)
        model.compile(loss="mse", optimizer="nadam")
        history = model.fit(X_train_scaled, y_train, epochs=5)
        score = model.evaluate(X_test_scaled, y_test)
        y_pred = model.predict(X_new_scaled)

        model.save(path)
        model = keras.models.load_model(path)
        history = model.fit(X_train_scaled, y_train, epochs=5)

        block1 = self.ResidualBlock(2, 30)
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal"),
            block1, block1, block1, block1,
            self.ResidualBlock(2, 30),
            keras.layers.Dense(1)
        ])
        model.compile(loss="mse", optimizer="nadam")
        history = model.fit(X_train_scaled, y_train, epochs=5)
        score = model.evaluate(X_test_scaled, y_test)
        y_pred = model.predict(X_new_scaled)

        model = self.ReconstructingRegressor(1)
        model.compile(loss="mse", optimizer="nadam")
        history = model.fit(X_train_scaled, y_train, epochs=2)
        y_pred = model.predict(X_test_scaled)

    def CustomLayer(self):
        housing = fetch_california_housing()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        input_shape = X_train.shape[1:]

        path = "tmp/models/my_model_with_a_custom_metric.h5"

        exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))

        print(exponential_layer([-1., 0., 1.]))

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        # model = keras.models.Sequential(
        #     [
        #         keras.layers.Dense(30, activation="relu", input_shape=input_shape),
        #         keras.layers.Dense(1),
        #         exponential_layer
        #     ]
        # )
        # model.compile(loss="mse", optimizer="sgd")
        # model.fit(X_train_scaled, y_train, epochs=5, validation_data=(X_valid_scaled, y_valid))
        # model.evaluate(X_test_scaled, y_test)

        # model = keras.models.Sequential([
        #     self.MyDense(30, activation="relu", input_shape=input_shape),
        #     self.MyDense(1)
        # ])
        # model.compile(loss="mse", optimizer="nadam")
        # model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid))
        # model.evaluate(X_test_scaled, y_test)
        # model.save(path)
        # model = keras.models.load_model("my_model_with_a_custom_layer.h5",
        #                                 custom_objects={"MyDense": self.MyDense})

        X_train_scaled_A, X_train_scaled_B = self.split_data(X_train_scaled)
        X_valid_scaled_A, X_valid_scaled_B = self.split_data(X_valid_scaled)
        X_test_scaled_A, X_test_scaled_B = self.split_data(X_test_scaled)

        outputs1, outputs2 = self.MyMultiLayer()((X_train_scaled_A, X_train_scaled_B))

        # input_A = keras.layers.Input(shape=X_train_scaled_A.shape[-1])
        # input_B = keras.layers.Input(shape=X_train_scaled_B.shape[-1])
        # hidden_A, hidden_B = self.MyMultiLayer()((input_A, input_B))
        # hidden_A = keras.layers.Dense(30, activation='selu')(hidden_A)
        # hidden_B = keras.layers.Dense(30, activation='selu')(hidden_B)
        # concat = keras.layers.Concatenate()((hidden_A, hidden_B))
        # output = keras.layers.Dense(1)(concat)
        # model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
        #
        # model.compile(loss='mse', optimizer='nadam')
        # model.fit((X_train_scaled_A, X_train_scaled_B), y_train, epochs=2,
        #           validation_data=((X_valid_scaled_A, X_valid_scaled_B), y_valid))

        model = keras.models.Sequential([
            self.AddGaussianNoise(stddev=1.0),
            keras.layers.Dense(30, activation="selu"),
            keras.layers.Dense(1)
        ])
        model.compile(loss="mse", optimizer="nadam")
        model.fit(X_train_scaled, y_train, epochs=2,
                  validation_data=(X_valid_scaled, y_valid))
        model.evaluate(X_test_scaled, y_test)

    # custom metrics
    # loses and metrics are conceptually not the same thing:
    # loses are used by gradient descent to train a model, so they must be differentiable
    # it is also not interpretable by humans
    # in contrast, metrics are used to evaluate a model, easily interpretable, can be non-differentiable
    # or have 0 gradients everywhere.
    def CustomMetrix(self):
        housing = fetch_california_housing()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        input_shape = X_train.shape[1:]

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)
        m = self.HuberMetric(2.)

        # print(m(tf.constant([[2.]]), tf.constant([[10.]])))
        m(tf.constant([[0.], [5.]]), tf.constant([[1.], [9.25]]))
        print(m.result())
        print(m.variables)
        print(m.reset_state())
        print(m.variables)

        input_shape = X_train.shape[1:]
        path = "tmp/models/my_model_with_a_custom_metric.h5"
        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                               input_shape=input_shape),
            keras.layers.Dense(1),
        ])
        model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[self.HuberMetric(2.0)])
        print(model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2))

        model.save(path)
        model = keras.models.load_model(path,
                                        custom_objects={"huber_fn": create_huber(2.0),
                                                        "HuberMetric": self.HuberMetric})
        model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)

        print(model.metrics[-1].threshold)

    class HuberMetric(keras.metrics.Metric):
        def __init__(self, threshold=1.0, **kwargs):
            super().__init__(**kwargs)  # handles base args (e.g., dtype)
            self.threshold = threshold
            self.huber_fn = create_huber(threshold)
            self.total = self.add_weight("total", initializer="zeros")
            self.count = self.add_weight("count", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            metric = self.huber_fn(y_true, y_pred)
            self.total.assign_add(tf.reduce_sum(metric))
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

        def result(self):
            return self.total / self.count

        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "threshold": self.threshold}

    def otherCustomFunctions(self):
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        fashion_mnist = keras.datasets.fashion_mnist
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
        y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

        pixel_means = X_train.mean(axis=0, keepdims=True)
        pixel_stds = X_train.std(axis=0, keepdims=True)
        X_train_scaled = (X_train - pixel_means) / pixel_stds
        X_valid_scaled = (X_valid - pixel_means) / pixel_stds

        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                               input_shape=self.input_shape),
            keras.layers.Dense(1, activation=self.my_softplus,
                               kernel_regularizer=self.my_l1_regularizer,
                               kernel_constraint=self.my_positive_weights,
                               kernel_initializer=self.my_glorot_initializer),
        ])
        model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
        model.fit(X_train_scaled, y_train, epochs=2,
                  validation_data=(X_valid_scaled, y_valid))
        path = 'tmp/models/model.save("my_model_with_many_custom_parts.h5'
        model.save(path)

        model = keras.models.load_model(
            path,
            custom_objects={
                "my_l1_regularizer": self.my_l1_regularizer,
                "my_positive_weights": self.my_positive_weights,
                "my_glorot_initializer": self.my_glorot_initializer,
                "my_softplus": self.my_softplus,
            })

        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                               input_shape=[32, 32, 3]),
            keras.layers.Dense(1, activation=self.my_softplus,
                               kernel_regularizer=MyL1Regularizer(0.01),
                               kernel_constraint=self.my_positive_weights,
                               kernel_initializer=self.my_glorot_initializer),
        ])
        model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
        model.fit(X_train_scaled, y_train, epochs=2,
                  validation_data=(X_valid_scaled, y_valid))
        model.save(path)
        model = keras.models.load_model(
            path,
            custom_objects={
                "MyL1Regularizer": MyL1Regularizer,
                "my_positive_weights": self.my_positive_weights,
                "my_glorot_initializer": self.my_glorot_initializer,
                "my_softplus": self.my_softplus
            })

    def my_softplus(self, z):  # return value is just tf.nn.softplus(z)
        return tf.math.log(tf.exp(z) + 1.0)

    def my_glorot_initializer(self, shape, dtype=tf.float32):
        stddev = tf.sqrt(2. / (shape[0] + shape[1]))
        return tf.random.normal(shape, stddev=stddev, dtype=dtype)

    def my_l1_regularizer(self, weights):
        return tf.reduce_sum(tf.abs(0.01 * weights))

    def my_positive_weights(self, weights):  # return value is just tf.nn.relu(weights)
        return tf.where(weights < 0., tf.zeros_like(weights), weights)

    def usingTfLikeNumpy(self):
        print("ok")
        # consts operation
        t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
        print(t)  # as matrix
        print(t.shape)  # as matrix
        print(t.dtype)  # as matrix
        print(tf.constant(32), "\n")

        # indexing
        print(t[:, 1:])
        print(t[..., 1, tf.newaxis])
        print(t, "\n")  # as matrix

        # all sorts of tensor
        print(t + 10)
        print(tf.square(t))
        print(t @ tf.transpose(t), "\n")

        # keras low level api
        K = keras.backend
        print(K.square(K.transpose(t)), "\n")

        # type conversions
        # they are generally hurt performance, can go unnoticed when they are done automatically
        # print(tf.constant(2.) + tf.constant(2))  # in tensorflow it is an error
        # print(tf.constant(2.) + tf.constant(2, dtype=tf.float64))  # in tensorflow it is an error
        # if we really need we can use tf.cast
        t2 = tf.constant(40., dtype=tf.float64)
        print(tf.constant(2.0) + tf.cast(t2, tf.float32), "\n")

        # variable
        v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
        print(v)

    def huber_fn(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < 1
        squared_loss = tf.square(error) / 2
        linear_loss = tf.abs(error) - 0.5
        return tf.where(is_small_error, squared_loss, linear_loss)

    def customLossFunction(self):
        housing = fetch_california_housing()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        input_shape = X_train.shape[1:]

        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                               input_shape=input_shape),
            keras.layers.Dense(1),
        ])
        model.compile(loss=self.huber_fn, optimizer="nadam", metrics=["mae"])
        model.fit(X_train_scaled, y_train, epochs=2,
                  validation_data=(X_valid_scaled, y_valid))

        plt.figure(figsize=(8, 3.5))
        z = np.linspace(-4, 4, 200)
        plt.plot(z, self.huber_fn(0, z), "b-", linewidth=2, label="huber($z$)")
        plt.plot(z, z ** 2 / 2, "b:", linewidth=1, label=r"$\frac{1}{2}z^2$")
        plt.plot([-1, -1], [0, self.huber_fn(0., -1.)], "r--")
        plt.plot([1, 1], [0, self.huber_fn(0., 1.)], "r--")
        plt.gca().axhline(y=0, color='k')
        plt.gca().axvline(x=0, color='k')
        plt.axis([-4, 4, 0, 4])
        plt.grid(True)
        plt.xlabel("$z$")
        plt.legend(fontsize=14)
        plt.title("Huber loss", fontsize=14)
        plt.show()


class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))

    def get_config(self):
        return {"factor": self.factor}


def program1():
    TrainingExample()


if __name__ == '__main__':
    program1()
