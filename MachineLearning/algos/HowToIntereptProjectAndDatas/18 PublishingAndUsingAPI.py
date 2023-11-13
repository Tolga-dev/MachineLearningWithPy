# training and deploying tensorflow models at scale

# deploying a model to a mobile or embedded device

# traning models acroos multiple devices
# model parallelism
#
# data parallelism


import os
import shutil
import joblib
import keras
import numpy as np
import tensorflow as tf
import json
import requests
from tensorflow_serving.apis import predict_pb2
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

train_model_cache_download = joblib.Memory('./tmp/ReinforcementLearning/train_model_cache_download')


@train_model_cache_download.cache
def getDataImdbFrom():
    pass


class TrainingExercises:
    def __init__(self):
        print("ok1")
        # creating a temp model
        (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
        X_train_full = X_train_full[..., np.newaxis].astype(np.float32) / 255.
        X_test = X_test[..., np.newaxis].astype(np.float32) / 255.
        X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
        y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
        X_new = X_test[:3]

        # np.random.seed(42)
        # tf.random.set_seed(42)
        #
        # model = keras.models.Sequential([
        #     keras.layers.Flatten(input_shape=[28, 28, 1]),
        #     keras.layers.Dense(100, activation="relu"),
        #     keras.layers.Dense(10, activation="softmax")
        # ])
        # model.compile(loss="sparse_categorical_crossentropy",
        #               optimizer=keras.optimizers.SGD(learning_rate=1e-2),
        #               metrics=["accuracy"])
        # model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))
        # model_version = "0001"
        # model_name = "my_mnist_model"
        # model_path = os.path.join(model_name, model_version)
        #
        # model_path = "models/my_mnist_model/0001"
        # # print(np.round(model.predict(X_new), 2))
        # print(model_path)

        # saving model
        # shutil.rmtree(model_name)
        # tf.saved_model.save(model, model_path)
        # for root, dirs, files in os.walk(model_name):
        #     indent = '    ' * root.count(os.sep)
        #     print('{}{}/'.format(indent, os.path.basename(root)))
        #     for filename in files:
        #         print('{}{}'.format(indent + '    ', filename))

        # saved_model_cli show --dir my_mnist_model/0001 --all

        # np.save("tmp/my_mnist_tests.npy", X_new)
        #
        # input_data_json = json.dumps({
        #     "signature_name": "serving_default",
        #     "instances": X_new.tolist(),
        # })
        # print(input_data_json)
        #
        # SERVER_URL = 'http://localhost:8501/v1/models/my_mnist_model:predict'
        # response = requests.post(SERVER_URL, data=input_data_json)
        # response.raise_for_status()  # raise an exception in case of error
        # response = response.json()
        # y_proba = np.array(response["predictions"])
        # print(y_proba.round(2))

        # querying tf serving through the gRPC api


def program1():
    TrainingExercises()


if __name__ == '__main__':
    program1()
