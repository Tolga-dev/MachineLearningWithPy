

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



def program1():
    TrainingExercises()


if __name__ == '__main__':
    program1()
