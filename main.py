import joblib

train_model_cache_download = joblib.Memory('./tmp/ReinforcementLearning/train_model_cache_download')


@train_model_cache_download.cache
def getDataImdbFrom():
    pass


class TrainingExercises:
    def __init__(self):
        print("ok1")


def program1():
    TrainingExercises()


if __name__ == '__main__':
    program1()
