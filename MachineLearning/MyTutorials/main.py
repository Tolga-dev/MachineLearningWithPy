import os


PROJECT_ROOT_DIR = ""
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "MachineLearning/data")
os.makedirs(IMAGES_PATH, exist_ok=True)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


class LearningState:
    def __init__(self):
        print("ok")


LearningState()






