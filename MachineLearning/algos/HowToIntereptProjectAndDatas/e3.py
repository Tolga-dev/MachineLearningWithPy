# Support Vector Machines (SVM) (well suited for classifications of complex small - medium-sized datesets)

# linear svm classifications
# svm classifier as fitting the widest possible street between the classes = large margin classification

# if we impose that all instance must be in same and near on a side, ths is called hard margin classification.
# this method works if data is linearly separable
#






import joblib as joblib

train_model_cache_voting_classifier = joblib.Memory('./tmp/TrainingModelsCache/train_model_cache_voting_classifier')


@train_model_cache_voting_classifier.cache
def GeneralVotingClassifier():
    pass


class VotingClassifier:
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
