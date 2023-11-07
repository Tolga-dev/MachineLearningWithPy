import sys
import tarfile
import urllib.request
import pandas as pd
from pandas.plotting import scatter_matrix

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit

assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
from zlib import crc32
# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


class LearningSet1:
    def __init__(self):
        pass

    def load_housing_data(self, housing_path=HOUSING_PATH):
        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)

    def load_housing_data_info(self):
        return self.load_housing_data().info()

    def get_attribute_data(self, attribute):
        return self.load_housing_data()[attribute].value_counts()

    def get_describe(self):
        return self.load_housing_data().describe()

    def get_plot_hist(self, bins=50, figsize=(20, 15)):
        self.load_housing_data().hist(bins=bins, figsize=figsize)
        plt.show()

    def get_plot_hist_by_data(self, data, bins=50, figsize=(20, 15)):
        data.hist(bins=bins, figsize=figsize)
        plt.show()

    def split_train_test(self, data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    def test_set_check(self, identifier, test_ratio):
        return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32

    def split_train_test_by_id(self, data, test_ratio, id_column):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: self.test_set_check(id_, test_ratio))
        return data.loc[~in_test_set], data.loc[in_test_set]

    def scatter_matrix_from_attribute(self, housing, figsize=(12, 8)):
        scatter_matrix(housing, figsize=figsize)
        plt.show()




import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from MachineLearning.algos.HowToIntereptProjectAndDatas.ex1 import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from MachineLearning.algos.HowToIntereptProjectAndDatas.transformers import CombinedAttributesAdder
class LearningSet2:


    def __init__(self):
        l = LearningSet1()
        house_data = l.load_housing_data()
        housing_with_id = house_data.reset_index()
        train_set, test_set = l.split_train_test_by_id(housing_with_id, 0.2, "index")

        house_data["income_cat"] = \
            pd.cut(house_data["median_income"],
                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        strat_train_set = None
        strat_test_set = None

        for train_index, test_index in split.split(house_data, house_data["income_cat"]):
            strat_train_set = house_data.loc[train_index]
            strat_test_set = house_data.loc[test_index]

        # print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
        # print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))

        # for set_ in (strat_train_set, strat_test_set): # dropping
        #     set_.drop("income_cat", axis=1, inplace=True)
        housing = strat_train_set.copy()
        # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
        # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        #              s=housing["population"] / 100, label="population", figsize=(10, 7),
        #              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
        #              )
        # plt.legend()
        # plt.show()
        # print(housing.info())
        #  attributes = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
        #    l.scatter_matrix_from_attribute(housing[attributes])
        # housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

        housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
        housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
        housing["population_per_household"] = housing["population"] / housing["households"]

        # plt.show()

        housing = housing.drop('ocean_proximity', axis=1)

        #    corr = housing.corr()
        #   print(corr['median_house_value'].sort_values(ascending=False))

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

        median = housing["total_bedrooms"].median()
        housing["total_bedrooms"].fillna(median, inplace=True)

        imputer = SimpleImputer(strategy="median")
        ordinal_encoder = OrdinalEncoder()

        checkOceanProximity = housing[["ocean_proximity"]]
        housing_cat_encoded = ordinal_encoder.fit_transform(checkOceanProximity)

        #    print(housing_cat_encoded[:10])
        #   print(checkOceanProximity.head(10))
        cat_encoder = OneHotEncoder()
        housing_cat_1hot = cat_encoder.fit_transform(checkOceanProximity)
        # print(housing_cat_1hot.toarray())

        # print(cat_encoder.categories_)

        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
        housing_extra_attribs = attr_adder.transform(housing.values)
        # print(housing_extra_attribs)

        housing_num = housing.drop("ocean_proximity", axis=1)
        imputer.fit(housing_num)

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

        housing_num_tr = num_pipeline.fit_transform(housing_num)

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

        housing_prepared = full_pipeline.fit_transform(housing)
        #    print(housing_prepared.shape)
        # lin_reg = LinearRegression()
        # lin_reg.fit(housing_prepared, housing_labels)
        some_data = housing[:5]
        some_label = housing_labels.iloc[:5]
        some_data_prepared = full_pipeline.transform(some_data)
        #    print("predictions: ", lin_reg.predict(some_data_prepared))
        #   print("labels: ", list(some_label))

        #  buy_house_predictions = lin_reg.predict(housing_prepared)
        # lin_mse = mean_squared_error(housing_labels, buy_house_predictions)
        # lin_mse = np.sqrt(lin_mse)
        # print(lin_mse)
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(housing_prepared, housing_labels)
        buy_house_predictions = tree_reg.predict(housing_prepared)
        tree_mse = mean_squared_error(housing_labels, buy_house_predictions)
        tree_mse = np.sqrt(tree_mse)
        print(tree_mse)

        # scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
        #                          scoring="neg_mean_squared_error", cv=10)
        # tree_rmse_scores = np.sqrt(-scores)
        #
        # print("scores", tree_rmse_scores)
        # print("mean ", tree_rmse_scores.mean())
        # print("standard deviation ", tree_rmse_scores.std())

        param_grid = [
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]

        forest_reg = RandomForestRegressor()
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)
        grid_search.fit(housing_prepared, housing_labels)

        print(grid_search.best_params_)
        print(grid_search.best_estimator_)
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
        final_model = grid_search.best_estimator_
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        X_test_prepared = full_pipeline.transform(X_test)
        final_predictions = final_model.predict(X_test_prepared)
        final_mse = mean_squared_error(y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)  # => evaluates to 47,730.2
        print(final_rmse)

        confidence = 0.95
        squared_errors = (final_predictions - y_test) ** 2
        print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                       loc=squared_errors.mean(),
                                       scale=stats.sem(squared_errors))))

    #    X = imputer.transform(housing_num)
    # print(X)
